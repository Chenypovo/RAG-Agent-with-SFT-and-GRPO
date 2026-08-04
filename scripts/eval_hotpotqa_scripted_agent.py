"""Evaluate a deterministic two-hop controller without an answer-generating model."""

import argparse
import json
import os
import sys
from pathlib import Path

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.agent_rl.adapters import build_minimal_registry  # noqa: E402
from app.agent_rl.env import PersonalRAGEnv  # noqa: E402
from app.agent_rl.evaluation import evaluate_retrieval_baseline  # noqa: E402
from app.agent_rl.rewards import RewardConfig  # noqa: E402
from app.agent_rl.scripted_agent import (  # noqa: E402
    ScriptedAgentConfig,
    ScriptedTwoHopPolicy,
    run_scripted_episode,
)
from app.agent_rl.tasks import load_tasks  # noqa: E402
from app.agent_rl.verifiers import aggregate_verifications  # noqa: E402
from app.vectordb.bm25_store import BM25Store  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a scripted HotpotQA retrieval controller")
    parser.add_argument("--data-dir", default="data/agent_rl/hotpotqa_smoke")
    parser.add_argument("--partition", choices=["all", "train", "validation", "test"], default="all")
    parser.add_argument("--retrieval-hops", type=int, default=2)
    parser.add_argument("--first-hop-k", type=int, default=7)
    parser.add_argument("--follow-up-k", type=int, default=1)
    parser.add_argument("--output", default="")
    parser.add_argument("--trajectories", default="")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    task_name = "tasks.jsonl" if args.partition == "all" else f"tasks_{args.partition}.jsonl"
    tasks = load_tasks(data_dir / task_name)
    store = BM25Store.load(str(data_dir / "bm25.json"))
    config = ScriptedAgentConfig(
        retrieval_hops=args.retrieval_hops,
        first_hop_k=args.first_hop_k,
        follow_up_k=args.follow_up_k,
    )
    policy = ScriptedTwoHopPolicy(config)
    registry = build_minimal_registry(
        lambda query: store.search(
            query=query,
            top_k=max(config.first_hop_k, config.follow_up_k),
        )
    )
    env = PersonalRAGEnv(
        registry=registry,
        finalize_fn=lambda task, events: "",
        allowed_tools=("retrieve_docs",),
        max_steps=config.retrieval_hops + 1,
        reward_config=RewardConfig(task_success=0.0, premature_final_answer=0.0),
        env_version="personal-rag-scripted-retrieval-v0",
    )

    rollouts = [run_scripted_episode(env, task, policy) for task in tasks]
    total_budget = config.total_retrieval_budget
    one_shot = evaluate_retrieval_baseline(
        tasks,
        lambda query, top_k: store.search(query=query, top_k=top_k),
        ks=(total_budget,),
    )
    report = {
        "agent": "ScriptedTwoHopPolicy",
        "answer_mode": "abstain_no_generator",
        "data_dir": str(data_dir),
        "n_tasks": len(tasks),
        "partition": args.partition,
        "retrieval_hops": config.retrieval_hops,
        "first_hop_k": config.first_hop_k,
        "follow_up_k": config.follow_up_k,
        "total_retrieval_budget": total_budget,
        "mean_environment_return": (
            sum(rollout.total_reward for rollout in rollouts) / max(len(rollouts), 1)
        ),
        "scripted_metrics": aggregate_verifications(
            rollout.verification for rollout in rollouts
        ),
        "one_shot_same_budget_metrics": one_shot["metrics"],
    }

    output_path = Path(args.output) if args.output else data_dir / f"scripted_agent_{args.partition}.json"
    trajectory_path = (
        Path(args.trajectories)
        if args.trajectories
        else data_dir / f"scripted_trajectories_{args.partition}.jsonl"
    )
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with trajectory_path.open("w", encoding="utf-8") as handle:
        for rollout in rollouts:
            handle.write(json.dumps(rollout.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
