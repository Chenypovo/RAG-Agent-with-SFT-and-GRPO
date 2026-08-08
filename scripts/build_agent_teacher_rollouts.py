"""Build verifiable scripted teacher rollouts with a frozen HF finalizer."""

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
from app.agent_rl.artifacts import (  # noqa: E402
    dependency_versions,
    ensure_outputs_available,
    validate_evaluation_inputs,
)
from app.agent_rl.env import PersonalRAGEnv  # noqa: E402
from app.agent_rl.evaluation import aggregate_policy_rollouts  # noqa: E402
from app.agent_rl.finalizers import (  # noqa: E402
    FROZEN_FINALIZER_VERSION,
    FrozenAnswerFinalizer,
)
from app.agent_rl.hf_backend import TransformersChatBackend  # noqa: E402
from app.agent_rl.rollouts import run_policy_episode  # noqa: E402
from app.agent_rl.scripted_agent import ScriptedAgentConfig  # noqa: E402
from app.agent_rl.tasks import load_tasks  # noqa: E402
from app.agent_rl.teacher import (  # noqa: E402
    SCRIPTED_TEACHER_POLICY_VERSION,
    ScriptedTeacherPolicy,
)
from app.vectordb.bm25_store import BM25Store  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build observation-only scripted teacher rollouts for controller SFT"
    )
    parser.add_argument("--data-dir", default="data/agent_rl/hotpotqa_smoke")
    parser.add_argument(
        "--partition",
        choices=["all", "train", "validation", "test"],
        default="test",
    )
    parser.add_argument("--finalizer-model", required=True)
    parser.add_argument("--finalizer-revision", default="")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
    )
    parser.add_argument("--finalizer-max-new-tokens", type=int, default=64)
    parser.add_argument("--retrieval-hops", type=int, default=2)
    parser.add_argument("--first-hop-k", type=int, default=7)
    parser.add_argument("--follow-up-k", type=int, default=1)
    parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="")
    parser.add_argument("--report", default="")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.finalizer_model.strip():
        parser.error("finalizer-model must not be empty")
    if (
        args.finalizer_max_new_tokens <= 0
        or args.retrieval_hops <= 0
        or args.first_hop_k <= 0
        or args.follow_up_k <= 0
        or args.max_tasks < 0
    ):
        parser.error(
            "token limit, retrieval-hops and retrieval k values must be positive; "
            "max-tasks cannot be negative"
        )
    return args


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    task_name = "tasks.jsonl" if args.partition == "all" else f"tasks_{args.partition}.jsonl"
    output_path = (
        Path(args.output)
        if args.output
        else data_dir / f"teacher_rollouts_{args.partition}.jsonl"
    )
    report_path = (
        Path(args.report)
        if args.report
        else data_dir / f"teacher_rollouts_{args.partition}.report.json"
    )
    ensure_outputs_available((output_path, report_path), overwrite=args.overwrite)
    provenance = validate_evaluation_inputs(data_dir, task_filename=task_name)

    tasks = load_tasks(data_dir / task_name)
    if args.max_tasks:
        tasks = tasks[: args.max_tasks]

    config = ScriptedAgentConfig(
        retrieval_hops=args.retrieval_hops,
        first_hop_k=args.first_hop_k,
        follow_up_k=args.follow_up_k,
    )
    store = BM25Store.load(str(data_dir / "bm25.json"))
    registry = build_minimal_registry(
        lambda query: store.search(
            query=query,
            top_k=max(config.first_hop_k, config.follow_up_k),
        )
    )
    finalizer_revision = args.finalizer_revision.strip() or None
    finalizer_backend = TransformersChatBackend(
        args.finalizer_model.strip(),
        device_map=args.device_map,
        dtype=args.dtype,
        enable_thinking=False,
        seed=args.seed,
        revision=finalizer_revision,
    )
    finalizer = FrozenAnswerFinalizer(
        finalizer_backend.make_complete_fn(
            max_new_tokens=args.finalizer_max_new_tokens,
            temperature=0.0,
        )
    )
    policy = ScriptedTeacherPolicy(config)
    env = PersonalRAGEnv(
        registry=registry,
        finalize_fn=finalizer,
        allowed_tools=("retrieve_docs",),
        max_steps=config.retrieval_hops + 1,
        seed=args.seed,
        env_version="personal-rag-hotpotqa-teacher-v1",
    )
    rollouts = [
        run_policy_episode(env, task, policy, seed=args.seed)
        for task in tasks
    ]

    report = {
        "schema_version": "agent-teacher-rollouts-report-v1",
        "agent": "ScriptedTeacherPolicy",
        "policy_version": SCRIPTED_TEACHER_POLICY_VERSION,
        "finalizer": "FrozenAnswerFinalizer",
        "finalizer_version": FROZEN_FINALIZER_VERSION,
        "finalizer_model": args.finalizer_model.strip(),
        "finalizer_revision": finalizer_revision,
        "finalizer_resolved_commit": finalizer_backend.resolved_commit,
        "thinking_enabled": False,
        "device_map": args.device_map,
        "dtype": args.dtype,
        "seed": args.seed,
        "data_dir": str(data_dir),
        "partition": args.partition,
        "output": str(output_path),
        "evaluation": aggregate_policy_rollouts(rollouts),
        "config": {
            "data_dir": str(data_dir),
            "partition": args.partition,
            "task_filename": task_name,
            "finalizer_model": args.finalizer_model.strip(),
            "finalizer_revision": finalizer_revision,
            "finalizer_max_new_tokens": args.finalizer_max_new_tokens,
            "thinking_enabled": False,
            "device_map": args.device_map,
            "dtype": args.dtype,
            "retrieval_hops": config.retrieval_hops,
            "first_hop_k": config.first_hop_k,
            "follow_up_k": config.follow_up_k,
            "total_retrieval_budget": config.total_retrieval_budget,
            "max_steps": env.max_steps,
            "max_tasks": args.max_tasks,
            "evaluated_tasks": len(tasks),
            "allowed_tools": ["retrieve_docs"],
            "seed": args.seed,
            "policy_version": SCRIPTED_TEACHER_POLICY_VERSION,
            "finalizer_version": FROZEN_FINALIZER_VERSION,
            "env_version": env.env_version,
        },
        "dependencies": dependency_versions(("rank-bm25", "torch", "transformers")),
        **provenance,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for rollout in rollouts:
            handle.write(
                json.dumps(rollout.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
            )
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
