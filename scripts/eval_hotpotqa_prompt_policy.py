"""Evaluate a prompt-only controller with a frozen answer generator on HotpotQA."""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.agent.llm import make_complete_fn  # noqa: E402
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
from app.agent_rl.policies import PROMPT_ONLY_POLICY_VERSION, PromptOnlyPolicy  # noqa: E402
from app.agent_rl.rollouts import run_policy_episode  # noqa: E402
from app.agent_rl.tasks import load_tasks  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.vectordb.bm25_store import BM25Store  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the prompt-only HotpotQA controller")
    parser.add_argument("--data-dir", default="data/agent_rl/hotpotqa_smoke")
    parser.add_argument("--partition", choices=["all", "train", "validation", "test"], default="test")
    parser.add_argument("--controller-model", default="")
    parser.add_argument(
        "--controller-adapter",
        default="",
        help="Optional local PEFT LoRA adapter directory for the controller only",
    )
    parser.add_argument("--finalizer-model", required=True)
    parser.add_argument("--controller-revision", default="")
    parser.add_argument("--finalizer-revision", default="")
    parser.add_argument("--env-file", default="")
    parser.add_argument("--completion-backend", choices=["openai", "transformers"], default="openai")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--controller-max-new-tokens", type=int, default=256)
    parser.add_argument("--finalizer-max-new-tokens", type=int, default=128)
    parser.add_argument("--controller-temperature", type=float, default=0.7)
    parser.add_argument("--controller-top-p", type=float, default=0.8)
    parser.add_argument("--controller-top-k", type=int, default=20)
    parser.add_argument("--retrieval-top-k", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="")
    parser.add_argument("--trajectories", default="")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if (
        args.retrieval_top_k <= 0
        or args.max_steps <= 0
        or args.max_tasks < 0
        or args.controller_max_new_tokens <= 0
        or args.finalizer_max_new_tokens <= 0
        or args.controller_temperature < 0.0
        or not 0.0 < args.controller_top_p <= 1.0
        or args.controller_top_k < 0
    ):
        parser.error("retrieval-top-k and max-steps must be positive; max-tasks cannot be negative")
    if not args.finalizer_model.strip():
        parser.error("finalizer-model must not be empty")
    if args.completion_backend == "openai" and (
        args.controller_revision.strip()
        or args.finalizer_revision.strip()
        or args.controller_adapter.strip()
    ):
        parser.error("model revisions and controller adapters require the transformers backend")
    if args.env_file:
        env_path = Path(args.env_file)
        if not env_path.is_file():
            parser.error(f"env file does not exist: {env_path}")
        load_dotenv(env_path, override=False)

    settings = get_settings()
    controller_model = args.controller_model or settings.llm_model
    finalizer_model = args.finalizer_model.strip()
    controller_revision = args.controller_revision.strip() or None
    finalizer_revision = args.finalizer_revision.strip() or None
    controller_adapter = args.controller_adapter.strip() or None
    data_dir = Path(args.data_dir)
    task_name = "tasks.jsonl" if args.partition == "all" else f"tasks_{args.partition}.jsonl"
    output_path = Path(args.output) if args.output else data_dir / f"prompt_policy_{args.partition}.json"
    trajectory_path = (
        Path(args.trajectories)
        if args.trajectories
        else data_dir / f"prompt_policy_trajectories_{args.partition}.jsonl"
    )
    ensure_outputs_available((output_path, trajectory_path), overwrite=args.overwrite)
    provenance = validate_evaluation_inputs(data_dir, task_filename=task_name)

    tasks = load_tasks(data_dir / task_name)
    if args.max_tasks:
        tasks = tasks[: args.max_tasks]

    store = BM25Store.load(str(data_dir / "bm25.json"))
    registry = build_minimal_registry(
        lambda query: store.search(query=query, top_k=args.retrieval_top_k)
    )
    if args.completion_backend == "transformers":
        controller_backend = TransformersChatBackend(
            controller_model,
            device_map=args.device_map,
            dtype=args.dtype,
            enable_thinking=False,
            seed=args.seed,
            revision=controller_revision,
            adapter_path=controller_adapter,
        )
        controller_complete = controller_backend.make_complete_fn(
            max_new_tokens=args.controller_max_new_tokens,
            temperature=args.controller_temperature,
            top_p=args.controller_top_p,
            top_k=args.controller_top_k,
        )
        if (
            controller_adapter is None
            and finalizer_model == controller_model
            and finalizer_revision == controller_revision
        ):
            finalizer_backend = controller_backend
        else:
            finalizer_backend = TransformersChatBackend(
                finalizer_model,
                device_map=args.device_map,
                dtype=args.dtype,
                enable_thinking=False,
                seed=args.seed,
                revision=finalizer_revision,
            )
        finalizer_complete = finalizer_backend.make_complete_fn(
            max_new_tokens=args.finalizer_max_new_tokens
        )
    else:
        controller_backend = None
        finalizer_backend = None
        controller_complete = make_complete_fn(
            model=controller_model,
            temperature=args.controller_temperature,
        )
        finalizer_complete = make_complete_fn(model=finalizer_model, temperature=0.0)

    policy = PromptOnlyPolicy(controller_complete)
    finalizer = FrozenAnswerFinalizer(finalizer_complete)
    env = PersonalRAGEnv(
        registry=registry,
        finalize_fn=finalizer,
        allowed_tools=("retrieve_docs",),
        max_steps=args.max_steps,
        seed=args.seed,
        env_version="personal-rag-hotpotqa-prompt-v1",
    )
    rollouts = [run_policy_episode(env, task, policy, seed=args.seed) for task in tasks]

    report = {
        "schema_version": "prompt-policy-eval-v1",
        "agent": "PromptOnlyPolicy",
        "completion_backend": args.completion_backend,
        "controller_model": controller_model,
        "finalizer_model": finalizer_model,
        "controller_revision": controller_revision,
        "finalizer_revision": finalizer_revision,
        "controller_adapter": controller_adapter,
        "controller_adapter_provenance": (
            controller_backend.adapter_provenance if controller_backend is not None else None
        ),
        "controller_resolved_commit": (
            controller_backend.resolved_commit if controller_backend is not None else None
        ),
        "finalizer_resolved_commit": (
            finalizer_backend.resolved_commit if finalizer_backend is not None else None
        ),
        "controller_temperature": args.controller_temperature,
        "controller_top_p": args.controller_top_p,
        "controller_top_k": args.controller_top_k,
        "finalizer_temperature": 0.0,
        "controller_max_new_tokens": args.controller_max_new_tokens,
        "finalizer_max_new_tokens": args.finalizer_max_new_tokens,
        "thinking_enabled": False,
        "device_map": args.device_map if args.completion_backend == "transformers" else None,
        "dtype": args.dtype if args.completion_backend == "transformers" else None,
        "policy_version": PROMPT_ONLY_POLICY_VERSION,
        "finalizer_version": FROZEN_FINALIZER_VERSION,
        "env_version": env.env_version,
        "seed": args.seed,
        "data_dir": str(data_dir),
        "partition": args.partition,
        "retrieval_top_k": args.retrieval_top_k,
        "max_steps": args.max_steps,
        "evaluation": aggregate_policy_rollouts(rollouts),
        "config": {
            "data_dir": str(data_dir),
            "partition": args.partition,
            "task_filename": task_name,
            "completion_backend": args.completion_backend,
            "controller_model": controller_model,
            "finalizer_model": finalizer_model,
            "controller_revision": controller_revision,
            "finalizer_revision": finalizer_revision,
            "controller_adapter": controller_adapter,
            "controller_adapter_provenance": (
                controller_backend.adapter_provenance
                if controller_backend is not None
                else None
            ),
            "controller_temperature": args.controller_temperature,
            "controller_top_p": args.controller_top_p,
            "controller_top_k": args.controller_top_k,
            "finalizer_temperature": 0.0,
            "controller_max_new_tokens": args.controller_max_new_tokens,
            "finalizer_max_new_tokens": args.finalizer_max_new_tokens,
            "thinking_enabled": False,
            "device_map": args.device_map if args.completion_backend == "transformers" else None,
            "dtype": args.dtype if args.completion_backend == "transformers" else None,
            "retrieval_top_k": args.retrieval_top_k,
            "max_steps": args.max_steps,
            "max_tasks": args.max_tasks,
            "evaluated_tasks": len(tasks),
            "seed": args.seed,
            "policy_version": PROMPT_ONLY_POLICY_VERSION,
            "finalizer_version": FROZEN_FINALIZER_VERSION,
            "env_version": env.env_version,
            "allowed_tools": ["retrieve_docs"],
        },
        "dependencies": dependency_versions(
            (
                ("python-dotenv", "rank-bm25", "torch", "transformers", "peft")
                if controller_adapter is not None
                else ("python-dotenv", "rank-bm25", "torch", "transformers")
            )
            if args.completion_backend == "transformers"
            else ("openai", "python-dotenv", "rank-bm25")
        ),
        **provenance,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trajectory_path.parent.mkdir(parents=True, exist_ok=True)
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
