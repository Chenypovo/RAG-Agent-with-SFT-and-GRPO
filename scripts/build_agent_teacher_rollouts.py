"""Build verifier-selected LLM teacher rollouts for controller SFT.

The default path samples several complete, observation-only trajectories from a
frozen teacher model and selects one after endpoint verification.  The previous
fixed-hop scripted teacher remains available only as a reproducibility baseline.
"""

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Iterable

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.agent_rl.retrieval_setup import build_retrieval_registry  # noqa: E402
from app.agent_rl.artifacts import (  # noqa: E402
    dependency_versions,
    ensure_outputs_available,
    sha256_file,
    validate_evaluation_inputs,
)
from app.agent_rl.env import PersonalRAGEnv  # noqa: E402
from app.agent_rl.evaluation import aggregate_policy_rollouts  # noqa: E402
from app.agent_rl.finalizers import (  # noqa: E402
    FROZEN_FINALIZER_VERSION,
    FrozenAnswerFinalizer,
)
from app.agent_rl.hf_backend import TransformersChatBackend  # noqa: E402
from app.agent_rl.rollouts import PolicyRollout, run_policy_episode  # noqa: E402
from app.agent_rl.run_journal import CandidateJournal, atomic_json, rollout_from_checkpoint  # noqa: E402
from app.agent_rl.behaviour_audit import audit_behaviour  # noqa: E402
from app.agent_rl.scripted_agent import ScriptedAgentConfig  # noqa: E402
from app.agent_rl.tasks import load_tasks  # noqa: E402
from app.agent_rl.teacher import (  # noqa: E402
    SCRIPTED_TEACHER_POLICY_VERSION,
    VERIFIER_GUIDED_TEACHER_POLICY_VERSION,
    ScriptedTeacherPolicy,
    VerifierGuidedTeacherPolicy,
    generate_teacher_candidates,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build adaptive best-of-N LLM teacher rollouts for controller SFT; "
            "gold labels are used only after each complete rollout"
        )
    )
    parser.add_argument("--data-dir", default="data/agent_rl/hotpotqa_smoke")
    parser.add_argument(
        "--partition",
        choices=["all", "train", "validation", "test"],
        default="train",
    )
    parser.add_argument(
        "--teacher-mode",
        choices=["llm_best_of_n", "scripted"],
        default="llm_best_of_n",
        help="scripted reproduces the legacy fixed-hop baseline only",
    )
    parser.add_argument("--finalizer-model", required=True)
    parser.add_argument("--finalizer-revision", default="")
    parser.add_argument(
        "--teacher-model",
        default="",
        help="defaults to --finalizer-model so one frozen model can be reused",
    )
    parser.add_argument("--teacher-revision", default="")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
    )
    parser.add_argument("--finalizer-max-new-tokens", type=int, default=64)
    parser.add_argument("--teacher-max-new-tokens", type=int, default=128)
    parser.add_argument("--teacher-temperature", type=float, default=0.7)
    parser.add_argument("--teacher-top-p", type=float, default=0.8)
    parser.add_argument("--teacher-top-k", type=int, default=20)
    parser.add_argument("--candidates-per-task", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--retrieval-top-k", type=int, default=8)
    parser.add_argument("--retrieval-backend", choices=["bm25", "hybrid", "hybrid-rerank"])
    parser.add_argument("--retrieval-candidate-top-k", type=int, default=15)
    parser.add_argument("--reranker-top-k", type=int, default=6)
    parser.add_argument("--retrieval-rrf-k", type=int, default=60)
    parser.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--embedding-device", default="cuda")
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--reranker-model", default="BAAI/bge-reranker-base")
    parser.add_argument("--reranker-device", default="cuda")
    parser.add_argument("--reranker-batch-size", type=int, default=16)
    parser.add_argument("--lancedb-uri-name", default="lancedb")
    parser.add_argument("--lancedb-table", default="chunks")
    parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="")
    parser.add_argument("--report", default="")
    parser.add_argument("--selection-audit", default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true", help="resume the same candidate journal and RNG state")

    # Legacy scripted-teacher options. They are ignored by llm_best_of_n.
    parser.add_argument("--retrieval-hops", type=int, default=2)
    parser.add_argument("--first-hop-k", type=int, default=7)
    parser.add_argument("--follow-up-k", type=int, default=1)
    args = parser.parse_args()
    if args.retrieval_backend is None:
        args.retrieval_backend = "bm25" if args.teacher_mode == "scripted" else "hybrid-rerank"

    if not args.finalizer_model.strip():
        parser.error("finalizer-model must not be empty")
    if (
        args.finalizer_max_new_tokens <= 0
        or args.teacher_max_new_tokens <= 0
        or args.candidates_per_task <= 0
        or args.max_steps <= 1
        or args.retrieval_top_k <= 0
        or args.max_tasks < 0
        or args.retrieval_hops <= 0
        or args.first_hop_k <= 0
        or args.follow_up_k <= 0
    ):
        parser.error(
            "token limits, candidate count, max steps, retrieval hops and k values "
            "must be positive; max steps must exceed 1 and max-tasks cannot be negative"
        )
    if args.teacher_temperature <= 0.0:
        parser.error("teacher-temperature must be positive for best-of-N sampling")
    if not 0.0 < args.teacher_top_p <= 1.0:
        parser.error("teacher-top-p must be in (0, 1]")
    if args.teacher_top_k < 0:
        parser.error("teacher-top-k must be non-negative")
    if args.teacher_mode == "llm_best_of_n":
        if args.candidates_per_task > 4:
            parser.error("adaptive teacher permits at most four candidates")
        if args.max_steps > 5:
            parser.error("adaptive teacher permits at most five actions")
        teacher_model = args.teacher_model.strip() or args.finalizer_model.strip()
        teacher_revision = args.teacher_revision.strip() or args.finalizer_revision.strip()
        if (teacher_model, teacher_revision) != (
            args.finalizer_model.strip(), args.finalizer_revision.strip()
        ):
            parser.error("teacher and finalizer must use the same model path and revision")
    return args


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("teacher generation requires the authorised AutoDL CUDA environment")
    torch.cuda.reset_peak_memory_stats()
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
    selection_path = (
        Path(args.selection_audit)
        if args.selection_audit
        else output_path.with_name(f"{output_path.stem}.selection.jsonl")
    )
    guarded_outputs = [output_path, report_path]
    if args.teacher_mode == "llm_best_of_n":
        guarded_outputs.append(selection_path)
        guarded_outputs.append(output_path.with_name(output_path.stem + ".all-candidates.jsonl"))
    ensure_outputs_available(guarded_outputs, overwrite=args.overwrite)
    provenance = validate_evaluation_inputs(data_dir, task_filename=task_name)

    tasks = load_tasks(data_dir / task_name)
    if args.max_tasks:
        tasks = tasks[: args.max_tasks]

    retrieval_config = {
        "backend": args.retrieval_backend,
        "top_k": args.retrieval_top_k,
        "candidate_top_k": args.retrieval_candidate_top_k,
        "output_top_k": args.reranker_top_k,
        "rrf_k": args.retrieval_rrf_k,
        "embedding_model": args.embedding_model,
        "embedding_device": args.embedding_device,
        "embedding_batch_size": args.embedding_batch_size,
        "reranker_model": args.reranker_model,
        "reranker_device": args.reranker_device,
        "reranker_batch_size": args.reranker_batch_size,
        "lancedb_uri_name": args.lancedb_uri_name,
        "lancedb_table": args.lancedb_table,
        "parent_child_expansion": False,
    }
    registry, retrieval_artifacts = build_retrieval_registry(data_dir, retrieval_config)
    finalizer_revision = args.finalizer_revision.strip() or None
    finalizer_backend = TransformersChatBackend(
        args.finalizer_model.strip(),
        device_map=args.device_map,
        dtype=args.dtype,
        enable_thinking=False,
        seed=args.seed,
        revision=finalizer_revision,
    )
    finalizer_backend.model.requires_grad_(False)
    finalizer = FrozenAnswerFinalizer(
        finalizer_backend.make_complete_fn(
            max_new_tokens=args.finalizer_max_new_tokens,
            temperature=0.0,
        )
    )

    selection_audits = []
    candidate_rollouts_generated = 0
    journal = None
    if args.teacher_mode == "llm_best_of_n":
        teacher_model = args.teacher_model.strip() or args.finalizer_model.strip()
        teacher_revision = args.teacher_revision.strip() or finalizer_revision
        # parse_args enforces identical model identity; never allocate a second 7B.
        teacher_backend = finalizer_backend
        policy = VerifierGuidedTeacherPolicy(
            teacher_backend.make_complete_fn(
                max_new_tokens=args.teacher_max_new_tokens,
                temperature=args.teacher_temperature,
                top_p=args.teacher_top_p,
                top_k=args.teacher_top_k,
            )
        )
        env = PersonalRAGEnv(
            registry=registry,
            finalize_fn=finalizer,
            allowed_tools=("retrieve_docs",),
            max_steps=args.max_steps,
            seed=args.seed,
            env_version="personal-rag-hotpotqa-llm-teacher-v1",
        )
        journal_dir = output_path.with_suffix(".candidates")
        protocol = {
            "args": {k: v for k, v in vars(args).items() if k not in {"resume", "overwrite", "output", "report", "selection_audit"}},
            "inputs": provenance,
            "retrieval": retrieval_artifacts,
            "policy_code": {
                name: sha256_file(Path(PROJECT_ROOT) / name) for name in (
                    "app/agent_rl/policies.py", "app/agent_rl/teacher.py", "app/agent_rl/env.py",
                    "app/agent_rl/finalizers.py", "app/agent_rl/hf_backend.py",
                    "app/agent_rl/retrieval.py", "app/agent_rl/retrieval_setup.py",
                )
            },
            "dependencies": dependency_versions(("torch", "transformers", "lancedb", "rank-bm25")),
        }
        journal = CandidateJournal(journal_dir, protocol, resume=args.resume)
        journal.restore()
        model_setup_seconds = time.perf_counter() - started
        rollouts = []
        for task_index, task in enumerate(tasks):
            task_started = time.perf_counter()
            selection = generate_teacher_candidates(
                env,
                task,
                policy,
                candidates_per_task=args.candidates_per_task,
                seed=args.seed + task_index * args.candidates_per_task,
                existing_candidates=journal.candidates_for(task_index, task.task_id),
                on_candidate=lambda index, candidate, seconds: journal.append(task_index, index, candidate, seconds),
            )
            rollouts.append(selection.selected)
            selection_audits.append(selection.to_audit_dict())
            candidate_rollouts_generated += len(selection.candidates)
            progress = {
                "completed_tasks": task_index + 1, "total_tasks": len(tasks),
                "candidate_count": candidate_rollouts_generated,
                "task_seconds": time.perf_counter() - task_started,
                "elapsed_seconds_this_process": time.perf_counter() - started,
                "generation_seconds": sum(r["runtime_seconds"] for r in journal.records),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            }
            atomic_json(journal_dir / "progress.json", progress)
            # Selection is independently reviewable before the full run ends.
            atomic_json(journal_dir / f"selected-{task_index:05d}.json", selection.to_audit_dict())
            print(json.dumps(progress), flush=True)
        policy_version = VERIFIER_GUIDED_TEACHER_POLICY_VERSION
        agent_name = "VerifierGuidedTeacherPolicy"
        teacher_resolved_commit = teacher_backend.resolved_commit
    else:
        scripted_config = ScriptedAgentConfig(
            retrieval_hops=args.retrieval_hops,
            first_hop_k=args.first_hop_k,
            follow_up_k=args.follow_up_k,
        )
        policy = ScriptedTeacherPolicy(scripted_config)
        env = PersonalRAGEnv(
            registry=registry,
            finalize_fn=finalizer,
            allowed_tools=("retrieve_docs",),
            max_steps=scripted_config.retrieval_hops + 1,
            seed=args.seed,
            env_version="personal-rag-hotpotqa-scripted-teacher-v1",
        )
        rollouts = [
            run_policy_episode(env, task, policy, seed=args.seed)
            for task in tasks
        ]
        candidate_rollouts_generated = len(rollouts)
        policy_version = SCRIPTED_TEACHER_POLICY_VERSION
        agent_name = "ScriptedTeacherPolicy"
        teacher_model = None
        teacher_revision = None
        teacher_resolved_commit = None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for rollout in rollouts:
            handle.write(
                json.dumps(rollout.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
            )
    selection_artifact = None
    if selection_audits:
        selection_path.parent.mkdir(parents=True, exist_ok=True)
        with selection_path.open("w", encoding="utf-8") as handle:
            for audit in selection_audits:
                handle.write(
                    json.dumps(audit, ensure_ascii=False, sort_keys=True) + "\n"
                )
        selection_artifact = {
            "path": str(selection_path),
            "sha256": sha256_file(selection_path),
        }

    all_candidates = [rollout_from_checkpoint(r["rollout"]) for r in journal.records] if journal else rollouts
    candidate_path = output_path.with_name(output_path.stem + ".all-candidates.jsonl")
    if journal:
        with candidate_path.open("x", encoding="utf-8") as handle:
            for record, rollout in zip(journal.records, all_candidates):
                handle.write(json.dumps({
                    "task_index": record["task_index"], "candidate_index": record["candidate_index"],
                    "runtime_seconds": record["runtime_seconds"], "rollout": rollout.to_dict(),
                }, ensure_ascii=False, sort_keys=True) + "\n")

    report = {
        "schema_version": "agent-teacher-rollouts-report-v2",
        "agent": agent_name,
        "teacher_mode": args.teacher_mode,
        "policy_version": policy_version,
        "policy_code_sha256": sha256_file(Path(PROJECT_ROOT) / "app/agent_rl/policies.py"),
        "teacher_model": teacher_model,
        "teacher_revision": teacher_revision,
        "teacher_resolved_commit": teacher_resolved_commit,
        "finalizer": "FrozenAnswerFinalizer",
        "finalizer_version": FROZEN_FINALIZER_VERSION,
        "finalizer_model": args.finalizer_model.strip(),
        "finalizer_revision": finalizer_revision,
        "finalizer_resolved_commit": finalizer_backend.resolved_commit,
        "thinking_enabled": False,
        "shared_teacher_finalizer_instance": args.teacher_mode == "llm_best_of_n" and teacher_backend is finalizer_backend,
        "retrieval": retrieval_artifacts,
        "device_map": args.device_map,
        "dtype": args.dtype,
        "seed": args.seed,
        "data_dir": str(data_dir),
        "partition": args.partition,
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
        "selection_audit": selection_artifact,
        "evaluation": aggregate_policy_rollouts(rollouts),
        "candidate_rollouts_generated": candidate_rollouts_generated,
        "selected_action_statistics": _action_statistics(rollouts),
        "selected_behaviour": audit_behaviour(rollouts, gold_by_task={t.task_id: t.gold_evidence_ids for t in tasks}),
        "all_candidate_behaviour": audit_behaviour(all_candidates, gold_by_task={t.task_id: t.gold_evidence_ids for t in tasks}),
        "all_candidate_evaluation": aggregate_policy_rollouts(all_candidates),
        "all_candidate_artifact": {"path": str(candidate_path), "sha256": sha256_file(candidate_path)} if journal else None,
        "runtime": {
            "total_seconds_this_process": time.perf_counter() - started,
            "model_setup_seconds": model_setup_seconds if journal else None,
            "generation_seconds": sum(r["runtime_seconds"] for r in journal.records) if journal else None,
            "candidate_seconds": [r["runtime_seconds"] for r in journal.records] if journal else None,
            "peak_allocated_bytes": max([torch.cuda.max_memory_allocated()] + [r["peak_allocated_bytes"] for r in journal.records]) if journal else torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": max([torch.cuda.max_memory_reserved()] + [r["peak_reserved_bytes"] for r in journal.records]) if journal else torch.cuda.max_memory_reserved(),
            "seconds_per_task": {
                t.task_id: sum(r["runtime_seconds"] for r in journal.records if r["task_index"] == i)
                for i, t in enumerate(tasks)
            } if journal else None,
        },
        "candidate_journal": str(journal.directory) if journal else None,
        "mean_candidates_per_task": candidate_rollouts_generated / len(tasks) if tasks else None,
        "candidate_early_stop_task_count": sum(len(a["candidates"]) < args.candidates_per_task for a in selection_audits),
        "config": {
            "data_dir": str(data_dir),
            "partition": args.partition,
            "task_filename": task_name,
            "teacher_mode": args.teacher_mode,
            "teacher_model": teacher_model,
            "teacher_revision": teacher_revision,
            "teacher_max_new_tokens": args.teacher_max_new_tokens,
            "teacher_temperature": args.teacher_temperature,
            "teacher_top_p": args.teacher_top_p,
            "teacher_top_k": args.teacher_top_k,
            "candidates_per_task": (
                args.candidates_per_task if args.teacher_mode == "llm_best_of_n" else 1
            ),
            "early_stop_on_complete_evidence": args.teacher_mode == "llm_best_of_n",
            "finalizer_model": args.finalizer_model.strip(),
            "finalizer_revision": finalizer_revision,
            "finalizer_max_new_tokens": args.finalizer_max_new_tokens,
            "device_map": args.device_map,
            "dtype": args.dtype,
            "max_steps": env.max_steps,
            "retrieval_top_k": args.retrieval_top_k,
            "retrieval": retrieval_config,
            "retrieval_hops": (
                args.retrieval_hops if args.teacher_mode == "scripted" else None
            ),
            "first_hop_k": args.first_hop_k if args.teacher_mode == "scripted" else None,
            "follow_up_k": args.follow_up_k if args.teacher_mode == "scripted" else None,
            "max_tasks": args.max_tasks,
            "evaluated_tasks": len(tasks),
            "allowed_tools": ["retrieve_docs"],
            "seed": args.seed,
            "policy_version": policy_version,
            "finalizer_version": FROZEN_FINALIZER_VERSION,
            "env_version": env.env_version,
        },
        "dependencies": dependency_versions(("rank-bm25", "torch", "transformers")),
        **provenance,
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


def _action_statistics(rollouts: Iterable[PolicyRollout]) -> dict:
    action_counts: Counter[str] = Counter()
    retrieval_call_histogram: Counter[str] = Counter()
    decision_count_histogram: Counter[str] = Counter()
    for rollout in rollouts:
        retrieval_calls = 0
        for decision in rollout.decisions:
            action = decision.action or {}
            name = "invalid_json_action" if decision.parse_error else str(action.get("tool") or "final_answer")
            action_counts[name] += 1
            retrieval_calls += int(name == "retrieve_docs")
        retrieval_call_histogram[str(retrieval_calls)] += 1
        decision_count_histogram[str(len(rollout.decisions))] += 1
    return {
        "action_counts": dict(sorted(action_counts.items())),
        "retrieval_calls_per_episode": dict(sorted(retrieval_call_histogram.items())),
        "decisions_per_episode": dict(sorted(decision_count_histogram.items())),
    }


if __name__ == "__main__":
    main()
