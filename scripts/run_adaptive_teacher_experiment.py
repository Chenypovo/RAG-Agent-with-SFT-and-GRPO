"""Run the authorised E1/E2 sequence once, stopping on errors or the 24h gate."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.agent_rl.artifacts import sha256_file, validate_evaluation_inputs
from app.agent_rl.run_journal import atomic_json


def estimate_cost(pilot, *, tasks=1617):
    candidate_seconds = pilot["runtime"]["generation_seconds"] / pilot["candidate_rollouts_generated"]
    times = sorted(pilot["runtime"]["seconds_per_task"].values())
    candidate_times = sorted(pilot["runtime"]["candidate_seconds"])
    p90_candidate_seconds = candidate_times[min(len(candidate_times) - 1, int(len(candidate_times) * 0.9))]
    upper_candidate_seconds = max(candidate_seconds, p90_candidate_seconds)
    expected_teacher_seconds = candidate_seconds * 4 * tasks
    upper_teacher_seconds = upper_candidate_seconds * 4 * tasks
    # Three 1k evaluations at the observed conservative 7B trajectory cost,
    # plus two hours reserved for SFT/GRPO and startup. Report the assumption.
    remaining_workflow_upper_seconds = upper_teacher_seconds + upper_candidate_seconds * 3000 + 7200
    remaining_workflow_expected_seconds = expected_teacher_seconds + candidate_seconds * 3000 + 7200
    return {"pilot_tasks": len(times), "pilot_candidates": pilot["candidate_rollouts_generated"],
        "mean_candidate_seconds": candidate_seconds, "pilot_p90_candidate_seconds": p90_candidate_seconds,
        "full_teacher_expected_hours": expected_teacher_seconds / 3600,
        "full_teacher_conservative_hours": upper_teacher_seconds / 3600,
        "remaining_workflow_conservative_hours": remaining_workflow_upper_seconds / 3600,
        "remaining_workflow_expected_hours": remaining_workflow_expected_seconds / 3600,
        "continue": remaining_workflow_expected_seconds <= 86400,
        "assumptions": "1617 tasks x maximum 4 candidates, without credit for early candidate stop; measured mean and p90 candidate times; three evaluations x 1000 episodes costed at the 7B teacher rate plus 2h SFT/GRPO/startup allowance"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--through", choices=["pilot", "all"], default="all")
    args = parser.parse_args()
    root = Path(args.run_root).resolve()
    workspace = root / "workspace"
    results = root / "results"
    results.mkdir(exist_ok=False)
    logs = results / "logs"; logs.mkdir()
    model_manifest = json.loads((root / "audit/model_download_result.json").read_text())
    if len(model_manifest) != 4 or any(r.get("status") != "verified" for r in model_manifest):
        raise RuntimeError("all four fixed model downloads must be verified first")
    models = {r["repo"].split("/")[-1]: r for r in model_manifest}
    seven = models["Qwen2.5-7B-Instruct"]
    controller = models["Qwen3-1.7B"]
    train = workspace / "data/agent_rl/hotpotqa_train_2k_v3"
    dev = workspace / "data/agent_rl/hotpotqa_validation_1k"
    for directory, task_file, count in [(train, "tasks_train.jsonl", 1617), (dev, "tasks_test.jsonl", 1000)]:
        validate_evaluation_inputs(directory, task_filename=task_file)
        assert sum(bool(line.strip()) for line in (directory / task_file).read_text().splitlines()) == count
    source_hashes = {str(p.relative_to(workspace)): sha256_file(p)
        for directory in ("app", "scripts", "configs/agent_rl")
        for p in sorted((workspace / directory).rglob("*")) if p.is_file() and "__pycache__" not in p.parts}
    atomic_json(results / "source_hashes.json", source_hashes)
    atomic_json(results / "models.json", model_manifest)
    python = str(root / "venv/bin/python")
    env = dict(os.environ, OMP_NUM_THREADS="8", TOKENIZERS_PARALLELISM="false",
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    stages = []

    def run(name, arguments):
        started = time.time()
        stage = {"name": name, "started_unix": started, "command": [python, *map(str, arguments)], "status": "running"}
        atomic_json(results / "status.json", {"stage": stage, "completed_stages": stages})
        with (logs / (name + ".log")).open("x") as handle:
            completed = subprocess.run(stage["command"], cwd=workspace, env=env, stdout=handle, stderr=subprocess.STDOUT)
        stage.update(seconds=time.time() - started, returncode=completed.returncode,
            status="completed" if completed.returncode == 0 else "failed")
        stages.append(stage)
        atomic_json(results / "stages.json", stages)
        atomic_json(results / "status.json", {"stage": stage, "completed_stages": stages})
        print(json.dumps(stage), flush=True)
        if completed.returncode:
            raise RuntimeError(f"stage failed without automatic retry: {name}; inspect log/checkpoint")

    run("real_model_smoke", ["scripts/smoke_agent_blackwell.py", "--models-root", root / "models", "--output", results / "real_model_smoke.json"])
    for name, directory in [("train_index", train), ("dev_index", dev)]:
        run(name, ["scripts/build_agent_rl_hybrid_index.py", "--data-dir", directory,
            "--embedding-model", models["bge-small-en-v1.5"]["path"], "--embedding-device", "cuda",
            "--embedding-batch-size", "64", "--vector-store", "lancedb"])

    teacher_args = ["scripts/build_agent_teacher_rollouts.py", "--data-dir", train, "--partition", "train",
        "--teacher-mode", "llm_best_of_n", "--teacher-model", seven["path"], "--teacher-revision", seven["revision"],
        "--finalizer-model", seven["path"], "--finalizer-revision", seven["revision"], "--dtype", "bfloat16",
        "--max-steps", "5", "--teacher-max-new-tokens", "128", "--finalizer-max-new-tokens", "64",
        "--teacher-temperature", "0.7", "--teacher-top-p", "0.8", "--teacher-top-k", "20", "--seed", "42",
        "--retrieval-backend", "hybrid-rerank", "--retrieval-candidate-top-k", "15", "--reranker-top-k", "6",
        "--retrieval-rrf-k", "60", "--embedding-model", models["bge-small-en-v1.5"]["path"],
        "--embedding-device", "cuda", "--reranker-model", models["bge-reranker-base"]["path"], "--reranker-device", "cuda"]
    run("pilot50", [*teacher_args, "--max-tasks", "50", "--candidates-per-task", "2",
        "--output", results / "pilot50/rollouts.jsonl", "--report", results / "pilot50/report.json",
        "--selection-audit", results / "pilot50/selection.jsonl"])
    pilot = json.loads((results / "pilot50/report.json").read_text())
    estimate = estimate_cost(pilot)
    atomic_json(results / "pilot_cost_gate.json", estimate)
    if not estimate["continue"]:
        atomic_json(results / "status.json", {"status": "paused_24h_cost_gate", "cost": estimate})
        return
    if args.through == "pilot":
        atomic_json(results / "status.json", {"status": "pilot_complete", "cost": estimate})
        return
    run("full_teacher1617", [*teacher_args, "--max-tasks", "1617", "--candidates-per-task", "4",
        "--output", results / "e1_teacher/rollouts.jsonl", "--report", results / "e1_teacher/report.json",
        "--selection-audit", results / "e1_teacher/selection.jsonl"])
    run("sft_data", ["scripts/build_agent_sft_data.py", "--input", results / "e1_teacher/rollouts.jsonl",
        "--output", results / "e1_sft_data/train.jsonl", "--report", results / "e1_sft_data/report.json",
        "--success-metric", "CompleteSentenceEvidence", "--pre-final-tool-repeat", "1"])
    config = "configs/agent_rl/adaptive_teacher_20260905/"
    run("sft", ["scripts/train_agent_sft.py", "--config", config + "sft.json"])
    run("reward_audit", ["scripts/audit_agent_rewards.py", "--config", config + "grpo.json", "--output", results / "reward_audit.json"])
    run("grpo20", ["scripts/train_agent_grpo.py", "--config", config + "grpo.json"])
    comparisons = []
    for name, adapter in [("prompt", None), ("e1_sft", results / "e1_sft/adapter"), ("e2_grpo", results / "e2_grpo/adapter/policy")]:
        report = results / "evaluation" / (name + ".json")
        trajectories = report.with_suffix(".jsonl")
        options = ["scripts/eval_hotpotqa_prompt_policy.py", "--completion-backend", "transformers",
            "--controller-model", controller["path"], "--controller-revision", controller["revision"],
            "--finalizer-model", seven["path"], "--finalizer-revision", seven["revision"],
            "--data-dir", dev, "--partition", "test", "--max-tasks", "1000", "--seed", "42",
            "--dtype", "bfloat16", "--max-steps", "5", "--controller-max-new-tokens", "128",
            "--finalizer-max-new-tokens", "64", "--controller-temperature", "0.7", "--controller-top-p", "0.8", "--controller-top-k", "20",
            "--retrieval-backend", "hybrid-rerank", "--vector-store", "lancedb", "--retrieval-candidate-top-k", "15",
            "--retrieval-rrf-k", "60", "--reranker-top-k", "6", "--embedding-model", models["bge-small-en-v1.5"]["path"],
            "--embedding-device", "cuda", "--reranker-model", models["bge-reranker-base"]["path"], "--reranker-device", "cuda",
            "--output", report, "--trajectories", trajectories]
        if adapter:
            options += ["--controller-adapter", adapter]
        run("eval_" + name, options)
        comparisons += ["--run", name, report, trajectories]
    run("paired_bootstrap", ["scripts/compare_agent_evals.py", *comparisons, "--bootstrap-samples", "2000", "--seed", "42",
        "--output", results / "paired_comparison.json"])
    atomic_json(results / "status.json", {"status": "experiments_complete_reporting_pending", "completed_stages": stages})


if __name__ == "__main__":
    main()
