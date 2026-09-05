"""Read-only scientific provenance checks after the fixed E1/E2 run."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.agent_rl.artifacts import sha256_file
from app.agent_rl.behaviour_audit import audit_behaviour
from app.agent_rl.evaluation import aggregate_policy_rollouts
from app.agent_rl.policies import PromptOnlyPolicy, _SYSTEM_PROMPT
from app.agent_rl.run_journal import atomic_json, rollout_from_checkpoint
from app.agent_rl.teacher import select_teacher_rollout
from app.agent_rl.tasks import load_tasks
from app.agent_rl.verifiers import verify_task


def read_json(path):
    return json.loads(path.read_text())


def rows(path):
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def audit_teacher(folder, *, expected_tasks, candidate_budget):
    report = read_json(folder / "report.json")
    selected = {r["task_id"]: r for r in rows(folder / "rollouts.jsonl")}
    selection = {r["task_id"]: r for r in rows(folder / "selection.jsonl")}
    assert len(selected) == len(selection) == expected_tasks
    assert report["shared_teacher_finalizer_instance"] is True
    assert report["teacher_model"] == report["finalizer_model"]
    assert report["teacher_resolved_commit"] == report["finalizer_resolved_commit"]
    renderer = PromptOnlyPolicy(lambda _s, _u: "")
    count = 0
    prompt_count = 0
    histogram = Counter()
    paths = sorted((folder / "rollouts.candidates").glob("candidate-*.json"))

    def saved_records():
        nonlocal count
        for path in paths:
            envelope = read_json(path)
            record = envelope["payload"]
            assert hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest() == envelope["sha256"]
            assert record["sequence"] == count
            count += 1
            yield record

    tasks_seen = 0
    for task_index, group in itertools.groupby(saved_records(), key=lambda r: r["task_index"]):
        assert task_index == tasks_seen
        records = list(group)
        assert 1 <= len(records) <= candidate_budget
        candidates = []
        for index, record in enumerate(records):
            assert record["candidate_index"] == index
            rollout = rollout_from_checkpoint(record["rollout"])
            assert rollout.seed == 42 + task_index * candidate_budget + index
            assert 1 <= len(rollout.decisions) <= 5
            for decision, transition in zip(rollout.decisions, rollout.transitions):
                assert decision.system_prompt == _SYSTEM_PROMPT
                assert decision.user_prompt == renderer._render_observation(transition["observation_before"])
                if decision.action and decision.action.get("tool") not in (None, "retrieve_docs"):
                    assert transition["reward_breakdown"].get("invalid_action", 0) < 0
                prompt_count += 1
            candidates.append(rollout)
        chosen = select_teacher_rollout(candidates)
        assert chosen.to_audit_dict() == selection[chosen.task_id]
        assert chosen.selected.to_dict() == selected[chosen.task_id]
        # A clean complete candidate terminates sampling at the endpoint.
        for prior in chosen.to_audit_dict()["candidates"][:-1]:
            assert not (prior["clean"] and prior["verification"]["CompleteSentenceEvidence"] >= 1)
        histogram[len(records)] += 1
        tasks_seen += 1
    assert tasks_seen == expected_tasks and count == report["candidate_rollouts_generated"]
    task_path = Path(report["data_dir"]) / report["config"]["task_filename"]
    gold = {task.task_id: task.gold_evidence_ids for task in load_tasks(task_path)}
    return {"tasks": tasks_seen, "candidate_count": count, "candidate_count_histogram": dict(histogram),
        "candidate_hashes_valid": True, "candidate_limits_and_seed_schedule_valid": True,
        "selection_recomputed_identically": True, "audited_prompts": prompt_count,
        "prompts_equal_observation_only_renderer": True, "shared_teacher_finaliser_verified": True,
        "selected_behaviour_recomputed": audit_behaviour(selected.values(), gold_by_task=gold),
        "all_candidate_behaviour_recomputed": audit_behaviour(
            (row["rollout"] for row in rows(folder / "rollouts.all-candidates.jsonl")), gold_by_task=gold),
        "qualification": "Gold text can occur naturally in retrieved corpus evidence; this check verifies the authorised observation renderer, not a misleading ban on matching answer strings."}


def audit_evaluation(folder, name, report):
    """Reconcile saved episode checkpoints, JSONL, gold verification and reports."""
    journal = folder / (name + ".episodes")
    protocol = read_json(journal / "protocol.json")
    assert hashlib.sha256(json.dumps(protocol["protocol"], sort_keys=True).encode()).hexdigest() == protocol["sha256"]
    assert protocol["protocol"]["prompt_code_sha256"] == report["policy_code_sha256"]
    assert protocol["protocol"]["adapter"] == report["controller_adapter_provenance"]
    task_path = Path(report["data_dir"]) / report["config"]["task_filename"]
    tasks = load_tasks(task_path)
    saved = list(rows(folder / (name + ".jsonl")))
    paths = sorted(journal.glob("candidate-*.json"))
    assert len(tasks) == len(saved) == len(paths) == report["evaluation"]["n_tasks"] == 1000
    assert len({row["task_id"] for row in saved}) == 1000
    renderer = PromptOnlyPolicy(lambda _s, _u: "")
    rollouts = []
    prompt_count = 0
    for index, (path, row, task) in enumerate(zip(paths, saved, tasks)):
        envelope = read_json(path)
        record = envelope["payload"]
        assert hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest() == envelope["sha256"]
        assert record["sequence"] == record["task_index"] == index
        assert record["candidate_index"] == 0
        rollout = rollout_from_checkpoint(record["rollout"])
        assert rollout.to_dict() == row
        assert rollout.task_id == task.task_id and rollout.seed == report["seed"] == 42
        assert 1 <= len(rollout.decisions) == len(rollout.transitions) <= report["max_steps"] == 5
        assert verify_task(task, predicted_answer=rollout.answer, evidence_ids=rollout.evidence_ids).to_dict() == row["verification"]
        for decision, transition in zip(rollout.decisions, rollout.transitions):
            assert decision.system_prompt == _SYSTEM_PROMPT
            assert decision.user_prompt == renderer._render_observation(transition["observation_before"])
            prompt_count += 1
        rollouts.append(rollout)
    assert aggregate_policy_rollouts(rollouts) == report["evaluation"]
    gold = {task.task_id: task.gold_evidence_ids for task in tasks}
    assert audit_behaviour(rollouts, gold_by_task=gold) == report["behaviour"]
    return {"tasks": len(tasks), "audited_prompts": prompt_count,
        "episode_hashes_and_sequences_valid": True, "jsonl_equals_episode_checkpoints": True,
        "same_order_as_fixed_benchmark": True, "seed_and_action_budget_valid": True,
        "prompts_equal_observation_only_renderer": True, "endpoint_verification_recomputed_identically": True,
        "headline_and_behaviour_metrics_recomputed_identically": True,
        "report_sha256": sha256_file(folder / (name + ".json")),
        "trajectories_sha256": sha256_file(folder / (name + ".jsonl")),
        "protocol_sha256": protocol["sha256"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()
    root = Path(args.run_root)
    results = root / "results"
    output = results / "final_integrity_audit.json"
    if output.exists():
        raise FileExistsError(output)
    report = {"pilot": audit_teacher(results / "pilot50", expected_tasks=50, candidate_budget=2),
        "teacher": audit_teacher(results / "e1_teacher", expected_tasks=1617, candidate_budget=4)}
    evaluations = {name: read_json(results / "evaluation" / (name + ".json"))
        for name in ("prompt", "e1_sft", "e2_grpo")}
    report["evaluation_integrity"] = {name: audit_evaluation(results / "evaluation", name, evaluation)
        for name, evaluation in evaluations.items()}
    excluded = {"controller_adapter", "controller_adapter_provenance"}
    common_configs = [{k: v for k, v in row["config"].items() if k not in excluded} for row in evaluations.values()]
    assert all(row == common_configs[0] for row in common_configs)
    for field in ("data_manifest_sha256", "retrieval_artifact_sha256", "policy_code_sha256", "finalizer_resolved_commit"):
        values = [row[field] for row in evaluations.values()]
        assert all(value == values[0] for value in values), field
    task_sets = [{r["task_id"] for r in rows(results / "evaluation" / (name + ".jsonl"))} for name in evaluations]
    assert len(task_sets[0]) == 1000 and all(ids == task_sets[0] for ids in task_sets)
    report["evaluation_comparability"] = {"same_1000_task_ids": True, "same_config_except_adapter": True,
        "same_model_revision_prompt_and_retrieval_hashes": True, "common_config": common_configs[0]}
    workspace = root / "workspace"
    before = read_json(workspace / "configs/agent_rl/qwen3_1.7b_grpo.json")
    after = read_json(workspace / "configs/agent_rl/adaptive_teacher_20260905/grpo.json")
    for field in ("reward", "training", "rollout"):
        assert before[field] == after[field], field
    report["grpo_original_settings_preserved"] = True
    sft_rows = list(rows(results / "e1_sft_data/train.jsonl"))
    task_ids = {row["task_id"] for row in sft_rows}
    kept = [row for row in rows(results / "e1_teacher/rollouts.jsonl") if row["task_id"] in task_ids]
    assert len(kept) == len(task_ids)
    report["sft_kept_trajectory_behaviour"] = audit_behaviour(kept)
    report["sft_task_count"] = len(task_ids)
    report["source_manifest_sha256"] = sha256_file(results / "source_hashes.json")
    report["model_manifest_sha256"] = sha256_file(results / "models.json")
    original_sources = read_json(results / "source_hashes.json")
    changed = {path: {"at_launch": digest, "now": sha256_file(workspace / path)}
        for path, digest in original_sources.items() if sha256_file(workspace / path) != digest}
    assert set(changed).issubset({"app/agent_rl/behaviour_audit.py"})
    report["source_changes_since_launch"] = changed
    report["diagnostic_correction"] = "Unsupported tool names, including a tool named final_answer, are separate from explicit stop actions; retrieval attempts exclude these names. Scientific generation, reward and selection code were unchanged. Original reports are retained."
    report["audit_script_sha256"] = sha256_file(Path(__file__))
    atomic_json(output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
