"""Describe stopping decisions after fixed retrieval counts, using endpoint-only gold labels."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.agent_rl.artifacts import sha256_file
from app.agent_rl.run_journal import atomic_json
from app.agent_rl.tasks import load_tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()
    results = Path(args.run_root) / "results"
    summary_path = results / "stopping_decision_audit.json"
    records_path = results / "stopping_decisions.jsonl"
    if summary_path.exists() or records_path.exists():
        raise FileExistsError("stopping audit already exists")
    all_records = []
    summaries = {}
    inputs = {}
    for name in ("prompt", "e1_sft", "e2_grpo"):
        path = results / "evaluation" / (name + ".jsonl")
        report = json.loads((path.with_suffix(".json")).read_text())
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        tasks = {task.task_id: task for task in load_tasks(Path(report["data_dir"]) / report["config"]["task_filename"])}
        assert len(rows) == len(tasks) == 1000
        counts = defaultdict(Counter)
        examples = defaultdict(list)
        two_records = []
        for row in sorted(rows, key=lambda item: item["task_id"]):
            gold = set(tasks[row["task_id"]].gold_evidence_ids)
            assert gold
            calls = 0
            checked = set()
            for index, (decision, transition) in enumerate(zip(row["decisions"], row["transitions"])):
                action = decision.get("action") or {}
                if calls in range(1, 5) and calls not in checked:
                    checked.add(calls)
                    evidence = set(transition["observation_before"].get("evidence_ids", []))
                    complete = gold.issubset(evidence)
                    kind = "stop" if action.get("final_answer") else "continue" if action.get("tool") == "retrieve_docs" else "invalid"
                    state = "complete" if complete else "incomplete"
                    bucket = counts[(calls, state)]
                    bucket["opportunities"] += 1
                    bucket[kind] += 1
                    record = {"policy": name, "task_id": row["task_id"], "decision_index_zero_based": index,
                        "prior_retrieval_attempts": calls, "gold_complete_before_decision": complete,
                        "next_decision": kind, "next_action": action, "parse_error": decision.get("parse_error"),
                        "gold_evidence_ids": sorted(gold), "evidence_ids_before_decision": sorted(evidence),
                        "final_complete_evidence": row["verification"]["CompleteSentenceEvidence"],
                        "final_joint_success": row["verification"]["JointSuccess"]}
                    all_records.append(record)
                    if calls == 2:
                        two_records.append(record)
                        category = state + ":" + kind
                        if kind == "continue" and not complete:
                            category += ":joint_success" if record["final_joint_success"] else ":no_joint_success"
                        if len(examples[category]) < 2:
                            examples[category].append({**record, "question": tasks[row["task_id"]].question,
                                "final_answer": row["answer"], "stop_reason": row["stop_reason"],
                                "all_actions": [d.get("action") for d in row["decisions"]],
                                "raw_trajectory_file": str(path)})
                calls += int(action.get("tool") == "retrieve_docs")
        incomplete = [row for row in two_records if not row["gold_complete_before_decision"]]
        original = report["behaviour"]["decision_after_two_calls_with_incomplete_evidence"]
        assert len(incomplete) == original.get("opportunities_with_incomplete_evidence", 0)
        assert sum(row["next_decision"] == "continue" for row in incomplete) == original.get("continue_with_incomplete_evidence", 0)
        assert sum(row["next_decision"] == "stop" for row in incomplete) == original.get("stop_with_incomplete_evidence", 0)
        summaries[name] = {"conditional_decisions": [
            {"prior_retrieval_attempts": calls, "evidence_state": state,
                **{key: count[key] for key in ("opportunities", "stop", "continue", "invalid")},
                "stop_rate": count["stop"] / count["opportunities"]}
            for (calls, state), count in sorted(counts.items())],
            "tasks_with_post_two_decision": len(two_records),
            "tasks_ending_before_post_two_decision": 1000 - len(two_records),
            "examples": dict(examples)}
        inputs[name] = {"path": str(path), "sha256": sha256_file(path)}
    with records_path.open("x") as handle:
        for row in all_records:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    report = {"scope": "Descriptive post-episode labels only; no model calls, interventions or added scientific trials.",
        "definition": "First next decision after each retrieval-attempt count; duplicate attempts count; gold sentence IDs label complete/incomplete evidence after the recorded episode has ended.",
        "limitation": "Conditional groups can contain different tasks across policies. Rates describe observed behaviour and do not identify a causal effect of evidence completeness.",
        "example_selection": "First two task IDs in lexicographic order for each policy and post-two-decision category; successes and failures use separate categories.",
        "inputs": inputs, "policies": summaries, "n_decision_records": len(all_records),
        "records_sha256": sha256_file(records_path), "script_sha256": sha256_file(Path(__file__))}
    atomic_json(summary_path, report)
    print(json.dumps({"summary": str(summary_path), "records": str(records_path), "n_decision_records": len(all_records)}))


if __name__ == "__main__":
    main()
