import importlib.util
import json
import sys
from pathlib import Path

import pytest

from app.agent_rl.comparison import (
    COMPARISON_SCHEMA_VERSION,
    compare_evaluation_runs,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "compare_agent_evals.py"
SPEC = importlib.util.spec_from_file_location("compare_agent_evals", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
compare_agent_evals = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = compare_agent_evals
SPEC.loader.exec_module(compare_agent_evals)


def _write_run(tmp_path, label, rows):
    trajectory_path = tmp_path / f"{label}.jsonl"
    trajectory_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    metrics = {}
    for name in (
        "AnswerEM",
        "AnswerF1",
        "JointSuccess",
        "CompleteSentenceEvidence",
        "CompleteDocumentEvidence",
    ):
        metrics[name] = sum(row["verification"][name] for row in rows) / len(rows)
    metrics.update({
        "MeanToolCalls": sum(
            sum(bool(transition["action"].get("tool")) for transition in row["transitions"])
            for row in rows
        ) / len(rows),
        "MeanEnvironmentReturn": sum(row["total_reward"] for row in rows) / len(rows),
        "BudgetStopRate": sum(row["stop_reason"] == "budget" for row in rows) / len(rows),
        "DuplicateCallRate": 0.0,
        "InvalidActionRate": 0.0,
    })
    report_path = tmp_path / f"{label}.json"
    report_path.write_text(json.dumps({
        "schema_version": "prompt-policy-eval-v1",
        "data_manifest_sha256": "same-manifest",
        "evaluation": {"n_tasks": len(rows), "metrics": metrics},
    }))
    return label, report_path, trajectory_path


def _row(task_id, answer_em, *, tool_calls, total_reward, budget=False):
    verification = {
        "AnswerEM": float(answer_em),
        "AnswerF1": float(answer_em),
        "JointSuccess": float(answer_em),
        "CompleteSentenceEvidence": float(answer_em),
        "CompleteDocumentEvidence": float(answer_em),
    }
    transitions = [
        {
            "action": {"tool": "retrieve_docs", "args": {"query": f"q{index}"}},
            "reward_breakdown": {"duplicate_call": 0.0, "invalid_action": 0.0},
        }
        for index in range(tool_calls)
    ]
    return {
        "task_id": task_id,
        "verification": verification,
        "transitions": transitions,
        "total_reward": total_reward,
        "stop_reason": "budget" if budget else "final_answer",
    }


def test_comparison_outputs_core_deltas_paired_ci_and_counts(tmp_path):
    prompt = _write_run(tmp_path, "prompt", [
        _row("task-a", 1, tool_calls=2, total_reward=1.0),
        _row("task-b", 0, tool_calls=2, total_reward=0.0, budget=True),
        _row("task-c", 1, tool_calls=1, total_reward=1.0),
    ])
    sft = _write_run(tmp_path, "sft", [
        _row("task-c", 0, tool_calls=1, total_reward=0.0),
        _row("task-a", 1, tool_calls=1, total_reward=1.0),
        _row("task-b", 1, tool_calls=3, total_reward=1.0),
    ])

    report = compare_evaluation_runs([prompt, sft], bootstrap_samples=200, seed=7)

    assert report["schema_version"] == COMPARISON_SCHEMA_VERSION
    assert report["comparability"] == {
        "task_ids_identical": True,
        "n_shared_tasks": 3,
        "data_manifest_sha256_identical": True,
    }
    comparison = report["comparisons"][0]
    assert comparison["core_metric_differences"]["AnswerEM"]["difference"] == 0.0
    answer = comparison["paired_metrics"]["AnswerEM"]
    assert answer["paired_counts"] == {
        "candidate_better": 1,
        "baseline_better": 1,
        "ties": 1,
        "candidate_higher": 1,
        "baseline_higher": 1,
    }
    assert answer["bootstrap_95_ci"]["samples"] == 200
    assert answer["bootstrap_95_ci"]["low"] <= 0.0 <= answer["bootstrap_95_ci"]["high"]
    tool_calls = comparison["paired_metrics"]["MeanToolCalls"]
    assert tool_calls["higher_is_better"] is False
    assert tool_calls["paired_counts"]["candidate_better"] == 1
    assert tool_calls["paired_counts"]["baseline_better"] == 1


def test_three_runs_produce_every_pairwise_comparison(tmp_path):
    rows = [_row("task-a", 1, tool_calls=1, total_reward=1.0)]
    runs = [_write_run(tmp_path, label, rows) for label in ("prompt", "sft", "grpo")]

    report = compare_evaluation_runs(runs, bootstrap_samples=100)

    assert [(row["baseline"], row["candidate"]) for row in report["comparisons"]] == [
        ("prompt", "sft"),
        ("prompt", "grpo"),
        ("sft", "grpo"),
    ]


def test_comparison_rejects_non_identical_task_sets(tmp_path):
    prompt = _write_run(
        tmp_path, "prompt", [_row("task-a", 1, tool_calls=1, total_reward=1.0)]
    )
    sft = _write_run(
        tmp_path, "sft", [_row("task-b", 1, tool_calls=1, total_reward=1.0)]
    )

    with pytest.raises(ValueError, match="identical task_id sets"):
        compare_evaluation_runs([prompt, sft], bootstrap_samples=100)


def test_comparison_rejects_duplicate_task_ids(tmp_path):
    duplicated = _row("task-a", 1, tool_calls=1, total_reward=1.0)
    prompt = _write_run(tmp_path, "prompt", [duplicated, duplicated])
    sft = _write_run(
        tmp_path, "sft", [_row("task-a", 1, tool_calls=1, total_reward=1.0)]
    )

    with pytest.raises(ValueError, match="duplicate task_id"):
        compare_evaluation_runs([prompt, sft], bootstrap_samples=100)


def test_cli_writes_machine_readable_report(tmp_path):
    prompt = _write_run(
        tmp_path, "prompt", [_row("task-a", 0, tool_calls=2, total_reward=0.0)]
    )
    sft = _write_run(
        tmp_path, "sft", [_row("task-a", 1, tool_calls=1, total_reward=1.0)]
    )
    output_path = tmp_path / "comparison.json"

    result = compare_agent_evals.main([
        "--run", *(str(value) for value in prompt),
        "--run", *(str(value) for value in sft),
        "--bootstrap-samples", "100",
        "--output", str(output_path),
    ])

    assert result == 0
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["comparisons"][0]["paired_metrics"]["AnswerEM"]["difference"] == 1.0
