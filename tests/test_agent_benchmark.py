from pathlib import Path

from app.eval.agent_benchmark import (
    detect_real_environment,
    judge_answer,
    load_benchmark,
    run_benchmark,
    score_tool_calls,
)


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "eval" / "agent_benchmark_v1.json"


def test_frozen_dataset_is_valid_and_has_required_cohorts():
    data, digest = load_benchmark(str(DATASET))
    enabled = [task for task in data["tasks"] if task.get("enabled", True)]
    cohorts = [task["cohort"] for task in enabled]

    assert len(enabled) >= 20
    assert cohorts.count("shared_core") >= 8
    assert cohorts.count("extended_tools") + cohorts.count("robustness") >= 7
    workspace_tasks = [
        task for task in enabled
        if task["category"].startswith("workspace_")
    ]
    assert len(workspace_tasks) >= 3
    assert all(task["cohort"] == "extended_tools" for task in workspace_tasks)
    assert len(digest) == 64


def test_labelled_number_rejects_incidental_or_negated_number():
    rule = {"kind": "labelled_number", "value": 42, "gold": "最终答案：42"}

    assert judge_answer("计算过程出现 42，但最终答案：41", rule)["correct"] is False
    assert judge_answer("答案不是 42", rule)["correct"] is False
    assert judge_answer("18+24=42。\n最终答案：42", rule)["correct"] is True


def test_refusal_requires_marker_and_forbids_factual_guess():
    rule = {
        "kind": "refusal",
        "markers": ["证据不足"],
        "forbidden_values": ["密码是"],
        "gold": "证据不足。",
    }

    assert judge_answer("证据不足。", rule)["correct"] is True
    assert judge_answer("我猜密码是 1234。", rule)["correct"] is False
    assert judge_answer("证据不足，但密码是 1234。", rule)["correct"] is False


def test_refusal_can_reject_fabricated_dates_by_pattern():
    rule = {
        "kind": "refusal",
        "markers": ["没有记录"],
        "forbidden_patterns": [r"\d{4}\s*年\s*\d{1,2}\s*月"],
        "gold": "没有记录。",
    }

    assert judge_answer("没有记录。", rule)["correct"] is True
    assert judge_answer("没有记录，但可能是2026年7月。", rule)["correct"] is False


def test_tool_metrics_use_call_multiplicity_not_only_unique_names():
    score = score_tool_calls(
        ["retrieve_docs", "calculator"],
        {"retrieve_docs": 2, "calculator": 1},
    )

    assert score["precision"] == 1.0
    assert score["recall"] == 2 / 3
    assert score["exact"] is False


def test_scripted_benchmark_is_deterministic_and_separates_cohorts():
    first = run_benchmark(str(DATASET), mode="scripted")
    second = run_benchmark(str(DATASET), mode="scripted")

    assert first == second
    assert first["result_usage"] == "regression_only_not_an_effectiveness_claim"
    assert first["agents"]["baseline"]["main_comparison"]["n_tasks"] == 13
    assert first["agents"]["loop"]["extended_capabilities"]["n_tasks"] == 10
    assert first["agents"]["loop"]["robustness"]["n_tasks"] == 3
    assert first["reserved_tasks"] == []


def test_scripted_loop_passes_strict_regression_and_baseline_exposes_limit():
    report = run_benchmark(str(DATASET), mode="scripted")
    baseline = report["agents"]["baseline"]
    loop = report["agents"]["loop"]

    assert loop["main_comparison"]["task_success"] == 1.0
    assert loop["extended_capabilities"]["task_success"] == 1.0
    assert loop["robustness"]["task_success"] == 1.0
    assert baseline["main_comparison"]["task_success"] < 1.0
    assert baseline["extended_capabilities"]["capability_coverage"] < 1.0

    baseline_stateful = next(
        row for row in baseline["per_task"] if row["task_id"] == "memory_write_then_read_trip"
    )
    loop_stateful = next(
        row for row in loop["per_task"] if row["task_id"] == "memory_write_then_read_trip"
    )
    assert baseline_stateful["success"] is False
    assert baseline_stateful["memory_state"]["correct"] is True
    assert baseline_stateful["evidence"]["correct"] is False
    assert loop_stateful["success"] is True


def test_workspace_extension_is_separate_and_loop_uses_frozen_fixtures():
    report = run_benchmark(str(DATASET), mode="scripted")
    baseline_rows = {
        row["task_id"]: row for row in report["agents"]["baseline"]["per_task"]
    }
    loop_rows = {
        row["task_id"]: row for row in report["agents"]["loop"]["per_task"]
    }
    workspace_ids = {
        "workspace_budget_lookup",
        "workspace_production_owner_locator",
        "workspace_missing_migration_window",
    }

    assert all(loop_rows[task_id]["success"] for task_id in workspace_ids)
    assert all(
        baseline_rows[task_id]["capability_coverage"] == 0.0
        for task_id in workspace_ids
    )
    assert all(
        loop_rows[task_id]["cohort"] == "extended_tools"
        for task_id in workspace_ids
    )


def test_multiple_calculations_order_recovery_and_duplicate_memory_are_audited():
    report = run_benchmark(str(DATASET), mode="scripted")
    rows = {
        row["task_id"]: row for row in report["agents"]["loop"]["per_task"]
    }

    shipping = rows["multi_constraint_shipping_choice"]
    assert shipping["evidence"]["required_calculations"] == [24, 92, 83]
    assert shipping["tools"]["sequence_exact"] is True

    recovery = rows["calculator_error_then_retry"]
    assert recovery["recovery"]["actual_failed_tool_calls"] == 1
    assert recovery["recovery"]["recovered_to_final_answer"] is True
    assert recovery["success"] is True

    duplicate_write = rows["duplicate_memory_write_meeting"]
    assert duplicate_write["memory_state"]["expected_count"] == 1
    assert duplicate_write["memory_state"]["actual_count"] == 1
    assert duplicate_write["success"] is True


def test_real_environment_probe_only_returns_boolean_presence_flags():
    environment = detect_real_environment()

    assert set(environment) == {
        "api_key_configured",
        "base_url_configured",
        "llm_model_configured",
        "ready_for_real_mode",
    }
    assert all(isinstance(value, bool) for value in environment.values())
