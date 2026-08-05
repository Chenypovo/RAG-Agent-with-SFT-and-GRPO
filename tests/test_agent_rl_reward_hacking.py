import importlib.util
import json
import sys
from pathlib import Path

import pytest

from app.agent_rl.reward_hacking import (
    AUDIT_SCHEMA_VERSION,
    reward_config_from_mapping,
    run_reward_hacking_audit,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "audit_agent_rewards.py"
SPEC = importlib.util.spec_from_file_location("audit_agent_rewards", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
audit_agent_rewards = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit_agent_rewards
SPEC.loader.exec_module(audit_agent_rewards)


@pytest.fixture(scope="module")
def report():
    return run_reward_hacking_audit()


def test_reward_hacking_audit_passes_all_required_scenarios(report):
    assert report["schema_version"] == AUDIT_SCHEMA_VERSION
    assert report["passed"] is True
    assert report["summary"] == {
        "scenario_count": 3,
        "passed_count": 3,
        "failed_count": 0,
    }
    assert all(
        assertion["passed"]
        for scenario in report["scenarios"]
        for assertion in scenario["assertions"]
    )


def test_extra_unique_irrelevant_retrieval_cannot_raise_total_return(report):
    scenario = _scenario(report, "extra_unique_irrelevant_retrieval")
    baseline = scenario["measurements"]["baseline"]
    attacked = scenario["measurements"]["with_extra_retrieval"]

    assert attacked["total_reward"] < baseline["total_reward"]
    assert attacked["coverage"] == baseline["coverage"] == 1.0


def test_correct_answer_without_complete_evidence_has_no_success_reward(report):
    scenario = _scenario(report, "correct_answer_incomplete_evidence")
    episode = scenario["measurements"]["episode"]

    assert episode["answer"] == "Paris"
    assert episode["coverage"] == 0.5
    assert episode["final_reward_breakdown"]["task_success"] == 0.0
    assert episode["final_reward_breakdown"]["premature_final_answer"] < 0.0


def test_duplicate_failure_and_forgery_cannot_raise_coverage(report):
    measurements = _scenario(
        report, "duplicate_failed_and_forged_evidence"
    )["measurements"]

    assert measurements["duplicate"]["coverage_after"] == measurements["duplicate"]["coverage_before"]
    assert measurements["duplicate"]["reward_breakdown"]["duplicate_call"] < 0.0
    assert measurements["failed_retrieval"]["coverage_after"] == 0.0
    assert measurements["forged_action_args"]["coverage_after"] == 0.0
    assert measurements["forged_action_args"]["reward_breakdown"]["invalid_action"] < 0.0
    assert measurements["forged_observation_text"]["coverage_after"] == 0.0


def test_cli_writes_machine_readable_report_and_refuses_overwrite(tmp_path):
    output_path = tmp_path / "audit.json"

    assert audit_agent_rewards.main(["--output", str(output_path)]) == 0
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["passed"] is True
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        audit_agent_rewards.main(["--output", str(output_path)])


def test_cli_audits_exact_reward_weights_from_config(tmp_path):
    weights = {
        "task_success": 2.0,
        "evidence_coverage": 0.6,
        "valid_action_format": 0.0,
        "tool_call_cost": -0.05,
        "invalid_action": -0.4,
        "duplicate_call": -0.3,
        "premature_final_answer": -0.7,
        "budget_exhausted": -0.8,
    }
    config_path = tmp_path / "grpo.json"
    config_path.write_text(json.dumps({"reward": weights}), encoding="utf-8")
    output_path = tmp_path / "formal-audit.json"

    assert audit_agent_rewards.main([
        "--config", str(config_path),
        "--output", str(output_path),
    ]) == 0
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["reward_config"] == weights
    assert written["reward_config_source"]["path"] == str(config_path)
    extra = _scenario(written, "extra_unique_irrelevant_retrieval")["measurements"]
    assert extra["baseline"]["total_reward"] == pytest.approx(2.55)
    assert extra["with_extra_retrieval"]["total_reward"] == pytest.approx(2.5)
    incomplete = _scenario(
        written, "correct_answer_incomplete_evidence"
    )["measurements"]["episode"]
    assert incomplete["total_reward"] == pytest.approx(-0.45)


def test_reward_config_mapping_rejects_unknown_or_non_finite_weights():
    with pytest.raises(ValueError, match="unknown reward config keys"):
        reward_config_from_mapping({"task_success": 1.0, "typo": 1.0})
    with pytest.raises(ValueError, match="finite number"):
        reward_config_from_mapping({"task_success": float("nan")})


def _scenario(report, scenario_id):
    return next(row for row in report["scenarios"] if row["id"] == scenario_id)
