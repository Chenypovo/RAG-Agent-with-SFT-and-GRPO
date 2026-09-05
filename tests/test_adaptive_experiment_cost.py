import importlib.util
from pathlib import Path


def test_cost_gate_uses_full_1617_by_four_without_reducing_budget():
    path = Path(__file__).resolve().parents[1] / "scripts/run_adaptive_teacher_experiment.py"
    spec = importlib.util.spec_from_file_location("experiment_cost", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    pilot = {"candidate_rollouts_generated": 100, "runtime": {
        "generation_seconds": 500, "candidate_seconds": [5] * 100,
        "seconds_per_task": {str(i): 10 for i in range(50)}}}
    cost = module.estimate_cost(pilot)
    assert cost["full_teacher_expected_hours"] == 1617 * 4 * 5 / 3600
    assert cost["continue"] is True
    pilot["runtime"]["generation_seconds"] = 10000
    pilot["runtime"]["candidate_seconds"] = [100] * 100
    assert module.estimate_cost(pilot)["continue"] is False
