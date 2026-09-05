import numpy as np
import pytest

from scripts.compare_adaptive_behaviour import paired_summary


def test_ratio_bootstrap_preserves_total_decision_denominator():
    # Enumerate every paired two-task resample. A mean of per-task rates
    # would give 0.5, but the official per-decision rate is 1 / 5 = 0.2.
    indices = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]])
    result = paired_summary([1, 0], [1, 4], [0, 0], [1, 4], indices)
    assert result["baseline"] == pytest.approx(0.2)
    assert result["difference"] == pytest.approx(-0.2)
    assert result["ci95_low"] == pytest.approx(-0.94)
    assert result["ci95_high"] == pytest.approx(-0.015)


def test_paired_identical_runs_have_exactly_zero_difference_and_interval():
    indices = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]])
    result = paired_summary([1, 0], [1, 4], [1, 0], [1, 4], indices)
    assert result["difference"] == result["ci95_low"] == result["ci95_high"] == 0
