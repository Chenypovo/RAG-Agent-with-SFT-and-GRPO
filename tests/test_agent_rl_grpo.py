import math

import pytest

from app.agent_rl.grpo import (
    grpo_clipped_loss,
    group_relative_advantages,
    mean_dict,
    sum_reward_components,
)


def test_group_relative_advantages_are_centered_within_task():
    advantages, stats = group_relative_advantages([0.0, 1.0, 2.0, 1.0])

    assert sum(advantages) == pytest.approx(0.0, abs=1e-9)
    assert stats == {
        "mean": 1.0,
        "std": math.sqrt(0.5),
        "min": 0.0,
        "max": 2.0,
        "zero_variance": False,
    }
    assert advantages[2] > advantages[1] > advantages[0]


def test_zero_variance_group_has_no_learning_signal():
    advantages, stats = group_relative_advantages([0.25, 0.25, 0.25, 0.25])

    assert advantages == [0.0, 0.0, 0.0, 0.0]
    assert stats["zero_variance"] is True


def test_group_relative_advantages_reject_invalid_groups():
    with pytest.raises(ValueError, match="at least two"):
        group_relative_advantages([1.0])
    with pytest.raises(ValueError, match="finite"):
        group_relative_advantages([0.0, float("nan")])


def test_clipped_loss_masks_prompts_by_accepting_action_tokens_only():
    torch = pytest.importorskip("torch")
    current = torch.tensor([-0.2, -0.4], requires_grad=True)
    old = torch.tensor([-0.3, -0.3])
    reference = torch.tensor([-0.35, -0.35])

    loss, diagnostics = grpo_clipped_loss(
        current,
        old,
        reference,
        advantage=1.0,
        clip_epsilon=0.2,
        beta=0.04,
    )
    loss.backward()

    assert current.grad is not None
    assert diagnostics["tokens"] == 2.0
    assert diagnostics["kl"] >= 0.0
    assert 0.0 <= diagnostics["clip_fraction"] <= 1.0


def test_clipped_loss_rejects_shape_mismatch():
    torch = pytest.importorskip("torch")
    with pytest.raises(ValueError, match="identical shapes"):
        grpo_clipped_loss(
            torch.tensor([-0.1, -0.2]),
            torch.tensor([-0.1]),
            torch.tensor([-0.1, -0.2]),
            advantage=1.0,
            clip_epsilon=0.2,
            beta=0.0,
        )


def test_clipped_loss_rejects_non_finite_log_probabilities():
    torch = pytest.importorskip("torch")
    with pytest.raises(ValueError, match="must be finite"):
        grpo_clipped_loss(
            torch.tensor([-0.1, float("nan")]),
            torch.tensor([-0.1, -0.2]),
            torch.tensor([-0.1, -0.2]),
            advantage=1.0,
            clip_epsilon=0.2,
            beta=0.04,
        )


def test_negative_advantage_uses_the_clipped_lower_ratio():
    torch = pytest.importorskip("torch")
    current = torch.tensor([math.log(0.5)], requires_grad=True)
    old = torch.tensor([0.0])
    reference = current.detach().clone()

    loss, diagnostics = grpo_clipped_loss(
        current,
        old,
        reference,
        advantage=-1.0,
        clip_epsilon=0.2,
        beta=0.0,
    )

    assert float(loss.detach()) == pytest.approx(0.8)
    assert diagnostics["clip_fraction"] == 1.0


def test_reward_component_aggregation_is_explicit():
    transitions = [
        {"reward_breakdown": {"tool_call_cost": -0.02, "invalid_action": 0.0}},
        {"reward_breakdown": {"task_success": 1.0, "evidence_coverage": 0.3}},
    ]

    assert sum_reward_components(transitions) == {
        "evidence_coverage": 0.3,
        "invalid_action": 0.0,
        "task_success": 1.0,
        "tool_call_cost": -0.02,
    }
    assert mean_dict([{"reward": 1.0}, {"reward": 3.0, "kl": 0.2}]) == {
        "kl": 0.1,
        "reward": 2.0,
    }
