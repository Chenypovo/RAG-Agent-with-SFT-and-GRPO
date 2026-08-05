from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, Mapping, Sequence, Tuple


def group_relative_advantages(
    rewards: Sequence[float],
    *,
    epsilon: float = 1e-4,
) -> Tuple[list[float], Dict[str, float | bool]]:
    """Normalize rewards inside one prompt/task group.

    GRPO needs several rollouts from the same initial state.  A group with no
    reward variance carries no learning signal and therefore receives zero
    advantages instead of amplifying numerical noise.
    """

    if len(rewards) < 2:
        raise ValueError("GRPO requires at least two rewards per group")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be positive and finite")
    values = [float(value) for value in rewards]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("group rewards must be finite")
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    std = math.sqrt(variance)
    zero_variance = std < epsilon
    advantages = (
        [0.0 for _ in values]
        if zero_variance
        else [(value - mean) / (std + epsilon) for value in values]
    )
    return advantages, {
        "mean": mean,
        "std": std,
        "min": min(values),
        "max": max(values),
        "zero_variance": zero_variance,
    }


def grpo_clipped_loss(
    current_log_probs: Any,
    old_log_probs: Any,
    reference_log_probs: Any,
    *,
    advantage: float,
    clip_epsilon: float,
    beta: float,
) -> Tuple[Any, Dict[str, float]]:
    """Return the token-mean clipped GRPO objective and diagnostics.

    The reference policy is the frozen initial SFT adapter, not the base model.
    Inputs are one-dimensional tensors containing only generated action tokens;
    prompt and tool-observation tokens are excluded from the loss.
    """

    import torch

    if not 0 < clip_epsilon < 1:
        raise ValueError("clip_epsilon must be in (0, 1)")
    if not math.isfinite(beta) or beta < 0:
        raise ValueError("beta must be non-negative and finite")
    tensors = (current_log_probs, old_log_probs, reference_log_probs)
    if any(getattr(value, "ndim", None) != 1 for value in tensors):
        raise ValueError("log-probability inputs must be one-dimensional")
    if any(value.numel() == 0 for value in tensors):
        raise ValueError("log-probability inputs must not be empty")
    if current_log_probs.shape != old_log_probs.shape or current_log_probs.shape != reference_log_probs.shape:
        raise ValueError("log-probability inputs must have identical shapes")
    if not math.isfinite(float(advantage)):
        raise ValueError("advantage must be finite")

    if not all(bool(torch.isfinite(value).all()) for value in tensors):
        raise ValueError("log-probability inputs must be finite")
    log_ratio = current_log_probs - old_log_probs
    # Ratios outside this interval are already far beyond the clipping range;
    # bounding the exponent prevents an invalid sample from producing inf/NaN.
    ratio = torch.exp(torch.clamp(log_ratio, min=-20.0, max=20.0))
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
    advantage_tensor = torch.as_tensor(
        float(advantage),
        dtype=current_log_probs.dtype,
        device=current_log_probs.device,
    )
    surrogate = torch.minimum(ratio * advantage_tensor, clipped_ratio * advantage_tensor)
    ref_log_ratio = torch.clamp(
        reference_log_probs - current_log_probs,
        min=-20.0,
        max=20.0,
    )
    # Schulman-style non-negative sampled KL estimator used by GRPO trainers.
    per_token_kl = torch.exp(ref_log_ratio) - ref_log_ratio - 1.0
    loss = (-surrogate + beta * per_token_kl).mean()
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("non-finite GRPO loss")
    with torch.no_grad():
        diagnostics = {
            "loss": float(loss.detach().cpu()),
            "kl": float(per_token_kl.mean().detach().cpu()),
            "clip_fraction": float((ratio != clipped_ratio).float().mean().detach().cpu()),
            "ratio_mean": float(ratio.mean().detach().cpu()),
            "sampled_token_nll": float((-current_log_probs).mean().detach().cpu()),
            "tokens": float(current_log_probs.numel()),
        }
    return loss, diagnostics


def sum_reward_components(
    transitions: Sequence[Mapping[str, Any]],
) -> Dict[str, float]:
    totals: Dict[str, float] = defaultdict(float)
    for transition in transitions:
        breakdown = transition.get("reward_breakdown", {})
        if not isinstance(breakdown, Mapping):
            continue
        for key, value in breakdown.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            number = float(value)
            if math.isfinite(number):
                totals[str(key)] += number
    return dict(sorted(totals.items()))


def mean_dict(rows: Sequence[Mapping[str, float]]) -> Dict[str, float]:
    keys = sorted({str(key) for row in rows for key in row})
    if not rows:
        return {}
    return {
        key: sum(float(row.get(key, 0.0)) for row in rows) / len(rows)
        for key in keys
    }
