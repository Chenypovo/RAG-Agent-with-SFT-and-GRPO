from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ToolEvent:
    step_index: int
    tool: str
    args: Dict[str, Any]
    observation: str
    ok: bool
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_index": self.step_index,
            "tool": self.tool,
            "args": dict(self.args),
            "observation": self.observation,
            "ok": self.ok,
            "data": self.data,
            "error": self.error,
        }


@dataclass
class EnvTransition:
    step_index: int
    observation_before: Dict[str, Any]
    action: Dict[str, Any]
    observation_after: Dict[str, Any]
    reward: float
    reward_breakdown: Dict[str, float]
    terminated: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_index": self.step_index,
            "observation_before": self.observation_before,
            "action": self.action,
            "observation_after": self.observation_after,
            "reward": self.reward,
            "reward_breakdown": dict(self.reward_breakdown),
            "terminated": self.terminated,
        }
