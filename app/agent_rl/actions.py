from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

_ACTION_KEYS = {"tool", "args", "final_answer"}


@dataclass(frozen=True)
class AgentAction:
    """Strict controller action used by SFT and RL rollouts.

    Free-form thoughts and plans deliberately stay outside the executable action
    space. Exactly one of ``tool`` or ``final_answer`` must be selected.
    """

    tool: Optional[str] = None
    args: Dict[str, Any] = field(default_factory=dict)
    final_answer: bool = False

    @classmethod
    def from_raw(cls, raw: "AgentAction | Mapping[str, Any]") -> "AgentAction":
        if isinstance(raw, cls):
            return raw
        if not isinstance(raw, Mapping):
            raise ValueError("action must be a JSON object")

        unknown = set(raw) - _ACTION_KEYS
        if unknown:
            raise ValueError(f"unknown action fields: {', '.join(sorted(unknown))}")

        final_answer = raw.get("final_answer") is True
        tool_value = raw.get("tool")
        tool = tool_value.strip() if isinstance(tool_value, str) else None
        if bool(tool) == final_answer:
            raise ValueError("action must choose exactly one tool call or final_answer")

        args = raw.get("args", {})
        if not isinstance(args, Mapping):
            raise ValueError("action args must be a JSON object")
        if final_answer and args:
            raise ValueError("final_answer does not accept args")
        return cls(tool=tool, args=dict(args), final_answer=final_answer)

    def to_dict(self) -> Dict[str, Any]:
        if self.final_answer:
            return {"final_answer": True}
        return {"tool": self.tool, "args": dict(self.args)}
