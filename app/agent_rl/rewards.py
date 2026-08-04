from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from app.agent_rl.tasks import AgentRLTask

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
_NORMALIZE_RE = re.compile(r"[\W_]+", flags=re.UNICODE)


@dataclass(frozen=True)
class RewardConfig:
    task_success: float = 1.0
    evidence_coverage: float = 0.30
    valid_action_format: float = 0.05
    tool_call_cost: float = -0.02
    invalid_action: float = -0.10
    duplicate_call: float = -0.10
    premature_final_answer: float = -0.30
    budget_exhausted: float = -0.30


@dataclass
class RewardBreakdown:
    task_success: float = 0.0
    evidence_coverage: float = 0.0
    valid_action_format: float = 0.0
    tool_call_cost: float = 0.0
    invalid_action: float = 0.0
    duplicate_call: float = 0.0
    premature_final_answer: float = 0.0
    budget_exhausted: float = 0.0

    @property
    def total(self) -> float:
        return sum(self.to_dict().values())

    def to_dict(self) -> dict[str, float]:
        return {
            "task_success": self.task_success,
            "evidence_coverage": self.evidence_coverage,
            "valid_action_format": self.valid_action_format,
            "tool_call_cost": self.tool_call_cost,
            "invalid_action": self.invalid_action,
            "duplicate_call": self.duplicate_call,
            "premature_final_answer": self.premature_final_answer,
            "budget_exhausted": self.budget_exhausted,
        }


def evidence_coverage(gold_ids: Iterable[str], observed_ids: Iterable[str]) -> float:
    gold = set(gold_ids)
    if not gold:
        return 1.0
    return len(gold & set(observed_ids)) / len(gold)


def answer_is_correct(task: AgentRLTask, answer: str) -> bool:
    if task.expected_value is not None:
        tolerance = max(1e-6, abs(task.expected_value) * 1e-3)
        return any(abs(float(x) - task.expected_value) <= tolerance for x in _NUMBER_RE.findall(answer or ""))

    normalized_answer = _normalize(answer)
    return bool(normalized_answer) and any(
        _normalize(gold) == normalized_answer for gold in task.gold_answers
    )


def _normalize(text: str) -> str:
    return _NORMALIZE_RE.sub("", (text or "").casefold())
