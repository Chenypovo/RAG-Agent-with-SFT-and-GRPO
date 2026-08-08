from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class AgentRLTask:
    """A verifier-ready task; policy rollouts are intentionally not stored here."""

    task_id: str
    question: str
    task_type: str = "qa"
    gold_answers: Tuple[str, ...] = ()
    gold_evidence_ids: Tuple[str, ...] = ()
    expected_value: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.task_id.strip():
            raise ValueError("task_id must not be empty")
        if not self.question.strip():
            raise ValueError("question must not be empty")
        if not self.gold_answers and self.expected_value is None:
            raise ValueError("task needs gold_answers or expected_value")

    @classmethod
    def from_dict(cls, row: Dict[str, Any]) -> "AgentRLTask":
        if not isinstance(row, dict):
            raise ValueError("task row must be a JSON object")
        answers = row.get("gold_answers", row.get("answers", []))
        evidence = row.get("gold_evidence_ids", row.get("gold_chunk_ids", []))
        expected = row.get("expected_value")
        return cls(
            task_id=str(row.get("task_id", "")).strip(),
            question=str(row.get("question", "")).strip(),
            task_type=str(row.get("task_type", row.get("type", "qa"))).strip() or "qa",
            gold_answers=tuple(str(x).strip() for x in (answers or []) if str(x).strip()),
            gold_evidence_ids=tuple(str(x).strip() for x in (evidence or []) if str(x).strip()),
            expected_value=float(expected) if expected is not None else None,
            metadata=dict(row.get("metadata") or {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "question": self.question,
            "task_type": self.task_type,
            "gold_answers": list(self.gold_answers),
            "gold_evidence_ids": list(self.gold_evidence_ids),
            "expected_value": self.expected_value,
            "metadata": dict(self.metadata),
        }


def load_tasks(path: str | Path) -> List[AgentRLTask]:
    tasks: List[AgentRLTask] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                tasks.append(AgentRLTask.from_dict(json.loads(line)))
            except Exception as exc:
                raise ValueError(f"invalid task at line {line_number}: {exc}") from exc
    return tasks
