from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence

from app.agent_rl.env import PersonalRAGEnv
from app.agent_rl.tasks import AgentRLTask
from app.agent_rl.verifiers import VerificationResult, verify_task

_RETRIEVED_LINE_RE = re.compile(r"^\[([^\]]+)\]\s*(.+)$")
_CAPITALIZED_PHRASE_RE = re.compile(
    r"\b[A-Z][A-Za-z0-9'().-]*(?:\s+(?:(?:of|the|and|de|van|von)\s+)?[A-Z][A-Za-z0-9'().-]*){0,4}"
)
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9'-]*")
_CUE_RE = re.compile(
    r"(?:also called|known as|named after|married to|wife of|husband of|part of|member of|"
    r"located in|headquartered in|based in|born in|founded by|soluble in)\s+([^.;]+)",
    flags=re.IGNORECASE,
)
_SPLIT_CANDIDATE_RE = re.compile(r",|\band\b|\bor\b", flags=re.IGNORECASE)
_STOPWORDS = {
    "about", "after", "also", "among", "because", "before", "being", "between", "called",
    "could", "does", "from", "have", "into", "named", "only", "other", "their", "there",
    "these", "this", "those", "through", "under", "using", "very", "were", "what", "when",
    "where", "which", "while", "with", "would", "written", "unknown", "retrieved", "chunks",
}


@dataclass(frozen=True)
class ScriptedAgentConfig:
    retrieval_hops: int = 2
    first_hop_k: int = 7
    follow_up_k: int = 1

    def __post_init__(self) -> None:
        if self.retrieval_hops < 1:
            raise ValueError("retrieval_hops must be positive")
        if self.first_hop_k < 1:
            raise ValueError("first_hop_k must be positive")
        if self.follow_up_k < 1:
            raise ValueError("follow_up_k must be positive")

    @property
    def total_retrieval_budget(self) -> int:
        return self.first_hop_k + (self.retrieval_hops - 1) * self.follow_up_k


class ScriptedTwoHopPolicy:
    """A deterministic observation-only controller for a no-GPU baseline."""

    def __init__(self, config: ScriptedAgentConfig | None = None) -> None:
        self.config = config or ScriptedAgentConfig()

    def act(self, observation: Mapping[str, Any]) -> Dict[str, Any]:
        question = str(observation.get("question", "")).strip()
        history = observation.get("history", [])
        retrieval_events = [
            event for event in history
            if isinstance(event, Mapping) and event.get("tool") == "retrieve_docs"
        ] if isinstance(history, list) else []

        if len(retrieval_events) >= self.config.retrieval_hops:
            return {"final_answer": True}
        if not retrieval_events:
            query = question
            top_k = self.config.first_hop_k
        else:
            query = build_follow_up_query(question, retrieval_events[-1])
            top_k = self.config.follow_up_k
        return {
            "tool": "retrieve_docs",
            "args": {"query": query, "k": top_k},
        }


@dataclass(frozen=True)
class ScriptedRollout:
    task_id: str
    actions: Sequence[Dict[str, Any]]
    evidence_ids: Sequence[str]
    total_reward: float
    stop_reason: str
    answer: str
    verification: VerificationResult

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "actions": list(self.actions),
            "evidence_ids": list(self.evidence_ids),
            "total_reward": self.total_reward,
            "stop_reason": self.stop_reason,
            "answer": self.answer,
            "verification": self.verification.to_dict(),
        }


def run_scripted_episode(
    env: PersonalRAGEnv,
    task: AgentRLTask,
    policy: ScriptedTwoHopPolicy,
) -> ScriptedRollout:
    observation = env.reset(task)
    total_reward = 0.0
    actions: List[Dict[str, Any]] = []
    info: Dict[str, Any] = {}

    while not observation["terminated"]:
        action = policy.act(observation)
        actions.append(action)
        observation, reward, _, info = env.step(action)
        total_reward += reward

    answer = str(info.get("answer", ""))
    evidence_ids = tuple(str(value) for value in info.get("evidence_ids", []))
    verification = verify_task(task, predicted_answer=answer, evidence_ids=evidence_ids)
    return ScriptedRollout(
        task_id=task.task_id,
        actions=tuple(actions),
        evidence_ids=evidence_ids,
        total_reward=total_reward,
        stop_reason=str(info.get("stop_reason", "")),
        answer=answer,
        verification=verification,
    )


def build_follow_up_query(question: str, retrieval_event: Mapping[str, Any]) -> str:
    """Extract a likely bridge entity from the visible retrieval observation."""
    observation = str(retrieval_event.get("observation", ""))
    retrieved_lines: List[tuple[str, str]] = []
    for raw_line in observation.splitlines():
        match = _RETRIEVED_LINE_RE.match(raw_line.strip())
        if match:
            retrieved_lines.append((match.group(1).strip(), match.group(2).strip()))

    question_tokens = {token.lower() for token in _WORD_RE.findall(question)}
    candidates: List[tuple[float, int, str]] = []
    seen: set[str] = set()

    def add(candidate: str, score: float, rank: int) -> None:
        cleaned = " ".join(_WORD_RE.findall(candidate)).strip(" -'")
        normalized = cleaned.lower()
        tokens = normalized.split()
        if not cleaned or normalized in seen or len(cleaned) > 80:
            return
        if all(token in question_tokens or token in _STOPWORDS for token in tokens):
            return
        seen.add(normalized)
        candidates.append((score + min(len(tokens), 4), -rank, cleaned))

    for rank, (citation, payload) in enumerate(retrieved_lines):
        source = citation.rsplit("#", 1)[0].rsplit("/", 1)[-1]
        source_title = re.sub(r"-[0-9a-f]{10}$", "", source).replace("_", " ")
        add(source_title, 52.0 - rank * 15.0, rank)
        for cue_match in _CUE_RE.finditer(payload):
            pieces = [piece.strip() for piece in _SPLIT_CANDIDATE_RE.split(cue_match.group(1))]
            for reverse_index, piece in enumerate(reversed(pieces)):
                nested = re.split(
                    r"(?:also called|known as|named after|soluble in)\s+",
                    piece,
                    flags=re.IGNORECASE,
                )[-1]
                add(nested, 100.0 - rank * 15.0 - reverse_index, rank)
        for phrase in _CAPITALIZED_PHRASE_RE.findall(payload):
            add(phrase, 50.1 - rank * 15.0, rank)

    if not candidates:
        for rank, (_, payload) in enumerate(retrieved_lines):
            novel_words = [
                word for word in _WORD_RE.findall(payload)
                if word.lower() not in question_tokens
                and word.lower() not in _STOPWORDS
                and len(word) >= 4
            ]
            for reverse_index, word in enumerate(reversed(novel_words)):
                add(word, 10.0 - reverse_index * 0.01, rank)

    if not candidates:
        return f"{question} supporting evidence"
    candidates.sort(reverse=True)
    return candidates[0][2]
