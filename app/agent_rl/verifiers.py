from __future__ import annotations

import re
import string
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, Sequence

from app.agent_rl.rewards import evidence_coverage
from app.agent_rl.tasks import AgentRLTask

_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_hotpot_answer(text: str) -> str:
    """Apply the normalization used by the official HotpotQA answer metrics."""
    lowered = (text or "").lower()
    no_punctuation = "".join(character for character in lowered if character not in string.punctuation)
    no_articles = _ARTICLES_RE.sub(" ", no_punctuation)
    return _WHITESPACE_RE.sub(" ", no_articles).strip()


def answer_exact_match(prediction: str, gold_answers: Sequence[str]) -> float:
    normalized = normalize_hotpot_answer(prediction)
    return float(bool(normalized) and any(
        normalized == normalize_hotpot_answer(answer) for answer in gold_answers
    ))


def answer_f1(prediction: str, gold_answers: Sequence[str]) -> float:
    prediction_tokens = normalize_hotpot_answer(prediction).split()
    if not prediction_tokens:
        return 0.0

    best = 0.0
    for answer in gold_answers:
        gold_tokens = normalize_hotpot_answer(answer).split()
        if not gold_tokens:
            continue
        common = Counter(prediction_tokens) & Counter(gold_tokens)
        overlap = sum(common.values())
        if overlap == 0:
            continue
        precision = overlap / len(prediction_tokens)
        recall = overlap / len(gold_tokens)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def evidence_document_id(evidence_id: str) -> str:
    source, separator, suffix = str(evidence_id).rpartition("#")
    return source if separator and suffix.isdigit() else str(evidence_id)


@dataclass(frozen=True)
class VerificationResult:
    answer_em: float
    answer_f1: float
    sentence_recall: float
    complete_sentence_evidence: float
    document_recall: float
    complete_document_evidence: float
    joint_success: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "AnswerEM": self.answer_em,
            "AnswerF1": self.answer_f1,
            "SentenceRecall": self.sentence_recall,
            "CompleteSentenceEvidence": self.complete_sentence_evidence,
            "DocumentRecall": self.document_recall,
            "CompleteDocumentEvidence": self.complete_document_evidence,
            "JointSuccess": self.joint_success,
        }


def verify_task(
    task: AgentRLTask,
    *,
    predicted_answer: str,
    evidence_ids: Iterable[str],
) -> VerificationResult:
    observed_sentences = set(str(value) for value in evidence_ids)
    observed_documents = {evidence_document_id(value) for value in observed_sentences}
    gold_documents = set(str(value) for value in task.metadata.get("gold_document_ids", []))
    if not gold_documents:
        gold_documents = {evidence_document_id(value) for value in task.gold_evidence_ids}

    sentence_recall = evidence_coverage(task.gold_evidence_ids, observed_sentences)
    document_recall = evidence_coverage(gold_documents, observed_documents)
    exact_match = answer_exact_match(predicted_answer, task.gold_answers)
    return VerificationResult(
        answer_em=exact_match,
        answer_f1=answer_f1(predicted_answer, task.gold_answers),
        sentence_recall=sentence_recall,
        complete_sentence_evidence=float(sentence_recall >= 1.0),
        document_recall=document_recall,
        complete_document_evidence=float(document_recall >= 1.0),
        joint_success=float(exact_match >= 1.0 and sentence_recall >= 1.0),
    )


def aggregate_verifications(results: Iterable[VerificationResult]) -> Dict[str, float]:
    rows = [result.to_dict() for result in results]
    if not rows:
        return {key: 0.0 for key in VerificationResult(0, 0, 0, 0, 0, 0, 0).to_dict()}
    return {
        key: sum(row[key] for row in rows) / len(rows)
        for key in rows[0]
    }
