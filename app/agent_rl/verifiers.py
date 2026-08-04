from __future__ import annotations

import re
import string
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, Sequence, Tuple

from app.agent_rl.tasks import AgentRLTask

_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")
_SPECIAL_ANSWERS = {"yes", "no", "noanswer"}


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


def answer_precision_recall_f1(
    prediction: str,
    gold_answers: Sequence[str],
) -> Tuple[float, float, float]:
    prediction_tokens = normalize_hotpot_answer(prediction).split()
    if not prediction_tokens:
        return 0.0, 0.0, 0.0

    best = (0.0, 0.0, 0.0)
    for answer in gold_answers:
        gold_tokens = normalize_hotpot_answer(answer).split()
        if not gold_tokens:
            continue
        prediction_special = prediction_tokens[0] if len(prediction_tokens) == 1 else ""
        gold_special = gold_tokens[0] if len(gold_tokens) == 1 else ""
        if (
            prediction_special in _SPECIAL_ANSWERS
            or gold_special in _SPECIAL_ANSWERS
        ) and prediction_tokens != gold_tokens:
            candidate = (0.0, 0.0, 0.0)
        else:
            overlap = sum((Counter(prediction_tokens) & Counter(gold_tokens)).values())
            if overlap == 0:
                candidate = (0.0, 0.0, 0.0)
            else:
                precision = overlap / len(prediction_tokens)
                recall = overlap / len(gold_tokens)
                candidate = (precision, recall, 2 * precision * recall / (precision + recall))
        if candidate[2] > best[2]:
            best = candidate
    return best


def answer_f1(prediction: str, gold_answers: Sequence[str]) -> float:
    return answer_precision_recall_f1(prediction, gold_answers)[2]


def evidence_document_id(evidence_id: str) -> str:
    source, separator, suffix = str(evidence_id).rpartition("#")
    return source if separator and suffix.isdigit() else str(evidence_id)


def set_precision_recall_f1(
    predicted: Iterable[str],
    gold: Iterable[str],
) -> Tuple[float, float, float, float]:
    predicted_set = set(predicted)
    gold_set = set(gold)
    if not predicted_set and not gold_set:
        return 1.0, 1.0, 1.0, 1.0
    overlap = len(predicted_set & gold_set)
    precision = overlap / len(predicted_set) if predicted_set else 0.0
    recall = overlap / len(gold_set) if gold_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    exact_match = float(predicted_set == gold_set)
    return precision, recall, f1, exact_match


@dataclass(frozen=True)
class VerificationResult:
    answer_em: float
    answer_f1: float
    sentence_precision: float
    sentence_recall: float
    sentence_f1: float
    sentence_em: float
    complete_sentence_evidence: float
    document_precision: float
    document_recall: float
    document_f1: float
    document_em: float
    complete_document_evidence: float
    joint_em: float
    joint_success: float
    joint_f1: float

    @classmethod
    def zero(cls) -> "VerificationResult":
        return cls(*(0.0 for _ in range(15)))

    def to_dict(self) -> Dict[str, float]:
        return {
            "AnswerEM": self.answer_em,
            "AnswerF1": self.answer_f1,
            "SentencePrecision": self.sentence_precision,
            "SentenceRecall": self.sentence_recall,
            "SentenceF1": self.sentence_f1,
            "SentenceEM": self.sentence_em,
            "CompleteSentenceEvidence": self.complete_sentence_evidence,
            "DocumentPrecision": self.document_precision,
            "DocumentRecall": self.document_recall,
            "DocumentF1": self.document_f1,
            "DocumentEM": self.document_em,
            "CompleteDocumentEvidence": self.complete_document_evidence,
            "JointEM": self.joint_em,
            "JointF1": self.joint_f1,
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
    gold_sentences = set(task.gold_evidence_ids)
    gold_documents = set(str(value) for value in task.metadata.get("gold_document_ids", []))
    if not gold_documents:
        gold_documents = {evidence_document_id(value) for value in gold_sentences}

    sentence_precision, sentence_recall, sentence_f1, sentence_em = set_precision_recall_f1(
        observed_sentences,
        gold_sentences,
    )
    document_precision, document_recall, document_f1, document_em = set_precision_recall_f1(
        observed_documents,
        gold_documents,
    )
    exact_match = answer_exact_match(predicted_answer, task.gold_answers)
    answer_precision, answer_recall, answer_score = answer_precision_recall_f1(
        predicted_answer,
        task.gold_answers,
    )
    joint_precision = answer_precision * sentence_precision
    joint_recall = answer_recall * sentence_recall
    joint_f1 = (
        2 * joint_precision * joint_recall / (joint_precision + joint_recall)
        if joint_precision + joint_recall
        else 0.0
    )
    return VerificationResult(
        answer_em=exact_match,
        answer_f1=answer_score,
        sentence_precision=sentence_precision,
        sentence_recall=sentence_recall,
        sentence_f1=sentence_f1,
        sentence_em=sentence_em,
        complete_sentence_evidence=float(sentence_recall >= 1.0),
        document_precision=document_precision,
        document_recall=document_recall,
        document_f1=document_f1,
        document_em=document_em,
        complete_document_evidence=float(document_recall >= 1.0),
        joint_em=exact_match * sentence_em,
        # Success is aligned with the environment reward: the answer must be
        # exactly correct and every required sentence must be retrieved. Extra
        # retrieved sentences reduce precision/JointEM but do not turn an
        # otherwise grounded answer into a failure.
        joint_success=exact_match * float(sentence_recall >= 1.0),
        joint_f1=joint_f1,
    )


def aggregate_verifications(results: Iterable[VerificationResult]) -> Dict[str, float]:
    rows = [result.to_dict() for result in results]
    if not rows:
        return {key: 0.0 for key in VerificationResult.zero().to_dict()}
    return {
        key: sum(row[key] for row in rows) / len(rows)
        for key in rows[0]
    }
