from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Sequence

from app.agent_rl.rewards import evidence_coverage
from app.agent_rl.tasks import AgentRLTask

RetrieveFn = Callable[[str, int], Sequence[Dict[str, Any]]]


def evaluate_retrieval_baseline(
    tasks: Iterable[AgentRLTask],
    retrieve_fn: RetrieveFn,
    *,
    ks: Sequence[int] = (4, 8, 20),
) -> Dict[str, Any]:
    normalized_ks = sorted(set(int(k) for k in ks))
    if not normalized_ks or normalized_ks[0] <= 0:
        raise ValueError("ks must contain positive integers")

    task_list = list(tasks)
    totals: Dict[int, Dict[str, float]] = {
        k: {
            "sentence_recall": 0.0,
            "complete_sentence": 0.0,
            "document_recall": 0.0,
            "complete_document": 0.0,
        }
        for k in normalized_ks
    }

    for task in task_list:
        ranked = list(retrieve_fn(task.question, normalized_ks[-1]))
        gold_sentences = task.gold_evidence_ids
        gold_documents = tuple(str(x) for x in task.metadata.get("gold_document_ids", []))
        for k in normalized_ks:
            selected = ranked[:k]
            sentence_ids = _ranked_sentence_ids(selected)
            document_ids = _ranked_document_ids(selected)
            sentence_recall = evidence_coverage(gold_sentences, sentence_ids)
            document_recall = evidence_coverage(gold_documents, document_ids)
            totals[k]["sentence_recall"] += sentence_recall
            totals[k]["complete_sentence"] += float(sentence_recall >= 1.0)
            totals[k]["document_recall"] += document_recall
            totals[k]["complete_document"] += float(document_recall >= 1.0)

    denominator = max(len(task_list), 1)
    metrics: Dict[str, float] = {}
    for k in normalized_ks:
        metrics[f"SentenceRecall@{k}"] = totals[k]["sentence_recall"] / denominator
        metrics[f"CompleteSentenceEvidence@{k}"] = totals[k]["complete_sentence"] / denominator
        metrics[f"DocumentRecall@{k}"] = totals[k]["document_recall"] / denominator
        metrics[f"CompleteDocumentEvidence@{k}"] = totals[k]["complete_document"] / denominator
    return {"n_tasks": len(task_list), "ks": normalized_ks, "metrics": metrics}


def _ranked_sentence_ids(items: Sequence[Dict[str, Any]]) -> List[str]:
    output: List[str] = []
    for item in items:
        meta = item.get("metadata", {}) if isinstance(item, dict) else {}
        if not isinstance(meta, dict):
            continue
        doc_id = meta.get("doc_id")
        if doc_id:
            _append_unique(output, str(doc_id))
            continue
        source, chunk_id = meta.get("source"), meta.get("chunk_id")
        if source is not None and chunk_id is not None:
            _append_unique(output, f"{source}#{chunk_id}")
    return output


def _ranked_document_ids(items: Sequence[Dict[str, Any]]) -> List[str]:
    output: List[str] = []
    for item in items:
        meta = item.get("metadata", {}) if isinstance(item, dict) else {}
        if isinstance(meta, dict) and meta.get("source") is not None:
            _append_unique(output, str(meta["source"]))
    return output


def _append_unique(values: List[str], value: str) -> None:
    if value not in values:
        values.append(value)
