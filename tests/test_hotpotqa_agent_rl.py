import json

import pytest

from app.agent_rl.adapters import build_bm25_registry
from app.agent_rl.datasets.hotpotqa import (
    convert_hotpot_rows,
    download_hotpot_rows,
    write_prepared_hotpotqa,
)
from app.agent_rl.env import PersonalRAGEnv
from app.agent_rl.evaluation import evaluate_retrieval_baseline
from app.agent_rl.tasks import load_tasks


def hotpot_row(row_id, second_title="Beta"):
    return {
        "id": row_id,
        "question": f"What links Alpha and {second_title}?",
        "answer": "linked",
        "type": "bridge",
        "level": "medium",
        "supporting_facts": {
            "title": ["Alpha", second_title],
            "sent_id": [0, 0],
        },
        "context": {
            "title": ["Alpha", second_title, "Distractor"],
            "sentences": [
                ["Alpha was founded in 1900.", "It has a second fact."],
                [f"{second_title} is linked to Alpha."],
                ["Nothing relevant is written here."],
            ],
        },
    }


def test_conversion_deduplicates_corpus_and_maps_sentence_evidence():
    prepared = convert_hotpot_rows([hotpot_row("one"), hotpot_row("two", "Gamma")])
    assert len(prepared.tasks) == 2
    assert len(prepared.corpus) == 5  # shared Alpha and Distractor records are deduplicated

    first = prepared.tasks[0]
    assert first.task_id == "hotpotqa_one"
    assert len(first.gold_evidence_ids) == 2
    assert all(evidence_id in {row["doc_id"] for row in prepared.corpus} for evidence_id in first.gold_evidence_ids)

    # Both tasks share a gold Alpha document, so gold-document grouping keeps them together.
    assert prepared.tasks[0].metadata["partition"] == prepared.tasks[1].metadata["partition"]


def test_dataset_viewer_pagination_contract():
    source = [hotpot_row("one"), hotpot_row("two"), hotpot_row("three")]
    calls = []

    def fetch(offset, length):
        calls.append((offset, length))
        return {"rows": [{"row_idx": i, "row": row} for i, row in enumerate(source[offset:offset + length], offset)]}

    rows = download_hotpot_rows(limit=3, page_size=2, fetch_page=fetch)
    assert [row["id"] for row in rows] == ["one", "two", "three"]
    assert calls == [(0, 2), (2, 1)]


def test_invalid_supporting_sentence_is_rejected():
    row = hotpot_row("bad")
    row["supporting_facts"]["sent_id"][0] = 99
    with pytest.raises(ValueError, match="supporting sentence is absent"):
        convert_hotpot_rows([row])


def test_written_dataset_loads_and_bm25_returns_stable_evidence(tmp_path):
    prepared = convert_hotpot_rows([hotpot_row("one")])
    manifest = write_prepared_hotpotqa(prepared, tmp_path, build_bm25=True)

    assert manifest["tasks"] == 1
    assert manifest["corpus_records"] == 4
    assert manifest["bm25_records"] == 4
    assert len(load_tasks(tmp_path / "tasks.jsonl")) == 1
    json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))

    registry = build_bm25_registry(str(tmp_path / "bm25.json"), top_k=2)
    result = registry.dispatch("retrieve_docs", {"query": "Alpha founded 1900"})
    assert result.ok and result.data["chunks"]
    returned_ids = {item["metadata"]["doc_id"] for item in result.data["chunks"]}
    assert prepared.tasks[0].gold_evidence_ids[0] in returned_ids

    env = PersonalRAGEnv(registry=registry, finalize_fn=lambda task, events: "unknown")
    env.reset(prepared.tasks[0])
    observation, _, done, info = env.step({
        "tool": "retrieve_docs",
        "args": {"query": "Alpha founded 1900"},
    })
    json.dumps(observation)
    assert not done and prepared.tasks[0].gold_evidence_ids[0] in info["evidence_ids"]


def test_retrieval_baseline_reports_sentence_and_document_metrics():
    prepared = convert_hotpot_rows([hotpot_row("one")])
    records = [{"metadata": row} for row in prepared.corpus]
    report = evaluate_retrieval_baseline(
        prepared.tasks,
        lambda query, top_k: records[:top_k],
        ks=(1, 4),
    )
    assert report["n_tasks"] == 1
    assert report["metrics"]["SentenceRecall@4"] >= report["metrics"]["SentenceRecall@1"]
    assert report["metrics"]["DocumentRecall@4"] >= report["metrics"]["DocumentRecall@1"]
