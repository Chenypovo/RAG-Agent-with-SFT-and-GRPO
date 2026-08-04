import json

import pytest

from app.agent_rl.adapters import build_bm25_registry
from app.agent_rl.datasets.hotpotqa import (
    convert_hotpot_rows,
    download_hotpot_rows,
    load_hotpot_parquet,
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

    # Both tasks share a visible Alpha document, so context-document grouping keeps them together.
    assert prepared.tasks[0].metadata["partition"] == prepared.tasks[1].metadata["partition"]


def test_partition_grouping_blocks_gold_document_reused_as_distractor():
    gold_row = hotpot_row("gold", "Shared Title")
    distractor_row = hotpot_row("distractor", "Gamma")
    distractor_row["supporting_facts"]["title"][0] = "Delta"
    distractor_row["context"]["title"][0] = "Delta"
    distractor_row["context"]["sentences"][0] = ["Delta was founded in 1900."]
    distractor_row["context"]["title"][2] = "ＳＨＡＲＥＤ   TITLE"
    distractor_row["context"]["sentences"][2] = [
        "Different content must not produce a different partition key."
    ]

    prepared = convert_hotpot_rows([gold_row, distractor_row])
    gold_task, distractor_task = prepared.tasks

    assert set(gold_task.metadata["gold_document_ids"]).isdisjoint(
        distractor_task.metadata["gold_document_ids"]
    )
    assert set(gold_task.metadata["context_document_keys"]) & set(
        distractor_task.metadata["context_document_keys"]
    )
    assert gold_task.metadata["partition"] == distractor_task.metadata["partition"]


def test_dataset_viewer_pagination_contract():
    source = [hotpot_row("one"), hotpot_row("two"), hotpot_row("three")]
    calls = []

    def fetch(offset, length):
        calls.append((offset, length))
        return {"rows": [{"row_idx": i, "row": row} for i, row in enumerate(source[offset:offset + length], offset)]}

    rows = download_hotpot_rows(limit=3, page_size=2, fetch_page=fetch)
    assert [row["id"] for row in rows] == ["one", "two", "three"]
    assert calls == [(0, 2), (2, 1)]


def test_parquet_loader_reads_only_requested_window(tmp_path):
    parquet = pytest.importorskip("pyarrow.parquet")
    pyarrow = pytest.importorskip("pyarrow")
    path = tmp_path / "rows.parquet"
    parquet.write_table(
        pyarrow.Table.from_pylist([{"id": str(index)} for index in range(8)]),
        path,
    )

    rows = load_hotpot_parquet(path, offset=3, limit=3, batch_size=2)

    assert [row["id"] for row in rows] == ["3", "4", "5"]


def test_invalid_supporting_sentence_is_rejected():
    row = hotpot_row("bad")
    row["supporting_facts"]["sent_id"][0] = 99
    with pytest.raises(ValueError, match="supporting sentence is absent"):
        convert_hotpot_rows([row])


def test_invalid_supporting_sentence_can_be_skipped_and_audited(tmp_path):
    bad = hotpot_row("5a7b23ca554299042af8f703")
    bad["answer"] = "SECRET_ANSWER_MUST_NOT_APPEAR_IN_SKIP_AUDIT"
    bad["context"]["title"][0] = "Minoru Suzuki"
    bad["supporting_facts"]["title"][0] = "Minoru Suzuki"
    bad["supporting_facts"]["sent_id"][0] = 2
    good = hotpot_row("good")

    prepared = convert_hotpot_rows(
        [bad, good],
        invalid_row_policy="skip",
    )
    manifest = write_prepared_hotpotqa(prepared, tmp_path, build_bm25=False)

    assert [task.task_id for task in prepared.tasks] == ["hotpotqa_good"]
    assert prepared.source_rows == 1
    assert prepared.requested_source_rows == 2
    assert manifest["requested_source_rows"] == 2
    assert manifest["source_rows"] == 1
    assert manifest["tasks"] == 1
    assert manifest["skipped_rows"] == 1
    assert manifest["skipped_row_details"] == [{
        "row_id": "5a7b23ca554299042af8f703",
        "reason": "supporting sentence is absent: Minoru Suzuki#2",
    }]
    assert "SECRET_ANSWER" not in json.dumps(manifest)
    assert all(row["title"] != "Minoru Suzuki" for row in prepared.corpus)


def test_unknown_invalid_row_policy_is_rejected():
    with pytest.raises(ValueError, match="unsupported invalid_row_policy"):
        convert_hotpot_rows([hotpot_row("one")], invalid_row_policy="ignore")


def test_official_held_out_rows_can_be_assigned_entirely_to_test():
    prepared = convert_hotpot_rows(
        [hotpot_row("heldout")],
        split="validation",
        partition_mode="all_test",
    )

    assert prepared.split == "validation"
    assert prepared.partition_strategy == "all_test_explicit"
    assert prepared.tasks[0].metadata["source_split"] == "validation"
    assert prepared.tasks[0].metadata["partition"] == "test"


def test_written_dataset_loads_and_bm25_returns_stable_evidence(tmp_path):
    prepared = convert_hotpot_rows([hotpot_row("one")])
    manifest = write_prepared_hotpotqa(prepared, tmp_path, build_bm25=True)

    assert manifest["tasks"] == 1
    assert manifest["requested_source_rows"] == 1
    assert manifest["source_rows"] == 1
    assert manifest["skipped_rows"] == 0
    assert manifest["skipped_row_details"] == []
    assert manifest["corpus_records"] == 4
    assert manifest["bm25_records"] == 4
    assert set(manifest["files"]) == {
        "bm25.json",
        "corpus.jsonl",
        "tasks.jsonl",
        "tasks_test.jsonl",
        "tasks_train.jsonl",
        "tasks_validation.jsonl",
    }
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


def test_skip_bm25_rejects_an_existing_stale_index(tmp_path):
    stale_index = tmp_path / "bm25.json"
    stale_index.write_text('{"stale": true}\n', encoding="utf-8")
    prepared = convert_hotpot_rows([hotpot_row("one")])

    with pytest.raises(FileExistsError, match="could become stale"):
        write_prepared_hotpotqa(prepared, tmp_path, build_bm25=False)

    assert json.loads(stale_index.read_text(encoding="utf-8")) == {"stale": True}
    assert not (tmp_path / "tasks.jsonl").exists()


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
