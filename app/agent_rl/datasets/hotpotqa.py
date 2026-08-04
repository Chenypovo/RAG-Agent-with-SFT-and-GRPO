from __future__ import annotations

import hashlib
import json
import re
import time
import unicodedata
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from app.agent_rl.tasks import AgentRLTask

DATASET_NAME = "hotpotqa/hotpot_qa"
DATASET_CONFIG = "distractor"
DATASET_SPLIT = "train"
DATASET_LICENSE = "CC BY-SA 4.0"
VIEWER_BASE_URL = "https://datasets-server.huggingface.co"
_SLUG_RE = re.compile(r"[^a-z0-9]+")

FetchPageFn = Callable[[int, int], Mapping[str, Any]]


@dataclass(frozen=True)
class PreparedHotpotQA:
    tasks: Sequence[AgentRLTask]
    corpus: Sequence[Dict[str, Any]]
    source_rows: int
    dataset: str = DATASET_NAME
    config: str = DATASET_CONFIG
    split: str = DATASET_SPLIT
    offset: int = 0


def fetch_viewer_page(
    offset: int,
    length: int,
    *,
    dataset: str = DATASET_NAME,
    config: str = DATASET_CONFIG,
    split: str = DATASET_SPLIT,
    timeout: float = 60.0,
    retries: int = 3,
) -> Mapping[str, Any]:
    if offset < 0:
        raise ValueError("offset must be non-negative")
    if not 1 <= length <= 100:
        raise ValueError("Dataset Viewer page length must be between 1 and 100")

    query = urlencode({
        "dataset": dataset,
        "config": config,
        "split": split,
        "offset": offset,
        "length": length,
    })
    request = Request(
        f"{VIEWER_BASE_URL}/rows?{query}",
        headers={"User-Agent": "Personal-RAG-AgentRL/0.1"},
    )
    last_error: Optional[Exception] = None
    for attempt in range(retries):
        try:
            with urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("Dataset Viewer returned a non-object payload")
            return payload
        except Exception as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(1.0)
    raise RuntimeError(f"failed to fetch Dataset Viewer page at offset {offset}: {last_error}")


def download_hotpot_rows(
    limit: int,
    *,
    offset: int = 0,
    page_size: int = 100,
    fetch_page: Optional[FetchPageFn] = None,
) -> List[Dict[str, Any]]:
    """Download raw rows using the Dataset Viewer pagination contract."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    if offset < 0:
        raise ValueError("offset must be non-negative")
    if not 1 <= page_size <= 100:
        raise ValueError("page_size must be between 1 and 100")

    fetch = fetch_page or (lambda page_offset, length: fetch_viewer_page(page_offset, length))
    rows: List[Dict[str, Any]] = []
    next_offset = offset
    while len(rows) < limit:
        length = min(page_size, limit - len(rows))
        payload = fetch(next_offset, length)
        page_rows = payload.get("rows", [])
        if not isinstance(page_rows, list):
            raise ValueError("Dataset Viewer payload has invalid rows")
        if not page_rows:
            break
        for wrapped in page_rows:
            raw = wrapped.get("row") if isinstance(wrapped, dict) and "row" in wrapped else wrapped
            if not isinstance(raw, dict):
                raise ValueError("Dataset Viewer row must be a JSON object")
            rows.append(dict(raw))
            if len(rows) == limit:
                break
        next_offset += len(page_rows)
        if len(page_rows) < length:
            break
    return rows


def convert_hotpot_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    dataset: str = DATASET_NAME,
    config: str = DATASET_CONFIG,
    split: str = DATASET_SPLIT,
    offset: int = 0,
) -> PreparedHotpotQA:
    corpus_by_id: Dict[str, Dict[str, Any]] = {}
    tasks: List[AgentRLTask] = []
    seen_task_ids: set[str] = set()
    source_rows = 0

    for raw_row in rows:
        source_rows += 1
        row = _unwrap_row(raw_row)
        row_id = str(row.get("id", "")).strip()
        if not row_id:
            raise ValueError(f"HotpotQA row {source_rows} has no id")
        task_id = f"hotpotqa_{row_id}"
        if task_id in seen_task_ids:
            raise ValueError(f"duplicate HotpotQA task id: {task_id}")
        seen_task_ids.add(task_id)

        titles, sentence_groups = _context_columns(row, row_id)
        title_to_source: Dict[str, str] = {}
        title_to_sentences: Dict[str, Sequence[str]] = {}
        context_document_ids: List[str] = []

        for title, sentences in zip(titles, sentence_groups):
            clean_title = str(title).strip()
            clean_sentences = [str(sentence).strip() for sentence in sentences]
            if not clean_title or not clean_sentences:
                raise ValueError(f"HotpotQA row {row_id} has an empty context document")
            source = _document_source(clean_title, clean_sentences)
            title_to_source[clean_title] = source
            title_to_sentences[clean_title] = clean_sentences
            context_document_ids.append(source)

            for sentence_id, sentence in enumerate(clean_sentences):
                if not sentence:
                    continue
                doc_id = f"{source}#{sentence_id}"
                record = {
                    "doc_id": doc_id,
                    "source": source,
                    "chunk_id": sentence_id,
                    "title": clean_title,
                    "text": f"{clean_title}\n{sentence}",
                    "modality": "text",
                    "file_type": "hotpotqa",
                    "dataset": dataset,
                }
                previous = corpus_by_id.get(doc_id)
                if previous is not None and previous != record:
                    raise ValueError(f"corpus collision for {doc_id}")
                corpus_by_id[doc_id] = record

        support = row.get("supporting_facts")
        if not isinstance(support, Mapping):
            raise ValueError(f"HotpotQA row {row_id} has invalid supporting_facts")
        support_titles = support.get("title", [])
        support_sentence_ids = support.get("sent_id", [])
        if not isinstance(support_titles, list) or not isinstance(support_sentence_ids, list):
            raise ValueError(f"HotpotQA row {row_id} has invalid supporting_facts columns")
        if len(support_titles) != len(support_sentence_ids):
            raise ValueError(f"HotpotQA row {row_id} has mismatched supporting_facts columns")

        evidence_ids: List[str] = []
        gold_document_ids: List[str] = []
        for title_value, sentence_id_value in zip(support_titles, support_sentence_ids):
            title = str(title_value).strip()
            if title not in title_to_source:
                raise ValueError(f"HotpotQA row {row_id} supporting title is absent from context: {title}")
            try:
                sentence_id = int(sentence_id_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"HotpotQA row {row_id} has non-integer supporting sentence id") from exc
            sentences = title_to_sentences[title]
            if sentence_id < 0 or sentence_id >= len(sentences) or not sentences[sentence_id]:
                raise ValueError(
                    f"HotpotQA row {row_id} supporting sentence is absent: {title}#{sentence_id}"
                )
            source = title_to_source[title]
            _append_unique(gold_document_ids, source)
            _append_unique(evidence_ids, f"{source}#{sentence_id}")

        question = str(row.get("question", "")).strip()
        answer = str(row.get("answer", "")).strip()
        task = AgentRLTask(
            task_id=task_id,
            question=question,
            task_type="multihop",
            gold_answers=(answer,),
            gold_evidence_ids=tuple(evidence_ids),
            metadata={
                "source_dataset": dataset,
                "source_config": config,
                "source_split": split,
                "source_row_id": row_id,
                "hotpot_type": str(row.get("type", "")),
                "level": str(row.get("level", "")),
                "context_document_ids": context_document_ids,
                "gold_document_ids": gold_document_ids,
            },
        )
        tasks.append(task)

    partitioned = _assign_gold_document_disjoint_partitions(tasks)
    corpus = [corpus_by_id[key] for key in sorted(corpus_by_id)]
    return PreparedHotpotQA(
        tasks=partitioned,
        corpus=corpus,
        source_rows=source_rows,
        dataset=dataset,
        config=config,
        split=split,
        offset=offset,
    )


def write_prepared_hotpotqa(
    prepared: PreparedHotpotQA,
    output_dir: str | Path,
    *,
    build_bm25: bool = True,
) -> Dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    tasks_path = output / "tasks.jsonl"
    corpus_path = output / "corpus.jsonl"
    _write_jsonl(tasks_path, (task.to_dict() for task in prepared.tasks))
    _write_jsonl(corpus_path, prepared.corpus)

    partition_counts: Dict[str, int] = {}
    for partition in ("train", "validation", "test"):
        selected = [task for task in prepared.tasks if task.metadata.get("partition") == partition]
        partition_counts[partition] = len(selected)
        _write_jsonl(output / f"tasks_{partition}.jsonl", (task.to_dict() for task in selected))

    bm25_records = 0
    if build_bm25:
        from app.vectordb.bm25_store import BM25Store

        store = BM25Store()
        store.build_from_records(prepared.corpus)
        store.save(str(output / "bm25.json"))
        bm25_records = len(store.metadatas)

    manifest = {
        "dataset": prepared.dataset,
        "config": prepared.config,
        "source_split": prepared.split,
        "offset": prepared.offset,
        "source_rows": prepared.source_rows,
        "tasks": len(prepared.tasks),
        "corpus_records": len(prepared.corpus),
        "bm25_records": bm25_records,
        "partition_counts": partition_counts,
        "partition_strategy": "gold_document_components_sha256_80_10_10",
        "evidence_granularity": "sentence",
        "license": DATASET_LICENSE,
        "source_url": f"https://huggingface.co/datasets/{prepared.dataset}",
        "files": {
            "tasks.jsonl": _sha256(tasks_path),
            "corpus.jsonl": _sha256(corpus_path),
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _unwrap_row(raw_row: Mapping[str, Any]) -> Dict[str, Any]:
    if "row" in raw_row:
        row = raw_row["row"]
        if not isinstance(row, dict):
            raise ValueError("wrapped Dataset Viewer row must contain an object")
        return dict(row)
    return dict(raw_row)


def _context_columns(row: Mapping[str, Any], row_id: str) -> tuple[List[Any], List[Any]]:
    context = row.get("context")
    if not isinstance(context, Mapping):
        raise ValueError(f"HotpotQA row {row_id} has invalid context")
    titles = context.get("title", [])
    sentences = context.get("sentences", [])
    if not isinstance(titles, list) or not isinstance(sentences, list) or len(titles) != len(sentences):
        raise ValueError(f"HotpotQA row {row_id} has mismatched context columns")
    if any(not isinstance(group, list) for group in sentences):
        raise ValueError(f"HotpotQA row {row_id} has invalid sentence groups")
    return titles, sentences


def _document_source(title: str, sentences: Sequence[str]) -> str:
    normalized_title = unicodedata.normalize("NFKC", title).strip()
    slug = _SLUG_RE.sub("_", normalized_title.casefold()).strip("_")[:56] or "untitled"
    content_key = normalized_title + "\n" + "\n".join(sentences)
    digest = hashlib.sha1(content_key.encode("utf-8")).hexdigest()[:10]
    return f"hotpotqa/{slug}-{digest}"


def _assign_gold_document_disjoint_partitions(tasks: Sequence[AgentRLTask]) -> List[AgentRLTask]:
    if not tasks:
        return []
    parents = list(range(len(tasks)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parents[root_right] = root_left

    first_by_document: Dict[str, int] = {}
    for index, task in enumerate(tasks):
        for document_id in task.metadata.get("gold_document_ids", []):
            previous = first_by_document.setdefault(str(document_id), index)
            union(index, previous)

    component_members: Dict[int, List[int]] = {}
    for index in range(len(tasks)):
        component_members.setdefault(find(index), []).append(index)

    partition_by_index: Dict[int, str] = {}
    for members in component_members.values():
        component_key = "|".join(sorted(tasks[index].task_id for index in members))
        bucket = int(hashlib.sha256(component_key.encode("utf-8")).hexdigest()[:8], 16) % 10
        partition = "train" if bucket < 8 else "validation" if bucket == 8 else "test"
        for index in members:
            partition_by_index[index] = partition

    output: List[AgentRLTask] = []
    for index, task in enumerate(tasks):
        metadata = dict(task.metadata)
        metadata["partition"] = partition_by_index[index]
        output.append(replace(task, metadata=metadata))
    return output


def _append_unique(values: List[str], value: str) -> None:
    if value not in values:
        values.append(value)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    text = "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
