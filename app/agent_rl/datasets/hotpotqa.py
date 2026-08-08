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
PARTITION_MODES = ("gold_document_hash", "all_train", "all_validation", "all_test")
INVALID_ROW_POLICIES = ("error", "skip")
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
    partition_strategy: str = "context_document_components_sha256_80_10_10"
    requested_source_rows: Optional[int] = None
    skipped_row_details: Sequence[Dict[str, str]] = ()


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
    dataset: str = DATASET_NAME,
    config: str = DATASET_CONFIG,
    split: str = DATASET_SPLIT,
    fetch_page: Optional[FetchPageFn] = None,
) -> List[Dict[str, Any]]:
    """Download raw rows using the Dataset Viewer pagination contract."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    if offset < 0:
        raise ValueError("offset must be non-negative")
    if not 1 <= page_size <= 100:
        raise ValueError("page_size must be between 1 and 100")

    fetch = fetch_page or (lambda page_offset, length: fetch_viewer_page(
        page_offset,
        length,
        dataset=dataset,
        config=config,
        split=split,
    ))
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


def load_hotpot_parquet(
    path: str | Path,
    *,
    limit: int,
    offset: int = 0,
    batch_size: int = 1024,
) -> List[Dict[str, Any]]:
    """Read a bounded row window without loading an entire Hub shard into memory."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    if offset < 0:
        raise ValueError("offset must be non-negative")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError("reading Parquet requires pyarrow") from exc

    rows: List[Dict[str, Any]] = []
    seen = 0
    for batch in parquet.ParquetFile(str(path)).iter_batches(batch_size=batch_size):
        batch_rows = batch.to_pylist()
        batch_end = seen + len(batch_rows)
        if batch_end <= offset:
            seen = batch_end
            continue
        start = max(offset - seen, 0)
        for row in batch_rows[start:]:
            if not isinstance(row, dict):
                raise ValueError("Parquet row must be a JSON object")
            rows.append(dict(row))
            if len(rows) >= limit:
                return rows
        seen = batch_end
    return rows


def convert_hotpot_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    dataset: str = DATASET_NAME,
    config: str = DATASET_CONFIG,
    split: str = DATASET_SPLIT,
    offset: int = 0,
    partition_mode: str = "gold_document_hash",
    invalid_row_policy: str = "error",
) -> PreparedHotpotQA:
    if partition_mode not in PARTITION_MODES:
        raise ValueError(f"unsupported partition_mode: {partition_mode}")
    if invalid_row_policy not in INVALID_ROW_POLICIES:
        raise ValueError(f"unsupported invalid_row_policy: {invalid_row_policy}")
    corpus_by_id: Dict[str, Dict[str, Any]] = {}
    tasks: List[AgentRLTask] = []
    seen_task_ids: set[str] = set()
    skipped_row_details: List[Dict[str, str]] = []
    requested_source_rows = 0

    for row_number, raw_row in enumerate(rows, start=1):
        requested_source_rows += 1
        row_id = _best_effort_row_id(raw_row, offset=offset, row_number=row_number)
        try:
            task, corpus_records = _convert_hotpot_row(
                raw_row,
                row_number=row_number,
                dataset=dataset,
                config=config,
                split=split,
            )
        except ValueError as exc:
            if invalid_row_policy == "error":
                raise
            skipped_row_details.append({
                "row_id": row_id,
                "reason": _invalid_row_reason(exc, row_id),
            })
            continue

        task_id = task.task_id
        if task_id in seen_task_ids:
            raise ValueError(f"duplicate HotpotQA task id: {task_id}")
        seen_task_ids.add(task_id)
        for record in corpus_records:
            doc_id = str(record["doc_id"])
            previous = corpus_by_id.get(doc_id)
            if previous is not None and previous != record:
                raise ValueError(f"corpus collision for {doc_id}")
            corpus_by_id[doc_id] = record
        tasks.append(task)

    if partition_mode == "gold_document_hash":
        partitioned = _assign_context_document_disjoint_partitions(tasks)
        partition_strategy = "context_document_components_sha256_80_10_10"
    else:
        target_partition = partition_mode.removeprefix("all_")
        partitioned = _assign_single_partition(tasks, target_partition)
        partition_strategy = f"all_{target_partition}_explicit"
    corpus = [corpus_by_id[key] for key in sorted(corpus_by_id)]
    return PreparedHotpotQA(
        tasks=partitioned,
        corpus=corpus,
        source_rows=len(tasks),
        dataset=dataset,
        config=config,
        split=split,
        offset=offset,
        partition_strategy=partition_strategy,
        requested_source_rows=requested_source_rows,
        skipped_row_details=tuple(skipped_row_details),
    )


def write_prepared_hotpotqa(
    prepared: PreparedHotpotQA,
    output_dir: str | Path,
    *,
    build_bm25: bool = True,
) -> Dict[str, Any]:
    output = Path(output_dir)
    bm25_path = output / "bm25.json"
    if not build_bm25 and bm25_path.exists():
        raise FileExistsError(
            f"refusing to skip BM25 while an existing index could become stale: {bm25_path}"
        )
    output.mkdir(parents=True, exist_ok=True)

    tasks_path = output / "tasks.jsonl"
    corpus_path = output / "corpus.jsonl"
    _write_jsonl(tasks_path, (task.to_dict() for task in prepared.tasks))
    _write_jsonl(corpus_path, prepared.corpus)

    partition_counts: Dict[str, int] = {}
    partition_paths: Dict[str, Path] = {}
    for partition in ("train", "validation", "test"):
        selected = [task for task in prepared.tasks if task.metadata.get("partition") == partition]
        partition_counts[partition] = len(selected)
        partition_path = output / f"tasks_{partition}.jsonl"
        partition_paths[partition] = partition_path
        _write_jsonl(partition_path, (task.to_dict() for task in selected))

    bm25_records = 0
    if build_bm25:
        from app.vectordb.bm25_store import BM25Store

        store = BM25Store()
        store.build_from_records(prepared.corpus)
        store.save(str(bm25_path))
        bm25_records = len(store.metadatas)

    file_hashes = {
        "tasks.jsonl": _sha256(tasks_path),
        "corpus.jsonl": _sha256(corpus_path),
        **{
            f"tasks_{partition}.jsonl": _sha256(path)
            for partition, path in partition_paths.items()
        },
    }
    if build_bm25:
        file_hashes["bm25.json"] = _sha256(bm25_path)

    manifest = {
        "dataset": prepared.dataset,
        "config": prepared.config,
        "source_split": prepared.split,
        "offset": prepared.offset,
        "requested_source_rows": (
            prepared.source_rows
            if prepared.requested_source_rows is None
            else prepared.requested_source_rows
        ),
        "source_rows": prepared.source_rows,
        "skipped_rows": len(prepared.skipped_row_details),
        "skipped_row_details": [dict(detail) for detail in prepared.skipped_row_details],
        "tasks": len(prepared.tasks),
        "corpus_records": len(prepared.corpus),
        "bm25_records": bm25_records,
        "partition_counts": partition_counts,
        "partition_strategy": prepared.partition_strategy,
        "evidence_granularity": "sentence",
        "license": DATASET_LICENSE,
        "source_url": f"https://huggingface.co/datasets/{prepared.dataset}",
        "files": file_hashes,
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


def _best_effort_row_id(
    raw_row: Mapping[str, Any],
    *,
    offset: int,
    row_number: int,
) -> str:
    """Identify an invalid source row without copying question or answer fields."""
    try:
        row = _unwrap_row(raw_row)
    except (TypeError, ValueError):
        row = {}
    row_id = str(row.get("id", "")).strip()
    return row_id or f"source-index-{offset + row_number - 1}"


def _invalid_row_reason(exc: ValueError, row_id: str) -> str:
    reason = str(exc)
    prefix = f"HotpotQA row {row_id} "
    if reason.startswith(prefix):
        reason = reason[len(prefix):]
    return reason


def _convert_hotpot_row(
    raw_row: Mapping[str, Any],
    *,
    row_number: int,
    dataset: str,
    config: str,
    split: str,
) -> tuple[AgentRLTask, List[Dict[str, Any]]]:
    """Validate one row fully before its task or corpus records are committed."""
    row = _unwrap_row(raw_row)
    row_id = str(row.get("id", "")).strip()
    if not row_id:
        raise ValueError(f"HotpotQA row {row_number} has no id")

    titles, sentence_groups = _context_columns(row, row_id)
    title_to_source: Dict[str, str] = {}
    title_to_sentences: Dict[str, Sequence[str]] = {}
    context_document_ids: List[str] = []
    context_document_keys: List[str] = []
    corpus_records: List[Dict[str, Any]] = []

    for title, sentences in zip(titles, sentence_groups):
        clean_title = str(title).strip()
        clean_sentences = [str(sentence).strip() for sentence in sentences]
        if not clean_title or not clean_sentences:
            raise ValueError(f"HotpotQA row {row_id} has an empty context document")
        source = _document_source(clean_title, clean_sentences)
        title_to_source[clean_title] = source
        title_to_sentences[clean_title] = clean_sentences
        context_document_ids.append(source)
        _append_unique(context_document_keys, _canonical_document_key(clean_title))

        for sentence_id, sentence in enumerate(clean_sentences):
            if not sentence:
                continue
            corpus_records.append({
                "doc_id": f"{source}#{sentence_id}",
                "source": source,
                "chunk_id": sentence_id,
                "title": clean_title,
                "text": f"{clean_title}\n{sentence}",
                "modality": "text",
                "file_type": "hotpotqa",
                "dataset": dataset,
            })

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

    task = AgentRLTask(
        task_id=f"hotpotqa_{row_id}",
        question=str(row.get("question", "")).strip(),
        task_type="multihop",
        gold_answers=(str(row.get("answer", "")).strip(),),
        gold_evidence_ids=tuple(evidence_ids),
        metadata={
            "source_dataset": dataset,
            "source_config": config,
            "source_split": split,
            "source_row_id": row_id,
            "hotpot_type": str(row.get("type", "")),
            "level": str(row.get("level", "")),
            "context_document_ids": context_document_ids,
            "context_document_keys": context_document_keys,
            "gold_document_ids": gold_document_ids,
        },
    )
    return task, corpus_records


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


def _canonical_document_key(title: str) -> str:
    normalized_title = " ".join(unicodedata.normalize("NFKC", title).split()).casefold()
    slug = _SLUG_RE.sub("_", normalized_title).strip("_")[:56] or "untitled"
    digest = hashlib.sha256(normalized_title.encode("utf-8")).hexdigest()[:16]
    return f"hotpotqa-title/{slug}-{digest}"


def _assign_context_document_disjoint_partitions(tasks: Sequence[AgentRLTask]) -> List[AgentRLTask]:
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
        for document_key in task.metadata.get("context_document_keys", []):
            previous = first_by_document.setdefault(str(document_key), index)
            union(index, previous)

    component_members: Dict[int, List[int]] = {}
    for index in range(len(tasks)):
        component_members.setdefault(find(index), []).append(index)

    partition_by_index: Dict[int, str] = {}
    for members in component_members.values():
        component_documents = {
            str(document_key)
            for index in members
            for document_key in tasks[index].metadata.get("context_document_keys", [])
        }
        component_key = "|".join(sorted(component_documents))
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


def _assign_single_partition(
    tasks: Sequence[AgentRLTask],
    partition: str,
) -> List[AgentRLTask]:
    output: List[AgentRLTask] = []
    for task in tasks:
        metadata = dict(task.metadata)
        metadata["partition"] = partition
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
