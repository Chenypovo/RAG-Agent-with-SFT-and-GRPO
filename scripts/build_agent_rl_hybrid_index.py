"""Build a sentence-preserving vector index for prepared Agent RL data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.agent_rl.artifacts import sha256_tree  # noqa: E402
from app.agent_rl.retrieval import LocalBGETextEncoder  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_corpus(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            source = record.get("source")
            chunk_id = record.get("chunk_id")
            text = str(record.get("text", "")).strip()
            if source is None or chunk_id is None or not text:
                raise ValueError(
                    f"{path}:{line_number} requires source, chunk_id and non-empty text"
                )
            expected_doc_id = f"{source}#{chunk_id}"
            if str(record.get("doc_id", expected_doc_id)) != expected_doc_id:
                raise ValueError(
                    f"{path}:{line_number} doc_id does not match source#chunk_id"
                )
            normalized = dict(record)
            normalized["doc_id"] = expected_doc_id
            records.append(normalized)
    if not records:
        raise ValueError(f"no corpus records found in {path}")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a local BGE vector index without changing HotpotQA sentence IDs"
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument("--embedding-device", default="cpu")
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--vector-store", choices=["lancedb", "faiss"], default="lancedb")
    parser.add_argument("--index-name", default="faiss.index")
    parser.add_argument("--metadata-name", default="faiss_metadatas.json")
    parser.add_argument("--lancedb-uri-name", default="lancedb")
    parser.add_argument("--lancedb-table", default="chunks")
    parser.add_argument("--report-name", default="hybrid_index_report.json")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.embedding_batch_size <= 0:
        parser.error("embedding-batch-size must be positive")
    data_dir = Path(args.data_dir)
    corpus_path = data_dir / "corpus.jsonl"
    if not corpus_path.is_file():
        parser.error(f"corpus does not exist: {corpus_path}")
    index_path = data_dir / args.index_name
    metadata_path = data_dir / args.metadata_name
    lancedb_uri = data_dir / args.lancedb_uri_name
    report_path = data_dir / args.report_name
    vector_outputs = (
        (lancedb_uri,)
        if args.vector_store == "lancedb"
        else (index_path, metadata_path)
    )
    existing = [
        path for path in (*vector_outputs, report_path) if path.exists()
    ]
    if existing and not args.overwrite:
        parser.error("outputs already exist: " + ", ".join(str(path) for path in existing))

    records = _load_corpus(corpus_path)
    encoder = LocalBGETextEncoder(
        args.embedding_model,
        device=args.embedding_device,
        batch_size=args.embedding_batch_size,
    )
    vectors = encoder.encode_passages([str(record["text"]) for record in records])
    if len(vectors) != len(records):
        raise RuntimeError("embedding count does not match corpus record count")

    if args.vector_store == "lancedb":
        from app.vectordb.lancedb_store import LanceDBStore

        store = LanceDBStore(uri=str(lancedb_uri), table_name=args.lancedb_table)
        store.add(vectors=vectors, metadatas=records)
        vector_artifacts = {
            "lancedb": {
                "path": str(lancedb_uri),
                "table": args.lancedb_table,
                "sha256": sha256_tree(lancedb_uri),
            }
        }
    else:
        from app.vectordb.faiss_store import FaissStore

        store = FaissStore(dim=len(vectors[0]))
        store.add(vectors=vectors, metadatas=records)
        store.save(str(index_path), str(metadata_path))
        vector_artifacts = {
            "index": {"path": str(index_path), "sha256": _sha256(index_path)},
            "metadata": {"path": str(metadata_path), "sha256": _sha256(metadata_path)},
        }

    model_config_path = Path(args.embedding_model) / "config.json"
    report = {
        "schema_version": "agent-rl-hybrid-index-v1",
        "data_dir": str(data_dir),
        "embedding": {
            "model": args.embedding_model,
            "device": args.embedding_device,
            "batch_size": args.embedding_batch_size,
            "pooling": "cls",
            "normalized": True,
            "model_config_sha256": (
                _sha256(model_config_path) if model_config_path.is_file() else None
            ),
        },
        "vector_store": args.vector_store,
        "records": len(records),
        "dimension": len(vectors[0]),
        "artifacts": {
            "corpus": {"path": str(corpus_path), "sha256": _sha256(corpus_path)},
            **vector_artifacts,
        },
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
