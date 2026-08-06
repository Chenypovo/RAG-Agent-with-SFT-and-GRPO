"""Evaluate one-shot hybrid retrieval and reranking on prepared HotpotQA."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.agent_rl.artifacts import (  # noqa: E402
    dependency_versions,
    ensure_outputs_available,
    sha256_file,
    sha256_tree,
    validate_evaluation_inputs,
)
from app.agent_rl.evaluation import evaluate_retrieval_baseline  # noqa: E402
from app.agent_rl.retrieval import HybridRerankRetriever  # noqa: E402
from app.agent_rl.tasks import load_tasks  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate HotpotQA hybrid retrieval")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--partition", choices=["all", "train", "validation", "test"], default="test")
    parser.add_argument("--ks", default="3,5,6,8")
    parser.add_argument("--candidate-top-k", type=int, default=15)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument("--embedding-device", default="cpu")
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--vector-store", choices=["lancedb", "faiss"], default="lancedb")
    parser.add_argument("--faiss-index-name", default="faiss.index")
    parser.add_argument("--faiss-metadata-name", default="faiss_metadatas.json")
    parser.add_argument("--lancedb-uri-name", default="lancedb")
    parser.add_argument("--lancedb-table", default="chunks")
    parser.add_argument("--reranker-model", default="")
    parser.add_argument("--reranker-device", default="")
    parser.add_argument("--reranker-batch-size", type=int, default=16)
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    ks = sorted({int(value.strip()) for value in args.ks.split(",") if value.strip()})
    if not ks or any(value <= 0 for value in ks):
        parser.error("ks must contain one or more positive integers")
    if args.candidate_top_k < max(ks):
        parser.error("candidate-top-k must be at least max(ks)")

    data_dir = Path(args.data_dir)
    task_name = "tasks.jsonl" if args.partition == "all" else f"tasks_{args.partition}.jsonl"
    index_path = data_dir / args.faiss_index_name
    metadata_path = data_dir / args.faiss_metadata_name
    lancedb_uri = data_dir / args.lancedb_uri_name
    if args.vector_store == "lancedb":
        if not lancedb_uri.is_dir():
            parser.error(f"LanceDB artifact does not exist: {lancedb_uri}")
        retrieval_artifacts = {
            args.lancedb_uri_name: sha256_tree(lancedb_uri),
        }
    else:
        for path in (index_path, metadata_path):
            if not path.is_file():
                parser.error(f"FAISS artifact does not exist: {path}")
        retrieval_artifacts = {
            args.faiss_index_name: sha256_file(index_path),
            args.faiss_metadata_name: sha256_file(metadata_path),
        }
    output_path = Path(args.output)
    ensure_outputs_available((output_path,), overwrite=args.overwrite)
    provenance = validate_evaluation_inputs(data_dir, task_filename=task_name)

    retriever = HybridRerankRetriever.from_paths(
        bm25_path=str(data_dir / "bm25.json"),
        vector_store=args.vector_store,
        faiss_index_path=str(index_path),
        faiss_metadata_path=str(metadata_path),
        lancedb_uri=str(lancedb_uri),
        lancedb_table=args.lancedb_table,
        embedding_model=args.embedding_model,
        embedding_device=args.embedding_device,
        embedding_batch_size=args.embedding_batch_size,
        candidate_top_k=args.candidate_top_k,
        output_top_k=max(ks),
        rrf_k=args.rrf_k,
        reranker_model=args.reranker_model,
        reranker_device=args.reranker_device or None,
        reranker_batch_size=args.reranker_batch_size,
    )
    cache: Dict[str, List[Dict[str, Any]]] = {}

    def retrieve(query: str, top_k: int) -> List[Dict[str, Any]]:
        if query not in cache:
            cache[query] = retriever.retrieve(query, top_k=max(ks))
        return cache[query][:top_k]

    tasks = load_tasks(data_dir / task_name)
    report = evaluate_retrieval_baseline(tasks, retrieve, ks=ks)
    report.update({
        "schema_version": "hotpotqa-hybrid-eval-v1",
        "retriever": "FAISS+BM25+RRF" + ("+reranker" if args.reranker_model else ""),
        "partition": args.partition,
        "data_dir": str(data_dir),
        "config": {
            "data_dir": str(data_dir),
            "partition": args.partition,
            "task_filename": task_name,
            "ks": ks,
            "candidate_top_k": args.candidate_top_k,
            "rrf_k": args.rrf_k,
            "vector_store": args.vector_store,
            "lancedb_table": (
                args.lancedb_table if args.vector_store == "lancedb" else None
            ),
            "embedding_model": args.embedding_model,
            "embedding_device": args.embedding_device,
            "reranker_model": args.reranker_model or None,
            "reranker_device": args.reranker_device or None,
            "reranker_batch_size": args.reranker_batch_size,
        },
        "retrieval_artifact_sha256": retrieval_artifacts,
        "dependencies": dependency_versions(
            (
                "lancedb" if args.vector_store == "lancedb" else "faiss-cpu",
                "rank-bm25",
                "torch",
                "transformers",
            )
        ),
        **provenance,
    })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
