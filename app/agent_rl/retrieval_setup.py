"""Configure the existing sentence-level retriever for post-training runs.

This module only wires the repository's stores, RRF and reranker together. It
does not introduce a second retrieval implementation or parent-child expansion.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from app.agent_rl.adapters import build_bm25_registry, build_hybrid_registry
from app.agent_rl.artifacts import sha256_file, sha256_tree


def build_retrieval_registry(data_dir: Path, config: Mapping[str, Any]):
    backend = config.get("backend", "bm25")
    if backend == "bm25":
        registry = build_bm25_registry(
            str(data_dir / "bm25.json"), top_k=int(config.get("top_k", 8))
        )
        return registry, {"backend": backend, "bm25_sha256": sha256_file(data_dir / "bm25.json")}
    if backend not in {"hybrid", "hybrid-rerank"}:
        raise ValueError(f"unsupported retrieval backend: {backend}")
    if config.get("parent_child_expansion", False):
        raise ValueError("AgentRL requires sentence-level evidence; parent-child expansion is forbidden")
    candidate_k = int(config.get("candidate_top_k", 15))
    output_k = int(config.get("output_top_k", 6))
    if not 0 < output_k <= candidate_k:
        raise ValueError("retrieval output_top_k must be positive and <= candidate_top_k")
    uri = data_dir / str(config.get("lancedb_uri_name", "lancedb"))
    if not uri.is_dir():
        raise FileNotFoundError(f"LanceDB artifact does not exist: {uri}")
    if not str(config.get("embedding_model", "")).strip():
        raise ValueError("hybrid retrieval requires embedding_model")
    if backend == "hybrid-rerank" and not str(config.get("reranker_model", "")).strip():
        raise ValueError("hybrid-rerank requires reranker_model")
    artifacts = {
        "backend": backend,
        "config": dict(config),
        "parent_child_expansion": False,
        "bm25_sha256": sha256_file(data_dir / "bm25.json"),
        "lancedb_sha256": sha256_tree(uri),
    }
    registry = build_hybrid_registry(
        bm25_path=str(data_dir / "bm25.json"),
        vector_store="lancedb",
        lancedb_uri=str(uri),
        lancedb_table=str(config.get("lancedb_table", "chunks")),
        embedding_model=str(config["embedding_model"]),
        embedding_device=str(config.get("embedding_device", "cuda")),
        embedding_batch_size=int(config.get("embedding_batch_size", 64)),
        candidate_top_k=candidate_k,
        output_top_k=output_k,
        rrf_k=int(config.get("rrf_k", 60)),
        reranker_model=str(config["reranker_model"]) if backend == "hybrid-rerank" else "",
        reranker_device=config.get("reranker_device") or None,
        reranker_batch_size=int(config.get("reranker_batch_size", 16)),
    )
    return registry, artifacts
