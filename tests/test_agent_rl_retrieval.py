from __future__ import annotations

import pytest

from app.agent_rl.retrieval import HybridRerankRetriever, HybridRetrievalConfig


def _item(doc_id: str, *, branch_score: float = 1.0):
    source, chunk_id = doc_id.rsplit("#", 1)
    return {
        "score": branch_score,
        "metadata": {
            "doc_id": doc_id,
            "source": source,
            "chunk_id": int(chunk_id),
            "text": f"text for {doc_id}",
        },
    }


class _VectorStore:
    def __init__(self):
        self.top_k = None

    def search(self, query_vector, top_k):
        assert query_vector == [0.1, 0.2]
        self.top_k = top_k
        return [_item("doc/a#0"), _item("doc/shared#1")]


class _BM25Store:
    def __init__(self):
        self.top_k = None

    def search(self, query, top_k):
        assert query == "bridge question"
        self.top_k = top_k
        return [_item("doc/shared#1"), _item("doc/b#2")]


class _Reranker:
    def __init__(self):
        self.seen = None
        self.top_k = None

    def rerank(self, query, retrieved, top_k):
        self.seen = list(retrieved)
        self.top_k = top_k
        return list(reversed(retrieved))[:top_k]


def test_hybrid_retriever_uses_fixed_candidate_pool_then_reranks():
    vector_store = _VectorStore()
    bm25_store = _BM25Store()
    reranker = _Reranker()
    retriever = HybridRerankRetriever(
        vector_store=vector_store,
        bm25_store=bm25_store,
        encode_query=lambda query: [0.1, 0.2],
        config=HybridRetrievalConfig(candidate_top_k=15, output_top_k=2),
        reranker=reranker,
    )

    results = retriever.retrieve("bridge question")

    assert vector_store.top_k == 15
    assert bm25_store.top_k == 15
    assert reranker.top_k == 2
    assert len(reranker.seen) == 3
    assert len(results) == 2
    assert results[0]["metadata"]["doc_id"] == "doc/b#2"


def test_hybrid_retriever_allows_prefix_ablation_without_changing_candidates():
    retriever = HybridRerankRetriever(
        vector_store=_VectorStore(),
        bm25_store=_BM25Store(),
        encode_query=lambda query: [0.1, 0.2],
        config=HybridRetrievalConfig(candidate_top_k=15, output_top_k=2),
    )

    results = retriever.retrieve("bridge question", top_k=3)

    assert len(results) == 3
    assert results[0]["metadata"]["doc_id"] == "doc/shared#1"


def test_hybrid_retrieval_config_rejects_output_larger_than_candidates():
    with pytest.raises(ValueError, match="cannot exceed"):
        HybridRetrievalConfig(candidate_top_k=5, output_top_k=6)
