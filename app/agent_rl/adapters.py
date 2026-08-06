from __future__ import annotations

from app.agent.agent import RetrieveDocsFn
from app.agent.registry import ToolRegistry
from app.agent.tools.calculator import CalculatorTool
from app.agent.tools.retrieve import RetrieveDocsTool


def build_minimal_registry(retrieve_docs_fn: RetrieveDocsFn) -> ToolRegistry:
    """Build the frozen Phase-1 action space: retrieval plus arithmetic."""
    registry = ToolRegistry()
    registry.register(RetrieveDocsTool(retrieve_docs_fn=retrieve_docs_fn))
    registry.register(CalculatorTool())
    return registry


def build_bm25_registry(index_path: str, top_k: int = 8) -> ToolRegistry:
    """Load a prepared BM25 corpus and expose it through the Phase-1 tools."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    from app.vectordb.bm25_store import BM25Store

    store = BM25Store.load(index_path)

    def retrieve(query: str):
        return store.search(query=query, top_k=top_k)

    return build_minimal_registry(retrieve)


def build_hybrid_registry(
    *,
    bm25_path: str,
    vector_store: str = "lancedb",
    faiss_index_path: str = "",
    faiss_metadata_path: str = "",
    lancedb_uri: str = "",
    lancedb_table: str = "chunks",
    embedding_model: str,
    embedding_device: str = "cpu",
    embedding_batch_size: int = 64,
    candidate_top_k: int = 15,
    output_top_k: int = 6,
    rrf_k: int = 60,
    reranker_model: str = "",
    reranker_device: str | None = None,
    reranker_batch_size: int = 16,
) -> ToolRegistry:
    """Expose sentence-level hybrid retrieval through the existing RL tool."""
    from app.agent_rl.retrieval import HybridRerankRetriever

    retriever = HybridRerankRetriever.from_paths(
        bm25_path=bm25_path,
        vector_store=vector_store,
        faiss_index_path=faiss_index_path,
        faiss_metadata_path=faiss_metadata_path,
        lancedb_uri=lancedb_uri,
        lancedb_table=lancedb_table,
        embedding_model=embedding_model,
        embedding_device=embedding_device,
        embedding_batch_size=embedding_batch_size,
        candidate_top_k=candidate_top_k,
        output_top_k=output_top_k,
        rrf_k=rrf_k,
        reranker_model=reranker_model,
        reranker_device=reranker_device,
        reranker_batch_size=reranker_batch_size,
    )
    return build_minimal_registry(retriever)
