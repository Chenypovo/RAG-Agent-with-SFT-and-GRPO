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
