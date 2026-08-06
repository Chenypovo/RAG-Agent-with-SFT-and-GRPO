from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

from app.retriever.hybrid import rrf_fuse


BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class LocalBGETextEncoder:
    """Local BGE encoder using normalized CLS embeddings."""

    def __init__(
        self,
        model_name: str,
        *,
        device: str = "cpu",
        batch_size: int = 64,
        max_length: int = 512,
        query_instruction: str = BGE_QUERY_INSTRUCTION,
    ) -> None:
        if not model_name.strip():
            raise ValueError("model_name must not be empty")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if max_length <= 0:
            raise ValueError("max_length must be positive")

        import torch
        from transformers import AutoModel, AutoTokenizer

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError("CUDA requested for embedding but is not available")
        self.model_name = model_name
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.max_length = max_length
        self.query_instruction = query_instruction
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()

    def encode_query(self, query: str) -> List[float]:
        text = f"{self.query_instruction}{query.strip()}"
        return self._encode([text])[0]

    def encode_passages(self, passages: Sequence[str]) -> List[List[float]]:
        return self._encode([str(text).strip() for text in passages])

    def _encode(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        torch = self._torch
        vectors: List[List[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.inference_mode():
                hidden = self.model(**encoded).last_hidden_state[:, 0]
                normalized = torch.nn.functional.normalize(hidden, p=2, dim=1)
            vectors.extend(normalized.float().cpu().tolist())
        return vectors


@dataclass(frozen=True)
class HybridRetrievalConfig:
    candidate_top_k: int = 15
    output_top_k: int = 6
    rrf_k: int = 60

    def __post_init__(self) -> None:
        if self.candidate_top_k <= 0:
            raise ValueError("candidate_top_k must be positive")
        if self.output_top_k <= 0:
            raise ValueError("output_top_k must be positive")
        if self.output_top_k > self.candidate_top_k:
            raise ValueError("output_top_k cannot exceed candidate_top_k")
        if self.rrf_k <= 0:
            raise ValueError("rrf_k must be positive")


class HybridRerankRetriever:
    """Dense-store + BM25 RRF retrieval with an optional cross-encoder reranker."""

    def __init__(
        self,
        *,
        vector_store: Any,
        bm25_store: Any,
        encode_query: Callable[[str], List[float]],
        config: HybridRetrievalConfig,
        reranker: Optional[Any] = None,
    ) -> None:
        self.vector_store = vector_store
        self.bm25_store = bm25_store
        self.encode_query = encode_query
        self.config = config
        self.reranker = reranker

    @classmethod
    def from_paths(
        cls,
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
        reranker_device: Optional[str] = None,
        reranker_batch_size: int = 16,
    ) -> "HybridRerankRetriever":
        from app.vectordb import load_doc_store
        from app.vectordb.bm25_store import BM25Store

        if vector_store not in {"lancedb", "faiss"}:
            raise ValueError("vector_store must be lancedb or faiss")
        encoder = LocalBGETextEncoder(
            embedding_model,
            device=embedding_device,
            batch_size=embedding_batch_size,
        )
        reranker = None
        if reranker_model.strip():
            from app.reranker.bge_reranker import BGEReranker

            reranker = BGEReranker(
                model_name=reranker_model,
                device=reranker_device,
                batch_size=reranker_batch_size,
            )
        return cls(
            vector_store=load_doc_store(
                vector_store,
                index_path=faiss_index_path or None,
                meta_path=faiss_metadata_path or None,
                lancedb_uri=lancedb_uri or None,
                lancedb_table=lancedb_table,
            ),
            bm25_store=BM25Store.load(bm25_path),
            encode_query=encoder.encode_query,
            config=HybridRetrievalConfig(
                candidate_top_k=candidate_top_k,
                output_top_k=output_top_k,
                rrf_k=rrf_k,
            ),
            reranker=reranker,
        )

    def retrieve(self, query: str, *, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        clean_query = str(query).strip()
        if not clean_query:
            return []
        requested_top_k = self.config.output_top_k if top_k is None else int(top_k)
        if requested_top_k <= 0:
            raise ValueError("top_k must be positive")
        requested_top_k = min(requested_top_k, self.config.candidate_top_k)

        query_vector = self.encode_query(clean_query)
        vector_items = self.vector_store.search(
            query_vector=query_vector,
            top_k=self.config.candidate_top_k,
        )
        bm25_items = self.bm25_store.search(
            query=clean_query,
            top_k=self.config.candidate_top_k,
        )
        candidates = rrf_fuse(
            vector_items,
            bm25_items,
            rrf_k=self.config.rrf_k,
            top_k=self.config.candidate_top_k,
        )
        if self.reranker is not None:
            return self.reranker.rerank(
                query=clean_query,
                retrieved=candidates,
                top_k=requested_top_k,
            )
        return candidates[:requested_top_k]

    def __call__(self, query: str) -> List[Dict[str, Any]]:
        return self.retrieve(query)
