# doc_qa/retrieval/retriever.py
"""Query embedding, FAISS similarity search, and cross-encoder reranking.

Two-stage retrieval: FAISS for approximate nearest-neighbor (fast, large recall),
then cross-encoder reranking for precise relevance scoring. The cross-encoder
scores actual (query, passage) pairs — significantly more accurate than embedding
similarity alone for targeted financial questions.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np

logger = logging.getLogger(__name__)

_embeddings: Optional[Any] = None


def _get_embeddings() -> "HuggingFaceEmbeddings":
    global _embeddings
    if _embeddings is None:
        from langchain_huggingface import HuggingFaceEmbeddings as _HFE
        _embeddings = _HFE(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return _embeddings


@dataclass
class RetrievedChunk:
    """A chunk enriched with retrieval scores and rank after reranking."""
    chunk_id: str
    doc_id: str
    filename: str
    file_type: str
    upload_timestamp: str
    page_count: int
    page_number: int
    chunk_index: int
    section_heading: str
    extraction_method: str
    content_type: str
    text: str
    char_count: int
    embedding_text: str
    bounding_box: Optional[tuple]
    faiss_score: float    # L2 distance (lower = more similar)
    reranker_score: float # cross-encoder score (higher = more relevant)
    rank: int             # 1-based position after reranking


def _embed_query(query: str) -> np.ndarray:
    """Embed a query string using the same model used at index time."""
    return np.array(_get_embeddings().embed_query(query), dtype="float32")


def _rerank(query: str, chunks: List[dict]) -> List[Tuple[dict, float]]:
    """Score (query, passage) pairs with a cross-encoder.

    CrossEncoder reads both query and passage together, producing a true
    relevance score rather than independent embedding similarity.
    ms-marco-MiniLM is trained on passage retrieval and generalises well
    to financial document Q&A.
    """
    from sentence_transformers import CrossEncoder
    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    pairs = [(query, c["text"]) for c in chunks]
    scores = model.predict(pairs)
    return list(zip(chunks, scores.tolist()))


def retrieve(
    query: str,
    index: faiss.Index,
    metadata_dict: Dict[str, dict],
    position_map: List[str],
    top_k: int = 10,
    rerank_top_n: int = 5,
) -> List[RetrievedChunk]:
    """Retrieve and rerank the most relevant chunks for a query.

    Args:
        query: The user's question string.
        index: Merged FAISS index from vectorstore.load_index().
        metadata_dict: chunk_id → metadata dict from vectorstore.load_index().
        position_map: Ordered list of chunk_ids matching FAISS integer positions.
        top_k: Number of candidates to retrieve from FAISS before reranking.
        rerank_top_n: Final number of chunks returned after reranking.

    Returns:
        List of RetrievedChunk sorted by reranker_score descending, length <= rerank_top_n.
    """
    # Step 1: embed query with the same model used at index time
    query_vec = _embed_query(query).reshape(1, -1)

    # Step 2: FAISS approximate nearest-neighbor search
    k = min(top_k, index.ntotal)
    scores, indices = index.search(query_vec, k)

    candidates = []
    for faiss_score, pos in zip(scores[0], indices[0]):
        if pos < 0 or pos >= len(position_map):
            continue
        chunk_id = position_map[pos]
        meta = metadata_dict.get(chunk_id)
        if meta:
            candidates.append({**meta, "faiss_score": float(faiss_score)})

    if not candidates:
        return []

    # Step 3: cross-encoder reranking — more accurate than embedding similarity
    scored = _rerank(query, candidates)
    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:rerank_top_n]

    # Step 4: assemble RetrievedChunk objects preserving both scores
    return [
        RetrievedChunk(
            chunk_id=meta["chunk_id"],
            doc_id=meta["doc_id"],
            filename=meta["filename"],
            file_type=meta["file_type"],
            upload_timestamp=meta["upload_timestamp"],
            page_count=meta["page_count"],
            page_number=meta["page_number"],
            chunk_index=meta["chunk_index"],
            section_heading=meta["section_heading"],
            extraction_method=meta["extraction_method"],
            content_type=meta["content_type"],
            text=meta["text"],
            char_count=meta["char_count"],
            embedding_text=meta["embedding_text"],
            bounding_box=meta.get("bounding_box"),
            faiss_score=meta["faiss_score"],
            reranker_score=reranker_score,
            rank=rank,
        )
        for rank, (meta, reranker_score) in enumerate(top, start=1)
    ]
