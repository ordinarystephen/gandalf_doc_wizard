# doc_qa/retrieval/retriever.py
"""Query embedding and FAISS similarity search.

Single-stage retrieval: FAISS top-n over OpenAI embeddings. The cross-encoder
reranker was removed when the local HuggingFace model stack was dropped —
OpenAI embeddings are strong enough that reranking is a marginal gain for
this use case, and removing it eliminates the last offline model dependency.

The ``reranker_score`` field on ``RetrievedChunk`` is preserved for backwards
compatibility with downstream callers (chain.py prompt header, confidence
scoring, CSV audit logs, UI badges). It is now populated with cosine
similarity derived from the FAISS L2 distance, since OpenAI embeddings are
unit-normalized: sim = 1 - L2² / 2. Confidence thresholds in
``doc_qa/utils/confidence.py`` have been recalibrated for this score range
(0.55 High / 0.40 Medium); refine against a real query log once one exists.

The embedder comes from ``doc_qa.retrieval._embeddings.get_embeddings`` so
index-time and query-time use exactly the same model — an index built with
one embedder can't be queried with another.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import faiss
import numpy as np

from doc_qa.retrieval._embeddings import get_embeddings

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """A chunk enriched with retrieval scores and rank."""
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
    faiss_score: float     # L2 distance (lower = more similar)
    reranker_score: float  # cosine similarity derived from L2 (higher = more relevant)
    rank: int              # 1-based position after retrieval


def _embed_query(query: str) -> np.ndarray:
    """Embed a query string using the same model used at index time."""
    return np.array(get_embeddings().embed_query(query), dtype="float32")


def _l2_to_cosine(l2: float) -> float:
    """Convert L2 distance to cosine similarity for unit-normalized vectors.

    OpenAI embeddings are length-1 by default, so ||a - b||² = 2 - 2·cos(a,b).
    Clamped to [0, 1] since FAISS can emit tiny negative or >2 values from
    float rounding on near-identical vectors.
    """
    sim = 1.0 - (l2 / 2.0)
    return max(0.0, min(1.0, sim))


def retrieve(
    query: str,
    index: faiss.Index,
    metadata_dict: Dict[str, dict],
    position_map: List[str],
    top_k: int = 10,  # kept for signature compatibility — unused without reranker
    rerank_top_n: int = 5,
) -> List[RetrievedChunk]:
    """Retrieve the most relevant chunks for a query via FAISS similarity.

    Args:
        query: The user's question string.
        index: Merged FAISS index from vectorstore.load_index().
        metadata_dict: chunk_id → metadata dict from vectorstore.load_index().
        position_map: Ordered list of chunk_ids matching FAISS integer positions.
        top_k: Accepted for signature compatibility — no longer used.
        rerank_top_n: Final number of chunks returned.

    Returns:
        List of RetrievedChunk sorted by similarity descending, length <= rerank_top_n.
    """
    query_vec = _embed_query(query).reshape(1, -1)

    k = min(rerank_top_n, index.ntotal)
    scores, indices = index.search(query_vec, k)

    results: List[RetrievedChunk] = []
    rank = 1
    for faiss_score, pos in zip(scores[0], indices[0]):
        if pos < 0 or pos >= len(position_map):
            continue
        chunk_id = position_map[pos]
        meta = metadata_dict.get(chunk_id)
        if not meta:
            continue
        results.append(
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
                faiss_score=float(faiss_score),
                reranker_score=_l2_to_cosine(float(faiss_score)),
                rank=rank,
            )
        )
        rank += 1

    return results
