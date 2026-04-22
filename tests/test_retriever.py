import sys, uuid; sys.path.insert(0, '.')
import numpy as np
from unittest.mock import patch


def _make_meta(n=5):
    """Build a metadata dict and position_map for n fake chunks."""
    meta = {}
    position_map = []
    for i in range(n):
        cid = str(uuid.uuid4())
        meta[cid] = {
            "chunk_id": cid, "doc_id": "d1", "filename": "a.pdf",
            "file_type": "pdf", "upload_timestamp": "2026-04-16T00:00:00Z",
            "page_count": 5, "page_number": i + 1, "chunk_index": i,
            "section_heading": "Financial Analysis",
            "extraction_method": "native_text", "content_type": "prose",
            "text": f"Cash flow is strong. Sentence {i}.",
            "char_count": 30, "bounding_box": None,
            "embedding_text": f"Cash flow is strong. Sentence {i}.",
        }
        position_map.append(cid)
    return meta, position_map


def _unit_vec(d, rng):
    v = rng.standard_normal(d).astype("float32")
    return v / np.linalg.norm(v)


def test_retrieve_returns_top_n_sorted_by_similarity():
    import faiss
    rng = np.random.default_rng(42)
    d = 1536  # matches text-embedding-3-small
    index = faiss.IndexFlatL2(d)
    vecs = np.stack([_unit_vec(d, rng) for _ in range(5)])
    index.add(vecs)
    meta, position_map = _make_meta(5)

    with patch("doc_qa.retrieval.retriever._embed_query",
               return_value=_unit_vec(d, rng)):
        from doc_qa.retrieval.retriever import retrieve
        results = retrieve(
            query="What is the DSCR?",
            index=index,
            metadata_dict=meta,
            position_map=position_map,
            top_k=5, rerank_top_n=3,
        )

    assert len(results) == 3
    assert results[0].rank == 1
    # FAISS returns ascending L2 distance → descending cosine similarity
    assert results[0].reranker_score >= results[1].reranker_score >= results[2].reranker_score


def test_retrieve_reranker_score_is_cosine_from_l2():
    """reranker_score should equal 1 - L2²/2 clamped to [0,1] (unit-norm vectors)."""
    import faiss
    rng = np.random.default_rng(7)
    d = 1536
    index = faiss.IndexFlatL2(d)
    vecs = np.stack([_unit_vec(d, rng) for _ in range(3)])
    index.add(vecs)
    meta, position_map = _make_meta(3)

    with patch("doc_qa.retrieval.retriever._embed_query",
               return_value=_unit_vec(d, rng)):
        from doc_qa.retrieval.retriever import retrieve
        results = retrieve("test", index, meta, position_map, rerank_top_n=3)

    assert len(results) == 3
    for r in results:
        assert r.chunk_id in meta
        assert r.faiss_score >= 0
        assert 0.0 <= r.reranker_score <= 1.0
        expected = max(0.0, min(1.0, 1.0 - (r.faiss_score / 2.0)))
        assert abs(r.reranker_score - expected) < 1e-6
        assert r.rank >= 1


def test_retrieve_handles_fewer_chunks_than_top_n():
    import faiss
    rng = np.random.default_rng(3)
    d = 1536
    index = faiss.IndexFlatL2(d)
    vecs = np.stack([_unit_vec(d, rng) for _ in range(2)])
    index.add(vecs)
    meta, position_map = _make_meta(2)

    with patch("doc_qa.retrieval.retriever._embed_query",
               return_value=_unit_vec(d, rng)):
        from doc_qa.retrieval.retriever import retrieve
        results = retrieve("test", index, meta, position_map, rerank_top_n=5)

    assert len(results) == 2
