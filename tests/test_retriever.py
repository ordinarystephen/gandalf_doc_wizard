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


def test_retrieve_returns_reranked_chunks():
    import faiss
    d = 384
    index = faiss.IndexFlatL2(d)
    vecs = np.random.rand(5, d).astype("float32")
    index.add(vecs)
    meta, position_map = _make_meta(5)

    with patch("doc_qa.retrieval.retriever._embed_query",
               return_value=np.random.rand(d).astype("float32")):
        with patch("doc_qa.retrieval.retriever._rerank",
                   side_effect=lambda q, chunks: [(c, 0.9 - i * 0.1)
                                                   for i, c in enumerate(chunks)]):
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
    assert results[0].reranker_score >= results[1].reranker_score


def test_retrieve_fields_populated():
    import faiss
    d = 384
    index = faiss.IndexFlatL2(d)
    vecs = np.random.rand(3, d).astype("float32")
    index.add(vecs)
    meta, position_map = _make_meta(3)

    with patch("doc_qa.retrieval.retriever._embed_query",
               return_value=np.random.rand(d).astype("float32")):
        with patch("doc_qa.retrieval.retriever._rerank",
                   side_effect=lambda q, chunks: [(c, 0.8) for c in chunks]):
            from doc_qa.retrieval.retriever import retrieve
            results = retrieve("test", index, meta, position_map, top_k=3, rerank_top_n=2)

    assert len(results) == 2
    for r in results:
        assert r.chunk_id in meta
        assert r.faiss_score >= 0
        assert r.reranker_score == 0.8
        assert r.rank >= 1
