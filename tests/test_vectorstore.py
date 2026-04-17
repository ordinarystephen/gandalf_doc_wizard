import sys, uuid; sys.path.insert(0, '.')
import pytest


def _make_chunks(n=3, doc_id="doc1"):
    from doc_qa.ingest.extractor import ProcessedChunk
    return [
        ProcessedChunk(
            doc_id=doc_id, filename="test.pdf", file_type="pdf",
            upload_timestamp="2026-04-16T00:00:00Z", page_count=5,
            page_number=i + 1, chunk_index=i, section_heading="Intro",
            extraction_method="native_text", content_type="prose",
            text=f"The borrower has strong cash flow. Sentence {i}.",
            char_count=40, bounding_box=None,
            chunk_id=str(uuid.uuid4()),
            embedding_text=f"The borrower has strong cash flow. Sentence {i}.",
        )
        for i in range(n)
    ]


def test_build_creates_index_and_sidecar(tmp_path):
    from doc_qa.retrieval.vectorstore import build_index
    chunks = _make_chunks(3)
    build_index(chunks, doc_id="doc1", data_dir=str(tmp_path))
    assert (tmp_path / "doc1.faiss").exists()
    assert (tmp_path / "doc1_meta.json").exists()


def test_load_returns_index_and_metadata(tmp_path):
    from doc_qa.retrieval.vectorstore import build_index, load_index
    chunks = _make_chunks(3)
    build_index(chunks, doc_id="doc1", data_dir=str(tmp_path))
    index, meta, position_map = load_index(["doc1"], data_dir=str(tmp_path))
    assert index is not None
    assert index.ntotal == 3
    assert len(meta) == 3
    assert len(position_map) == 3


def test_list_indexed_docs(tmp_path):
    from doc_qa.retrieval.vectorstore import build_index, list_indexed_docs
    chunks = _make_chunks(2)
    build_index(chunks, doc_id="docA", data_dir=str(tmp_path))
    docs = list_indexed_docs(data_dir=str(tmp_path))
    assert "docA" in docs


def test_load_multiple_docs(tmp_path):
    from doc_qa.retrieval.vectorstore import build_index, load_index
    build_index(_make_chunks(2, "d1"), doc_id="d1", data_dir=str(tmp_path))
    build_index(_make_chunks(2, "d2"), doc_id="d2", data_dir=str(tmp_path))
    index, meta, position_map = load_index(["d1", "d2"], data_dir=str(tmp_path))
    assert index.ntotal == 4
    assert len(meta) == 4
    assert len(position_map) == 4
