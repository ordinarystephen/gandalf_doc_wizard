import sys; sys.path.insert(0, '.')
import pytest
from pathlib import Path

SAMPLE_DOCX = Path("tests/fixtures/sample.docx")

@pytest.mark.skipif(not SAMPLE_DOCX.exists(), reason="fixture not generated")
def test_extract_docx_returns_chunks():
    from doc_qa.ingest.docx_extractor import extract_docx
    chunks = extract_docx(str(SAMPLE_DOCX))
    assert len(chunks) >= 1

@pytest.mark.skipif(not SAMPLE_DOCX.exists(), reason="fixture not generated")
def test_docx_headings_captured_as_section_heading():
    from doc_qa.ingest.docx_extractor import extract_docx
    chunks = extract_docx(str(SAMPLE_DOCX))
    headings = {c.section_heading for c in chunks}
    assert "Executive Summary" in headings or "Financial Analysis" in headings

@pytest.mark.skipif(not SAMPLE_DOCX.exists(), reason="fixture not generated")
def test_docx_table_extracted():
    from doc_qa.ingest.docx_extractor import extract_docx
    chunks = extract_docx(str(SAMPLE_DOCX))
    tables = [c for c in chunks if c.content_type == "table"]
    assert len(tables) >= 1
    assert "LTV" in tables[0].text

@pytest.mark.skipif(not SAMPLE_DOCX.exists(), reason="fixture not generated")
def test_docx_extraction_method_and_metadata():
    from doc_qa.ingest.docx_extractor import extract_docx
    chunks = extract_docx(str(SAMPLE_DOCX))
    for chunk in chunks:
        assert chunk.extraction_method == "python_docx"
        assert chunk.char_count == len(chunk.text)
        assert chunk.doc_id == ""  # stamped later by ingest_document()
