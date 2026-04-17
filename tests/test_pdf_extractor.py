import sys; sys.path.insert(0, '.')
import pytest
from pathlib import Path

SAMPLE_PDF = Path("tests/fixtures/sample.pdf")

@pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="fixture not generated")
def test_extract_pdf_returns_chunks():
    from doc_qa.ingest.pdf_extractor import extract_pdf
    chunks = extract_pdf(str(SAMPLE_PDF))
    assert len(chunks) >= 1

@pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="fixture not generated")
def test_extract_pdf_prose_content():
    from doc_qa.ingest.pdf_extractor import extract_pdf
    chunks = extract_pdf(str(SAMPLE_PDF))
    prose = [c for c in chunks if c.content_type == "prose"]
    assert len(prose) >= 1
    all_text = " ".join(c.text for c in prose)
    assert "DSCR" in all_text

@pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="fixture not generated")
def test_extract_pdf_section_heading():
    from doc_qa.ingest.pdf_extractor import extract_pdf
    chunks = extract_pdf(str(SAMPLE_PDF))
    headings = [c.section_heading for c in chunks]
    assert any("FINANCIAL" in h for h in headings)

@pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="fixture not generated")
def test_extract_pdf_chunk_metadata():
    from doc_qa.ingest.pdf_extractor import extract_pdf
    chunks = extract_pdf(str(SAMPLE_PDF))
    for chunk in chunks:
        assert chunk.page_number >= 1
        assert chunk.char_count == len(chunk.text)
        assert chunk.extraction_method in (
            "native_text", "pdfplumber_table", "camelot_table", "vision_ocr_gpt4o"
        )
