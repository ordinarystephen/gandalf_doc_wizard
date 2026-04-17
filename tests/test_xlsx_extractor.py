import sys; sys.path.insert(0, '.')
import pytest
from pathlib import Path

SAMPLE_XLSX = Path("tests/fixtures/sample.xlsx")

@pytest.mark.skipif(not SAMPLE_XLSX.exists(), reason="fixture not generated")
def test_extract_xlsx_all_sheets():
    from doc_qa.ingest.xlsx_extractor import extract_xlsx
    chunks = extract_xlsx(str(SAMPLE_XLSX))
    section_headings = {c.section_heading for c in chunks}
    assert "Sheet: Financials" in section_headings
    assert "Sheet: Covenants" in section_headings

@pytest.mark.skipif(not SAMPLE_XLSX.exists(), reason="fixture not generated")
def test_extract_xlsx_content():
    from doc_qa.ingest.xlsx_extractor import extract_xlsx
    chunks = extract_xlsx(str(SAMPLE_XLSX))
    all_text = " ".join(c.text for c in chunks)
    assert "DSCR" in all_text
    assert "Revenue" in all_text

@pytest.mark.skipif(not SAMPLE_XLSX.exists(), reason="fixture not generated")
def test_extract_xlsx_metadata():
    from doc_qa.ingest.xlsx_extractor import extract_xlsx
    chunks = extract_xlsx(str(SAMPLE_XLSX))
    for chunk in chunks:
        assert chunk.extraction_method == "openpyxl"
        assert chunk.content_type == "table"
        assert chunk.char_count == len(chunk.text)
        assert chunk.doc_id == ""
