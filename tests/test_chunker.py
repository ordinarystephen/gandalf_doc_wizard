import sys, uuid; sys.path.insert(0, '.')
from doc_qa.ingest.extractor import RawChunk


def _prose(text="The borrower has strong coverage ratios. " * 15) -> RawChunk:
    return RawChunk(
        doc_id="d1", filename="a.pdf", file_type="pdf",
        upload_timestamp="2026-04-16T00:00:00Z", page_count=5,
        page_number=1, chunk_index=0, section_heading="Financial Analysis",
        extraction_method="native_text", content_type="prose",
        text=text, char_count=len(text), bounding_box=None,
    )


def _table() -> RawChunk:
    text = "| Metric | Value |\n|--------|-------|\n| DSCR   | 1.35x |"
    return RawChunk(
        doc_id="d1", filename="a.pdf", file_type="pdf",
        upload_timestamp="2026-04-16T00:00:00Z", page_count=5,
        page_number=2, chunk_index=1, section_heading="Financial Analysis",
        extraction_method="pdfplumber_table", content_type="table",
        text=text, char_count=len(text), bounding_box=None,
    )


def test_table_not_split():
    from doc_qa.ingest.chunker import chunk_raw
    result = chunk_raw([_table()])
    assert len(result) == 1
    assert result[0].content_type == "table"
    assert result[0].text == _table().text


def test_table_embedding_text_has_section_prefix():
    from doc_qa.ingest.chunker import chunk_raw
    result = chunk_raw([_table()])
    assert result[0].embedding_text.startswith("Table from section: Financial Analysis")


def test_processed_chunk_has_chunk_id():
    from doc_qa.ingest.chunker import chunk_raw
    result = chunk_raw([_table()])
    assert len(result[0].chunk_id) > 10  # uuid4 is 36 chars


def test_prose_produces_at_least_one_output():
    from doc_qa.ingest.chunker import chunk_raw
    result = chunk_raw([_prose()])
    assert len(result) >= 1
    assert all(c.content_type == "prose" for c in result)


def test_all_metadata_preserved_on_table_chunk():
    from doc_qa.ingest.chunker import chunk_raw
    raw = _table()
    result = chunk_raw([raw])
    pc = result[0]
    assert pc.doc_id == raw.doc_id
    assert pc.filename == raw.filename
    assert pc.page_number == raw.page_number
    assert pc.section_heading == raw.section_heading
    assert pc.char_count == len(pc.text)
