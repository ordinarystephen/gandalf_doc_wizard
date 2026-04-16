import sys; sys.path.insert(0, '.')
from doc_qa.ingest.extractor import RawChunk, ProcessedChunk

def test_rawchunk_fields():
    chunk = RawChunk(
        doc_id="d1", filename="a.pdf", file_type="pdf",
        upload_timestamp="2026-04-16T00:00:00Z", page_count=10,
        page_number=1, chunk_index=0, section_heading="Intro",
        extraction_method="native_text", content_type="prose",
        text="Hello world", char_count=11, bounding_box=None,
    )
    assert chunk.char_count == 11
    assert chunk.content_type == "prose"

def test_processedchunk_fields():
    pc = ProcessedChunk(
        doc_id="d1", filename="a.pdf", file_type="pdf",
        upload_timestamp="2026-04-16T00:00:00Z", page_count=10,
        page_number=1, chunk_index=0, section_heading="Intro",
        extraction_method="native_text", content_type="prose",
        text="Hello world", char_count=11, bounding_box=None,
        chunk_id="c1", embedding_text="Hello world",
    )
    assert pc.chunk_id == "c1"
    assert pc.embedding_text == "Hello world"
