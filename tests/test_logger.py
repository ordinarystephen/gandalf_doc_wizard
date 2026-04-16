import os, sys, json, tempfile, pytest
sys.path.insert(0, '.')

@pytest.fixture
def tmp_logger(tmp_path):
    (tmp_path / "data").mkdir()
    from doc_qa.metadata.logger import QueryLogger
    return QueryLogger(db_path=str(tmp_path / "data" / "query_log.db"))

def test_logger_creates_db(tmp_path):
    (tmp_path / "data").mkdir()
    from doc_qa.metadata.logger import QueryLogger
    logger = QueryLogger(db_path=str(tmp_path / "data" / "query_log.db"))
    assert (tmp_path / "data" / "query_log.db").exists()

def test_log_and_retrieve(tmp_logger):
    record = {
        "query": "What is the DSCR?",
        "answer": "1.25x",
        "filename": "memo.pdf",
        "doc_id": "abc-123",
        "confidence_level": "High",
        "top_chunk_page": 3,
        "top_chunk_section": "Financial Analysis",
        "top_reranker_score": 0.92,
        "prompt_tokens": 800,
        "completion_tokens": 120,
        "latency_seconds": 1.4,
        "timestamp": "2026-04-16T10:00:00Z",
        "model_deployment": "gpt-4o",
        "extraction_methods_used": "native_text,pdfplumber_table",
        "chunk_ids_used": "c1,c2",
        "chunks_json": json.dumps([{"chunk_id": "c1", "text": "sample"}]),
    }
    tmp_logger.log(record)
    df = tmp_logger.get_recent(n=10)
    assert len(df) == 1
    assert df.iloc[0]["query"] == "What is the DSCR?"
