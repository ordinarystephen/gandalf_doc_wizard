# tests/test_batch.py
import sys, uuid; sys.path.insert(0, '.')
from unittest.mock import patch, MagicMock
from doc_qa.retrieval.retriever import RetrievedChunk
from doc_qa.qa.chain import AnswerResult

def _mock_chunk():
    return RetrievedChunk(
        chunk_id=str(uuid.uuid4()), doc_id="d1", filename="memo.pdf",
        file_type="pdf", upload_timestamp="2026-04-16T00:00:00Z",
        page_count=5, page_number=1, chunk_index=0,
        section_heading="Intro", extraction_method="native_text",
        content_type="prose", text="Revenue is $1M.",
        char_count=15, embedding_text="Revenue is $1M.",
        bounding_box=None, faiss_score=0.05, reranker_score=0.9, rank=1,
    )

def _mock_answer(q, filename, doc_id, chain):
    return AnswerResult(
        query=q, answer=f"Answer to: {q}",
        retrieved_chunks=[_mock_chunk()], prompt_tokens=100,
        completion_tokens=20, latency_seconds=1.0,
        timestamp="2026-04-16T10:00:00Z",
        model_deployment="gpt-4o", confidence_level="High",
    )

def test_run_batch_grid_shape():
    from doc_qa.qa.batch import run_batch

    questions = ["What is revenue?", "What is DSCR?"]
    doc_configs = [
        {"doc_id": "d1", "filename": "memo.pdf"},
        {"doc_id": "d2", "filename": "report.pdf"},
    ]

    with patch("doc_qa.qa.batch._answer_for_doc", side_effect=_mock_answer):
        answers_df, trace_df = run_batch(questions, doc_configs, chain=MagicMock())

    assert answers_df.shape == (2, 2)  # 2 questions x 2 docs
    assert "memo.pdf" in answers_df.columns
    assert len(trace_df) == 4  # 2 questions x 2 docs

def test_export_to_excel(tmp_path):
    import pandas as pd
    from doc_qa.qa.batch import export_to_excel
    answers_df = pd.DataFrame({"doc1.pdf": ["ans1", "ans2"]}, index=["q1", "q2"])
    trace_df = pd.DataFrame({"query": ["q1"], "answer": ["ans1"]})
    out = tmp_path / "export.xlsx"
    export_to_excel(answers_df, trace_df, str(out))
    assert out.exists()
