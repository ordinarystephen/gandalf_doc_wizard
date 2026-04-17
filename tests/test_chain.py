import sys, pytest; sys.path.insert(0, '.')
from unittest.mock import MagicMock, patch


def _chunk(text="The DSCR is 1.35x.", score=0.9, rank=1):
    from doc_qa.retrieval.retriever import RetrievedChunk
    return RetrievedChunk(
        chunk_id="c1", doc_id="d1", filename="memo.pdf", file_type="pdf",
        upload_timestamp="2026-04-16T00:00:00Z", page_count=10,
        page_number=3, chunk_index=0, section_heading="Financial Analysis",
        extraction_method="native_text", content_type="prose",
        text=text, char_count=len(text), embedding_text=text,
        bounding_box=None, faiss_score=0.1, reranker_score=score, rank=rank,
    )


def test_answer_result_dataclass():
    from doc_qa.qa.chain import AnswerResult
    ar = AnswerResult(
        query="What is the DSCR?", answer="1.35x",
        retrieved_chunks=[_chunk()], prompt_tokens=100,
        completion_tokens=20, latency_seconds=1.2,
        timestamp="2026-04-16T10:00:00Z",
        model_deployment="gpt-4o", confidence_level="High",
    )
    assert ar.confidence_level == "High"
    assert len(ar.retrieved_chunks) == 1


def test_qa_node_calls_llm_invoke():
    from doc_qa.qa.chain import qa_node, QAState

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(
        content="The DSCR is 1.35x.",
        usage_metadata={"input_tokens": 100, "output_tokens": 20},
    )

    state = QAState(
        query="What is the DSCR?",
        retrieved_chunks=[_chunk()],
        answer="", prompt_tokens=0, completion_tokens=0,
        latency_seconds=0.0, timestamp="", model_deployment="", confidence_level="",
    )

    with patch("doc_qa.qa.chain.time") as mock_time:
        mock_time.time.side_effect = [0.0, 1.5]
        result = qa_node(state, llm=mock_llm)

    mock_llm.invoke.assert_called_once()
    assert "1.35x" in result["answer"]
    assert result["confidence_level"] == "High"
    assert result["prompt_tokens"] == 100
    assert result["completion_tokens"] == 20


def test_answer_question_returns_result():
    from doc_qa.qa.chain import answer_question

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(
        content="The DSCR is 1.35x.",
        usage_metadata={"input_tokens": 80, "output_tokens": 15},
    )

    with patch("doc_qa.qa.chain.time") as mock_time:
        mock_time.time.side_effect = [0.0, 1.2]
        result = answer_question("What is the DSCR?", [_chunk()], mock_llm)

    assert result.answer == "The DSCR is 1.35x."
    assert result.latency_seconds == pytest.approx(1.2, abs=0.1)
    assert result.confidence_level == "High"
