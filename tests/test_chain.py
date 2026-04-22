import sys
import pytest
sys.path.insert(0, '.')

from unittest.mock import MagicMock, patch


def _chunk(text="The DSCR is 1.35x.", score=0.9, rank=1, chunk_id="c1"):
    from doc_qa.retrieval.retriever import RetrievedChunk
    return RetrievedChunk(
        chunk_id=chunk_id, doc_id="d1", filename="memo.pdf", file_type="pdf",
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
    assert ar.citations == []  # default empty list via __post_init__


def test_is_summarization_query_matches_common_phrasing():
    from doc_qa.qa.chain import is_summarization_query
    assert is_summarization_query("summarize this document")
    assert is_summarization_query("Give me a TL;DR")
    assert is_summarization_query("overview of the document please")
    assert not is_summarization_query("What is the DSCR in Q3?")


def _mock_structured_llm(answer_text, citations_data, usage=None):
    """Build a mock that supports llm.with_structured_output(...).invoke(...)."""
    from doc_qa.qa.chain import Citation, StructuredAnswer

    parsed = StructuredAnswer(
        answer=answer_text,
        citations=[Citation(**c) for c in citations_data],
    )
    raw = MagicMock()
    raw.content = answer_text
    raw.usage_metadata = usage or {"input_tokens": 100, "output_tokens": 20}

    structured = MagicMock()
    structured.invoke.return_value = {"raw": raw, "parsed": parsed, "parsing_error": None}

    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    # Plain-path fallback: llm.invoke(...) also works, returns raw directly.
    llm.invoke.return_value = raw
    return llm


def test_qa_node_structured_output_happy_path():
    from doc_qa.qa.chain import qa_node, State

    chunk = _chunk()
    llm = _mock_structured_llm(
        answer_text="The DSCR is 1.35x.",
        citations_data=[{
            "chunk_id": "c1", "page_number": 3,
            "section_heading": "Financial Analysis", "quote": "DSCR is 1.35x",
        }],
    )

    state = State(mode="qa", query="What is the DSCR?", retrieved_chunks=[chunk])
    with patch("doc_qa.qa.chain.time") as mock_time:
        mock_time.time.side_effect = [0.0, 1.5]
        result = qa_node(state, llm=llm)

    llm.with_structured_output.assert_called_once()
    assert "1.35x" in result["answer"]
    assert result["prompt_tokens"] == 100
    assert result["completion_tokens"] == 20
    assert result["latency_seconds"] == pytest.approx(1.5, abs=0.01)
    assert result["confidence_level"] == "High"
    assert len(result["citations"]) == 1
    assert result["citations"][0].chunk_id == "c1"


def test_qa_node_drops_hallucinated_citations():
    from doc_qa.qa.chain import qa_node, State

    chunk = _chunk(chunk_id="real_id")
    llm = _mock_structured_llm(
        answer_text="The DSCR is 1.35x.",
        citations_data=[
            {"chunk_id": "real_id", "page_number": 3, "section_heading": "X", "quote": "x"},
            {"chunk_id": "hallucinated", "page_number": 9, "section_heading": "", "quote": "y"},
        ],
    )

    state = State(mode="qa", query="q", retrieved_chunks=[chunk])
    result = qa_node(state, llm=llm)

    kept_ids = [c.chunk_id for c in result["citations"]]
    assert kept_ids == ["real_id"]


def test_qa_node_falls_back_to_plain_text_on_structured_failure():
    from doc_qa.qa.chain import qa_node, State

    raw = MagicMock()
    raw.content = "plain fallback answer"
    raw.usage_metadata = {"input_tokens": 50, "output_tokens": 10}

    llm = MagicMock()
    # Simulate structured parse returning parsed=None (malformed JSON etc).
    structured = MagicMock()
    structured.invoke.return_value = {
        "raw": raw, "parsed": None, "parsing_error": "bad json"
    }
    llm.with_structured_output.return_value = structured
    llm.invoke.return_value = raw

    state = State(mode="qa", query="q", retrieved_chunks=[_chunk()])
    result = qa_node(state, llm=llm)

    assert result["answer"] == "plain fallback answer"
    assert result["citations"] == []
    # Plain-text fallback calls llm.invoke, not the structured wrapper.
    llm.invoke.assert_called_once()


def test_answer_question_routes_through_graph():
    from doc_qa.qa.chain import answer_question

    chunk = _chunk()
    llm = _mock_structured_llm(
        answer_text="The DSCR is 1.35x.",
        citations_data=[{
            "chunk_id": "c1", "page_number": 3, "section_heading": "F", "quote": "q",
        }],
        usage={"input_tokens": 80, "output_tokens": 15},
    )

    result = answer_question("What is the DSCR?", [chunk], llm)

    assert result.answer == "The DSCR is 1.35x."
    assert result.prompt_tokens == 80
    assert result.completion_tokens == 15
    assert result.confidence_level == "High"
    assert len(result.citations) == 1
    assert result.citations[0].chunk_id == "c1"
