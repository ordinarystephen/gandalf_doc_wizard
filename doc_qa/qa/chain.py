# doc_qa/qa/chain.py
"""LangGraph QA node and AnswerResult dataclass.

Pattern: UI → service/orchestrator → graph state → wrapper prompt + inputs → llm.invoke()
Auth: DefaultAzureCredential + bearer token provider — no API key.
"""

import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from doc_qa.retrieval.retriever import RetrievedChunk
from doc_qa.utils.confidence import score_confidence

logger = logging.getLogger(__name__)

# Wrapper system prompt applied at the node level — not baked into the LLM constructor.
# This keeps the prompt version-controlled here rather than scattered across callers.
_WRAPPER_SYSTEM_PROMPT = """You are a precise financial document analyst. Answer questions
using only the context chunks provided below. Follow these rules strictly:

1. Cite the source section heading and page number inline when possible,
   e.g. "(Financial Analysis, p.3)".
2. For numeric values from tables, reproduce the exact figure — never round
   or paraphrase a number from a source table.
3. If the answer cannot be found in the provided context, respond with exactly:
   "I could not find sufficient information in the provided documents to answer
   this question." — never infer or hallucinate beyond the context.
4. Keep your answer focused on what the context supports.
"""


class QAState(dict):
    """Graph state passed through the LangGraph QA node.

    Subclasses dict so LangGraph can merge partial state updates.
    Type annotations are for documentation only — values are plain dict keys.
    """
    # Input fields (set before calling qa_node)
    query: str
    retrieved_chunks: List[RetrievedChunk]
    # Output fields (populated by qa_node)
    answer: str
    prompt_tokens: int
    completion_tokens: int
    latency_seconds: float
    timestamp: str
    model_deployment: str
    confidence_level: str


@dataclass
class AnswerResult:
    """Flat answer record returned by answer_question() for logging and UI display."""
    query: str
    answer: str
    retrieved_chunks: List[RetrievedChunk]
    prompt_tokens: int
    completion_tokens: int
    latency_seconds: float
    timestamp: str
    model_deployment: str
    confidence_level: str  # High | Medium | Low


def build_llm():
    """Construct AzureChatOpenAI authenticated via DefaultAzureCredential.

    Deployment and API version are read from env vars with sensible defaults.
    azure_endpoint is optional — Domino environments provide it via a local proxy
    automatically; set AZURE_OPENAI_ENDPOINT explicitly for local/non-Domino use.
    No API key — Azure AD bearer token is obtained at runtime.
    """
    from azure.identity import DefaultAzureCredential, get_bearer_token_provider
    from langchain_openai import AzureChatOpenAI

    credential = DefaultAzureCredential()
    token_provider = get_bearer_token_provider(
        credential, "https://cognitiveservices.azure.com/.default"
    )
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

    kwargs = dict(
        azure_deployment=deployment,
        azure_ad_token_provider=token_provider,
        api_version=os.getenv("OPENAI_API_VERSION", "2025-04-01-preview"),
        temperature=0,
    )
    if endpoint:
        kwargs["azure_endpoint"] = endpoint

    llm = AzureChatOpenAI(**kwargs)
    logger.info("LLM initialised: deployment=%s endpoint=%s", deployment, endpoint or "proxy")
    return llm


def _format_context(chunks: List[RetrievedChunk]) -> str:
    """Assemble context block ordered by reranker rank.

    Source metadata prefixed to each chunk lets the LLM cite specific pages
    and sections — the primary audit mechanism for credit analysts.
    """
    parts = []
    for c in sorted(chunks, key=lambda x: x.rank):
        header = (
            f"[Source: {c.filename} | Page {c.page_number} | "
            f"Section: {c.section_heading} | Method: {c.extraction_method} | "
            f"Reranker score: {c.reranker_score:.2f}]"
        )
        parts.append(f"{header}\n{c.text}")
    return "\n\n---\n\n".join(parts)


def qa_node(state: QAState, llm=None) -> Dict[str, Any]:
    """LangGraph node: apply wrapper prompt, call llm.invoke(), return updated state.

    Args:
        state: Graph state with 'query' and 'retrieved_chunks'.
        llm: AzureChatOpenAI instance. If None, build_llm() is called.

    Returns:
        Dict with updated state fields: answer, prompt_tokens, completion_tokens,
        latency_seconds, timestamp, model_deployment, confidence_level.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    if llm is None:
        llm = build_llm()

    chunks = state.get("retrieved_chunks", [])
    query = state["query"]
    context = _format_context(chunks)

    messages = [
        SystemMessage(content=_WRAPPER_SYSTEM_PROMPT),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}"),
    ]

    t0 = time.time()
    response = llm.invoke(messages)
    latency = time.time() - t0

    usage = getattr(response, "usage_metadata", {}) or {}
    top_score = chunks[0].reranker_score if chunks else 0.0
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "unknown")

    return {
        **state,
        "answer": response.content,
        "prompt_tokens": usage.get("input_tokens", 0),
        "completion_tokens": usage.get("output_tokens", 0),
        "latency_seconds": latency,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_deployment": deployment,
        "confidence_level": score_confidence(top_score),
    }


def answer_question(
    query: str,
    retrieved_chunks: List[RetrievedChunk],
    llm,
) -> AnswerResult:
    """Convenience wrapper: run qa_node and return a flat AnswerResult.

    Used by the Streamlit UI and batch runner without wiring a full LangGraph graph.
    """
    state = qa_node(
        QAState(
            query=query, retrieved_chunks=retrieved_chunks,
            answer="", prompt_tokens=0, completion_tokens=0,
            latency_seconds=0.0, timestamp="", model_deployment="",
            confidence_level="",
        ),
        llm=llm,
    )
    return AnswerResult(
        query=query,
        answer=state["answer"],
        retrieved_chunks=retrieved_chunks,
        prompt_tokens=state["prompt_tokens"],
        completion_tokens=state["completion_tokens"],
        latency_seconds=state["latency_seconds"],
        timestamp=state["timestamp"],
        model_deployment=state["model_deployment"],
        confidence_level=state["confidence_level"],
    )
