# doc_qa/qa/chain.py
"""LangGraph-backed Q&A with structured output and chunk-validated citations.

Architecture mirrors the Kronos AICE pattern:
  - Pydantic `Context` holds runtime params (model, temperature, api_version,
    summarize_max_chars) with env-var defaults.
  - Pydantic `State` carries messages (add_messages reducer), mode flag, query,
    retrieved chunks / doc ids, and all output fields.
  - A single `StateGraph` is compiled once at import and cached; a conditional
    edge on `state.mode` routes to `qa_node` or `summarize_node`.
  - `qa_node` uses `llm.with_structured_output(StructuredAnswer, include_raw=True)`
    with a plain-text fallback. Returned citations are validated against the
    retrieved chunk_ids — hallucinated ids are dropped, never displayed.

Top-level `answer_question()` and `summarize_documents()` remain as thin
wrappers so existing call sites (app.py, batch.py, tests) don't change.
"""

from __future__ import annotations

import logging
import os
import time
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Annotated, Any, List, Literal, Optional

from langgraph.graph.message import add_messages
from pydantic import BaseModel, ConfigDict, Field

from doc_qa.retrieval.retriever import RetrievedChunk
from doc_qa.utils.confidence import score_confidence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

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

CITATIONS: Each source block below starts with a header like
[chunk_id=XXX | Source: ... | Page N | Section: ...]. For every claim you make
in the answer, emit a Citation in the structured output with:
  - chunk_id: copy EXACTLY from the chunk header (never invent one)
  - page_number and section_heading: copy from the header
  - quote: a short exact string from the chunk that supports the claim
    (under 240 chars, copied verbatim)

Only cite chunk_ids that appear in the provided context.
"""

_SUMMARIZE_SYSTEM_PROMPT = """You are a precise financial document analyst. Produce a
structured summary of the document(s) provided below. Follow these rules:

1. Organize the summary by the major sections present in the document.
2. For each section, give a concise summary and list any key figures — reproduce
   numeric values exactly as they appear in the source.
3. Cite page numbers inline when referencing specific content, e.g. "(p.3)".
4. Flag anything notable — missing sections, inconsistent figures, or risks.
5. Base the summary strictly on the content provided — never infer beyond it.
"""

_SUMMARIZATION_KEYWORDS = (
    "summarize", "summarise",
    "summary of this", "summary of the document",
    "overview of this", "overview of the document",
    "what is this document", "what's this document",
    "what is this about", "what's this about",
    "tl;dr", "tldr",
)


def is_summarization_query(query: str) -> bool:
    """Detect whether a query should bypass retrieval and summarize full docs."""
    q = query.lower().strip()
    return any(kw in q for kw in _SUMMARIZATION_KEYWORDS)


# ---------------------------------------------------------------------------
# Pydantic models — Context, Citation, StructuredAnswer, State
# ---------------------------------------------------------------------------


class Context(BaseModel):
    """Runtime parameters for the LLM call, with env-var defaults.

    Mirrors the Kronos Context shape: one place to override model, temperature,
    API version, endpoint, and the summarization truncation cap. Instantiate
    via `Context.from_env()` or pass explicit overrides.
    """
    model: str = Field(
        default_factory=lambda: os.getenv("OPENAI_MODEL")
        or os.getenv("AZURE_OPENAI_DEPLOYMENT")
        or "gpt-4o"
    )
    temperature: float = 0.0
    api_version: Optional[str] = Field(
        default_factory=lambda: os.getenv("OPENAI_API_VERSION")
    )
    azure_endpoint: Optional[str] = Field(
        default_factory=lambda: os.getenv("AZURE_OPENAI_ENDPOINT")
    )
    summarize_max_chars: int = Field(
        default_factory=lambda: int(os.getenv("SUMMARIZE_MAX_CHARS", "100000"))
    )

    @classmethod
    def from_env(cls) -> "Context":
        return cls()


class Citation(BaseModel):
    """A single citation tying a claim in the answer back to one retrieved chunk.

    chunk_id must match exactly one chunk_id from the retrieved set — the graph
    drops citations whose chunk_id doesn't validate, so the UI never renders a
    hallucinated source.
    """
    chunk_id: str = Field(
        description="Exact chunk_id copied from the chunk header in the provided context."
    )
    page_number: int = Field(description="1-indexed page number of the cited content.")
    section_heading: str = Field(
        default="", description="Section heading of the cited chunk, or empty."
    )
    quote: str = Field(
        description="Short exact quote from the chunk (verbatim, under 240 chars) "
                    "that supports the claim."
    )


class StructuredAnswer(BaseModel):
    """Schema the LLM is asked to populate via with_structured_output."""
    answer: str = Field(description="Answer to the user's question in prose form.")
    citations: List[Citation] = Field(
        default_factory=list,
        description="Every chunk the answer draws on, one Citation per claim."
    )


# RetrievedChunk is a dataclass — pydantic needs arbitrary_types_allowed to
# hold it as a State field. That's fine: State is internal plumbing, not
# serialized anywhere.
class State(BaseModel):
    """Graph state — input + output fields. `mode` selects the node to run."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    messages: Annotated[List[Any], add_messages] = Field(default_factory=list)
    mode: Literal["qa", "summarize"]

    # Inputs (populated before invoke)
    query: str = ""
    retrieved_chunks: List[RetrievedChunk] = Field(default_factory=list)
    doc_ids: List[str] = Field(default_factory=list)
    summarize_max_chars: Optional[int] = None  # per-call override of Context default

    # Outputs (populated by the node)
    answer: str = ""
    citations: List[Citation] = Field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_seconds: float = 0.0
    timestamp: str = ""
    model_deployment: str = ""
    confidence_level: str = ""


@dataclass
class AnswerResult:
    """Flat answer record returned to callers. Unchanged field set — batch.py,
    app.py, and audit logging already depend on it."""
    query: str
    answer: str
    retrieved_chunks: List[RetrievedChunk]
    prompt_tokens: int
    completion_tokens: int
    latency_seconds: float
    timestamp: str
    model_deployment: str
    confidence_level: str  # High | Medium | Low
    citations: List[Citation] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.citations is None:
            self.citations = []


# ---------------------------------------------------------------------------
# LLM construction + override for tests
# ---------------------------------------------------------------------------

# ContextVar so a caller (tests, batch runner) can inject a specific LLM for
# the duration of one invoke() call without mutating module-level state.
_llm_override: ContextVar[Optional[Any]] = ContextVar("_llm_override", default=None)

_LLM_SINGLETON: Optional[Any] = None


def build_llm(context: Optional[Context] = None):
    """Create the chat LLM, matching the Kronos AICE formula.

    Auto-selects auth based on env:
      - AZURE_OPENAI_DEPLOYMENT set → AzureChatOpenAI + DefaultAzureCredential
        bearer token (production / Domino / work — no API key).
      - Else → ChatOpenAI + OPENAI_API_KEY (local dev).

    Uses the Context for model/temperature defaults.
    """
    ctx = context or Context.from_env()

    if os.getenv("AZURE_OPENAI_DEPLOYMENT"):
        from azure.identity import DefaultAzureCredential, get_bearer_token_provider
        from langchain_openai import AzureChatOpenAI

        token_provider = get_bearer_token_provider(
            DefaultAzureCredential(),
            "https://cognitiveservices.azure.com/.default",  # literal — do not change
        )
        kwargs = dict(
            azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
            api_version=os.environ["OPENAI_API_VERSION"],
            azure_ad_token_provider=token_provider,
            temperature=ctx.temperature,
        )
        # Only set azure_endpoint if the env provides one — some platforms
        # (Domino proxy) inject it for you.
        if os.getenv("AZURE_OPENAI_ENDPOINT"):
            kwargs["azure_endpoint"] = os.environ["AZURE_OPENAI_ENDPOINT"]
        llm = AzureChatOpenAI(**kwargs)
        logger.info("LLM initialised (Azure): deployment=%s", kwargs["azure_deployment"])
        return llm

    # Local dev fallback — OpenAI API key.
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model=ctx.model,
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=ctx.temperature,
    )
    logger.info("LLM initialised (OpenAI): model=%s", ctx.model)
    return llm


def _get_llm():
    """Return the LLM for the current call — override first, else singleton."""
    override = _llm_override.get()
    if override is not None:
        return override
    global _LLM_SINGLETON
    if _LLM_SINGLETON is None:
        _LLM_SINGLETON = build_llm()
    return _LLM_SINGLETON


# ---------------------------------------------------------------------------
# Context formatting
# ---------------------------------------------------------------------------


def _format_context(chunks: List[RetrievedChunk]) -> str:
    """Assemble the Q&A context block. chunk_id is in the header so the LLM
    can echo it back in structured citations."""
    parts = []
    for c in sorted(chunks, key=lambda x: x.rank):
        header = (
            f"[chunk_id={c.chunk_id} | Source: {c.filename} | Page {c.page_number} | "
            f"Section: {c.section_heading} | Method: {c.extraction_method} | "
            f"Reranker score: {c.reranker_score:.2f}]"
        )
        parts.append(f"{header}\n{c.text}")
    return "\n\n---\n\n".join(parts)


def _validate_citations(
    citations: List[Citation], chunks: List[RetrievedChunk]
) -> List[Citation]:
    """Drop citations whose chunk_id isn't in the retrieved set.

    Why: we instruct the LLM to echo chunk_ids from the context header. A
    chunk_id that doesn't match means the LLM hallucinated a source — surfacing
    it in the UI would poison the audit trail. Log and drop.
    """
    valid_ids = {c.chunk_id for c in chunks}
    kept: List[Citation] = []
    for cit in citations:
        if cit.chunk_id in valid_ids:
            kept.append(cit)
        else:
            logger.warning(
                "Dropping hallucinated citation with unknown chunk_id=%s", cit.chunk_id
            )
    return kept


def _deployment_name() -> str:
    """Resolve model name for audit logging across local/Azure."""
    return (
        os.getenv("AZURE_OPENAI_DEPLOYMENT")
        or os.getenv("OPENAI_MODEL")
        or "unknown"
    )


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def qa_node(state: State, llm=None) -> dict:
    """LangGraph node: run structured-output Q&A against retrieved chunks.

    Returns a dict of state updates. LangGraph coerces dict input at the graph
    boundary, so this node always sees a hydrated State.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    chunks: List[RetrievedChunk] = state.retrieved_chunks or []
    query: str = state.query

    if llm is None:
        llm = _get_llm()

    context_block = _format_context(chunks)
    messages = [
        SystemMessage(content=_WRAPPER_SYSTEM_PROMPT),
        HumanMessage(content=f"Context:\n{context_block}\n\nQuestion: {query}"),
    ]

    t0 = time.time()
    answer_text = ""
    citations: List[Citation] = []
    raw_response = None

    try:
        structured_llm = llm.with_structured_output(
            StructuredAnswer, include_raw=True
        )
        result = structured_llm.invoke(messages)
        raw_response = result.get("raw") if isinstance(result, dict) else None
        parsed = result.get("parsed") if isinstance(result, dict) else None
        parsing_error = result.get("parsing_error") if isinstance(result, dict) else None

        if parsed is None:
            raise ValueError(f"structured parse returned None: {parsing_error}")

        answer_text = parsed.answer
        citations = _validate_citations(parsed.citations, chunks)
    except Exception as exc:
        # Silent fallback to plain text per Kronos pattern. Citations will be
        # empty — UI shows all retrieved chunks instead of a filtered set.
        logger.warning("Structured output failed (%s); falling back to plain text", exc)
        raw_response = llm.invoke(messages)
        answer_text = raw_response.content if raw_response is not None else ""
        citations = []

    latency = time.time() - t0
    usage = getattr(raw_response, "usage_metadata", {}) or {}
    top_score = chunks[0].reranker_score if chunks else 0.0

    return {
        "answer": answer_text,
        "citations": citations,
        "prompt_tokens": usage.get("input_tokens", 0),
        "completion_tokens": usage.get("output_tokens", 0),
        "latency_seconds": latency,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_deployment": _deployment_name(),
        "confidence_level": score_confidence(top_score),
    }


def summarize_node(state: State, llm=None) -> dict:
    """LangGraph node: summarize all chunks for the given doc_ids in order."""
    from doc_qa.retrieval.vectorstore import load_index
    from langchain_core.messages import HumanMessage, SystemMessage

    query: str = state.query
    doc_ids: List[str] = state.doc_ids or []
    max_chars_override = state.summarize_max_chars

    max_chars = (
        max_chars_override
        if max_chars_override is not None
        else Context.from_env().summarize_max_chars
    )

    if llm is None:
        llm = _get_llm()

    _, metadata_dict, _ = load_index(doc_ids)
    ordered = sorted(
        metadata_dict.values(),
        key=lambda m: (m["doc_id"], m["page_number"], m["chunk_index"]),
    )

    parts: List[str] = []
    used: List[dict] = []
    total = 0
    truncated = False
    for m in ordered:
        header = (
            f"[chunk_id={m['chunk_id']} | Source: {m['filename']} | "
            f"Page {m['page_number']} | Section: {m['section_heading']} | "
            f"Method: {m['extraction_method']}]"
        )
        block = f"{header}\n{m['text']}"
        if total + len(block) > max_chars:
            truncated = True
            break
        parts.append(block)
        used.append(m)
        total += len(block)

    context_block = "\n\n---\n\n".join(parts)
    if truncated:
        context_block += (
            "\n\n[Note: document content truncated to fit context window — "
            "later sections may not appear in the summary.]"
        )

    messages = [
        SystemMessage(content=_SUMMARIZE_SYSTEM_PROMPT),
        HumanMessage(content=f"Document content:\n{context_block}\n\nTask: {query}"),
    ]

    t0 = time.time()
    response = llm.invoke(messages)
    latency = time.time() - t0

    usage = getattr(response, "usage_metadata", {}) or {}

    synthetic_chunks = [
        RetrievedChunk(
            chunk_id=m["chunk_id"],
            doc_id=m["doc_id"],
            filename=m["filename"],
            file_type=m["file_type"],
            upload_timestamp=m["upload_timestamp"],
            page_count=m["page_count"],
            page_number=m["page_number"],
            chunk_index=m["chunk_index"],
            section_heading=m["section_heading"],
            extraction_method=m["extraction_method"],
            content_type=m["content_type"],
            text=m["text"],
            char_count=m["char_count"],
            embedding_text=m["embedding_text"],
            bounding_box=m.get("bounding_box"),
            faiss_score=0.0,
            reranker_score=1.0,
            rank=rank,
        )
        for rank, m in enumerate(used, start=1)
    ]

    return {
        "answer": response.content,
        "citations": [],
        "retrieved_chunks": synthetic_chunks,
        "prompt_tokens": usage.get("input_tokens", 0),
        "completion_tokens": usage.get("output_tokens", 0),
        "latency_seconds": latency,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_deployment": _deployment_name(),
        "confidence_level": "High",  # whole-document context — no retrieval uncertainty
    }


# ---------------------------------------------------------------------------
# StateGraph — compiled once, cached, routed by state.mode
# ---------------------------------------------------------------------------

_COMPILED_GRAPH = None


def _route(state: State) -> str:
    return state.mode


def _build_graph():
    from langgraph.graph import StateGraph, START, END

    graph = StateGraph(State)
    graph.add_node("qa", qa_node)
    graph.add_node("summarize", summarize_node)
    graph.add_conditional_edges(
        START, _route, {"qa": "qa", "summarize": "summarize"}
    )
    graph.add_edge("qa", END)
    graph.add_edge("summarize", END)
    return graph.compile()


def _get_graph():
    global _COMPILED_GRAPH
    if _COMPILED_GRAPH is None:
        _COMPILED_GRAPH = _build_graph()
    return _COMPILED_GRAPH


# ---------------------------------------------------------------------------
# Public API — thin wrappers over graph.invoke
# ---------------------------------------------------------------------------


def _to_answer_result(
    final: dict, query: str, retrieved_chunks: List[RetrievedChunk]
) -> AnswerResult:
    """Extract graph output into the flat AnswerResult callers expect."""
    # LangGraph returns a dict-like mapping of state fields. Pull either the
    # node-returned retrieved_chunks (summarize path) or the ones we passed in.
    out_chunks = final.get("retrieved_chunks") or retrieved_chunks
    return AnswerResult(
        query=query,
        answer=final.get("answer", ""),
        retrieved_chunks=out_chunks,
        prompt_tokens=final.get("prompt_tokens", 0),
        completion_tokens=final.get("completion_tokens", 0),
        latency_seconds=final.get("latency_seconds", 0.0),
        timestamp=final.get("timestamp", ""),
        model_deployment=final.get("model_deployment", ""),
        confidence_level=final.get("confidence_level", ""),
        citations=final.get("citations", []) or [],
    )


def answer_question(
    query: str,
    retrieved_chunks: List[RetrievedChunk],
    llm=None,
) -> AnswerResult:
    """Run a Q&A query through the graph and return a flat AnswerResult.

    `llm` is optional: if provided, it's used for this call only (via the
    ContextVar override). Otherwise the graph uses the module singleton
    built by build_llm().
    """
    token = _llm_override.set(llm) if llm is not None else None
    try:
        state = State(mode="qa", query=query, retrieved_chunks=retrieved_chunks)
        final = _get_graph().invoke(state)
    finally:
        if token is not None:
            _llm_override.reset(token)
    return _to_answer_result(final, query, retrieved_chunks)


def summarize_documents(
    query: str,
    doc_ids: List[str],
    llm=None,
    max_chars: Optional[int] = None,
) -> AnswerResult:
    """Run a summarization query through the graph and return a flat AnswerResult."""
    token = _llm_override.set(llm) if llm is not None else None
    try:
        state = State(
            mode="summarize",
            query=query,
            doc_ids=doc_ids,
            summarize_max_chars=max_chars,
        )
        final = _get_graph().invoke(state)
    finally:
        if token is not None:
            _llm_override.reset(token)
    return _to_answer_result(final, query, final.get("retrieved_chunks", []))


# ---------------------------------------------------------------------------
# MLflow ResponsesAgent wrapper (Kronos pattern)
# ---------------------------------------------------------------------------
#
# Truly dormant at runtime: the wrapper stores a callable, not the compiled
# graph, so `set_model` doesn't force graph construction at import time. The
# graph is resolved lazily inside `predict_stream` — which only fires when
# MLflow serves the model. Local/Streamlit imports pay nothing.
#
# Deploy via `mlflow.pyfunc.log_model(python_model="doc_qa/qa/chain.py")`.
# Adapts Gandalf's retrieval flow to the Responses Agent interface:
#   - Query: last user message in request.input
#   - doc_ids: request.custom_inputs["doc_ids"]

try:
    import mlflow
    from mlflow.pyfunc import ResponsesAgent
    from mlflow.types.responses import (
        output_to_responses_items_stream,
        to_chat_completions_input,
        ResponsesAgentResponse,
    )
except ImportError:
    # mlflow not installed — wrapper stays dormant, module imports cleanly.
    pass
else:

    class GraphResponsesAgent(ResponsesAgent):
        """MLflow ResponsesAgent wrapping the Gandalf StateGraph.

        Attribute name `agent` is mandatory — MLflow introspects it. We bind
        it to a callable that builds the graph on first access so importing
        this module doesn't trigger graph construction.
        """

        def __init__(self, agent_factory):
            self._agent_factory = agent_factory
            self._agent = None

        @property
        def agent(self):
            if self._agent is None:
                self._agent = self._agent_factory()
            return self._agent

        def predict(self, request):
            outputs = [
                e.item
                for e in self.predict_stream(request)
                if e.type == "response.output_item.done"
            ]
            return ResponsesAgentResponse(
                output=outputs,
                custom_outputs=request.custom_inputs,
            )

        def predict_stream(self, request):
            msgs = to_chat_completions_input([i.model_dump() for i in request.input])
            query = ""
            for m in reversed(msgs):
                if m.get("role") == "user":
                    query = m.get("content", "")
                    break

            custom = request.custom_inputs or {}
            doc_ids = custom.get("doc_ids", []) or []
            mode = custom.get("mode", "qa")

            if mode == "summarize":
                state_in = {
                    "messages": msgs,
                    "mode": "summarize",
                    "query": query,
                    "doc_ids": doc_ids,
                    "summarize_max_chars": custom.get("summarize_max_chars"),
                }
            else:
                from doc_qa.retrieval.retriever import retrieve
                from doc_qa.retrieval.vectorstore import load_index

                index, meta_dict, position_map = load_index(doc_ids)
                chunks = retrieve(query, index, meta_dict, position_map)
                state_in = {
                    "messages": msgs,
                    "mode": "qa",
                    "query": query,
                    "retrieved_chunks": chunks,
                }

            for _, events in self.agent.stream(
                state_in, context=request.metadata, stream_mode=["updates"]
            ):
                for node_data in events.values():
                    yield from output_to_responses_items_stream(
                        node_data.get("messages", [])
                    )

    try:
        mlflow.models.set_model(GraphResponsesAgent(_get_graph))
    except (AttributeError, TypeError) as _exc:
        # Surface MLflow API drift loudly without breaking module import.
        logger.warning("MLflow set_model failed (API drift?): %s", _exc)
