# app.py
"""Streamlit entry point for the Document Q&A system.

Two modes:
  Chat — free-form questions against indexed documents
  Batch — upload a question sheet and get an answer grid
"""

import json
import logging
import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()  # load AZURE_OPENAI_* variables from .env file


@st.cache_data(show_spinner=False, max_entries=64)
def _cached_render_pdf_page(file_path: str, page_number: int) -> bytes:
    """Streamlit-cached wrapper around the PDF page renderer.

    Streamlit reruns the script on every interaction — without caching, each
    expander click would re-rasterize every page. Cache key is (path, page).
    """
    from doc_qa.utils.pdf_render import render_pdf_page
    return render_pdf_page(file_path, page_number)

logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Session state keys — all keys are defined here for documentation
# ---------------------------------------------------------------------------
# "uploaded_docs"  : dict {filename: {"doc_id": str, "status": "indexing"|"ready"}}
# "chat_history"   : list of {"role": "user"|"assistant", "content": str, "meta": dict|None}
# "qa_chain"       : chat LLM instance (AzureChatOpenAI or ChatOpenAI — env-driven, lazy-loaded)
# "batch_questions": list[str] from uploaded question sheet
# "batch_results"  : (answers_df, trace_df, export_bytes) from last batch run

if "uploaded_docs" not in st.session_state:
    st.session_state.uploaded_docs = {}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "batch_questions" not in st.session_state:
    st.session_state.batch_questions = []
if "batch_results" not in st.session_state:
    st.session_state.batch_results = None


def get_llm():
    """Lazy-load the chat LLM once and cache in session state.

    build_llm() auto-selects AzureChatOpenAI (work/Domino) or ChatOpenAI
    (local) based on env vars — see doc_qa/qa/chain.py::build_llm.
    """
    if st.session_state.qa_chain is None:
        from doc_qa.qa.chain import build_llm
        st.session_state.qa_chain = build_llm()
    return st.session_state.qa_chain


_SOURCES_DIR = Path("data/sources")


def ingest_file(uploaded_file) -> tuple[str, str]:
    """Write an uploaded file to disk, run ingest pipeline, build FAISS index.

    The original file is retained at data/sources/{doc_id}{ext} so the UI can
    render source pages later for audit — the primary audit feature for credit
    analysts. Returns (doc_id, persisted_source_path).
    """
    from doc_qa.ingest.extractor import ingest_document
    from doc_qa.ingest.chunker import chunk_raw
    from doc_qa.retrieval.vectorstore import build_index

    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(uploaded_file.getbuffer())
        tmp_path = f.name

    raw = ingest_document(tmp_path, uploaded_file.name)
    processed = chunk_raw(raw)
    if not processed:
        os.unlink(tmp_path)
        raise ValueError(f"No content extracted from {uploaded_file.name}")
    doc_id = processed[0].doc_id
    build_index(processed, doc_id)

    _SOURCES_DIR.mkdir(parents=True, exist_ok=True)
    source_path = _SOURCES_DIR / f"{doc_id}{suffix}"
    os.replace(tmp_path, source_path)
    return doc_id, str(source_path)


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("Document Q&A")

    # Mode toggle — drives which panel is shown in the main area
    mode = st.radio("Mode", ["Chat", "Batch"], horizontal=True)

    st.markdown("---")
    st.subheader("Upload documents")

    # Multi-file uploader — accepts the three supported formats
    new_files = st.file_uploader(
        "PDF, DOCX, or XLSX",
        type=["pdf", "docx", "xlsx"],
        accept_multiple_files=True,
        key="doc_uploader",
    )

    # Ingest any newly uploaded files that aren't already tracked
    if new_files:
        for f in new_files:
            if f.name not in st.session_state.uploaded_docs:
                st.session_state.uploaded_docs[f.name] = {
                    "doc_id": None, "status": "indexing",
                    "file_path": None, "file_type": Path(f.name).suffix.lstrip(".").lower(),
                }
                with st.spinner(f"Indexing {f.name}…"):
                    try:
                        doc_id, source_path = ingest_file(f)
                        st.session_state.uploaded_docs[f.name] = {
                            "doc_id": doc_id, "status": "ready",
                            "file_path": source_path,
                            "file_type": Path(f.name).suffix.lstrip(".").lower(),
                        }
                    except Exception as exc:
                        st.session_state.uploaded_docs[f.name] = {
                            "doc_id": None, "status": f"error: {exc}",
                            "file_path": None,
                            "file_type": Path(f.name).suffix.lstrip(".").lower(),
                        }

    # Show status badges for each uploaded document
    for name, info in st.session_state.uploaded_docs.items():
        color = {"ready": "green", "indexing": "orange"}.get(info["status"], "red")
        st.markdown(f":{color}[{info['status']}] {name}")

    st.markdown("---")

    # Active docs selector — query all or a chosen subset
    ready_docs = [
        {
            "filename": n, "doc_id": i["doc_id"],
            "file_path": i.get("file_path"), "file_type": i.get("file_type"),
        }
        for n, i in st.session_state.uploaded_docs.items()
        if i["status"] == "ready"
    ]
    active_doc_names = st.multiselect(
        "Active documents",
        options=[d["filename"] for d in ready_docs],
        default=[d["filename"] for d in ready_docs],
    )
    active_docs = [d for d in ready_docs if d["filename"] in active_doc_names]

    # Batch mode: question sheet uploader
    if mode == "Batch":
        st.markdown("---")
        st.subheader("Question sheet")
        q_file = st.file_uploader("Upload xlsx (requires 'questions' column)", type=["xlsx"])
        if q_file:
            import pandas as pd
            df = pd.read_excel(q_file)
            if "questions" in df.columns:
                st.session_state.batch_questions = df["questions"].dropna().tolist()
                st.success(f"{len(st.session_state.batch_questions)} questions loaded")
            else:
                st.error("Sheet must have a 'questions' column")


# ---------------------------------------------------------------------------
# MAIN PANEL
# ---------------------------------------------------------------------------

if mode == "Chat":
    st.header("Chat")

    from doc_qa.utils.confidence import confidence_color

    # Render full chat history (user + assistant messages)
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            # Show source chips and trace only for assistant messages
            if msg["role"] == "assistant" and msg.get("meta"):
                meta = msg["meta"]
                chunks = meta.get("retrieved_chunks", [])

                # Source chips: one per retrieved chunk
                if chunks:
                    chip_cols = st.columns(min(len(chunks), 5))
                    for i, chunk in enumerate(chunks[:5]):
                        with chip_cols[i]:
                            st.caption(
                                f"**{chunk['filename']}** · p.{chunk['page_number']} · "
                                f"{chunk['reranker_score']:.2f} · {chunk['extraction_method']}"
                            )

                # Confidence badge
                level = meta.get("confidence_level", "Low")
                color = confidence_color(level)
                st.markdown(
                    f'<span style="background:{color};color:white;padding:2px 8px;'
                    f'border-radius:4px;font-size:0.8em">{level} confidence</span>',
                    unsafe_allow_html=True,
                )

                # Expandable trace drawer — header row plus one card per source chunk
                with st.expander("Answer trace"):
                    st.write(f"**Document:** {meta.get('filename', 'N/A')}")
                    st.write(f"**Model:** {meta.get('model_deployment')} | "
                             f"**Tokens:** {meta.get('prompt_tokens')}+{meta.get('completion_tokens')} | "
                             f"**Latency:** {meta.get('latency_seconds', 0):.1f}s | "
                             f"**Time:** {meta.get('timestamp')}")

                    # Structured citations (if the LLM emitted any via
                    # with_structured_output). Each citation has already been
                    # validated against the retrieved chunk set; the chunk_id
                    # here is guaranteed to match one of the source cards below.
                    citations = meta.get("citations") or []
                    cited_ids = {cit["chunk_id"] for cit in citations}
                    if citations:
                        st.markdown("**Citations**")
                        for cit in citations:
                            st.markdown(
                                f"- p.{cit['page_number']} · "
                                f"{cit.get('section_heading') or 'no heading'} — "
                                f"\"{cit['quote']}\""
                            )
                        st.markdown("---")

                    # Per-chunk source cards: extracted text on the left, page render
                    # on the right. Primary audit mechanism for credit analysts —
                    # "where exactly in the document did the AI read this?"
                    for i, chunk in enumerate(chunks, start=1):
                        cited_tag = " · [cited]" if chunk["chunk_id"] in cited_ids else ""
                        label = (
                            f"Source {i} — {chunk['filename']} · p.{chunk['page_number']}"
                            f" · {chunk.get('section_heading') or 'no heading'}{cited_tag}"
                        )
                        with st.expander(label):
                            st.caption(
                                f"Method: {chunk['extraction_method']} · "
                                f"Type: {chunk.get('content_type', 'n/a')} · "
                                f"Chars: {chunk['char_count']} · "
                                f"Reranker: {chunk['reranker_score']:.3f}"
                            )
                            text_col, img_col = st.columns([1, 1])
                            with text_col:
                                st.markdown("**Extracted text**")
                                st.text_area(
                                    label="chunk_text",
                                    value=chunk.get("text", ""),
                                    height=300,
                                    key=f"chunk_text_{chunk['chunk_id']}_{i}",
                                    label_visibility="collapsed",
                                )
                            with img_col:
                                st.markdown(f"**Page {chunk['page_number']} preview**")
                                file_type = (chunk.get("file_type") or "").lower()
                                file_path = chunk.get("file_path")
                                if file_type == "pdf" and file_path:
                                    try:
                                        img = _cached_render_pdf_page(
                                            file_path, chunk["page_number"]
                                        )
                                        st.image(img, use_container_width=True)
                                    except Exception as exc:
                                        st.warning(f"Could not render page: {exc}")
                                else:
                                    st.info(
                                        "Page preview is PDF-only — DOCX and XLSX "
                                        "sources show extracted text only."
                                    )

    # Chat input box
    if question := st.chat_input("Ask a question about your documents…"):
        if not active_docs:
            st.warning("Upload and select at least one document first.")
        else:
            st.session_state.chat_history.append({"role": "user", "content": question, "meta": None})
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Searching and answering…"):
                    try:
                        from doc_qa.retrieval.vectorstore import load_index
                        from doc_qa.retrieval.retriever import retrieve
                        from doc_qa.qa.chain import (
                            answer_question, is_summarization_query, summarize_documents,
                        )
                        from doc_qa.metadata.logger import QueryLogger

                        doc_ids = [d["doc_id"] for d in active_docs]

                        # Summarization bypasses retrieval: a "summarize" query won't
                        # semantically match any specific chunk, so top-k retrieval
                        # starves the LLM. Send whole documents instead.
                        if is_summarization_query(question):
                            result = summarize_documents(question, doc_ids, get_llm())
                            chunks = result.retrieved_chunks
                        else:
                            index, meta_dict, position_map = load_index(doc_ids)
                            chunks = retrieve(question, index, meta_dict, position_map)
                            result = answer_question(question, chunks, get_llm())

                        st.markdown(result.answer)

                        # Log to SQLite audit trail
                        QueryLogger().log({
                            "query": result.query,
                            "answer": result.answer,
                            "filename": "; ".join(d["filename"] for d in active_docs),
                            "doc_id": "; ".join(doc_ids),
                            "confidence_level": result.confidence_level,
                            "top_chunk_page": chunks[0].page_number if chunks else "",
                            "top_chunk_section": chunks[0].section_heading if chunks else "",
                            "top_reranker_score": chunks[0].reranker_score if chunks else 0,
                            "prompt_tokens": result.prompt_tokens,
                            "completion_tokens": result.completion_tokens,
                            "latency_seconds": result.latency_seconds,
                            "timestamp": result.timestamp,
                            "model_deployment": result.model_deployment,
                            "extraction_methods_used": ",".join(
                                dict.fromkeys(c.extraction_method for c in chunks)),
                            "chunk_ids_used": ",".join(c.chunk_id for c in chunks),
                            "chunks_json": json.dumps([{
                                "chunk_id": c.chunk_id, "text": c.text[:200]
                            } for c in chunks]),
                        })

                        # Store metadata in session state for the trace drawer.
                        # file_path/file_type are copied per-chunk so each chunk is
                        # self-contained — the trace drawer needs no doc-lookup later.
                        file_lookup = {d["filename"]: d for d in active_docs}
                        meta_payload = {
                            "model_deployment": result.model_deployment,
                            "timestamp": result.timestamp,
                            "prompt_tokens": result.prompt_tokens,
                            "completion_tokens": result.completion_tokens,
                            "latency_seconds": result.latency_seconds,
                            "confidence_level": result.confidence_level,
                            "filename": "; ".join(d["filename"] for d in active_docs),
                            "citations": [
                                {
                                    "chunk_id": cit.chunk_id,
                                    "page_number": cit.page_number,
                                    "section_heading": cit.section_heading,
                                    "quote": cit.quote,
                                }
                                for cit in (result.citations or [])
                            ],
                            "retrieved_chunks": [
                                {
                                    "filename": c.filename,
                                    "page_number": c.page_number,
                                    "section_heading": c.section_heading,
                                    "extraction_method": c.extraction_method,
                                    "content_type": c.content_type,
                                    "char_count": c.char_count,
                                    "reranker_score": c.reranker_score,
                                    "chunk_id": c.chunk_id,
                                    "text": c.text,
                                    "file_path": file_lookup.get(c.filename, {}).get("file_path"),
                                    "file_type": file_lookup.get(c.filename, {}).get("file_type"),
                                }
                                for c in chunks
                            ],
                        }
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": result.answer,
                            "meta": meta_payload,
                        })
                    except Exception as exc:
                        st.error(f"Error: {exc}")
                        logging.exception("Chat error")

else:  # Batch mode
    from doc_qa.qa.batch import run_batch, export_to_excel
    st.header("Batch Q&A")
    st.info("Upload a question sheet (xlsx with a 'questions' column) and select "
            "documents. Click Run Batch to generate the answer grid.")

    # Preview the loaded question sheet
    if st.session_state.batch_questions:
        import pandas as pd
        st.subheader("Question preview")
        st.dataframe(pd.DataFrame({"questions": st.session_state.batch_questions}),
                     use_container_width=True)

    if st.button("Run Batch", disabled=not (active_docs and st.session_state.batch_questions)):
        progress = st.progress(0)
        status = st.status("Running batch…", expanded=True)

        def _cb(done, total):
            progress.progress(done / total)
            status.write(f"Completed {done}/{total}")

        try:
            answers_df, trace_df = run_batch(
                st.session_state.batch_questions,
                active_docs,
                chain=get_llm(),
                progress_callback=_cb,
            )
            with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
                export_path = tmp.name
            export_to_excel(answers_df, trace_df, export_path)
            with open(export_path, "rb") as fh:
                export_bytes = fh.read()
            os.unlink(export_path)
            st.session_state.batch_results = (answers_df, trace_df, export_bytes)
            status.update(label="Batch complete!", state="complete")
        except Exception as exc:
            status.update(label=f"Batch failed: {exc}", state="error")

    if st.session_state.batch_results:
        answers_df, trace_df, export_bytes = st.session_state.batch_results
        st.subheader("Answer grid")
        st.dataframe(answers_df, use_container_width=True)

        st.download_button(
            "Download Excel export",
            data=export_bytes,
            file_name="batch_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
