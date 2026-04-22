# Gandalf — Document Extraction & Q&A System

## Project Overview
Streamlit-based document Q&A system for credit analysts. Users upload PDF/DOCX/XLSX files, which are chunked and indexed in per-document FAISS indexes. Questions are answered through a compiled LangGraph `StateGraph` that calls Azure OpenAI (DefaultAzureCredential + bearer token) at work or standard OpenAI + API key at home — same code path, env-driven selection. Full source tracing and SQLite audit logging throughout.

## Branches

| Branch | Purpose |
|--------|---------|
| `main` | Production — Azure OpenAI via DefaultAzureCredential (Domino/work) |
| `local-testing` | Local dev — standard OpenAI via OPENAI_API_KEY |
| `overhaul` | Active refactor — Kronos-style LangGraph StateGraph + Pydantic Context/State + structured output; HF embeddings/reranker replaced with OpenAI embeddings API |

## Current Branch: `overhaul`

Forked from `local-testing`. Destination is work/Domino — the branch exists to fix HuggingFace issues that were blocking the app at work and to align the LLM/MLflow plumbing with the Kronos AICE pattern.

What's done:
1. **StateGraph rewrite** — compiled `StateGraph` singleton with `mode`-based conditional routing, Pydantic `Context` (runtime config) + `State` (graph state w/ `add_messages` reducer), `Citation` + `StructuredAnswer` models, and `with_structured_output(StructuredAnswer, include_raw=True)` with plain-text fallback and hallucinated-`chunk_id` drop.
2. **HuggingFace removed** — local MiniLM embedder and CrossEncoder reranker dropped. FAISS now runs on OpenAI/Azure embeddings (`text-embedding-3-small`, 1536-dim). Retrieval is single-stage FAISS top-n; `reranker_score` field repurposed to cosine similarity derived from L2 distance.
3. **Kronos LLM formula in `build_llm()`** — `AzureChatOpenAI` + `DefaultAzureCredential` + bearer token (`https://cognitiveservices.azure.com/.default`), uses `api_version=`, `azure_endpoint` omitted when the env doesn't provide one. Auto-falls back to `ChatOpenAI` + `OPENAI_API_KEY` when `AZURE_OPENAI_DEPLOYMENT` is unset, so the same branch runs at home for local testing.
4. **Kronos MLflow wrapper** — bottom of `chain.py` registers `GraphResponsesAgent(ResponsesAgent)` via `mlflow.models.set_model(...)` inside a `try/except ImportError` so the module imports cleanly without mlflow installed. Deployable as-is via `mlflow.pyfunc.log_model(python_model="doc_qa/qa/chain.py")`.

All 42 tests pass. Not yet tested end-to-end on real Azure — that's the next step when back on work/Domino.

## Running the App

```bash
cd /Users/stephencostello/Documents/Dev_Work/gandalf
source .venv/bin/activate
streamlit run app.py
```

The Azure and OpenAI code paths live in the same files and auto-select based on env vars — no editing required to switch machines.

**Local testing `.env` requires:**
```
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o
```

**Work/Domino `.env` requires:**
```
AZURE_OPENAI_DEPLOYMENT=gpt-4o
AZURE_EMBEDDING_DEPLOYMENT=text-embedding-3-small
OPENAI_API_VERSION=2025-04-01-preview
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/   # optional — omit if Domino proxy injects it
MLFLOW_TRACKING_URI=databricks   # optional — only if AICE-style tracking is enabled
```

Auth is picked in [chain.py::build_llm](doc_qa/qa/chain.py) and both embedders: `AZURE_OPENAI_DEPLOYMENT` set → `AzureChatOpenAI` + `DefaultAzureCredential` + bearer token (scope `https://cognitiveservices.azure.com/.default`); otherwise `ChatOpenAI` + `OPENAI_API_KEY`. Embeddings follow the same rule keyed on `AZURE_EMBEDDING_DEPLOYMENT`. No HuggingFace env vars needed on either branch.

## Running Tests

```bash
python3 -m pytest tests/ -v
```

## Key Post-Implementation Changes (since original build)

| File | Change |
|------|--------|
| `doc_qa/qa/chain.py` | **Overhaul rewrite.** Pydantic `Context` (model/temperature/api_version/azure_endpoint/summarize_max_chars with env-var defaults) + `State` (messages w/ `add_messages`, `mode: Literal["qa","summarize"]`, i/o fields) + `Citation` + `StructuredAnswer`. Compiled `StateGraph` singleton routed by `state.mode`. `qa_node` uses `llm.with_structured_output(StructuredAnswer, include_raw=True)` with plain-text fallback; `_validate_citations()` drops hallucinated `chunk_id`s. `answer_question()` / `summarize_documents()` are thin graph wrappers; `llm` kwarg honored per-call via `ContextVar` (for tests). **`build_llm()` follows the Kronos formula** — `AzureChatOpenAI` + `DefaultAzureCredential` + bearer token when `AZURE_OPENAI_DEPLOYMENT` is set, else `ChatOpenAI` + `OPENAI_API_KEY`. Bottom of the file registers a `GraphResponsesAgent(ResponsesAgent)` via `mlflow.models.set_model(...)` (guarded `try/except`) so the module is deployable as an MLflow pyfunc without editing. |
| `doc_qa/ingest/chunker.py` | Replaced `SemanticChunker` with `RecursiveCharacterTextSplitter` (chunk_size=800, overlap=100) for speed. Embedding model cached at module level |
| `doc_qa/ingest/pdf_extractor.py` | Camelot skipped on pages with <200 chars to prevent hang on image-heavy pages |
| `doc_qa/retrieval/vectorstore.py` | **Overhaul:** embedder swapped from local MiniLM (HuggingFace) to OpenAI `text-embedding-3-small` via API. `_get_embeddings()` auto-switches to `AzureOpenAIEmbeddings` + `DefaultAzureCredential` bearer token when `AZURE_EMBEDDING_DEPLOYMENT` is set — same pattern as `chain.build_llm()`. Model name in module-level `EMBEDDING_MODEL` constant — adjust there to switch. |
| `doc_qa/retrieval/retriever.py` | **Overhaul:** same embedder + auto-switch as vectorstore. CrossEncoder reranker fully removed — retrieval is now single-stage FAISS top-n over OpenAI embeddings. `RetrievedChunk.reranker_score` field preserved for downstream compatibility but now populated with cosine similarity derived from FAISS L2 distance (`1 - L2²/2`, unit-norm). Confidence thresholds in `utils/confidence.py` (0.85/0.60) were calibrated for CrossEncoder outputs and may need recalibration. |
| `doc_qa/utils/pdf_render.py` | NEW — `render_pdf_page()` rasterizes PDF pages to PNG via pymupdf for the trace-drawer source preview |
| `app.py` | Source files persisted to `data/sources/{doc_id}.{ext}` after ingest; trace drawer rebuilt as per-chunk expanders with extracted text + rendered page image (PDF only). `@st.cache_data` wraps page rendering. Summarization routed via `is_summarization_query()`. **Overhaul:** trace drawer now shows a "Citations" section (validated quotes) + `[cited]` tag on chunks the LLM actually used. |
| `tests/test_chain.py` | **Overhaul rewrite.** Covers structured-output happy path, hallucinated-citation drop, plain-text fallback, and graph routing via `answer_question`. |
| `tests/test_retriever.py` | **Overhaul:** rewritten around FAISS-only retrieval. Asserts similarity-descending order and `reranker_score = 1 - L2²/2` clamped to [0,1]. |
| `tests/test_vectorstore.py` | **Overhaul:** mocks `_get_embeddings` to return deterministic unit vectors so `build_index` doesn't hit the OpenAI API during tests. |
| `requirements.txt` | **Overhaul:** dropped `langchain-huggingface`, `sentence-transformers`, `torchvision`. Embeddings now come from OpenAI/Azure via `langchain-openai`. Pinned `langgraph-checkpoint==2.1.1` and uncommented `mlflow[databricks]==3.7.0` to match the Kronos pinned set. |

## Summarization Path

Queries containing "summarize", "summary of this/the document", "overview of this/the document", "what is this document", "tl;dr", etc. bypass FAISS retrieval. The full document (all chunks in document order) is sent to the LLM with a summarization-specific system prompt. Truncation cap defaults to 100,000 chars (~25k tokens) to stay under OpenAI Tier 1's 30k TPM rate limit. Override via `.env`:

```
SUMMARIZE_MAX_CHARS=400000   # bump to ~100k tokens once on paid tier
```

## Source Preview (Trace Drawer)

After every Chat answer, the "Answer trace" expander contains one card per retrieved chunk: left side shows extracted text, right side renders the actual PDF page via pymupdf at 110 DPI. Page renders are cached via `@st.cache_data` (max 64 entries) so re-opening expanders is instant. DOCX/XLSX sources show text only — page preview is PDF-only. Source files live at `data/sources/{doc_id}.{ext}`.

## OpenAI Tier 1 Rate Limits (applies only in OpenAI fallback mode)

When running against your personal `OPENAI_API_KEY` at home, new accounts start at Tier 1 with **30,000 TPM** on gpt-4o:
- `SUMMARIZE_MAX_CHARS=100000` default (~25k input tokens, leaves headroom)
- Full-doc summarization of 25+ page PDFs will be truncated; output note warns the user
- Upgrade path: $50 spent + 7 days since first payment → Tier 2 (500k TPM)

Azure (work/Domino) does not hit this — the organization's Azure quota is the only limit.

## Pending: Ingestion Speedups (planned, not yet implemented)

Three changes queued to speed up initial ingestion (deferred while overhaul branch lands):
1. **Parallelize per-page extraction** in `pdf_extractor.py` using `ProcessPoolExecutor` — pages are independent; expected 3-5× speedup on multi-core
2. **Vision OCR toggle** — sidebar checkbox "Extract image tables (slower)" default off; skips per-image GPT-4o vision calls that dominate latency on image-heavy PDFs
3. **Live progress bar** in `app.py` replacing the spinner — show "Extracting page N/total" so long ingests feel responsive

## Pending: Port overhaul to `main`

`overhaul` now contains all the work that `main` needs — StateGraph rewrite, OpenAI/Azure embedder auto-switch, MLflow wrapper, HF removed. Porting to `main` should be a straight merge: no code edits required, only env vars change at deploy time. Once confirmed end-to-end on work (Azure auth + MLflow), fast-forward `main` to `overhaul`.

## Test Flakiness (historical)

The full-suite segfault in `tests/test_vectorstore.py` — caused by native-library import-ordering between `langgraph`/`pydantic` and `transformers` on macOS + LibreSSL 2.8.3 — should no longer reproduce on the `overhaul` branch now that HuggingFace/transformers has been removed from the stack. If the full suite runs clean, the `models/` folder can also be deleted to reclaim disk.

## Legacy Model Paths (no longer loaded)

The local MiniLM embedder and CrossEncoder reranker were used on `main` and `local-testing`. They are not loaded on `overhaul`. The `models/` folder can be deleted once `overhaul` ships back to `main` (plus the Azure equivalent is wired up there):

| Model | Local path (unused on overhaul) |
|-------|----------------------------------|
| `all-MiniLM-L6-v2` | `models/all-MiniLM-L6-v2/.../c9745ed1d9f207416be6d2e6f8de32d1f16199bf` |
| `ms-marco-MiniLM-L-6-v2` | `models/ms-marco-MiniLM-L-6-v2/snapshots/c5ee24cb16019beea0893ab7796b1df96625c6b8` |

## Debugging Extraction

```bash
python3 << 'EOF'
from doc_qa.ingest.extractor import ingest_document
from doc_qa.ingest.chunker import chunk_raw

raw = ingest_document('tests/fixtures/review_example.pdf', 'review_example.pdf')
print(f'Raw chunks: {len(raw)}')
for c in raw:
    print(f'--- p{c.page_number} [{c.content_type}] [{c.extraction_method}] ---')
    print(c.text)
    print()

processed = chunk_raw(raw)
print(f'Processed chunks: {len(processed)}')
EOF
```

## Debugging Retrieval

```bash
python3 << 'EOF'
from doc_qa.ingest.extractor import ingest_document
from doc_qa.ingest.chunker import chunk_raw
from doc_qa.retrieval.vectorstore import build_index, load_index
from doc_qa.retrieval.retriever import retrieve

raw = ingest_document('tests/fixtures/review_example.pdf', 'review_example.pdf')
processed = chunk_raw(raw)
doc_id = processed[0].doc_id
build_index(processed, doc_id)

index, meta_dict, position_map = load_index([doc_id])
chunks = retrieve("YOUR QUESTION HERE", index, meta_dict, position_map)

print(f"Retrieved {len(chunks)} chunks")
for c in chunks:
    print(f"\n--- rank {c.rank} | reranker: {c.reranker_score:.3f} | p{c.page_number} | {c.extraction_method} ---")
    print(c.text[:300])
EOF
```

## Key Architecture

- **Auth**: single code path, env-driven. `AZURE_OPENAI_DEPLOYMENT` set → `DefaultAzureCredential` + `get_bearer_token_provider("https://cognitiveservices.azure.com/.default")` → `AzureChatOpenAI`/`AzureOpenAIEmbeddings`. Else → `OPENAI_API_KEY` → `ChatOpenAI`/`OpenAIEmbeddings`.
- **Env vars (work)**: `AZURE_OPENAI_DEPLOYMENT`, `AZURE_EMBEDDING_DEPLOYMENT`, `OPENAI_API_VERSION`, optional `AZURE_OPENAI_ENDPOINT`, optional `MLFLOW_TRACKING_URI`
- **Env vars (local)**: `OPENAI_API_KEY`, `OPENAI_MODEL`
- **QA layer (overhaul)**: Compiled `StateGraph` singleton — `START → conditional(_route on state.mode) → qa_node | summarize_node → END`. `qa_node` calls `llm.with_structured_output(StructuredAnswer, include_raw=True)` so the LLM returns `answer + citations[]`; citations are validated against retrieved `chunk_id`s and hallucinated ones are dropped. Silent plain-text fallback on parse failure.
- **MLflow deployment**: bottom of `chain.py` registers `GraphResponsesAgent(ResponsesAgent)` via `mlflow.models.set_model(...)` — guarded `try/except`, no-op without mlflow installed. Deployable via `mlflow.pyfunc.log_model(python_model="doc_qa/qa/chain.py")`. The wrapper adapts Responses Agent input (`request.input`, `request.custom_inputs["doc_ids"]`) to Gandalf's state shape and runs retrieval inside `predict_stream`.
- **Embeddings**: OpenAI `text-embedding-3-small` via API (1536-dim). Model name in `doc_qa/retrieval/vectorstore.py::EMBEDDING_MODEL` — must match `retriever.py::EMBEDDING_MODEL`. No local HF models.
- **FAISS**: One index per doc — `data/{doc_id}.faiss` + `data/{doc_id}_meta.json`. Indexes built under MiniLM (384-dim) are incompatible — delete `data/*.faiss` + `data/*_meta.json` and re-upload docs after this swap; source files under `data/sources/` are preserved.
- **Chunking**: `RecursiveCharacterTextSplitter` (chunk_size=800, overlap=100) for prose; tables whole
- **Retrieval**: FAISS top-n over OpenAI embeddings (single-stage; CrossEncoder rerank removed on overhaul). `RetrievedChunk.reranker_score` now holds cosine similarity derived from L2 distance.
- **UI**: Streamlit two-panel — Chat (Q&A + trace drawer w/ Citations section + `[cited]` chunk tags) + Batch (question sheet → answer grid → Excel)

## Shared Types

```python
# doc_qa/ingest/extractor.py
RawChunk       # 13 fields: doc_id, filename, file_type, upload_timestamp, page_count,
               # page_number, chunk_index, section_heading, extraction_method,
               # content_type, text, char_count, bounding_box
ProcessedChunk # all RawChunk fields + chunk_id (uuid4) + embedding_text

# doc_qa/retrieval/retriever.py
RetrievedChunk # all ProcessedChunk fields + faiss_score, reranker_score, rank

# doc_qa/qa/chain.py
Context          # pydantic — model, temperature, api_version, azure_endpoint,
                 # summarize_max_chars (env-var defaults via Context.from_env())
State            # pydantic — messages (add_messages reducer), mode ("qa"|"summarize"),
                 # query, retrieved_chunks, doc_ids, summarize_max_chars (override),
                 # answer, citations, prompt_tokens, completion_tokens, latency_seconds,
                 # timestamp, model_deployment, confidence_level
Citation         # pydantic — chunk_id, page_number, section_heading, quote
StructuredAnswer # pydantic — answer: str, citations: list[Citation]
AnswerResult     # dataclass — query, answer, retrieved_chunks, prompt_tokens,
                 # completion_tokens, latency_seconds, timestamp, model_deployment,
                 # confidence_level, citations (validated)
```

## GitHub
- Repo: https://github.com/ordinarystephen/gandalf_doc_wizard.git
