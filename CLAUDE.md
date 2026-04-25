# Gandalf — Document Extraction & Q&A System

Streamlit Q&A app for credit analysts. Uploads PDF/DOCX/XLSX, chunks and indexes per-document FAISS, answers questions through a compiled LangGraph `StateGraph` calling Azure OpenAI. Full source tracing + SQLite audit log.

## Branches

| Branch | Purpose |
|---|---|
| `main` | Old production — still HF-based, will be replaced |
| `local-testing` | Local dev — OpenAI key path |
| `overhaul` | Full Kronos refactor with Azure↔OpenAI auto-switch |
| `domino` | **Current.** Slim Azure-only deploy off `overhaul`. No `models/`, no key fallback — fail-fast on missing env. |

## Running

```bash
source .venv/bin/activate
streamlit run app.py
python3 -m pytest tests/ -v        # 42 tests
```

## Required env (Azure-only on this branch)

```
AZURE_OPENAI_DEPLOYMENT=gpt-4o
AZURE_EMBEDDING_DEPLOYMENT=text-embedding-3-small
OPENAI_API_VERSION=2025-04-01-preview
# AZURE_OPENAI_ENDPOINT=...    # only if Domino proxy doesn't inject it
# AZURE_OPENAI_VISION_DEPLOYMENT=gpt-4o   # vision OCR; defaults to chat deployment
# MLFLOW_TRACKING_URI=databricks
# SUMMARIZE_MAX_CHARS=400000   # default 100000
```

`build_llm()` ([chain.py](doc_qa/qa/chain.py)) and `get_embeddings()` ([_embeddings.py](doc_qa/retrieval/_embeddings.py)) raise `RuntimeError` listing missing vars. Both use `DefaultAzureCredential` + `get_bearer_token_provider("https://cognitiveservices.azure.com/.default")` — no API key.

## Architecture

- **QA graph**: compiled `StateGraph` singleton. `START → conditional(state.mode) → qa_node | summarize_node → END`.
- **Structured output**: `qa_node` uses `llm.with_structured_output(StructuredAnswer, include_raw=True)`. Citations validated against retrieved `chunk_id`s — hallucinated ids dropped silently. Plain-text fallback on parse failure.
- **Embeddings**: Azure `text-embedding-3-small` (1536-dim). FAISS `IndexFlatL2`, one index per doc → `data/{doc_id}.faiss` + `data/{doc_id}_meta.json`. Source files in `data/sources/{doc_id}.{ext}`.
- **Retrieval**: single-stage FAISS top-n. `RetrievedChunk.reranker_score = 1 - L2²/2` (cosine sim from unit-norm L2). Confidence thresholds in [confidence.py](doc_qa/utils/confidence.py): 0.55 High / 0.40 Medium.
- **Chunking**: `RecursiveCharacterTextSplitter` (chunk_size=800, overlap=100) for prose; tables whole.
- **Summarization**: queries matching summarize/overview/tldr keywords ([chain.py::is_summarization_query](doc_qa/qa/chain.py)) bypass FAISS — full doc concatenated, capped by `SUMMARIZE_MAX_CHARS`.
- **MLflow**: bottom of [chain.py](doc_qa/qa/chain.py) registers `GraphResponsesAgent(ResponsesAgent)` via `mlflow.models.set_model(_get_graph)` — lazy factory, guarded `try/except ImportError`. Deployable via `mlflow.pyfunc.log_model(python_model="doc_qa/qa/chain.py")`.
- **UI**: two-panel Streamlit. Chat (Q&A + trace drawer with Citations section + `[cited]` chunk tags + PDF page render via pymupdf at 110 DPI, cached). Batch (question xlsx → answer grid → export).

## Shared types

```python
# doc_qa/ingest/extractor.py
RawChunk        # 13 fields incl. text, page_number, content_type, bounding_box
ProcessedChunk  # RawChunk + chunk_id (uuid4) + embedding_text

# doc_qa/retrieval/retriever.py
RetrievedChunk  # ProcessedChunk + faiss_score, reranker_score, rank

# doc_qa/qa/chain.py
Context          # model, temperature, api_version, azure_endpoint, summarize_max_chars
State            # messages (add_messages), mode "qa"|"summarize", query,
                 # retrieved_chunks, doc_ids, answer, citations, telemetry fields
Citation         # chunk_id, page_number, section_heading, quote
StructuredAnswer # answer: str, citations: list[Citation]
AnswerResult     # dataclass returned to callers
```

## Debugging

```bash
# Extraction
python3 -c "
from doc_qa.ingest.extractor import ingest_document
from doc_qa.ingest.chunker import chunk_raw
raw = ingest_document('tests/fixtures/review_example.pdf', 'review_example.pdf')
print(f'{len(raw)} raw, {len(chunk_raw(raw))} processed')
"

# Retrieval
python3 -c "
from doc_qa.ingest.extractor import ingest_document
from doc_qa.ingest.chunker import chunk_raw
from doc_qa.retrieval.vectorstore import build_index, load_index
from doc_qa.retrieval.retriever import retrieve
processed = chunk_raw(ingest_document('tests/fixtures/review_example.pdf', 'review_example.pdf'))
build_index(processed, processed[0].doc_id)
idx, meta, pos = load_index([processed[0].doc_id])
for c in retrieve('YOUR QUESTION', idx, meta, pos):
    print(f'rank {c.rank} | {c.reranker_score:.3f} | p{c.page_number}')
    print(c.text[:300])
"
```

## Pending

- **End-to-end Azure validation** in Domino — primary next step
- **Promote `domino` to `main`** once validated (purge HF blobs from `main` history with `git filter-repo --path models/ --invert-paths` if you want truly slim)
- **Ingestion speedups** (deferred): parallel per-page extraction, vision-OCR sidebar toggle, live progress bar replacing the spinner

## Repo
https://github.com/ordinarystephen/gandalf_doc_wizard.git
