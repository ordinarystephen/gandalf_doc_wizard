# Codebase Guide

What every `.py` file does and how data flows through the system.

---

## Data Flow

```
User uploads file
      ↓
extractor.py          — routes to the right extractor based on file type
      ↓
pdf_extractor.py      — extracts prose, tables, and image regions from PDFs
docx_extractor.py     — extracts paragraphs and tables from Word files
xlsx_extractor.py     — extracts sheets from Excel files
image_table.py        — sends image regions to GPT-4o vision OCR
      ↓
chunker.py            — splits prose into semantic chunks; keeps tables whole
      ↓
vectorstore.py        — embeds chunks and saves a FAISS index to disk

User asks a question
      ↓
vectorstore.py        — loads the index from disk
retriever.py          — finds top-k chunks via FAISS, reranks with CrossEncoder
chain.py              — builds prompt, calls Azure OpenAI, returns answer + metadata
batch.py              — (Batch mode only) loops over all question × doc combinations
      ↓
logger.py             — writes every query and its metadata to SQLite
      ↓
app.py                — renders the answer, confidence badge, and trace drawer
```

---

## Entry Point

### `app.py`
The Streamlit application. Everything starts here.
- **Sidebar:** file uploader → triggers ingest pipeline → shows status badges per doc; active doc selector
- **Chat tab:** takes a question → loads index → retrieves chunks → calls `answer_question()` → renders answer, source chips, confidence badge, and expandable trace drawer
- **Batch tab:** loads a question sheet (xlsx) → calls `run_batch()` with a progress bar → shows answer grid → Excel download button
- Lazy-loads the Azure OpenAI LLM once and caches it in session state

---

## Ingest Layer — `doc_qa/ingest/`

### `extractor.py`
The router. Accepts any supported file and dispatches to the right extractor.
- Defines `RawChunk` — the shared data structure every extractor returns (13 fields: doc_id, filename, file type, page number, section heading, extraction method, content type, text, etc.)
- Defines `ProcessedChunk` — a `RawChunk` plus a `chunk_id` (UUID) and `embedding_text` (what actually gets embedded)
- `ingest_document(file_path, filename)` — looks at the file extension and calls `extract_pdf`, `extract_docx`, or `extract_xlsx`

### `pdf_extractor.py`
Handles PDF files. The most complex extractor.
- Per page: detects image regions → tries vision OCR; detects tables → tries camelot lattice then stream fallback; extracts remaining prose with pdfplumber
- Detects section headings (bold + short lines) and attaches them to subsequent chunks
- Returns a list of `RawChunk` objects, one per content region

### `docx_extractor.py`
Handles Word (.docx) files.
- Iterates the document body directly (paragraphs and tables)
- "Heading" styled paragraphs update the current heading but don't become chunks
- Tables are converted to markdown-style text and stored as a single chunk

### `xlsx_extractor.py`
Handles Excel (.xlsx) files.
- One chunk per sheet — the entire sheet is stored as a single text block
- Section heading is set to `"Sheet: {name}"`

### `image_table.py`
Handles image regions found inside PDFs.
- Crops the image region from the page using PyMuPDF, sends PNG bytes to GPT-4o vision via Azure OpenAI
- If the model returns `"NO_TABLE"` or any error occurs, returns `None` (never crashes the parent extractor)
- `_call_vision_api()` is split out as its own function so tests can mock it without needing Azure credentials

### `chunker.py`
Turns `RawChunk` objects into `ProcessedChunk` objects ready for embedding.
- Prose chunks: run through `SemanticChunker` (percentile=85) — splits on semantic boundaries, not fixed token counts
- Table chunks: kept whole and never split; `embedding_text` is prefixed with the section heading so retrieval context is richer
- Assigns a UUID (`chunk_id`) to each processed chunk

---

## Retrieval Layer — `doc_qa/retrieval/`

### `vectorstore.py`
Manages FAISS indexes on disk.
- `build_index(chunks, doc_id)` — embeds all chunks, saves `data/{doc_id}.faiss` and `data/{doc_id}_meta.json`
- `load_index(doc_ids)` — loads and merges one or more indexes into a single FAISS index; returns `(index, metadata_dict, position_map)`
- `position_map` is an ordered list of `chunk_id`s that maps FAISS integer positions back to chunk metadata
- `list_indexed_docs()` — returns all doc_ids that have a `.faiss` file on disk

### `retriever.py`
Runs the two-stage retrieval pipeline.
- Stage 1: embeds the query → searches FAISS → returns top-k candidates
- Stage 2: scores all candidates with a CrossEncoder reranker → returns top-n by reranker score
- `_embed_query()` and `_rerank()` are split out so tests can mock them
- Returns a list of `RetrievedChunk` objects (all `ProcessedChunk` fields + `faiss_score`, `reranker_score`, `rank`)
- The embedding model is cached at module level so it isn't reloaded on every call

---

## QA Layer — `doc_qa/qa/`

### `chain.py`
The Azure OpenAI integration and LangGraph node.
- `build_llm()` — creates an `AzureChatOpenAI` instance authenticated via `DefaultAzureCredential` (no API key needed)
- `_WRAPPER_SYSTEM_PROMPT` — the financial analyst system prompt applied to every query
- `QAState` — the LangGraph state dict passed through the node
- `qa_node(state, llm)` — the LangGraph node: builds messages from retrieved chunks + question, calls `llm.invoke()`, records token counts and latency
- `answer_question(query, retrieved_chunks, llm)` — convenience wrapper that calls `qa_node` and returns an `AnswerResult` dataclass

### `batch.py`
Runs Q&A across an N×M grid of questions × documents.
- `_answer_for_doc(question, filename, doc_id, chain)` — loads the index for one document, retrieves, and answers; split out so tests can mock it
- `run_batch(question_list, doc_configs, chain, progress_callback)` — loops every (question, doc) pair; errors are caught per cell and stored as `"ERROR: ..."` strings so one failure never crashes the whole batch; calls `progress_callback(done, total)` for the Streamlit progress bar
- `export_to_excel(answers_df, trace_df, output_path)` — writes a two-sheet Excel workbook: "Answers" (the grid) and "Trace" (full audit metadata per cell)

---

## Metadata Layer — `doc_qa/metadata/`

### `logger.py`
SQLite audit log.
- `QueryLogger(db_path)` — creates `data/query_log.db` on first use (including the `data/` directory)
- `log(record)` — writes one row per query with 15 audit fields (query, answer, filename, confidence, tokens, latency, chunk IDs, etc.)
- `get_recent(n)` — returns the last n log entries as a DataFrame

---

## Utilities — `doc_qa/utils/`

### `confidence.py`
Two small functions used by the UI.
- `score_confidence(top_reranker_score)` — maps a reranker score to `"High"`, `"Medium"`, or `"Low"` based on fixed thresholds (0.85 / 0.60)
- `confidence_color(confidence_level)` — returns a hex color for the confidence badge in the UI

---

## Tests — `tests/`

| File | What it tests |
|------|---------------|
| `test_confidence.py` | Score thresholds and color mapping |
| `test_logger.py` | SQLite write and read-back |
| `test_extractor_router.py` | File type routing in `extractor.py` |
| `test_pdf_extractor.py` | PDF chunk extraction against a real fixture PDF |
| `test_docx_extractor.py` | DOCX chunk extraction |
| `test_xlsx_extractor.py` | XLSX sheet extraction |
| `test_image_table.py` | Vision OCR with mocked `_call_vision_api` |
| `test_chunker.py` | Semantic chunking of prose and table pass-through |
| `test_vectorstore.py` | FAISS build, save, load, and merge |
| `test_retriever.py` | Two-stage retrieval with mocked embed and rerank |
| `test_chain.py` | `qa_node` and `answer_question` with mocked LLM |
| `test_batch.py` | Batch grid shape and Excel export |
| `make_fixtures.py` | Script that generates the test fixture files in `tests/fixtures/` |
