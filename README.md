# Gandalf — Document Q&A System

Streamlit-based document Q&A system for credit analysts. Upload PDF, DOCX, or XLSX files; ask questions in plain language; get answers with full source tracing and an audit log.

---

## Architecture

```
Upload → ingest_document() → chunk_raw() → build_index()
                                           ↓
Question → load_index() → retrieve() → qa_node() → answer + trace
                          FAISS top-k  CrossEncoder
                          rerank top-n
```

**Key components**

| Layer | Module | Responsibility |
|-------|--------|----------------|
| Ingest | `doc_qa/ingest/` | Extract text/tables from PDF, DOCX, XLSX |
| Chunking | `doc_qa/ingest/chunker.py` | SemanticChunker (prose) + whole-table storage |
| Vector store | `doc_qa/retrieval/vectorstore.py` | FAISS IndexFlatL2, one index per document |
| Retrieval | `doc_qa/retrieval/retriever.py` | FAISS top-k → CrossEncoder rerank → top-n |
| QA | `doc_qa/qa/chain.py` | LangGraph `qa_node`, `answer_question` wrapper |
| Batch | `doc_qa/qa/batch.py` | N×M batch runner, Excel export |
| Audit | `doc_qa/metadata/logger.py` | SQLite audit log (`data/query_log.db`) |
| UI | `app.py` | Streamlit two-panel: Chat + Batch |

---

## Tech Stack

- **Python 3.10+**, **Streamlit**
- **Azure OpenAI** via `AzureChatOpenAI` — `DefaultAzureCredential` (no API key)
- **LangGraph** node pattern: `qa_node(state, llm)` calling `llm.invoke(messages)`
- **HuggingFace Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (local, no API cost)
- **FAISS** `IndexFlatL2` — one index per document: `data/{doc_id}.faiss` + `data/{doc_id}_meta.json`
- **SemanticChunker** (percentile=85) for prose; tables stored whole and never split
- **CrossEncoder** reranking: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **pdfplumber** + **camelot** (lattice → stream fallback) for PDF table extraction
- **GPT-4o vision OCR** for image regions in PDFs

---

## Prerequisites

- Python 3.10+
- Azure OpenAI deployment accessible via `DefaultAzureCredential` (`az login` or managed identity)
- The following environment variables (copy `.env.example` to `.env`):

| Variable | Description |
|----------|-------------|
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI resource endpoint |
| `AZURE_OPENAI_DEPLOYMENT` | Deployment name (e.g. `gpt-4o`) |
| `OPENAI_API_VERSION` | API version (e.g. `2024-02-15-preview`) |

---

## Setup

```bash
# 1. Clone and enter the repo
git clone <repo-url>
cd gandalf

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env with your Azure OpenAI values

# 5. Authenticate with Azure
az login
```

---

## Run

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Tests

```bash
pytest tests/ -v
```

---

## Extraction Methods

| Method | Source |
|--------|--------|
| `native_text` | Direct text extraction via pdfplumber (prose) |
| `camelot_lattice` | PDF table extraction — lattice mode |
| `camelot_stream` | PDF table extraction — stream mode fallback |
| `vision_ocr` | GPT-4o vision for image regions in PDFs |
| `docx_native` | python-docx paragraph and table extraction |
| `openpyxl` | Excel sheet extraction |

---

## Confidence Levels

| Level | Reranker Score | Meaning |
|-------|---------------|---------|
| **High** | ≥ 0.85 | Answer is well-supported by retrieved chunks |
| **Medium** | ≥ 0.60 | Partial support — review source chunks |
| **Low** | < 0.60 | Limited support — verify manually |

---

## Trace Metadata Fields

Each answer exposes the following audit fields (visible in the Chat trace drawer and in the Batch Excel export):

| Field | Description |
|-------|-------------|
| `query` | The question asked |
| `answer` | The model's answer |
| `filename` | Source document name |
| `doc_id` | Internal document UUID |
| `confidence_level` | High / Medium / Low |
| `top_chunk_page` | Page number of the top retrieved chunk |
| `top_chunk_section` | Section heading of the top chunk |
| `top_reranker_score` | CrossEncoder score for the top chunk |
| `prompt_tokens` | Tokens in the prompt |
| `completion_tokens` | Tokens in the response |
| `latency_seconds` | End-to-end latency |
| `timestamp` | ISO 8601 UTC timestamp |
| `model_deployment` | Azure deployment name used |
| `extraction_methods_used` | Comma-separated extraction methods |
| `chunk_ids_used` | Comma-separated chunk UUIDs |

---

## Known Limitations

- FAISS is an in-process store. Index files persist on disk (`data/`), but the in-memory index is lost on server restart and must be reloaded.
- Vision OCR requires an Azure OpenAI GPT-4o deployment. If unavailable, image regions are skipped silently.
- Batch runs are sequential; large question × document grids take proportionally longer.
- PDF rendering quality depends on the source file. Scanned PDFs without embedded text require vision OCR.
