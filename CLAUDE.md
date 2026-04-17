# Gandalf — Document Extraction & Q&A System

## Project Overview
Streamlit-based document Q&A system for credit analysts. Users upload PDF/DOCX/XLSX files, which are chunked and indexed in per-document FAISS indexes. Questions are answered via Azure OpenAI (DefaultAzureCredential, LangGraph node) with full source tracing and SQLite audit logging.

## Branches

| Branch | Purpose |
|--------|---------|
| `main` | Production — Azure OpenAI via DefaultAzureCredential (Domino/work) |
| `local-testing` | Local dev — standard OpenAI via OPENAI_API_KEY |

## Current Branch: `local-testing`

Active branch for local testing. Key difference from `main`: `build_llm()` in `chain.py` uses `ChatOpenAI` with `OPENAI_API_KEY` instead of `AzureChatOpenAI`. Azure version is commented out directly below it for easy switching.

## Running the App

```bash
cd /Users/stephencostello/Documents/Dev_Work/gandalf
source .venv/bin/activate
streamlit run app.py
```

**Local testing `.env` requires:**
```
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o
TRANSFORMERS_OFFLINE=1
HF_HUB_OFFLINE=1
```

**Work/Domino `.env` requires:**
```
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt-4o
OPENAI_API_VERSION=2025-04-01-preview
TRANSFORMERS_OFFLINE=1
HF_HUB_OFFLINE=1
```

## Running Tests

```bash
python3 -m pytest tests/ -v
```

## Key Post-Implementation Changes (since original build)

| File | Change |
|------|--------|
| `doc_qa/qa/chain.py` | `build_llm()` uses `ChatOpenAI` on `local-testing`; Azure version commented out. `api_version=` (not `openai_api_version=`), optional `azure_endpoint` |
| `doc_qa/ingest/chunker.py` | Replaced `SemanticChunker` with `RecursiveCharacterTextSplitter` (chunk_size=800, overlap=100) for speed. Embedding model cached at module level |
| `doc_qa/ingest/pdf_extractor.py` | Camelot skipped on pages with <200 chars to prevent hang on image-heavy pages |
| `doc_qa/retrieval/vectorstore.py` | Embedding model loads from local path (offline) |
| `doc_qa/retrieval/retriever.py` | Embedding model and CrossEncoder load from local paths (offline) |
| `requirements.txt` | Pinned to work LangChain stack; MLflow/aice-mcflow commented out on local-testing branch |

## Local Model Paths

Both models are bundled in `models/` for offline use:

| Model | Local path |
|-------|-----------|
| `all-MiniLM-L6-v2` | `models/all-MiniLM-L6-v2/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf` |
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

- **Auth (work)**: `DefaultAzureCredential` + `get_bearer_token_provider` — no API key
- **Auth (local)**: `OPENAI_API_KEY` env var
- **Env vars (work)**: `AZURE_OPENAI_DEPLOYMENT`, `OPENAI_API_VERSION`, `AZURE_OPENAI_ENDPOINT`
- **Env vars (local)**: `OPENAI_API_KEY`, `OPENAI_MODEL`
- **QA layer**: LangGraph node (`qa_node`) — flow: UI → graph state → wrapper prompt → `llm.invoke()`
- **Embeddings**: `all-MiniLM-L6-v2` loaded from local `models/` folder (offline)
- **FAISS**: One index per doc — `data/{doc_id}.faiss` + `data/{doc_id}_meta.json`
- **Chunking**: `RecursiveCharacterTextSplitter` (chunk_size=800, overlap=100) for prose; tables whole
- **Retrieval**: FAISS top-k → CrossEncoder rerank → top-n
- **UI**: Streamlit two-panel — Chat (Q&A + trace drawer) + Batch (question sheet → answer grid → Excel)

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
AnswerResult   # query, answer, retrieved_chunks, prompt_tokens, completion_tokens,
               # latency_seconds, timestamp, model_deployment, confidence_level
```

## GitHub
- Repo: https://github.com/ordinarystephen/gandalf_doc_wizard.git
