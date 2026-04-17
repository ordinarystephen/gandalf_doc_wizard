# Gandalf — Document Extraction & Q&A System

## Project Overview
Streamlit-based document Q&A system for credit analysts. Users upload PDF/DOCX/XLSX files, which are semantically chunked and indexed in per-document FAISS indexes. Questions are answered via Azure OpenAI (DefaultAzureCredential, LangGraph node) with full source tracing and SQLite audit logging.

## Active Branch & Worktree
- Branch: `feature/doc-qa-system`
- Worktree: `~/.config/superpowers/worktrees/gandalf/doc-qa-system`
- Main repo: `/Users/stephencostello/Documents/Dev_Work/gandalf`
- **All implementation work is in the worktree**

## Implementation Plan
Full plan with all task code: `docs/superpowers/plans/2026-04-16-document-qa-system.md`

## Completed Tasks ✅ (Tasks 1–13)

| # | File | Status |
|---|------|--------|
| 1 | `requirements.txt`, `.env.example`, scaffold | ✅ done |
| 2 | `doc_qa/utils/confidence.py` | ✅ done |
| 3 | `doc_qa/metadata/logger.py` | ✅ done |
| 4 | `doc_qa/ingest/extractor.py` (RawChunk, ProcessedChunk, router) | ✅ done |
| 5 | `doc_qa/ingest/pdf_extractor.py` | ✅ done |
| 6 | `doc_qa/ingest/image_table.py` | ✅ done |
| 7 | `doc_qa/ingest/docx_extractor.py` | ✅ done |
| 8 | `doc_qa/ingest/xlsx_extractor.py` | ✅ done |
| 9 | `doc_qa/ingest/chunker.py` | ✅ done |
| 10 | `doc_qa/retrieval/vectorstore.py` | ✅ done |
| 11 | `doc_qa/retrieval/retriever.py` | ✅ done |
| 12 | `doc_qa/qa/chain.py` (QAState, AnswerResult, build_llm, qa_node, answer_question) | ✅ done |
| 13 | `doc_qa/qa/batch.py` (_answer_for_doc, run_batch, export_to_excel) | ✅ done |

## Next Task — PICK UP HERE
**Task 15: `README.md`**

See plan Task 15 for full content. Cover: project overview, prerequisites, install, env vars, running the app, running tests.

## Remaining Tasks
- [ ] **Task 15**: `README.md` ← NEXT

## Key Architecture
- **Auth**: `DefaultAzureCredential` + `get_bearer_token_provider(credential, "https://cognitiveservices.azure.com/.default")` — no API key
- **Env vars**: `AZURE_OPENAI_DEPLOYMENT`, `OPENAI_API_VERSION`, `AZURE_OPENAI_ENDPOINT`
- **QA layer**: LangGraph node (`qa_node`) — flow: UI → service → graph state → wrapper prompt → inputs → `llm.invoke()`
- **Embeddings**: `langchain_huggingface.HuggingFaceEmbeddings("sentence-transformers/all-MiniLM-L6-v2")` (local, no API cost)
- **FAISS**: One index per doc — `data/{doc_id}.faiss` + `data/{doc_id}_meta.json`
- **Chunking**: SemanticChunker (percentile=85) for prose; tables stored whole (never split)
- **Retrieval**: FAISS top-k → CrossEncoder(`cross-encoder/ms-marco-MiniLM-L-6-v2`) rerank → top-n
- **UI**: Streamlit two-panel — Chat (free-form Q&A with trace drawer) + Batch (question sheet → answer grid)

## Shared Types (already implemented)
```python
# doc_qa/ingest/extractor.py
RawChunk      # 13 fields: doc_id, filename, file_type, upload_timestamp, page_count,
              # page_number, chunk_index, section_heading, extraction_method,
              # content_type, text, char_count, bounding_box
ProcessedChunk # all RawChunk fields + chunk_id (uuid4) + embedding_text

# doc_qa/retrieval/retriever.py
RetrievedChunk # all ProcessedChunk fields + faiss_score, reranker_score, rank

# doc_qa/qa/chain.py  ← TO BE CREATED
AnswerResult   # query, answer, retrieved_chunks, prompt_tokens, completion_tokens,
               # latency_seconds, timestamp, model_deployment, confidence_level
```

## Execution Approach
Subagent-driven development: fresh subagent per task, spec review + code quality review after each.
To resume: invoke `superpowers:subagent-driven-development` skill, start at Task 12.
