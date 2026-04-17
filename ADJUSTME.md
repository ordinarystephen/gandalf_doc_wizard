# Tuning Guide

Quick reference for adjusting extraction, chunking, retrieval, and QA behaviour during testing.

**Tuning loop:** adjust a parameter → re-run the debug script → re-upload the file in the app → ask a test question. You only need to re-ingest the affected document, not everything.

---

## Chunking — `doc_qa/ingest/chunker.py`

| What to change | Where | Effect |
|----------------|-------|--------|
| `breakpoint_threshold_amount=85` | `chunk_raw()` | Lower → more, smaller chunks. Raise → fewer, larger chunks. |
| `embedding_text` prefix for tables | `chunk_raw()` | Controls what text is embedded for table chunks vs what is stored. |

---

## PDF Extraction — `doc_qa/ingest/pdf_extractor.py`

| What to change | Where | Effect |
|----------------|-------|--------|
| Heading detection logic | `extract_pdf()` | Currently flags bold + short lines as headings. Adjust to match your document style. |
| camelot fallback threshold | `extract_pdf()` | Controls when lattice mode falls back to stream mode for table extraction. |

---

## Retrieval Tuning — `doc_qa/retrieval/retriever.py`

| What to change | Where | Effect |
|----------------|-------|--------|
| `top_k=10` | `retrieve()` | Number of FAISS candidates before reranking. Raise if good chunks are being missed. |
| `rerank_top_n=5` | `retrieve()` | Number of chunks passed to the LLM. Raise for broader context, lower for tighter answers. |

---

## QA Prompt — `doc_qa/qa/chain.py`

| What to change | Where | Effect |
|----------------|-------|--------|
| `_WRAPPER_SYSTEM_PROMPT` | module level | The financial analyst instructions given to the model. Adjust tone, citation style, or domain focus. |

---

## Debug Script

Use this to inspect extraction output without running the full app or needing Azure:

```bash
python3 << 'EOF'
from doc_qa.ingest.extractor import ingest_document
from doc_qa.ingest.chunker import chunk_raw

raw = ingest_document('path/to/your.pdf', 'your.pdf')
print(f'Raw chunks: {len(raw)}')
for c in raw:
    print(f'--- p{c.page_number} [{c.content_type}] [{c.extraction_method}] ---')
    print(c.text)
    print()

processed = chunk_raw(raw)
print(f'Processed chunks: {len(processed)}')
EOF
```
