# Document Extraction & Q&A System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Streamlit-based document Q&A system that ingests PDF/DOCX/XLSX files, indexes them with FAISS, and answers questions using Azure OpenAI with full source tracing.

**Architecture:** Files are extracted via type-specific extractors, semantically chunked, embedded with a local sentence-transformer, and stored in per-document FAISS indexes. At query time, top-k chunks are retrieved and reranked by a cross-encoder before being passed to AzureChatOpenAI. Every answer carries rich metadata logged to SQLite.

**Tech Stack:** Python 3.11, Streamlit, LangChain, LangGraph, AzureChatOpenAI (DefaultAzureCredential), pdfplumber, pymupdf, camelot-py, python-docx, openpyxl, sentence-transformers, FAISS, SQLite

---

## File Map

| File | Responsibility |
|------|---------------|
| `requirements.txt` | All pinned dependencies |
| `.env.example` | Azure credential template |
| `utils/confidence.py` | Score → High/Medium/Low + color |
| `metadata/logger.py` | SQLite query audit log |
| `ingest/extractor.py` | `RawChunk`, `ProcessedChunk` dataclasses + file-type router |
| `ingest/pdf_extractor.py` | pdfplumber + camelot native PDF extraction |
| `ingest/image_table.py` | GPT-4o vision OCR for image-embedded tables |
| `ingest/docx_extractor.py` | python-docx paragraph + table extraction |
| `ingest/xlsx_extractor.py` | openpyxl sheet-by-sheet extraction |
| `ingest/chunker.py` | SemanticChunker for prose; passthrough for tables |
| `retrieval/vectorstore.py` | FAISS build/persist/load per doc_id |
| `retrieval/retriever.py` | Embed query → FAISS search → cross-encoder rerank |
| `qa/chain.py` | `AnswerResult` dataclass + LangGraph QA node (DefaultAzureCredential) |
| `qa/batch.py` | N-docs × M-questions grid runner + Excel export |
| `app.py` | Streamlit two-panel UI (Chat + Batch modes) |
| `tests/` | One test file per module |

---

### Task 1: Project scaffold + requirements

**Files:**
- Create: `requirements.txt`
- Create: `.env.example`
- Create: `doc_qa/` directory tree with `__init__.py` files
- Create: `data/.gitkeep`

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p doc_qa/{ingest,retrieval,qa,metadata,utils} data tests
touch doc_qa/__init__.py doc_qa/ingest/__init__.py doc_qa/retrieval/__init__.py
touch doc_qa/qa/__init__.py doc_qa/metadata/__init__.py doc_qa/utils/__init__.py
touch data/.gitkeep
```

- [ ] **Step 2: Write requirements.txt**

```
# Streamlit UI
streamlit>=1.32.0

# LangChain + LangGraph + Azure OpenAI
langchain>=0.2.0
langchain-openai>=0.1.0
langchain-community>=0.2.0
langchain-experimental>=0.0.58
langgraph>=0.1.0
openai>=1.30.0

# Azure AD authentication (no API key — uses DefaultAzureCredential)
azure-identity>=1.16.0

# PDF extraction
pdfplumber>=0.11.0
pymupdf>=1.24.0
camelot-py[cv]>=0.11.0

# Word document extraction
python-docx>=1.1.0

# Excel extraction and export
openpyxl>=3.1.0
pandas>=2.2.0
tabulate>=0.9.0

# Embeddings (local, no API cost)
sentence-transformers>=3.0.0
faiss-cpu>=1.8.0

# Metadata and persistence
python-dotenv>=1.0.0

# Utilities
numpy>=1.26.0
Pillow>=10.0.0
```

- [ ] **Step 3: Write .env.example**

```
# Azure OpenAI — no API key needed; uses DefaultAzureCredential (az login / managed identity)
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt-4o
OPENAI_API_VERSION=2024-02-01

# Vision deployment — can be same as above if GPT-4o
AZURE_OPENAI_VISION_DEPLOYMENT=gpt-4o

# Optional: Azure Document Intelligence (only needed for scanned PDFs)
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=
AZURE_DOCUMENT_INTELLIGENCE_KEY=
```

- [ ] **Step 4: Install dependencies**

```bash
pip install -r requirements.txt
```

Expected: All packages install without conflict. If camelot-py fails on ghostscript, install it via: `brew install ghostscript` (macOS) or `apt-get install ghostscript` (Linux).

- [ ] **Step 5: Verify key imports**

```bash
python -c "import streamlit, langchain, pdfplumber, fitz, camelot, docx, openpyxl, pandas, sentence_transformers, faiss; print('all imports OK')"
```

Expected output: `all imports OK`

- [ ] **Step 6: Commit**

```bash
git add requirements.txt .env.example doc_qa/ data/ tests/
git commit -m "chore: scaffold project structure and requirements"
```

---

### Task 2: utils/confidence.py

**Files:**
- Create: `doc_qa/utils/confidence.py`
- Create: `tests/test_confidence.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_confidence.py
import sys; sys.path.insert(0, '.')
from doc_qa.utils.confidence import score_confidence, confidence_color

def test_high_confidence():
    assert score_confidence(0.85) == "High"
    assert score_confidence(1.0) == "High"

def test_medium_confidence():
    assert score_confidence(0.60) == "Medium"
    assert score_confidence(0.84) == "Medium"

def test_low_confidence():
    assert score_confidence(0.59) == "Low"
    assert score_confidence(0.0) == "Low"

def test_confidence_colors():
    assert confidence_color("High") == "#27500A"
    assert confidence_color("Medium") == "#633806"
    assert confidence_color("Low") == "#791F1F"
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_confidence.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# doc_qa/utils/confidence.py
"""Confidence scoring from cross-encoder reranker scores."""


def score_confidence(top_reranker_score: float) -> str:
    """Map a reranker score to High/Medium/Low.

    Thresholds are conservative for financial docs: a low-confidence
    answer about a DSCR or repayment source could have material consequences.
    """
    if top_reranker_score >= 0.85:
        return "High"
    if top_reranker_score >= 0.60:
        return "Medium"
    return "Low"


def confidence_color(confidence_level: str) -> str:
    """Return a dark hex color for the confidence badge."""
    return {"High": "#27500A", "Medium": "#633806", "Low": "#791F1F"}[confidence_level]
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_confidence.py -v
```

Expected: 4 tests PASSED

- [ ] **Step 5: Commit**

```bash
git add doc_qa/utils/confidence.py tests/test_confidence.py
git commit -m "feat: add confidence scoring utility"
```

---

### Task 3: metadata/logger.py

**Files:**
- Create: `doc_qa/metadata/logger.py`
- Create: `tests/test_logger.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_logger.py
import os, sys, json, tempfile, pytest
sys.path.insert(0, '.')

@pytest.fixture
def tmp_logger(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    from doc_qa.metadata.logger import QueryLogger
    return QueryLogger(db_path=str(tmp_path / "data" / "query_log.db"))

def test_logger_creates_db(tmp_path):
    (tmp_path / "data").mkdir()
    from doc_qa.metadata.logger import QueryLogger
    logger = QueryLogger(db_path=str(tmp_path / "data" / "query_log.db"))
    assert (tmp_path / "data" / "query_log.db").exists()

def test_log_and_retrieve(tmp_logger):
    record = {
        "query": "What is the DSCR?",
        "answer": "1.25x",
        "filename": "memo.pdf",
        "doc_id": "abc-123",
        "confidence_level": "High",
        "top_chunk_page": 3,
        "top_chunk_section": "Financial Analysis",
        "top_reranker_score": 0.92,
        "prompt_tokens": 800,
        "completion_tokens": 120,
        "latency_seconds": 1.4,
        "timestamp": "2026-04-16T10:00:00Z",
        "model_deployment": "gpt-4o",
        "extraction_methods_used": "native_text,pdfplumber_table",
        "chunk_ids_used": "c1,c2",
        "chunks_json": json.dumps([{"chunk_id": "c1", "text": "sample"}]),
    }
    tmp_logger.log(record)
    df = tmp_logger.get_recent(n=10)
    assert len(df) == 1
    assert df.iloc[0]["query"] == "What is the DSCR?"
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_logger.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# doc_qa/metadata/logger.py
"""Durable SQLite audit log for every Q&A query."""

import sqlite3
import logging
import pandas as pd
from typing import Any, Dict

logger = logging.getLogger(__name__)

# All columns stored in the query_log table. chunks_json holds chunk-level
# detail as a JSON string so the table stays flat without a join.
_COLUMNS = [
    "query", "answer", "filename", "doc_id", "confidence_level",
    "top_chunk_page", "top_chunk_section", "top_reranker_score",
    "prompt_tokens", "completion_tokens", "latency_seconds",
    "timestamp", "model_deployment", "extraction_methods_used",
    "chunk_ids_used", "chunks_json",
]

_CREATE_SQL = f"""
CREATE TABLE IF NOT EXISTS query_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    {', '.join(f'{c} TEXT' for c in _COLUMNS)}
)
"""


class QueryLogger:
    """Write and read Q&A audit records from a local SQLite database.

    SQLite is chosen over a flat file because it supports concurrent writes
    from multiple Streamlit sessions and enables future filtering queries.
    """

    def __init__(self, db_path: str = "data/query_log.db") -> None:
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(_CREATE_SQL)
            conn.commit()

    def log(self, record: Dict[str, Any]) -> None:
        """Insert one answer record into the audit log."""
        values = [str(record.get(c, "")) for c in _COLUMNS]
        placeholders = ", ".join("?" * len(_COLUMNS))
        sql = f"INSERT INTO query_log ({', '.join(_COLUMNS)}) VALUES ({placeholders})"
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(sql, values)
                conn.commit()
        except Exception as exc:
            logger.error("Failed to log query: %s", exc)

    def get_recent(self, n: int = 50) -> pd.DataFrame:
        """Return the n most recent log entries as a DataFrame."""
        sql = f"SELECT * FROM query_log ORDER BY id DESC LIMIT {int(n)}"
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(sql, conn)
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_logger.py -v
```

Expected: 2 tests PASSED

- [ ] **Step 5: Commit**

```bash
git add doc_qa/metadata/logger.py tests/test_logger.py
git commit -m "feat: add SQLite query audit logger"
```

---

### Task 4: ingest/extractor.py — dataclasses + router

**Files:**
- Create: `doc_qa/ingest/extractor.py`
- Create: `tests/test_extractor_router.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_extractor_router.py
import sys; sys.path.insert(0, '.')
from doc_qa.ingest.extractor import RawChunk, ProcessedChunk

def test_rawchunk_fields():
    chunk = RawChunk(
        doc_id="d1", filename="a.pdf", file_type="pdf",
        upload_timestamp="2026-04-16T00:00:00Z", page_count=10,
        page_number=1, chunk_index=0, section_heading="Intro",
        extraction_method="native_text", content_type="prose",
        text="Hello world", char_count=11, bounding_box=None,
    )
    assert chunk.char_count == 11
    assert chunk.content_type == "prose"

def test_processedchunk_fields():
    from doc_qa.ingest.extractor import ProcessedChunk
    pc = ProcessedChunk(
        doc_id="d1", filename="a.pdf", file_type="pdf",
        upload_timestamp="2026-04-16T00:00:00Z", page_count=10,
        page_number=1, chunk_index=0, section_heading="Intro",
        extraction_method="native_text", content_type="prose",
        text="Hello world", char_count=11, bounding_box=None,
        chunk_id="c1", embedding_text="Hello world",
    )
    assert pc.chunk_id == "c1"
    assert pc.embedding_text == "Hello world"
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_extractor_router.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement dataclasses and router**

```python
# doc_qa/ingest/extractor.py
"""File-type router and shared dataclasses for the ingest pipeline."""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class RawChunk:
    """A single extracted unit before embedding metadata is added.

    Every extractor returns a list of RawChunks. Fields here are set
    by the extractor; chunk_id and embedding_text are added by chunker.py.
    """
    doc_id: str               # uuid4 generated once per document at ingest
    filename: str             # original upload filename
    file_type: str            # pdf | docx | xlsx
    upload_timestamp: str     # ISO 8601 UTC — when the file was ingested
    page_count: int           # total pages (0 for xlsx)
    page_number: int          # 1-based page number (0 for xlsx)
    chunk_index: int          # 0-based position within the document
    section_heading: str      # nearest heading above this chunk
    extraction_method: str    # native_text | pdfplumber_table | camelot_table | vision_ocr_gpt4o | python_docx | openpyxl
    content_type: str         # prose | table | image_table
    text: str                 # the extracted text content
    char_count: int           # len(text) — used for chunk size diagnostics
    bounding_box: Optional[tuple]  # (x0, y0, x1, y1) in PDF points, None for non-PDF


@dataclass
class ProcessedChunk:
    """A RawChunk augmented with a unique chunk_id and embedding_text.

    chunker.py produces these from RawChunks. All fields are preserved
    so the retrieval layer never needs to look back at the raw data.
    """
    doc_id: str
    filename: str
    file_type: str
    upload_timestamp: str
    page_count: int
    page_number: int
    chunk_index: int
    section_heading: str
    extraction_method: str
    content_type: str
    text: str
    char_count: int
    bounding_box: Optional[tuple]
    chunk_id: str             # uuid4 — stable identifier for this chunk
    embedding_text: str       # text actually sent to the embedding model


def ingest_document(file_path: str, filename: str) -> List[RawChunk]:
    """Route a file to the correct extractor and attach document-level metadata.

    Args:
        file_path: Absolute path to the file on disk.
        filename: Original filename (used in metadata, not for routing).

    Returns:
        List of RawChunk objects with doc_id and upload_timestamp populated.
    """
    ext = Path(filename).suffix.lower()
    doc_id = str(uuid.uuid4())
    upload_timestamp = datetime.now(timezone.utc).isoformat()

    # Route by extension — each extractor knows only its own format
    if ext == ".pdf":
        from doc_qa.ingest.pdf_extractor import extract_pdf
        raw_chunks = extract_pdf(file_path)
    elif ext == ".docx":
        from doc_qa.ingest.docx_extractor import extract_docx
        raw_chunks = extract_docx(file_path)
    elif ext == ".xlsx":
        from doc_qa.ingest.xlsx_extractor import extract_xlsx
        raw_chunks = extract_xlsx(file_path)
    else:
        logger.warning("Unsupported file type: %s", ext)
        return []

    # Stamp every chunk with document-level identity fields
    for chunk in raw_chunks:
        chunk.doc_id = doc_id
        chunk.filename = filename
        chunk.file_type = ext.lstrip(".")
        chunk.upload_timestamp = upload_timestamp

    logger.info("Ingested %d chunks from %s (doc_id=%s)", len(raw_chunks), filename, doc_id)
    return raw_chunks
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_extractor_router.py -v
```

Expected: 2 tests PASSED

- [ ] **Step 5: Commit**

```bash
git add doc_qa/ingest/extractor.py tests/test_extractor_router.py
git commit -m "feat: add RawChunk/ProcessedChunk dataclasses and file router"
```

---

### Task 5: ingest/pdf_extractor.py

**Files:**
- Create: `doc_qa/ingest/pdf_extractor.py`
- Create: `tests/fixtures/sample.pdf` (generated programmatically)
- Create: `tests/test_pdf_extractor.py`

- [ ] **Step 1: Create a minimal test PDF fixture**

```python
# Run once to generate: python tests/make_fixtures.py
# tests/make_fixtures.py
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

def make_sample_pdf(path="tests/fixtures/sample.pdf"):
    import os; os.makedirs("tests/fixtures", exist_ok=True)
    c = canvas.Canvas(path, pagesize=letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(72, 720, "FINANCIAL ANALYSIS")
    c.setFont("Helvetica", 11)
    c.drawString(72, 700, "The borrower demonstrates strong cash flow coverage.")
    c.drawString(72, 685, "DSCR is 1.35x based on trailing twelve months.")
    c.save()

if __name__ == "__main__":
    make_sample_pdf()
```

```bash
pip install reportlab
python tests/make_fixtures.py
```

- [ ] **Step 2: Write failing tests**

```python
# tests/test_pdf_extractor.py
import sys; sys.path.insert(0, '.')
import pytest
from pathlib import Path

SAMPLE_PDF = Path("tests/fixtures/sample.pdf")

@pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="fixture not generated")
def test_extract_pdf_returns_chunks():
    from doc_qa.ingest.pdf_extractor import extract_pdf
    chunks = extract_pdf(str(SAMPLE_PDF))
    assert len(chunks) >= 1

@pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="fixture not generated")
def test_extract_pdf_metadata():
    from doc_qa.ingest.pdf_extractor import extract_pdf
    chunks = extract_pdf(str(SAMPLE_PDF))
    prose = [c for c in chunks if c.content_type == "prose"]
    assert len(prose) >= 1
    assert prose[0].page_number == 1
    assert "DSCR" in " ".join(c.text for c in prose)

@pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="fixture not generated")
def test_extract_pdf_section_heading():
    from doc_qa.ingest.pdf_extractor import extract_pdf
    chunks = extract_pdf(str(SAMPLE_PDF))
    # The all-caps heading "FINANCIAL ANALYSIS" should be detected
    headings = [c.section_heading for c in chunks]
    assert any("FINANCIAL" in h for h in headings)
```

- [ ] **Step 3: Run to verify failure**

```bash
pytest tests/test_pdf_extractor.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 4: Implement pdf_extractor.py**

```python
# doc_qa/ingest/pdf_extractor.py
"""Native PDF extraction: prose text, native tables, and image-region detection."""

import logging
import re
from typing import List, Optional, Tuple

import fitz  # pymupdf — used for high-resolution pixmap cropping
import pdfplumber

from doc_qa.ingest.extractor import RawChunk

logger = logging.getLogger(__name__)

# Minimum image dimensions to consider as a data table (not a logo/signature)
_MIN_IMAGE_WIDTH = 100
_MIN_IMAGE_HEIGHT = 50

# Patterns that identify section headings in financial documents
_HEADING_PATTERNS = [
    re.compile(r"^[A-Z][A-Z\s]{4,}$"),   # ALL CAPS lines
    re.compile(r"^(Sources of Repayment|Financial Analysis|Executive Summary|"
               r"Collateral|Guarantors?|Transaction Overview|Loan Structure|"
               r"Credit Risk|Conditions)", re.IGNORECASE),
]


def _is_heading(line: str) -> bool:
    """Return True if a line looks like a section heading."""
    line = line.strip()
    return any(p.match(line) for p in _HEADING_PATTERNS)


def extract_pdf(file_path: str) -> List[RawChunk]:
    """Extract prose, tables, and image-table regions from a native PDF.

    Args:
        file_path: Path to the PDF file.

    Returns:
        List of RawChunk objects (prose and table content_types).
        doc_id/filename/file_type/upload_timestamp are left as empty strings
        and stamped by ingest_document() in extractor.py.
    """
    chunks: List[RawChunk] = []
    chunk_index = 0
    current_heading = ""

    try:
        pdf_plumber = pdfplumber.open(file_path)
        pdf_fitz = fitz.open(file_path)
        page_count = len(pdf_plumber.pages)
    except Exception as exc:
        logger.error("Cannot open PDF %s: %s", file_path, exc)
        return []

    with pdf_plumber:
        for page_num, page in enumerate(pdf_plumber.pages, start=1):
            table_bboxes: List[Tuple] = []

            # --- STEP 1: detect image regions and dispatch to vision OCR ---
            # pdfplumber gives bounding boxes; pymupdf crops the pixmap.
            for img in page.images:
                if img["width"] < _MIN_IMAGE_WIDTH or img["height"] < _MIN_IMAGE_HEIGHT:
                    continue  # skip decorative images
                try:
                    fitz_page = pdf_fitz[page_num - 1]
                    # Convert pdfplumber coords (bottom-left origin) to fitz (top-left)
                    x0 = img["x0"]
                    y0 = page.height - img["y1"]
                    x1 = img["x1"]
                    y1 = page.height - img["y0"]
                    rect = fitz.Rect(x0, y0, x1, y1)
                    pixmap = fitz_page.get_pixmap(clip=rect, dpi=150)

                    from doc_qa.ingest.image_table import extract_image_table
                    img_chunk = extract_image_table(
                        pixmap=pixmap,
                        page_number=page_num,
                        bounding_box=(x0, y0, x1, y1),
                        filename="",
                    )
                    if img_chunk:
                        img_chunk.chunk_index = chunk_index
                        img_chunk.section_heading = current_heading
                        img_chunk.page_count = page_count
                        chunks.append(img_chunk)
                        chunk_index += 1
                        table_bboxes.append((img["x0"], img["y0"], img["x1"], img["y1"]))
                except Exception as exc:
                    logger.warning("Image region extraction failed on page %d: %s", page_num, exc)

            # --- STEP 2: extract native tables ---
            # pdfplumber first; camelot as fallback for complex layouts.
            try:
                tables = page.extract_tables()
            except Exception:
                tables = []

            if not tables:
                tables = _extract_with_camelot(file_path, page_num)

            for table in tables:
                try:
                    import pandas as pd
                    df = pd.DataFrame(table[1:], columns=table[0])
                    df = df.dropna(how="all").fillna(method="ffill", axis=1)
                    md_text = df.to_markdown(index=False)
                    chunk = RawChunk(
                        doc_id="", filename="", file_type="",
                        upload_timestamp="", page_count=page_count,
                        page_number=page_num, chunk_index=chunk_index,
                        section_heading=current_heading,
                        extraction_method="pdfplumber_table",
                        content_type="table", text=md_text,
                        char_count=len(md_text), bounding_box=None,
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                except Exception as exc:
                    logger.warning("Table serialization failed page %d: %s", page_num, exc)

            # --- STEP 3: extract prose text ---
            raw_text = page.extract_text() or ""
            lines = raw_text.splitlines()
            prose_lines = []
            for line in lines:
                if _is_heading(line):
                    # Flush accumulated prose before starting new section
                    if prose_lines:
                        text = "\n".join(prose_lines).strip()
                        if text:
                            chunk = RawChunk(
                                doc_id="", filename="", file_type="",
                                upload_timestamp="", page_count=page_count,
                                page_number=page_num, chunk_index=chunk_index,
                                section_heading=current_heading,
                                extraction_method="native_text",
                                content_type="prose", text=text,
                                char_count=len(text), bounding_box=None,
                            )
                            chunks.append(chunk)
                            chunk_index += 1
                        prose_lines = []
                    current_heading = line.strip()
                else:
                    prose_lines.append(line)

            # Flush remaining prose from this page
            if prose_lines:
                text = "\n".join(prose_lines).strip()
                if text:
                    chunk = RawChunk(
                        doc_id="", filename="", file_type="",
                        upload_timestamp="", page_count=page_count,
                        page_number=page_num, chunk_index=chunk_index,
                        section_heading=current_heading,
                        extraction_method="native_text",
                        content_type="prose", text=text,
                        char_count=len(text), bounding_box=None,
                    )
                    chunks.append(chunk)
                    chunk_index += 1

    pdf_fitz.close()
    logger.info("Extracted %d chunks from PDF %s", len(chunks), file_path)
    return chunks


def _extract_with_camelot(file_path: str, page_num: int) -> list:
    """Fallback table extraction using camelot-py.

    Tries lattice mode first (visible grid lines, common in exported
    financial models), then stream mode (whitespace-delimited).
    """
    try:
        import camelot
        # Lattice mode for Excel-exported tables with visible borders
        tables = camelot.read_pdf(file_path, pages=str(page_num), flavor="lattice")
        if tables and tables[0].parsing_report.get("accuracy", 0) >= 80:
            return [t.df.values.tolist() for t in tables]
        # Stream mode for narrative documents with whitespace-separated tables
        tables = camelot.read_pdf(file_path, pages=str(page_num), flavor="stream")
        return [t.df.values.tolist() for t in tables]
    except Exception as exc:
        logger.warning("camelot extraction failed page %d: %s", page_num, exc)
        return []
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_pdf_extractor.py -v
```

Expected: all tests PASSED (or SKIPPED if fixture missing — run `python tests/make_fixtures.py` first)

- [ ] **Step 6: Commit**

```bash
git add doc_qa/ingest/pdf_extractor.py tests/test_pdf_extractor.py tests/make_fixtures.py tests/fixtures/
git commit -m "feat: add PDF extractor with pdfplumber + camelot fallback"
```

---

### Task 6: ingest/image_table.py

**Files:**
- Create: `doc_qa/ingest/image_table.py`
- Create: `tests/test_image_table.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_image_table.py
import sys; sys.path.insert(0, '.')
from unittest.mock import MagicMock, patch

def _make_mock_pixmap():
    """Minimal fitz.Pixmap mock that returns valid PNG bytes."""
    import struct, zlib
    # 1x1 white PNG
    png_header = b'\x89PNG\r\n\x1a\n'
    ihdr = b'\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde'
    idat_data = zlib.compress(b'\x00\xff\xff\xff')
    idat = struct.pack('>I', len(idat_data)) + b'IDAT' + idat_data + struct.pack('>I', 0)
    iend = b'\x00\x00\x00\x00IEND\xaeB`\x82'
    png_bytes = png_header + ihdr + idat + iend
    pixmap = MagicMock()
    pixmap.tobytes.return_value = png_bytes
    return pixmap

def test_no_table_returns_none():
    with patch("doc_qa.ingest.image_table._call_vision_api", return_value="NO_TABLE"):
        from doc_qa.ingest.image_table import extract_image_table
        result = extract_image_table(
            pixmap=_make_mock_pixmap(), page_number=1,
            bounding_box=(0, 0, 100, 100), filename="test.pdf"
        )
        assert result is None

def test_table_response_returns_chunk():
    md_table = "| Col1 | Col2 |\n|------|------|\n| A    | 1.0  |"
    with patch("doc_qa.ingest.image_table._call_vision_api", return_value=md_table):
        from doc_qa.ingest.image_table import extract_image_table
        chunk = extract_image_table(
            pixmap=_make_mock_pixmap(), page_number=2,
            bounding_box=(10, 20, 300, 400), filename="test.pdf"
        )
        assert chunk is not None
        assert chunk.content_type == "image_table"
        assert chunk.extraction_method == "vision_ocr_gpt4o"
        assert chunk.page_number == 2

def test_api_failure_returns_none():
    with patch("doc_qa.ingest.image_table._call_vision_api", side_effect=Exception("API down")):
        from doc_qa.ingest.image_table import extract_image_table
        result = extract_image_table(
            pixmap=_make_mock_pixmap(), page_number=1,
            bounding_box=(0, 0, 100, 100), filename="test.pdf"
        )
        assert result is None
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_image_table.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# doc_qa/ingest/image_table.py
"""Vision OCR extraction for financial tables embedded as images in PDFs.

We use GPT-4o vision rather than pytesseract because tesseract does character
recognition but has no understanding of table structure, merged cells, or
financial formatting. GPT-4o reconstructs the semantic structure of the table.
"""

import base64
import logging
import os
from typing import Optional

from doc_qa.ingest.extractor import RawChunk

logger = logging.getLogger(__name__)

_VISION_SYSTEM_PROMPT = (
    "You are a financial document parser. You will be given an image of a "
    "financial table or model pasted into a document. Extract the complete "
    "table as a markdown table. Preserve all row labels, column headers, "
    "numeric values, subtotals, and formatting indicators (e.g. bold rows "
    "typically indicate subtotals or totals). If the image does not contain "
    "a table, return the string NO_TABLE."
)


def _call_vision_api(png_bytes: bytes) -> str:
    """Send a PNG image to Azure OpenAI GPT-4o and return the text response."""
    from langchain_openai import AzureChatOpenAI
    from langchain_core.messages import HumanMessage

    b64 = base64.b64encode(png_bytes).decode("utf-8")

    # Vision deployment may differ from the main LLM deployment
    from azure.identity import DefaultAzureCredential, get_bearer_token_provider
    credential = DefaultAzureCredential()
    token_provider = get_bearer_token_provider(
        credential, "https://cognitiveservices.azure.com/.default"
    )
    vision_deployment = os.getenv(
        "AZURE_OPENAI_VISION_DEPLOYMENT",
        os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
    )
    client = AzureChatOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        azure_deployment=vision_deployment,
        openai_api_version=os.getenv("OPENAI_API_VERSION", "2024-02-01"),
        azure_ad_token_provider=token_provider,
    )
    message = HumanMessage(content=[
        {"type": "text", "text": _VISION_SYSTEM_PROMPT},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
    ])
    response = client.invoke([message])
    return response.content.strip()


def extract_image_table(
    pixmap,
    page_number: int,
    bounding_box: tuple,
    filename: str,
) -> Optional[RawChunk]:
    """Extract a financial table from a PDF image region via GPT-4o vision.

    Args:
        pixmap: fitz.Pixmap of the image region.
        page_number: 1-based page number (for metadata).
        bounding_box: (x0, y0, x1, y1) in PDF coordinate space.
        filename: Source filename (for metadata).

    Returns:
        RawChunk with content_type='image_table', or None if no table found
        or if the API call fails. Never raises — errors are logged as warnings.
    """
    try:
        png_bytes = pixmap.tobytes("png")
        response = _call_vision_api(png_bytes)

        if response == "NO_TABLE":
            logger.info("Vision OCR: no table found on page %d of %s", page_number, filename)
            return None

        return RawChunk(
            doc_id="", filename=filename, file_type="pdf",
            upload_timestamp="", page_count=0,
            page_number=page_number, chunk_index=0,
            section_heading="",
            extraction_method="vision_ocr_gpt4o",
            content_type="image_table",
            text=response,
            char_count=len(response),
            bounding_box=bounding_box,
        )
    except Exception as exc:
        logger.warning("Vision OCR failed on page %d of %s: %s", page_number, filename, exc)
        return None
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_image_table.py -v
```

Expected: 3 tests PASSED

- [ ] **Step 5: Commit**

```bash
git add doc_qa/ingest/image_table.py tests/test_image_table.py
git commit -m "feat: add GPT-4o vision OCR extractor for image-embedded tables"
```

---

### Task 7: ingest/docx_extractor.py

**Files:**
- Create: `doc_qa/ingest/docx_extractor.py`
- Create: `tests/fixtures/sample.docx` (generated programmatically)
- Create: `tests/test_docx_extractor.py`

- [ ] **Step 1: Generate DOCX fixture**

Add to `tests/make_fixtures.py`:

```python
def make_sample_docx(path="tests/fixtures/sample.docx"):
    from docx import Document
    doc = Document()
    doc.add_heading("Executive Summary", level=1)
    doc.add_paragraph("This loan supports the acquisition of an industrial property.")
    doc.add_heading("Financial Analysis", level=1)
    doc.add_paragraph("The DSCR covenant threshold is 1.20x per the credit agreement.")
    table = doc.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Metric"
    table.cell(0, 1).text = "Value"
    table.cell(1, 0).text = "LTV"
    table.cell(1, 1).text = "65%"
    doc.save(path)

if __name__ == "__main__":
    make_sample_pdf()
    make_sample_docx()
```

```bash
python tests/make_fixtures.py
```

- [ ] **Step 2: Write failing tests**

```python
# tests/test_docx_extractor.py
import sys; sys.path.insert(0, '.')
import pytest
from pathlib import Path

SAMPLE_DOCX = Path("tests/fixtures/sample.docx")

@pytest.mark.skipif(not SAMPLE_DOCX.exists(), reason="fixture not generated")
def test_extract_docx_returns_chunks():
    from doc_qa.ingest.docx_extractor import extract_docx
    chunks = extract_docx(str(SAMPLE_DOCX))
    assert len(chunks) >= 1

@pytest.mark.skipif(not SAMPLE_DOCX.exists(), reason="fixture not generated")
def test_docx_headings_detected():
    from doc_qa.ingest.docx_extractor import extract_docx
    chunks = extract_docx(str(SAMPLE_DOCX))
    headings = {c.section_heading for c in chunks}
    assert "Executive Summary" in headings or "Financial Analysis" in headings

@pytest.mark.skipif(not SAMPLE_DOCX.exists(), reason="fixture not generated")
def test_docx_table_extracted():
    from doc_qa.ingest.docx_extractor import extract_docx
    chunks = extract_docx(str(SAMPLE_DOCX))
    tables = [c for c in chunks if c.content_type == "table"]
    assert len(tables) >= 1
    assert "LTV" in tables[0].text
```

- [ ] **Step 3: Run to verify failure**

```bash
pytest tests/test_docx_extractor.py -v
```

- [ ] **Step 4: Implement**

```python
# doc_qa/ingest/docx_extractor.py
"""Word document extraction preserving document order of prose and tables.

Word preserves document order in the python-docx object model, which PDFs
do not — this makes DOCX extraction simpler and more reliable for structured
credit documents.
"""

import logging
from typing import List

import pandas as pd
from docx import Document
from docx.oxml.ns import qn

from doc_qa.ingest.extractor import RawChunk

logger = logging.getLogger(__name__)


def _table_to_markdown(table) -> str:
    """Convert a python-docx Table to a markdown string.

    Word tables can have merged cells. python-docx exposes them as cells
    with identical text across the span — we forward-fill to handle this.
    """
    rows = []
    for row in table.rows:
        rows.append([cell.text.strip() for cell in row.cells])
    if not rows:
        return ""
    df = pd.DataFrame(rows[1:], columns=rows[0])
    # Forward-fill duplicated values from merged cells
    df = df.replace("", pd.NA).ffill(axis=1).fillna("")
    return df.to_markdown(index=False)


def extract_docx(file_path: str) -> List[RawChunk]:
    """Extract paragraphs and tables from a Word document in document order.

    Args:
        file_path: Path to the .docx file.

    Returns:
        List of RawChunk objects with extraction_method='python_docx'.
    """
    try:
        doc = Document(file_path)
    except Exception as exc:
        logger.error("Cannot open DOCX %s: %s", file_path, exc)
        return []

    chunks: List[RawChunk] = []
    chunk_index = 0
    current_heading = ""

    # Iterate body children in document order (paragraphs and tables interleaved)
    for child in doc.element.body:
        tag = child.tag.split("}")[-1]  # strip XML namespace

        if tag == "p":  # paragraph
            para_text = child.text_content() if hasattr(child, "text_content") else ""
            # Reconstruct from runs to handle mixed formatting
            from docx.text.paragraph import Paragraph
            para = Paragraph(child, doc)
            para_text = para.text.strip()
            style_name = para.style.name if para.style else ""

            if not para_text:
                continue

            if "Heading" in style_name:
                # Headings delimit sections — update the running heading tracker
                current_heading = para_text
                continue

            chunk = RawChunk(
                doc_id="", filename="", file_type="docx",
                upload_timestamp="", page_count=0,
                page_number=0, chunk_index=chunk_index,
                section_heading=current_heading,
                extraction_method="python_docx",
                content_type="prose",
                text=para_text, char_count=len(para_text),
                bounding_box=None,
            )
            chunks.append(chunk)
            chunk_index += 1

        elif tag == "tbl":  # table
            from docx.table import Table
            table = Table(child, doc)
            md_text = _table_to_markdown(table)
            if not md_text:
                continue
            chunk = RawChunk(
                doc_id="", filename="", file_type="docx",
                upload_timestamp="", page_count=0,
                page_number=0, chunk_index=chunk_index,
                section_heading=current_heading,
                extraction_method="python_docx",
                content_type="table",
                text=md_text, char_count=len(md_text),
                bounding_box=None,
            )
            chunks.append(chunk)
            chunk_index += 1

    logger.info("Extracted %d chunks from DOCX %s", len(chunks), file_path)
    return chunks
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_docx_extractor.py -v
```

Expected: all PASSED

- [ ] **Step 6: Commit**

```bash
git add doc_qa/ingest/docx_extractor.py tests/test_docx_extractor.py tests/make_fixtures.py
git commit -m "feat: add Word document extractor"
```

---

### Task 8: ingest/xlsx_extractor.py

**Files:**
- Create: `doc_qa/ingest/xlsx_extractor.py`
- Create: `tests/fixtures/sample.xlsx` (generated programmatically)
- Create: `tests/test_xlsx_extractor.py`

- [ ] **Step 1: Generate XLSX fixture**

Add to `tests/make_fixtures.py`:

```python
def make_sample_xlsx(path="tests/fixtures/sample.xlsx"):
    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Financials"
    ws.append(["Metric", "2023", "2024"])
    ws.append(["Revenue", 1000000, 1200000])
    ws.append(["EBITDA", 300000, 380000])
    ws2 = wb.create_sheet("Covenants")
    ws2.append(["Covenant", "Threshold", "Actual"])
    ws2.append(["DSCR", "1.20x", "1.35x"])
    wb.save(path)
```

```bash
python tests/make_fixtures.py
```

- [ ] **Step 2: Write failing tests**

```python
# tests/test_xlsx_extractor.py
import sys; sys.path.insert(0, '.')
import pytest
from pathlib import Path

SAMPLE_XLSX = Path("tests/fixtures/sample.xlsx")

@pytest.mark.skipif(not SAMPLE_XLSX.exists(), reason="fixture not generated")
def test_extract_xlsx_all_sheets():
    from doc_qa.ingest.xlsx_extractor import extract_xlsx
    chunks = extract_xlsx(str(SAMPLE_XLSX))
    sheet_names = {c.section_heading for c in chunks}
    assert "Sheet: Financials" in sheet_names
    assert "Sheet: Covenants" in sheet_names

@pytest.mark.skipif(not SAMPLE_XLSX.exists(), reason="fixture not generated")
def test_extract_xlsx_content():
    from doc_qa.ingest.xlsx_extractor import extract_xlsx
    chunks = extract_xlsx(str(SAMPLE_XLSX))
    all_text = " ".join(c.text for c in chunks)
    assert "DSCR" in all_text
    assert "Revenue" in all_text

@pytest.mark.skipif(not SAMPLE_XLSX.exists(), reason="fixture not generated")
def test_extract_xlsx_extraction_method():
    from doc_qa.ingest.xlsx_extractor import extract_xlsx
    chunks = extract_xlsx(str(SAMPLE_XLSX))
    assert all(c.extraction_method == "openpyxl" for c in chunks)
    assert all(c.content_type == "table" for c in chunks)
```

- [ ] **Step 3: Implement**

```python
# doc_qa/ingest/xlsx_extractor.py
"""Excel workbook extraction — every sheet is treated as a table.

Excel files are entirely structured tabular data, so all content
is extracted as table chunks without prose chunking.
"""

import logging
from typing import List

import pandas as pd

from doc_qa.ingest.extractor import RawChunk

logger = logging.getLogger(__name__)


def extract_xlsx(file_path: str) -> List[RawChunk]:
    """Extract every sheet from an Excel workbook as a markdown table chunk.

    Args:
        file_path: Path to the .xlsx file.

    Returns:
        One RawChunk per sheet with content_type='table' and
        section_heading='Sheet: {sheet_name}'.
    """
    try:
        xl = pd.ExcelFile(file_path, engine="openpyxl")
    except Exception as exc:
        logger.error("Cannot open XLSX %s: %s", file_path, exc)
        return []

    chunks: List[RawChunk] = []
    for idx, sheet_name in enumerate(xl.sheet_names):
        try:
            df = xl.parse(sheet_name)
            # Drop fully empty rows/columns from sparse sheets
            df = df.dropna(how="all").dropna(axis=1, how="all")
            # Strip leading/trailing whitespace from string cells
            df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
            if df.empty:
                continue
            md_text = df.to_markdown(index=False)
            section = f"Sheet: {sheet_name}"
            chunk = RawChunk(
                doc_id="", filename="", file_type="xlsx",
                upload_timestamp="", page_count=0,
                page_number=0, chunk_index=idx,
                section_heading=section,
                extraction_method="openpyxl",
                content_type="table",
                text=md_text, char_count=len(md_text),
                bounding_box=None,
            )
            chunks.append(chunk)
        except Exception as exc:
            logger.warning("Failed to extract sheet '%s' from %s: %s", sheet_name, file_path, exc)

    logger.info("Extracted %d sheet chunks from XLSX %s", len(chunks), file_path)
    return chunks
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_xlsx_extractor.py -v
```

Expected: all PASSED

- [ ] **Step 5: Commit**

```bash
git add doc_qa/ingest/xlsx_extractor.py tests/test_xlsx_extractor.py
git commit -m "feat: add Excel extractor (all sheets as table chunks)"
```

---

### Task 9: ingest/chunker.py

**Files:**
- Create: `doc_qa/ingest/chunker.py`
- Create: `tests/test_chunker.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_chunker.py
import sys; sys.path.insert(0, '.')
from doc_qa.ingest.extractor import RawChunk

def _make_prose_chunk(text="The borrower has strong coverage. " * 20) -> RawChunk:
    return RawChunk(
        doc_id="d1", filename="a.pdf", file_type="pdf",
        upload_timestamp="2026-04-16T00:00:00Z", page_count=5,
        page_number=1, chunk_index=0, section_heading="Financial Analysis",
        extraction_method="native_text", content_type="prose",
        text=text, char_count=len(text), bounding_box=None,
    )

def _make_table_chunk() -> RawChunk:
    text = "| Metric | Value |\n|--------|-------|\n| DSCR   | 1.35x |"
    return RawChunk(
        doc_id="d1", filename="a.pdf", file_type="pdf",
        upload_timestamp="2026-04-16T00:00:00Z", page_count=5,
        page_number=2, chunk_index=1, section_heading="Financial Analysis",
        extraction_method="pdfplumber_table", content_type="table",
        text=text, char_count=len(text), bounding_box=None,
    )

def test_table_chunk_not_split():
    from doc_qa.ingest.chunker import chunk_raw
    table = _make_table_chunk()
    result = chunk_raw([table])
    # Tables must never be split — exactly one output chunk per input table
    assert len(result) == 1
    assert result[0].content_type == "table"
    assert result[0].text == table.text

def test_table_embedding_text_has_heading_prefix():
    from doc_qa.ingest.chunker import chunk_raw
    table = _make_table_chunk()
    result = chunk_raw([table])
    assert result[0].embedding_text.startswith("Table from section: Financial Analysis")

def test_processed_chunk_has_chunk_id():
    from doc_qa.ingest.chunker import chunk_raw
    result = chunk_raw([_make_table_chunk()])
    assert len(result[0].chunk_id) > 0

def test_prose_chunk_produces_output():
    from doc_qa.ingest.chunker import chunk_raw
    result = chunk_raw([_make_prose_chunk()])
    assert len(result) >= 1
    assert all(c.content_type == "prose" for c in result)
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_chunker.py -v
```

- [ ] **Step 3: Implement**

```python
# doc_qa/ingest/chunker.py
"""Semantic chunking for prose; passthrough for tables and image tables.

SemanticChunker is used over RecursiveCharacterTextSplitter because fixed-size
splitting cuts mid-thought. Semantic splitting preserves complete ideas, which
is critical when a section like "Sources of Repayment" spans several sentences
that must stay together for accurate retrieval.
"""

import logging
import uuid
from typing import List

from doc_qa.ingest.extractor import ProcessedChunk, RawChunk

logger = logging.getLogger(__name__)


def _get_embeddings():
    """Return a shared HuggingFaceEmbeddings instance (no API cost)."""
    from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def _chunk_prose(raw: RawChunk) -> List[ProcessedChunk]:
    """Split prose using SemanticChunker with percentile breakpointing.

    breakpoint_threshold_amount=85 means we split when semantic similarity
    drops below the 85th percentile — more aggressive splitting produces
    more focused chunks for targeted Q&A.
    """
    try:
        from langchain_experimental.text_splitter import SemanticChunker
        splitter = SemanticChunker(
            embeddings=_get_embeddings(),
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=85,
        )
        texts = splitter.split_text(raw.text)
    except Exception as exc:
        logger.warning("SemanticChunker failed, using full text: %s", exc)
        texts = [raw.text]

    results = []
    for i, text in enumerate(texts):
        results.append(ProcessedChunk(
            doc_id=raw.doc_id, filename=raw.filename, file_type=raw.file_type,
            upload_timestamp=raw.upload_timestamp, page_count=raw.page_count,
            page_number=raw.page_number, chunk_index=raw.chunk_index + i,
            section_heading=raw.section_heading,
            extraction_method=raw.extraction_method,
            content_type=raw.content_type,
            text=text, char_count=len(text),
            bounding_box=raw.bounding_box,
            chunk_id=str(uuid.uuid4()),
            embedding_text=text,  # prose embeds as-is
        ))
    return results


def _chunk_table(raw: RawChunk) -> ProcessedChunk:
    """Store a table as a single chunk — never split tables.

    Splitting a financial table mid-row destroys its structure; the LLM
    needs the complete table with headers to answer numeric questions.
    The section heading prefix in embedding_text helps retrieval surface
    this table when questions reference the section name.
    """
    embedding_text = f"Table from section: {raw.section_heading}\n{raw.text}"
    return ProcessedChunk(
        doc_id=raw.doc_id, filename=raw.filename, file_type=raw.file_type,
        upload_timestamp=raw.upload_timestamp, page_count=raw.page_count,
        page_number=raw.page_number, chunk_index=raw.chunk_index,
        section_heading=raw.section_heading,
        extraction_method=raw.extraction_method,
        content_type=raw.content_type,
        text=raw.text, char_count=raw.char_count,
        bounding_box=raw.bounding_box,
        chunk_id=str(uuid.uuid4()),
        embedding_text=embedding_text,
    )


def chunk_raw(raw_chunks: List[RawChunk]) -> List[ProcessedChunk]:
    """Convert RawChunks to ProcessedChunks with chunk_ids and embedding_text.

    Args:
        raw_chunks: Output from any extractor module.

    Returns:
        List of ProcessedChunk objects ready for FAISS indexing.
    """
    results: List[ProcessedChunk] = []
    for raw in raw_chunks:
        if raw.content_type == "prose":
            results.extend(_chunk_prose(raw))
        else:
            # Tables and image_tables are stored whole
            results.append(_chunk_table(raw))
    logger.info("Chunking produced %d ProcessedChunks from %d RawChunks",
                len(results), len(raw_chunks))
    return results
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_chunker.py -v
```

Expected: all PASSED (prose test may be slow — SemanticChunker downloads model on first run)

- [ ] **Step 5: Commit**

```bash
git add doc_qa/ingest/chunker.py tests/test_chunker.py
git commit -m "feat: add semantic chunker (prose) and table passthrough"
```

---

### Task 10: retrieval/vectorstore.py

**Files:**
- Create: `doc_qa/retrieval/vectorstore.py`
- Create: `tests/test_vectorstore.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_vectorstore.py
import sys, json, uuid
sys.path.insert(0, '.')
import pytest

def _make_chunks(n=3):
    from doc_qa.ingest.extractor import ProcessedChunk
    return [
        ProcessedChunk(
            doc_id="doc1", filename="test.pdf", file_type="pdf",
            upload_timestamp="2026-04-16T00:00:00Z", page_count=5,
            page_number=i+1, chunk_index=i, section_heading="Intro",
            extraction_method="native_text", content_type="prose",
            text=f"The borrower has strong cash flow metrics. Chunk {i}.",
            char_count=40, bounding_box=None,
            chunk_id=str(uuid.uuid4()),
            embedding_text=f"The borrower has strong cash flow metrics. Chunk {i}.",
        )
        for i in range(n)
    ]

def test_build_and_load_index(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    from doc_qa.retrieval.vectorstore import build_index, load_index
    chunks = _make_chunks(3)
    build_index(chunks, doc_id="doc1", data_dir=str(tmp_path))
    assert (tmp_path / "doc1.faiss").exists()
    assert (tmp_path / "doc1_meta.json").exists()

    index, meta = load_index(["doc1"], data_dir=str(tmp_path))
    assert index is not None
    assert len(meta) == 3  # one entry per chunk

def test_list_indexed_docs(tmp_path):
    from doc_qa.retrieval.vectorstore import build_index, list_indexed_docs
    chunks = _make_chunks(2)
    build_index(chunks, doc_id="docA", data_dir=str(tmp_path))
    docs = list_indexed_docs(data_dir=str(tmp_path))
    assert "docA" in docs
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_vectorstore.py -v
```

- [ ] **Step 3: Implement**

```python
# doc_qa/retrieval/vectorstore.py
"""FAISS index management — one index file per document.

One index per document allows adding/removing documents without rebuilding
the entire index. The JSON sidecar preserves chunk metadata so we never
need to re-embed to look up source information.
"""

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np

from doc_qa.ingest.extractor import ProcessedChunk

logger = logging.getLogger(__name__)

_DEFAULT_DATA_DIR = "data"


def _get_embeddings():
    from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def build_index(
    processed_chunks: List[ProcessedChunk],
    doc_id: str,
    data_dir: str = _DEFAULT_DATA_DIR,
) -> None:
    """Embed chunks and persist a FAISS index + JSON metadata sidecar.

    Args:
        processed_chunks: Output from chunker.chunk_raw().
        doc_id: Unique document identifier (used as filename stem).
        data_dir: Directory to write .faiss and _meta.json files.
    """
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    embedder = _get_embeddings()
    texts = [c.embedding_text for c in processed_chunks]

    logger.info("Embedding %d chunks for doc_id=%s", len(texts), doc_id)
    vectors = np.array(embedder.embed_documents(texts), dtype="float32")

    # Build a flat L2 index — sufficient for per-document indexes of <50k chunks
    dimension = vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)

    faiss.write_index(index, str(Path(data_dir) / f"{doc_id}.faiss"))

    # Sidecar stores all metadata keyed by position so we can reconstruct
    # the chunk object from a FAISS result index without re-embedding
    meta = {i: asdict(c) for i, c in enumerate(processed_chunks)}
    with open(Path(data_dir) / f"{doc_id}_meta.json", "w") as f:
        json.dump(meta, f)

    logger.info("Index saved: %s/%s.faiss", data_dir, doc_id)


def load_index(
    doc_ids: List[str],
    data_dir: str = _DEFAULT_DATA_DIR,
) -> Tuple[faiss.Index, Dict[str, dict]]:
    """Load and merge FAISS indexes for multiple documents.

    Args:
        doc_ids: List of doc_id strings to load.
        data_dir: Directory containing .faiss and _meta.json files.

    Returns:
        (merged_index, metadata_dict) where metadata_dict maps chunk_id → metadata.
    """
    indexes = []
    metadata: Dict[str, dict] = {}

    for doc_id in doc_ids:
        faiss_path = Path(data_dir) / f"{doc_id}.faiss"
        meta_path = Path(data_dir) / f"{doc_id}_meta.json"
        if not faiss_path.exists():
            logger.warning("No index found for doc_id=%s", doc_id)
            continue
        idx = faiss.read_index(str(faiss_path))
        indexes.append(idx)
        with open(meta_path) as f:
            doc_meta = json.load(f)
        for entry in doc_meta.values():
            metadata[entry["chunk_id"]] = entry

    if not indexes:
        raise ValueError(f"No valid indexes found for doc_ids: {doc_ids}")

    # Merge into a single index by combining all vectors
    merged = faiss.IndexFlatL2(indexes[0].d)
    for idx in indexes:
        vectors = faiss.rev_swig_ptr(idx.get_xb(), idx.ntotal * idx.d)
        vectors = np.frombuffer(vectors, dtype="float32").reshape(idx.ntotal, idx.d).copy()
        merged.add(vectors)

    return merged, metadata


def list_indexed_docs(data_dir: str = _DEFAULT_DATA_DIR) -> List[str]:
    """Return doc_ids for all documents that have been indexed."""
    return [p.stem for p in Path(data_dir).glob("*.faiss")]
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_vectorstore.py -v
```

Expected: all PASSED

- [ ] **Step 5: Commit**

```bash
git add doc_qa/retrieval/vectorstore.py tests/test_vectorstore.py
git commit -m "feat: add per-document FAISS index with JSON metadata sidecar"
```

---

### Task 11: retrieval/retriever.py

**Files:**
- Create: `doc_qa/retrieval/retriever.py`
- Create: `tests/test_retriever.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_retriever.py
import sys, uuid
sys.path.insert(0, '.')
from unittest.mock import patch, MagicMock
import numpy as np

def _make_meta(n=5):
    return {
        str(uuid.uuid4()): {
            "chunk_id": str(uuid.uuid4()),
            "doc_id": "d1", "filename": "a.pdf", "file_type": "pdf",
            "upload_timestamp": "2026-04-16T00:00:00Z", "page_count": 5,
            "page_number": i+1, "chunk_index": i,
            "section_heading": "Financial Analysis",
            "extraction_method": "native_text", "content_type": "prose",
            "text": f"Cash flow is strong. Sentence {i}.",
            "char_count": 30, "bounding_box": None,
            "embedding_text": f"Cash flow is strong. Sentence {i}.",
        }
        for i in range(n)
    }

def test_retrieve_returns_reranked_chunks(tmp_path):
    import faiss
    # Build a tiny real FAISS index with random vectors
    d = 384
    index = faiss.IndexFlatL2(d)
    vecs = np.random.rand(5, d).astype("float32")
    index.add(vecs)

    meta = {i: list(_make_meta(5).values())[i] for i in range(5)}
    # Build chunk_id-keyed meta dict
    meta_by_chunk_id = {v["chunk_id"]: v for v in meta.values()}

    with patch("doc_qa.retrieval.retriever._embed_query",
               return_value=np.random.rand(d).astype("float32")):
        with patch("doc_qa.retrieval.retriever._rerank",
                   side_effect=lambda q, chunks: [(c, 0.8 - i*0.1) for i, c in enumerate(chunks)]):
            from doc_qa.retrieval.retriever import retrieve
            results = retrieve(
                query="What is the DSCR?",
                index=index,
                metadata_dict=meta_by_chunk_id,
                index_position_map=list(meta_by_chunk_id.keys()),
                top_k=5, rerank_top_n=3,
            )
    assert len(results) == 3
    assert results[0].rank == 1
    assert results[0].reranker_score >= results[1].reranker_score
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_retriever.py -v
```

- [ ] **Step 3: Implement**

```python
# doc_qa/retrieval/retriever.py
"""Query embedding, FAISS similarity search, and cross-encoder reranking.

Two-stage retrieval: FAISS for approximate nearest-neighbor (fast, large recall),
then cross-encoder reranking (slow, precise) on the top-k candidates.
The cross-encoder scores actual (query, passage) pairs for relevance, which is
significantly more accurate than embedding similarity alone — especially for
targeted financial questions like "what is the DSCR covenant threshold".
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """A ProcessedChunk enriched with retrieval scores and rank."""
    chunk_id: str
    doc_id: str
    filename: str
    file_type: str
    upload_timestamp: str
    page_count: int
    page_number: int
    chunk_index: int
    section_heading: str
    extraction_method: str
    content_type: str
    text: str
    char_count: int
    embedding_text: str
    bounding_box: Optional[tuple]
    faiss_score: float    # L2 distance from query embedding (lower = closer)
    reranker_score: float # cross-encoder relevance score (higher = more relevant)
    rank: int             # 1-based position after reranking


def _embed_query(query: str) -> np.ndarray:
    """Embed a query string using the same model used at index time."""
    from langchain_community.embeddings import HuggingFaceEmbeddings
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return np.array(embedder.embed_query(query), dtype="float32")


def _rerank(query: str, chunks: List[dict]) -> List[Tuple[dict, float]]:
    """Score (query, passage) pairs with a cross-encoder model.

    CrossEncoder is used because it reads both the query and the passage
    together, producing a true relevance score rather than an independent
    embedding similarity.
    """
    from sentence_transformers import CrossEncoder
    # ms-marco-MiniLM is trained on MS MARCO passage retrieval tasks,
    # which generalizes well to document Q&A scenarios
    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    pairs = [(query, c["text"]) for c in chunks]
    scores = model.predict(pairs)
    return list(zip(chunks, scores.tolist()))


def retrieve(
    query: str,
    index: faiss.Index,
    metadata_dict: Dict[str, dict],
    index_position_map: List[str],
    top_k: int = 10,
    rerank_top_n: int = 5,
) -> List[RetrievedChunk]:
    """Retrieve and rerank the most relevant chunks for a query.

    Args:
        query: The user's question string.
        index: Merged FAISS index from vectorstore.load_index().
        metadata_dict: chunk_id → metadata dict from vectorstore.load_index().
        index_position_map: Ordered list of chunk_ids matching FAISS index positions.
        top_k: Number of candidates to retrieve from FAISS before reranking.
        rerank_top_n: Final number of chunks to return after reranking.

    Returns:
        List of RetrievedChunk objects sorted by reranker_score descending.
    """
    # Step 1: embed query using the same model as the index
    query_vec = _embed_query(query).reshape(1, -1)

    # Step 2: FAISS approximate nearest-neighbor search
    scores, indices = index.search(query_vec, min(top_k, index.ntotal))
    candidates = []
    for faiss_score, pos in zip(scores[0], indices[0]):
        if pos < 0 or pos >= len(index_position_map):
            continue
        chunk_id = index_position_map[pos]
        meta = metadata_dict.get(chunk_id)
        if meta:
            candidates.append({**meta, "faiss_score": float(faiss_score)})

    if not candidates:
        return []

    # Step 3: cross-encoder reranking — more accurate than embedding similarity
    scored = _rerank(query, candidates)
    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:rerank_top_n]

    # Step 4: assemble RetrievedChunk objects with both scores preserved
    results = []
    for rank, (meta, reranker_score) in enumerate(top, start=1):
        results.append(RetrievedChunk(
            chunk_id=meta["chunk_id"],
            doc_id=meta["doc_id"],
            filename=meta["filename"],
            file_type=meta["file_type"],
            upload_timestamp=meta["upload_timestamp"],
            page_count=meta["page_count"],
            page_number=meta["page_number"],
            chunk_index=meta["chunk_index"],
            section_heading=meta["section_heading"],
            extraction_method=meta["extraction_method"],
            content_type=meta["content_type"],
            text=meta["text"],
            char_count=meta["char_count"],
            embedding_text=meta["embedding_text"],
            bounding_box=meta.get("bounding_box"),
            faiss_score=meta["faiss_score"],
            reranker_score=reranker_score,
            rank=rank,
        ))
    return results
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_retriever.py -v
```

Expected: all PASSED

- [ ] **Step 5: Commit**

```bash
git add doc_qa/retrieval/retriever.py tests/test_retriever.py
git commit -m "feat: add retriever with FAISS search + cross-encoder reranking"
```

---

### Task 12: qa/chain.py — LangGraph QA node

**Files:**
- Create: `doc_qa/qa/chain.py`
- Create: `tests/test_chain.py`

Authentication uses `DefaultAzureCredential` + `get_bearer_token_provider` (no API key).
The QA logic lives in a **LangGraph node** function following the pattern:
`UI → service/orchestrator → graph state → wrapper prompt + inputs → llm.invoke()`.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_chain.py
import sys, pytest; sys.path.insert(0, '.')
from unittest.mock import patch, MagicMock

def _make_retrieved_chunk(text="The DSCR is 1.35x.", score=0.9, rank=1):
    from doc_qa.retrieval.retriever import RetrievedChunk
    return RetrievedChunk(
        chunk_id="c1", doc_id="d1", filename="memo.pdf", file_type="pdf",
        upload_timestamp="2026-04-16T00:00:00Z", page_count=10,
        page_number=3, chunk_index=0, section_heading="Financial Analysis",
        extraction_method="native_text", content_type="prose",
        text=text, char_count=len(text), embedding_text=text,
        bounding_box=None, faiss_score=0.1,
        reranker_score=score, rank=rank,
    )

def test_answer_result_fields():
    from doc_qa.qa.chain import AnswerResult
    chunk = _make_retrieved_chunk()
    ar = AnswerResult(
        query="What is the DSCR?", answer="1.35x",
        retrieved_chunks=[chunk], prompt_tokens=100,
        completion_tokens=20, latency_seconds=1.2,
        timestamp="2026-04-16T10:00:00Z",
        model_deployment="gpt-4o", confidence_level="High",
    )
    assert ar.confidence_level == "High"
    assert len(ar.retrieved_chunks) == 1

def test_qa_node_calls_llm():
    """The LangGraph node must call llm.invoke() and populate state fields."""
    from doc_qa.qa.chain import build_llm, qa_node, QAState

    mock_response = MagicMock()
    mock_response.content = "The DSCR is 1.35x."
    mock_response.usage_metadata = {"input_tokens": 100, "output_tokens": 20}

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_response

    chunk = _make_retrieved_chunk()
    state: QAState = {
        "query": "What is the DSCR?",
        "retrieved_chunks": [chunk],
        "answer": "",
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "latency_seconds": 0.0,
        "timestamp": "",
        "model_deployment": "",
        "confidence_level": "",
    }

    with patch("doc_qa.qa.chain.time") as mock_time:
        mock_time.time.side_effect = [0.0, 1.5]
        result_state = qa_node(state, llm=mock_llm)

    assert "1.35x" in result_state["answer"]
    assert result_state["confidence_level"] == "High"
    mock_llm.invoke.assert_called_once()

def test_answer_question_returns_result():
    from doc_qa.qa.chain import answer_question

    mock_response = MagicMock()
    mock_response.content = "The DSCR is 1.35x."
    mock_response.usage_metadata = {"input_tokens": 80, "output_tokens": 15}
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_response

    chunk = _make_retrieved_chunk()
    with patch("doc_qa.qa.chain.time") as mock_time:
        mock_time.time.side_effect = [0.0, 1.2]
        result = answer_question("What is the DSCR?", [chunk], mock_llm)

    assert result.answer == "The DSCR is 1.35x."
    assert result.latency_seconds == pytest.approx(1.2, abs=0.1)
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_chain.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# doc_qa/qa/chain.py
"""LangGraph QA node and AnswerResult dataclass.

Pattern: UI → service/orchestrator → graph state → wrapper prompt + inputs → llm.invoke()
Authentication: DefaultAzureCredential + bearer token provider (no API key).
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

# ---------------------------------------------------------------------------
# Wrapper system prompt — applied at the node level, not baked into the LLM
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
"""


# ---------------------------------------------------------------------------
# LangGraph state schema
# ---------------------------------------------------------------------------
class QAState(dict):
    """Typed graph state passed through the LangGraph QA node.

    All fields are plain dict keys so LangGraph can merge partial updates.
    """
    query: str              # the user's question
    retrieved_chunks: list  # List[RetrievedChunk] from retriever
    answer: str             # populated by qa_node
    prompt_tokens: int
    completion_tokens: int
    latency_seconds: float
    timestamp: str
    model_deployment: str
    confidence_level: str   # High | Medium | Low


@dataclass
class AnswerResult:
    """Flat answer record returned by answer_question() for logging and UI."""
    query: str
    answer: str
    retrieved_chunks: List[RetrievedChunk]
    prompt_tokens: int
    completion_tokens: int
    latency_seconds: float
    timestamp: str
    model_deployment: str
    confidence_level: str


def build_llm():
    """Construct AzureChatOpenAI authenticated via DefaultAzureCredential.

    Reads env vars: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT,
    OPENAI_API_VERSION. No API key — uses Azure AD bearer token.
    """
    from azure.identity import DefaultAzureCredential, get_bearer_token_provider
    from langchain_openai import AzureChatOpenAI

    credential = DefaultAzureCredential()
    token_provider = get_bearer_token_provider(
        credential, "https://cognitiveservices.azure.com/.default"
    )
    deployment = os.environ["AZURE_OPENAI_DEPLOYMENT"]
    llm = AzureChatOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        azure_deployment=deployment,
        openai_api_version=os.environ.get("OPENAI_API_VERSION", "2024-02-01"),
        azure_ad_token_provider=token_provider,
        temperature=0,  # deterministic for auditability
    )
    logger.info("LLM initialised: deployment=%s", deployment)
    return llm


def _format_context(chunks: List[RetrievedChunk]) -> str:
    """Assemble a context block from retrieved chunks ordered by reranker rank.

    Source metadata prefixed to each chunk lets the LLM cite pages and sections
    in its answer, which is the primary audit mechanism for credit analysts.
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


def qa_node(state: QAState, llm=None) -> QAState:
    """LangGraph node: apply wrapper prompt + context, call llm.invoke().

    Args:
        state: Graph state containing 'query' and 'retrieved_chunks'.
        llm: AzureChatOpenAI instance (injected; built via build_llm() if None).

    Returns:
        Updated state with answer, token counts, latency, and confidence_level.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    if llm is None:
        llm = build_llm()

    chunks = state.get("retrieved_chunks", [])
    query = state["query"]
    context = _format_context(chunks)

    # Wrapper prompt applied at node level — not baked into the LLM constructor
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

    Used by the Streamlit UI and batch runner without wiring a full LangGraph.
    """
    state = qa_node(
        QAState(query=query, retrieved_chunks=retrieved_chunks,
                answer="", prompt_tokens=0, completion_tokens=0,
                latency_seconds=0.0, timestamp="",
                model_deployment="", confidence_level=""),
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
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_chain.py -v
```

Expected: all PASSED

- [ ] **Step 5: Commit**

```bash
git add doc_qa/qa/chain.py tests/test_chain.py
git commit -m "feat: add LangGraph QA node with DefaultAzureCredential auth"
```

---

### Task 13: qa/batch.py

**Files:**
- Create: `doc_qa/qa/batch.py`
- Create: `tests/test_batch.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_batch.py
import sys, uuid; sys.path.insert(0, '.')
from unittest.mock import patch, MagicMock
from doc_qa.retrieval.retriever import RetrievedChunk
from doc_qa.qa.chain import AnswerResult

def _mock_chunk():
    return RetrievedChunk(
        chunk_id=str(uuid.uuid4()), doc_id="d1", filename="memo.pdf",
        file_type="pdf", upload_timestamp="2026-04-16T00:00:00Z",
        page_count=5, page_number=1, chunk_index=0,
        section_heading="Intro", extraction_method="native_text",
        content_type="prose", text="Revenue is $1M.",
        char_count=15, embedding_text="Revenue is $1M.",
        bounding_box=None, faiss_score=0.05, reranker_score=0.9, rank=1,
    )

def _mock_answer(q, filename):
    return AnswerResult(
        query=q, answer=f"Answer to: {q}",
        retrieved_chunks=[_mock_chunk()], prompt_tokens=100,
        completion_tokens=20, latency_seconds=1.0,
        timestamp="2026-04-16T10:00:00Z",
        model_deployment="gpt-4o", confidence_level="High",
    )

def test_run_batch_grid_shape():
    from doc_qa.qa.batch import run_batch

    questions = ["What is revenue?", "What is DSCR?"]
    doc_configs = [
        {"doc_id": "d1", "filename": "memo.pdf"},
        {"doc_id": "d2", "filename": "report.pdf"},
    ]

    with patch("doc_qa.qa.batch._answer_for_doc", side_effect=_mock_answer):
        answers_df, trace_df = run_batch(questions, doc_configs, chain=MagicMock())

    assert answers_df.shape == (2, 2)  # 2 questions x 2 docs
    assert "memo.pdf" in answers_df.columns
    assert len(trace_df) == 4  # 2 questions x 2 docs

def test_export_to_excel(tmp_path):
    import pandas as pd
    from doc_qa.qa.batch import export_to_excel
    answers_df = pd.DataFrame({"doc1.pdf": ["ans1", "ans2"]}, index=["q1", "q2"])
    trace_df = pd.DataFrame({"query": ["q1"], "answer": ["ans1"]})
    out = tmp_path / "export.xlsx"
    export_to_excel(answers_df, trace_df, str(out))
    assert out.exists()
```

- [ ] **Step 2: Implement**

```python
# doc_qa/qa/batch.py
"""Batch runner: N documents × M questions → answer grid + trace log."""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def _answer_for_doc(question: str, filename: str, doc_id: str, chain) -> Any:
    """Answer one question against one document's FAISS index."""
    from doc_qa.retrieval.vectorstore import load_index
    from doc_qa.retrieval.retriever import retrieve
    from doc_qa.qa.chain import answer_question

    index, meta = load_index([doc_id])
    # Build position map: list of chunk_ids in FAISS index order
    position_map = list(meta.keys())
    chunks = retrieve(question, index, meta, position_map, top_k=10, rerank_top_n=5)
    return answer_question(question, chunks, chain)


def run_batch(
    question_list: List[str],
    doc_configs: List[Dict[str, str]],
    chain,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run Q&A for every (question, document) pair.

    Args:
        question_list: List of question strings.
        doc_configs: List of dicts with 'doc_id' and 'filename' keys.
        chain: AzureChatOpenAI chain from build_llm().
        progress_callback: Optional fn(current, total) for progress tracking.

    Returns:
        (answers_df, trace_df) — answers grid and full audit trace.
    """
    total = len(question_list) * len(doc_configs)
    done = 0

    # answers[question][filename] = answer_text
    answers: Dict[str, Dict[str, str]] = {q: {} for q in question_list}
    trace_rows = []

    for question in question_list:
        for doc in doc_configs:
            doc_id = doc["doc_id"]
            filename = doc["filename"]
            try:
                result = _answer_for_doc(question, filename, doc_id, chain)
                answers[question][filename] = result.answer

                top_chunk = result.retrieved_chunks[0] if result.retrieved_chunks else None
                trace_rows.append({
                    "question": question,
                    "filename": filename,
                    "doc_id": doc_id,
                    "answer": result.answer,
                    "confidence_level": result.confidence_level,
                    "top_chunk_page": top_chunk.page_number if top_chunk else "",
                    "top_chunk_section": top_chunk.section_heading if top_chunk else "",
                    "top_reranker_score": top_chunk.reranker_score if top_chunk else "",
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                    "latency_seconds": round(result.latency_seconds, 2),
                    "timestamp": result.timestamp,
                    "model_deployment": result.model_deployment,
                    "extraction_methods_used": ",".join(
                        {c.extraction_method for c in result.retrieved_chunks}
                    ),
                    "chunk_ids_used": ",".join(
                        c.chunk_id for c in result.retrieved_chunks
                    ),
                })
            except Exception as exc:
                logger.error("Batch error (%s, %s): %s", question[:30], filename, exc)
                answers[question][filename] = f"ERROR: {exc}"
                trace_rows.append({
                    "question": question, "filename": filename, "doc_id": doc_id,
                    "answer": f"ERROR: {exc}", "confidence_level": "Low",
                })

            done += 1
            if progress_callback:
                progress_callback(done, total)

    filenames = [d["filename"] for d in doc_configs]
    answers_df = pd.DataFrame(
        {fn: [answers[q].get(fn, "") for q in question_list] for fn in filenames},
        index=question_list,
    )
    trace_df = pd.DataFrame(trace_rows)
    return answers_df, trace_df


def export_to_excel(
    answers_df: pd.DataFrame,
    trace_df: pd.DataFrame,
    output_path: str,
) -> str:
    """Write answers grid and trace log to a formatted Excel workbook.

    Args:
        answers_df: Rows=questions, columns=filenames, cells=answer text.
        trace_df: One row per (question, doc) with all audit metadata.
        output_path: Path to write the .xlsx file.

    Returns:
        output_path (for convenience in the Streamlit download button).
    """
    import openpyxl
    from openpyxl.styles import PatternFill, Font
    from openpyxl.utils import get_column_letter

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        answers_df.to_excel(writer, sheet_name="Answers", index=True)
        trace_df.to_excel(writer, sheet_name="Trace", index=False)

        # Format Answers sheet
        ws_ans = writer.sheets["Answers"]
        ws_ans.freeze_panes = "B2"  # freeze header row and question column
        header_fill = PatternFill(fill_type="solid", fgColor="D9E1F2")
        alt_fill = PatternFill(fill_type="solid", fgColor="F2F2F2")
        for col_idx, col in enumerate(ws_ans.iter_cols(), start=1):
            max_len = max((len(str(cell.value or "")) for cell in col), default=10)
            ws_ans.column_dimensions[get_column_letter(col_idx)].width = min(max_len + 4, 60)
        for row_idx, row in enumerate(ws_ans.iter_rows(), start=1):
            for cell in row:
                if row_idx == 1:
                    cell.fill = header_fill
                    cell.font = Font(bold=True)
                elif row_idx % 2 == 0:
                    cell.fill = alt_fill

    logger.info("Exported batch results to %s", output_path)
    return output_path
```

- [ ] **Step 3: Run tests**

```bash
pytest tests/test_batch.py -v
```

Expected: all PASSED

- [ ] **Step 4: Commit**

```bash
git add doc_qa/qa/batch.py tests/test_batch.py
git commit -m "feat: add batch runner and Excel exporter"
```

---

### Task 14: app.py — Streamlit UI

**Files:**
- Create: `app.py`

- [ ] **Step 1: Write app.py**

```python
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

logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Session state keys — all keys are defined here for documentation
# ---------------------------------------------------------------------------
# "uploaded_docs"  : dict {filename: {"doc_id": str, "status": "indexing"|"ready"}}
# "chat_history"   : list of {"role": "user"|"assistant", "content": str, "meta": dict|None}
# "qa_chain"       : AzureChatOpenAI chain instance (lazy-loaded)
# "batch_questions": list[str] from uploaded question sheet
# "batch_results"  : (answers_df, trace_df) from last batch run

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
    """Lazy-load the AzureChatOpenAI LLM once and cache in session state."""
    if st.session_state.qa_chain is None:
        from doc_qa.qa.chain import build_llm
        st.session_state.qa_chain = build_llm()
    return st.session_state.qa_chain


def ingest_file(uploaded_file) -> str:
    """Write an uploaded file to disk, run ingest pipeline, build FAISS index.

    Returns the doc_id for the indexed document.
    """
    from doc_qa.ingest.extractor import ingest_document
    from doc_qa.ingest.chunker import chunk_raw
    from doc_qa.retrieval.vectorstore import build_index

    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as f:
        f.write(uploaded_file.getbuffer())
        tmp_path = f.name

    raw = ingest_document(tmp_path, uploaded_file.name)
    processed = chunk_raw(raw)
    doc_id = processed[0].doc_id if processed else None
    if doc_id:
        build_index(processed, doc_id)
    os.unlink(tmp_path)
    return doc_id


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
                st.session_state.uploaded_docs[f.name] = {"doc_id": None, "status": "indexing"}
                with st.spinner(f"Indexing {f.name}…"):
                    try:
                        doc_id = ingest_file(f)
                        st.session_state.uploaded_docs[f.name] = {
                            "doc_id": doc_id, "status": "ready"
                        }
                    except Exception as exc:
                        st.session_state.uploaded_docs[f.name] = {
                            "doc_id": None, "status": f"error: {exc}"
                        }

    # Show status badges for each uploaded document
    for name, info in st.session_state.uploaded_docs.items():
        color = {"ready": "green", "indexing": "orange"}.get(info["status"], "red")
        st.markdown(f":{color}[{info['status']}] {name}")

    st.markdown("---")

    # Active docs selector — query all or a chosen subset
    ready_docs = [
        {"filename": n, "doc_id": i["doc_id"]}
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

    # Render full chat history (user + assistant messages)
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            # Show source chips and trace only for assistant messages
            if msg["role"] == "assistant" and msg.get("meta"):
                meta = msg["meta"]
                chunks = meta.get("retrieved_chunks", [])

                # Source chips: one per retrieved chunk
                chip_cols = st.columns(min(len(chunks), 5))
                for i, chunk in enumerate(chunks[:5]):
                    with chip_cols[i]:
                        st.caption(
                            f"**{chunk['filename']}** · p.{chunk['page_number']} · "
                            f"{chunk['reranker_score']:.2f} · {chunk['extraction_method']}"
                        )

                # Confidence badge
                from doc_qa.utils.confidence import confidence_color
                level = meta.get("confidence_level", "Low")
                color = confidence_color(level)
                st.markdown(
                    f'<span style="background:{color};color:white;padding:2px 8px;'
                    f'border-radius:4px;font-size:0.8em">{level} confidence</span>',
                    unsafe_allow_html=True,
                )

                # Expandable trace drawer
                with st.expander("Answer trace"):
                    st.write(f"**Document:** {meta.get('filename', 'N/A')}")
                    st.write(f"**Model:** {meta.get('model_deployment')} | "
                             f"**Tokens:** {meta.get('prompt_tokens')}+{meta.get('completion_tokens')} | "
                             f"**Latency:** {meta.get('latency_seconds', 0):.1f}s | "
                             f"**Time:** {meta.get('timestamp')}")
                    if chunks:
                        import pandas as pd
                        trace_table = pd.DataFrame([{
                            "page": c["page_number"],
                            "section": c["section_heading"],
                            "method": c["extraction_method"],
                            "chars": c["char_count"],
                            "reranker": f"{c['reranker_score']:.3f}",
                        } for c in chunks])
                        st.dataframe(trace_table, use_container_width=True)

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
                        from doc_qa.qa.chain import answer_question
                        from doc_qa.metadata.logger import QueryLogger

                        doc_ids = [d["doc_id"] for d in active_docs]
                        index, meta_dict = load_index(doc_ids)
                        position_map = list(meta_dict.keys())
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
                                {c.extraction_method for c in chunks}),
                            "chunk_ids_used": ",".join(c.chunk_id for c in chunks),
                            "chunks_json": json.dumps([{
                                "chunk_id": c.chunk_id, "text": c.text[:200]
                            } for c in chunks]),
                        })

                        # Store metadata in session state for the trace drawer
                        meta_payload = {
                            "model_deployment": result.model_deployment,
                            "timestamp": result.timestamp,
                            "prompt_tokens": result.prompt_tokens,
                            "completion_tokens": result.completion_tokens,
                            "latency_seconds": result.latency_seconds,
                            "confidence_level": result.confidence_level,
                            "filename": "; ".join(d["filename"] for d in active_docs),
                            "retrieved_chunks": [
                                {
                                    "filename": c.filename,
                                    "page_number": c.page_number,
                                    "section_heading": c.section_heading,
                                    "extraction_method": c.extraction_method,
                                    "char_count": c.char_count,
                                    "reranker_score": c.reranker_score,
                                    "chunk_id": c.chunk_id,
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
        from doc_qa.qa.batch import run_batch, export_to_excel

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
            st.session_state.batch_results = (answers_df, trace_df)
            status.update(label="Batch complete!", state="complete")
        except Exception as exc:
            status.update(label=f"Batch failed: {exc}", state="error")

    if st.session_state.batch_results:
        answers_df, trace_df = st.session_state.batch_results
        st.subheader("Answer grid")
        st.dataframe(answers_df, use_container_width=True)

        # Export button — writes to a temp file and offers download
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as f:
            export_path = f.name
        export_to_excel(answers_df, trace_df, export_path)
        with open(export_path, "rb") as f:
            st.download_button(
                "Download Excel export",
                data=f.read(),
                file_name="batch_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
```

- [ ] **Step 2: Run the app**

```bash
streamlit run app.py
```

Expected: App opens at localhost:8501. Upload a test PDF, verify it shows "ready". Ask a question and verify the answer and trace drawer render.

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat: add Streamlit chat and batch UI"
```

---

### Task 15: README.md

- [ ] **Step 1: Write README** (create file, follow spec requirements)

Content must include: project overview, architecture pipeline diagram, setup instructions, chat/batch usage, metadata field explanations, extraction method descriptions, confidence level guide, known limitations.

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add README with setup, usage, and architecture"
```

---

## Self-Review

**Spec coverage check:**
- ✅ All 14 modules specified (utils, metadata, ingest×5, retrieval×2, qa×2, app)
- ✅ RawChunk and ProcessedChunk dataclasses with all required fields
- ✅ RetrievedChunk with faiss_score, reranker_score, rank
- ✅ AnswerResult with all metadata fields
- ✅ Vision OCR with exact system prompt from spec
- ✅ SemanticChunker with percentile=85, prose only
- ✅ Tables never split
- ✅ pdfplumber + camelot fallback (lattice then stream)
- ✅ FAISS per-doc with JSON sidecar
- ✅ CrossEncoder reranking
- ✅ Batch grid: rows=questions, cols=docs
- ✅ Excel export: two sheets (Answers + Trace)
- ✅ SQLite logger with all trace fields
- ✅ Confidence High/Medium/Low with exact thresholds
- ✅ .env.example (not .env) — env vars: AZURE_OPENAI_DEPLOYMENT, OPENAI_API_VERSION, no API key
- ✅ DefaultAzureCredential + bearer token provider in qa/chain.py and image_table.py
- ✅ LangGraph QAState + qa_node with llm.invoke() pattern
- ✅ TDD throughout

**Type consistency check:**
- `ingest_document()` returns `List[RawChunk]` ✅
- `chunk_raw()` accepts `List[RawChunk]`, returns `List[ProcessedChunk]` ✅
- `build_index()` accepts `List[ProcessedChunk]` ✅
- `load_index()` returns `(faiss.Index, Dict[str, dict])` ✅
- `retrieve()` accepts index + metadata_dict + index_position_map ✅
- `answer_question()` returns `AnswerResult` ✅
- `run_batch()` returns `(pd.DataFrame, pd.DataFrame)` ✅

**Gap found and addressed:** The `retrieve()` function needs an `index_position_map` argument (ordered list of chunk_ids matching FAISS integer positions) — this is included in both the implementation and the test.
