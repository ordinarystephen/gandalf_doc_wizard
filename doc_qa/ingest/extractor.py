# doc_qa/ingest/extractor.py
"""File-type router and shared dataclasses for the ingest pipeline."""

import logging
import uuid
from dataclasses import dataclass
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
    page_count: int           # total pages (0 for xlsx/docx)
    page_number: int          # 1-based page number (0 for xlsx/docx)
    chunk_index: int          # 0-based position within the document
    section_heading: str      # nearest heading above this chunk
    extraction_method: str    # native_text | pdfplumber_table | camelot_table | vision_ocr_gpt4o | python_docx | openpyxl
    content_type: str         # prose | table | image_table
    text: str                 # the extracted text content
    char_count: int           # len(text)
    bounding_box: Optional[tuple]  # (x0, y0, x1, y1) in PDF points, None for non-PDF


@dataclass
class ProcessedChunk:
    """A RawChunk augmented with chunk_id and embedding_text.

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
    """Route a file to the correct extractor and stamp document-level metadata.

    Args:
        file_path: Absolute path to the file on disk.
        filename: Original filename (used in metadata, not for routing).

    Returns:
        List of RawChunk objects with doc_id and upload_timestamp populated.
    """
    ext = Path(filename).suffix.lower()
    doc_id = str(uuid.uuid4())
    upload_timestamp = datetime.now(timezone.utc).isoformat()

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

    for chunk in raw_chunks:
        chunk.doc_id = doc_id
        chunk.filename = filename
        chunk.file_type = ext.lstrip(".")
        chunk.upload_timestamp = upload_timestamp

    logger.info("Ingested %d chunks from %s (doc_id=%s)", len(raw_chunks), filename, doc_id)
    return raw_chunks
