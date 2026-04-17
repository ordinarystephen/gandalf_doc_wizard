# doc_qa/ingest/pdf_extractor.py
"""Native PDF extraction: prose text, native tables (pdfplumber + camelot), image regions."""

import logging
import re
from typing import List, Optional

import fitz  # pymupdf — used for high-resolution pixmap cropping from PDF coordinates
import pdfplumber

from doc_qa.ingest.extractor import RawChunk

logger = logging.getLogger(__name__)

_MIN_IMAGE_WIDTH = 100
_MIN_IMAGE_HEIGHT = 50

_HEADING_PATTERNS = [
    re.compile(r"^[A-Z][A-Z\s]{4,}$"),
    re.compile(
        r"^(Sources of Repayment|Financial Analysis|Executive Summary|"
        r"Collateral|Guarantors?|Transaction Overview|Loan Structure|"
        r"Credit Risk|Conditions)",
        re.IGNORECASE,
    ),
]


def _is_heading(line: str) -> bool:
    line = line.strip()
    if not line:
        return False
    return any(p.match(line) for p in _HEADING_PATTERNS)


def _extract_with_camelot(file_path: str, page_num: int) -> list:
    """Fallback table extraction via camelot-py.

    Tries lattice mode first (visible grid lines, common in Excel-exported
    financial models), then stream mode (whitespace-delimited tables).
    """
    try:
        import camelot
        tables = camelot.read_pdf(file_path, pages=str(page_num), flavor="lattice")
        if tables and tables[0].parsing_report.get("accuracy", 0) >= 80:
            return [t.df.values.tolist() for t in tables]
        tables = camelot.read_pdf(file_path, pages=str(page_num), flavor="stream")
        return [t.df.values.tolist() for t in tables]
    except Exception as exc:
        logger.warning("camelot extraction failed on page %d: %s", page_num, exc)
        return []


def _table_to_markdown(table_data: list) -> Optional[str]:
    try:
        import pandas as pd
        if not table_data or not table_data[0]:
            return None
        df = pd.DataFrame(table_data[1:], columns=table_data[0])
        df = df.dropna(how="all").fillna("")
        return df.to_markdown(index=False)
    except Exception as exc:
        logger.warning("Table to markdown failed: %s", exc)
        return None


def extract_pdf(file_path: str) -> List[RawChunk]:
    """Extract prose, native tables, and image-table regions from a native PDF.

    Steps per page:
      1. Detect image regions → dispatch to vision OCR (image_table.py)
      2. Extract native tables via pdfplumber; fall back to camelot
      3. Extract remaining prose; detect section headings inline

    doc_id/filename/upload_timestamp are left empty — ingest_document() stamps them.
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

            # STEP 1: image regions → vision OCR
            # pdfplumber gives bounding boxes in PDF space; pymupdf crops the pixmap.
            for img in page.images:
                if img.get("width", 0) < _MIN_IMAGE_WIDTH or img.get("height", 0) < _MIN_IMAGE_HEIGHT:
                    continue
                try:
                    fitz_page = pdf_fitz[page_num - 1]
                    x0, y0, x1, y1 = img["x0"], img["y0"], img["x1"], img["y1"]
                    rect = fitz.Rect(x0, y0, x1, y1)
                    pixmap = fitz_page.get_pixmap(clip=rect, dpi=150)
                    from doc_qa.ingest.image_table import extract_image_table
                    img_chunk = extract_image_table(
                        pixmap=pixmap, page_number=page_num,
                        bounding_box=(x0, y0, x1, y1), filename="",
                    )
                    if img_chunk:
                        img_chunk.chunk_index = chunk_index
                        img_chunk.section_heading = current_heading
                        img_chunk.page_count = page_count
                        chunks.append(img_chunk)
                        chunk_index += 1
                except Exception as exc:
                    logger.warning("Image region failed on page %d: %s", page_num, exc)

            # STEP 2: native tables — pdfplumber first, camelot fallback
            try:
                raw_tables = page.extract_tables() or []
            except Exception:
                raw_tables = []

            if raw_tables:
                extraction_method = "pdfplumber_table"
            elif len(page.extract_text() or "") >= 200:
                raw_tables = _extract_with_camelot(file_path, page_num)
                extraction_method = "camelot_table"

            for table_data in raw_tables:
                md_text = _table_to_markdown(table_data)
                if not md_text:
                    continue
                chunks.append(RawChunk(
                    doc_id="", filename="", file_type="pdf",
                    upload_timestamp="", page_count=page_count,
                    page_number=page_num, chunk_index=chunk_index,
                    section_heading=current_heading,
                    extraction_method=extraction_method,
                    content_type="table", text=md_text,
                    char_count=len(md_text), bounding_box=None,
                ))
                chunk_index += 1

            # STEP 3: prose text with inline heading detection
            raw_text = page.extract_text() or ""
            prose_lines: List[str] = []

            for line in raw_text.splitlines():
                if _is_heading(line):
                    if prose_lines:
                        text = "\n".join(prose_lines).strip()
                        if text:
                            chunks.append(RawChunk(
                                doc_id="", filename="", file_type="pdf",
                                upload_timestamp="", page_count=page_count,
                                page_number=page_num, chunk_index=chunk_index,
                                section_heading=current_heading,
                                extraction_method="native_text",
                                content_type="prose", text=text,
                                char_count=len(text), bounding_box=None,
                            ))
                            chunk_index += 1
                        prose_lines = []
                    current_heading = line.strip()
                else:
                    prose_lines.append(line)

            if prose_lines:
                text = "\n".join(prose_lines).strip()
                if text:
                    chunks.append(RawChunk(
                        doc_id="", filename="", file_type="pdf",
                        upload_timestamp="", page_count=page_count,
                        page_number=page_num, chunk_index=chunk_index,
                        section_heading=current_heading,
                        extraction_method="native_text",
                        content_type="prose", text=text,
                        char_count=len(text), bounding_box=None,
                    ))
                    chunk_index += 1

    pdf_fitz.close()
    logger.info("Extracted %d chunks from PDF %s", len(chunks), file_path)
    return chunks
