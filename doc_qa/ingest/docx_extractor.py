# doc_qa/ingest/docx_extractor.py
"""Word document extraction preserving document order of prose and tables.

python-docx preserves document order in its object model, unlike PDFs where
content order must be inferred from coordinates. This makes DOCX extraction
simpler and more reliable for structured credit documents.
"""

import logging
from typing import List

import pandas as pd
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph

from doc_qa.ingest.extractor import RawChunk

logger = logging.getLogger(__name__)


def _table_to_markdown(table: Table) -> str:
    """Convert a python-docx Table to markdown.

    Word tables can have merged cells — python-docx exposes them as repeated
    cell text across the span. We forward-fill to handle this.
    """
    rows = [[cell.text.strip() for cell in row.cells] for row in table.rows]
    if not rows:
        return ""
    df = pd.DataFrame(rows[1:], columns=rows[0])
    df = df.replace("", pd.NA).ffill(axis=1).fillna("")
    return df.to_markdown(index=False)


def extract_docx(file_path: str) -> List[RawChunk]:
    """Extract paragraphs and tables from a Word document in document order.

    Args:
        file_path: Path to the .docx file.

    Returns:
        List of RawChunk objects with extraction_method='python_docx'.
        doc_id/filename/upload_timestamp are left as "" — stamped by ingest_document().
    """
    try:
        doc = Document(file_path)
    except Exception as exc:
        logger.error("Cannot open DOCX %s: %s", file_path, exc)
        return []

    chunks: List[RawChunk] = []
    chunk_index = 0
    current_heading = ""

    # Iterate body children directly to preserve prose/table interleaving
    for child in doc.element.body:
        tag = child.tag.split("}")[-1]  # strip XML namespace prefix

        if tag == "p":
            para = Paragraph(child, doc)
            text = para.text.strip()
            style_name = para.style.name if para.style else ""

            if not text:
                continue

            if "Heading" in style_name:
                # Headings delimit sections — update tracker, don't create a chunk
                current_heading = text
                continue

            chunks.append(RawChunk(
                doc_id="", filename="", file_type="docx",
                upload_timestamp="", page_count=0,
                page_number=0, chunk_index=chunk_index,
                section_heading=current_heading,
                extraction_method="python_docx",
                content_type="prose",
                text=text, char_count=len(text),
                bounding_box=None,
            ))
            chunk_index += 1

        elif tag == "tbl":
            table = Table(child, doc)
            md_text = _table_to_markdown(table)
            if not md_text:
                continue
            chunks.append(RawChunk(
                doc_id="", filename="", file_type="docx",
                upload_timestamp="", page_count=0,
                page_number=0, chunk_index=chunk_index,
                section_heading=current_heading,
                extraction_method="python_docx",
                content_type="table",
                text=md_text, char_count=len(md_text),
                bounding_box=None,
            ))
            chunk_index += 1

    logger.info("Extracted %d chunks from DOCX %s", len(chunks), file_path)
    return chunks
