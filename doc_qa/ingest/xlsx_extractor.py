# doc_qa/ingest/xlsx_extractor.py
"""Excel workbook extraction — every sheet is treated as a table.

Excel files are entirely structured tabular data, so all content is
extracted as table chunks without prose chunking.
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
        One RawChunk per non-empty sheet with content_type='table' and
        section_heading='Sheet: {sheet_name}'.
        doc_id/filename/upload_timestamp are left as "" — stamped by ingest_document().
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
            df = df.dropna(how="all").dropna(axis=1, how="all")
            df = df.apply(lambda col: col.map(
                lambda x: x.strip() if isinstance(x, str) else x
            ))
            if df.empty:
                continue
            md_text = df.to_markdown(index=False)
            section = f"Sheet: {sheet_name}"
            chunks.append(RawChunk(
                doc_id="", filename="", file_type="xlsx",
                upload_timestamp="", page_count=0,
                page_number=0, chunk_index=idx,
                section_heading=section,
                extraction_method="openpyxl",
                content_type="table",
                text=md_text, char_count=len(md_text),
                bounding_box=None,
            ))
        except Exception as exc:
            logger.warning(
                "Failed to extract sheet '%s' from %s: %s", sheet_name, file_path, exc
            )

    logger.info("Extracted %d sheet chunks from XLSX %s", len(chunks), file_path)
    return chunks
