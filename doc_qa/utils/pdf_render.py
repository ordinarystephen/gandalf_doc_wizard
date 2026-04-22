# doc_qa/utils/pdf_render.py
"""Rasterize PDF pages to PNG for UI source preview.

Used by the Streamlit trace drawer to show analysts the exact page a cited
chunk came from — the core audit-trail feature for credit review.
"""

import logging
from pathlib import Path

import fitz  # pymupdf — already a dependency for pdf_extractor

logger = logging.getLogger(__name__)


def render_pdf_page(file_path: str, page_number: int, dpi: int = 110) -> bytes:
    """Render a single 1-indexed PDF page to PNG bytes.

    dpi=110 keeps most letter-sized pages under ~250KB — readable in a
    Streamlit column without blowing up session state or bandwidth.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Source PDF missing: {file_path}")

    with fitz.open(str(path)) as doc:
        if not (1 <= page_number <= len(doc)):
            raise ValueError(
                f"page_number {page_number} out of range for {path.name} "
                f"({len(doc)} pages)"
            )
        page = doc[page_number - 1]
        pixmap = page.get_pixmap(dpi=dpi)
        return pixmap.tobytes("png")
