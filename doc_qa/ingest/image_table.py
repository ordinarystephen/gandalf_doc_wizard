# doc_qa/ingest/image_table.py
"""Vision OCR extraction for financial tables embedded as images in PDFs.

GPT-4o vision is used rather than pytesseract because tesseract does character
recognition but has no understanding of table structure, merged cells, or
financial formatting conventions. GPT-4o reconstructs the semantic structure.
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
    """Send a PNG to Azure OpenAI GPT-4o vision and return the text response.

    Mirrors the Kronos auth pattern used in chain.build_llm: bearer token from
    DefaultAzureCredential, `api_version=` (not the deprecated
    `openai_api_version=`), and `azure_endpoint` only set when the env provides
    one — some platforms (Domino proxy) inject it for you.

    Separated from extract_image_table so tests can mock it without needing
    Azure credentials.
    """
    from azure.identity import DefaultAzureCredential, get_bearer_token_provider
    from langchain_openai import AzureChatOpenAI
    from langchain_core.messages import HumanMessage

    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(),
        "https://cognitiveservices.azure.com/.default",
    )
    vision_deployment = os.getenv(
        "AZURE_OPENAI_VISION_DEPLOYMENT",
        os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
    )
    kwargs = dict(
        azure_deployment=vision_deployment,
        api_version=os.environ["OPENAI_API_VERSION"],
        azure_ad_token_provider=token_provider,
    )
    if os.getenv("AZURE_OPENAI_ENDPOINT"):
        kwargs["azure_endpoint"] = os.environ["AZURE_OPENAI_ENDPOINT"]
    client = AzureChatOpenAI(**kwargs)
    b64 = base64.b64encode(png_bytes).decode("utf-8")
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
        pixmap: fitz.Pixmap of the image region (from pdf_extractor.py).
        page_number: 1-based page number for metadata.
        bounding_box: (x0, y0, x1, y1) in PDF coordinate space.
        filename: Source filename for metadata.

    Returns:
        RawChunk with content_type='image_table', or None if no table found
        or if the API call fails. Never raises — failures are logged as warnings.
    """
    try:
        png_bytes = pixmap.tobytes("png")
        response = _call_vision_api(png_bytes)

        if response == "NO_TABLE":
            logger.info("Vision OCR: no table on page %d of %s", page_number, filename)
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
        logger.warning(
            "Vision OCR failed on page %d of %s: %s", page_number, filename, exc
        )
        return None
