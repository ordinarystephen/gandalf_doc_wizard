# doc_qa/ingest/chunker.py
"""Chunking for prose; passthrough for tables and image tables.

RecursiveCharacterTextSplitter is used for speed — no embedding calls during
ingestion. chunk_size=800 with chunk_overlap=100 keeps financial sentences
together while staying well under the embedding model's 256-token limit.
"""

import logging
import uuid
from typing import List

from doc_qa.ingest.extractor import ProcessedChunk, RawChunk

logger = logging.getLogger(__name__)


def _chunk_prose(raw: RawChunk) -> List[ProcessedChunk]:
    """Split prose using RecursiveCharacterTextSplitter."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    texts = splitter.split_text(raw.text) or [raw.text]

    return [
        ProcessedChunk(
            doc_id=raw.doc_id, filename=raw.filename, file_type=raw.file_type,
            upload_timestamp=raw.upload_timestamp, page_count=raw.page_count,
            page_number=raw.page_number, chunk_index=raw.chunk_index + i,
            section_heading=raw.section_heading,
            extraction_method=raw.extraction_method,
            content_type=raw.content_type,
            text=text, char_count=len(text),
            bounding_box=raw.bounding_box,
            chunk_id=str(uuid.uuid4()),
            embedding_text=text,
        )
        for i, text in enumerate(texts)
    ]


def _chunk_table(raw: RawChunk) -> ProcessedChunk:
    """Store a table as a single chunk — never split.

    Splitting a financial table mid-row destroys its structure. The LLM needs
    the complete table with headers to answer numeric questions accurately.
    The section heading prefix in embedding_text helps retrieval surface this
    table when questions reference the section name rather than table content.
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
            results.append(_chunk_table(raw))
    logger.info(
        "Chunking produced %d ProcessedChunks from %d RawChunks",
        len(results), len(raw_chunks),
    )
    return results
