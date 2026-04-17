# doc_qa/retrieval/vectorstore.py
"""FAISS index management — one index file per document.

One index per document allows adding or removing documents without rebuilding
the entire combined index. The JSON sidecar preserves chunk metadata so we
never need to re-embed to look up source information.
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
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def build_index(
    processed_chunks: List[ProcessedChunk],
    doc_id: str,
    data_dir: str = _DEFAULT_DATA_DIR,
) -> None:
    """Embed chunks and persist a FAISS flat index + JSON metadata sidecar.

    Args:
        processed_chunks: Output from chunker.chunk_raw().
        doc_id: Unique document identifier — used as the filename stem.
        data_dir: Directory to write {doc_id}.faiss and {doc_id}_meta.json.
    """
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    embedder = _get_embeddings()
    texts = [c.embedding_text for c in processed_chunks]

    logger.info("Embedding %d chunks for doc_id=%s", len(texts), doc_id)
    vectors = np.array(embedder.embed_documents(texts), dtype="float32")

    # IndexFlatL2 is exact search — sufficient for per-document indexes of <50k chunks
    dimension = vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)

    faiss.write_index(index, str(Path(data_dir) / f"{doc_id}.faiss"))

    # Sidecar: list of dicts in FAISS position order so retriever can reconstruct
    # a chunk object from an integer FAISS result without re-embedding
    meta_list = [asdict(c) for c in processed_chunks]
    with open(Path(data_dir) / f"{doc_id}_meta.json", "w") as f:
        json.dump(meta_list, f)

    logger.info("Index saved: %s/%s.faiss (%d vectors)", data_dir, doc_id, len(texts))


def load_index(
    doc_ids: List[str],
    data_dir: str = _DEFAULT_DATA_DIR,
) -> Tuple[faiss.Index, Dict[str, dict], List[str]]:
    """Load and merge FAISS indexes for multiple documents.

    Args:
        doc_ids: List of doc_id strings to load.
        data_dir: Directory containing .faiss and _meta.json files.

    Returns:
        (merged_index, metadata_dict, position_map) where:
          - merged_index: combined FAISS IndexFlatL2
          - metadata_dict: chunk_id → metadata dict
          - position_map: ordered list of chunk_ids matching FAISS positions
    """
    all_vectors: List[np.ndarray] = []
    metadata: Dict[str, dict] = {}
    position_map: List[str] = []  # index position → chunk_id
    dimension = None

    for doc_id in doc_ids:
        faiss_path = Path(data_dir) / f"{doc_id}.faiss"
        meta_path = Path(data_dir) / f"{doc_id}_meta.json"
        if not faiss_path.exists():
            logger.warning("No index found for doc_id=%s", doc_id)
            continue
        idx = faiss.read_index(str(faiss_path))
        dimension = idx.d

        # Extract raw vectors from the index
        n = idx.ntotal
        vecs = faiss.rev_swig_ptr(idx.get_xb(), n * dimension)
        vecs = np.frombuffer(vecs, dtype="float32").reshape(n, dimension).copy()
        all_vectors.append(vecs)

        with open(meta_path) as f:
            doc_meta = json.load(f)
        for entry in doc_meta:
            cid = entry["chunk_id"]
            metadata[cid] = entry
            position_map.append(cid)

    if not all_vectors:
        raise ValueError(f"No valid indexes found for doc_ids: {doc_ids}")

    merged = faiss.IndexFlatL2(dimension)
    merged.add(np.vstack(all_vectors))
    return merged, metadata, position_map


def list_indexed_docs(data_dir: str = _DEFAULT_DATA_DIR) -> List[str]:
    """Return doc_ids for all documents that have a .faiss index file."""
    return [p.stem for p in Path(data_dir).glob("*.faiss")]
