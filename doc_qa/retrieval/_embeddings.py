# doc_qa/retrieval/_embeddings.py
"""Shared embedder factory for indexing (vectorstore) and querying (retriever).

One place to construct the embedder, one cache. Both call sites MUST use the
same embedder — an index built with one model can't be queried with another
(the vectors wouldn't live in the same space, and FAISS dimension would
mismatch on load).

Azure-only on this branch — uses AzureOpenAIEmbeddings + bearer token from
DefaultAzureCredential. Required env: AZURE_EMBEDDING_DEPLOYMENT,
OPENAI_API_VERSION. Optional: AZURE_OPENAI_ENDPOINT.
"""

import os
from typing import Any, Optional

_cache: Optional[Any] = None


def get_embeddings():
    """Return the shared Azure embedder, building it on first call.

    Cached module-level so repeat calls (per-document ingest, per-query
    retrieval) don't re-authenticate. Tests should patch at the call site
    (``doc_qa.retrieval.vectorstore.get_embeddings`` /
    ``doc_qa.retrieval.retriever.get_embeddings``) rather than this module,
    which naturally bypasses the cache for the test's duration.
    """
    global _cache
    if _cache is not None:
        return _cache

    missing = [v for v in ("AZURE_EMBEDDING_DEPLOYMENT", "OPENAI_API_VERSION") if not os.getenv(v)]
    if missing:
        raise RuntimeError(
            f"Missing required Azure env var(s): {', '.join(missing)}. "
            "This branch is Azure-only — set them in the Domino project env."
        )

    from azure.identity import DefaultAzureCredential, get_bearer_token_provider
    from langchain_openai import AzureOpenAIEmbeddings

    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(),
        "https://cognitiveservices.azure.com/.default",
    )
    kwargs = dict(
        azure_deployment=os.environ["AZURE_EMBEDDING_DEPLOYMENT"],
        api_version=os.environ["OPENAI_API_VERSION"],
        azure_ad_token_provider=token_provider,
    )
    if os.getenv("AZURE_OPENAI_ENDPOINT"):
        kwargs["azure_endpoint"] = os.environ["AZURE_OPENAI_ENDPOINT"]
    _cache = AzureOpenAIEmbeddings(**kwargs)
    return _cache


def reset_cache() -> None:
    """Clear the cached embedder. Intended for tests that swap auth modes."""
    global _cache
    _cache = None
