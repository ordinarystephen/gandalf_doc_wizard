# doc_qa/retrieval/_embeddings.py
"""Shared embedder factory for indexing (vectorstore) and querying (retriever).

One place for the Azure/OpenAI auto-switch, one cache. Both call sites MUST
use the same embedder — an index built with one model can't be queried with
another (the vectors wouldn't live in the same space, and FAISS dimension
would mismatch on load).

Auth follows the Kronos pattern (same as chain.build_llm):
  - AZURE_EMBEDDING_DEPLOYMENT set → AzureOpenAIEmbeddings + bearer token
    from DefaultAzureCredential (production / Domino / work).
  - Else → OpenAIEmbeddings + OPENAI_API_KEY (local dev).
"""

import os
from typing import Any, Optional

# Module-level constant for the OpenAI-path model. For the Azure path this is
# ignored — the deployment name in AZURE_EMBEDDING_DEPLOYMENT is what matters.
# text-embedding-3-small: 1536-dim, ~$0.02/1M tokens (good default)
# text-embedding-3-large: 3072-dim, ~$0.13/1M tokens (marginal gain, 6x cost)
EMBEDDING_MODEL = "text-embedding-3-small"

_cache: Optional[Any] = None


def get_embeddings():
    """Return the shared embedder, building it on first call.

    Cached module-level so repeat calls (per-document ingest, per-query
    retrieval) don't re-authenticate. Tests should patch at the call site
    (``doc_qa.retrieval.vectorstore.get_embeddings`` /
    ``doc_qa.retrieval.retriever.get_embeddings``) rather than this module,
    which naturally bypasses the cache for the test's duration.
    """
    global _cache
    if _cache is not None:
        return _cache

    if os.getenv("AZURE_EMBEDDING_DEPLOYMENT"):
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
    else:
        from langchain_openai import OpenAIEmbeddings
        _cache = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    return _cache


def reset_cache() -> None:
    """Clear the cached embedder. Intended for tests that swap auth modes."""
    global _cache
    _cache = None
