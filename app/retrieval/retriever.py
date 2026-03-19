"""
Retrieval pipeline: vector search → optional reranking → formatted context.

IMPROVEMENTS over original:
- Reranker integration (original had only raw similarity search)
- Context formatting extracted here (was mixed into utils)
- Structured source extraction
"""

import logging
from typing import List, Dict, Any

from langchain_core.documents import Document

from app.config import VECTOR_TOP_K, RERANKER_ENABLED
from app.core.vectorstore import similarity_search
from app.retrieval.reranker import rerank

logger = logging.getLogger("tilon.retriever")


def retrieve(query: str) -> List[Document]:
    """
    Retrieve relevant documents for a query.

    1. Vector similarity search (top-K from ChromaDB)
    2. Rerank if enabled (narrows to top-N best matches)
    """
    # Fetch more candidates if reranker will narrow them down
    fetch_k = VECTOR_TOP_K * 2 if RERANKER_ENABLED else VECTOR_TOP_K

    docs = similarity_search(query, k=fetch_k)
    logger.info("Retrieved %d candidates for query: '%s'", len(docs), query[:80])

    if RERANKER_ENABLED and docs:
        docs = rerank(query, docs)
        logger.info("After reranking: %d documents", len(docs))

    return docs


def format_context(docs: List[Document]) -> str:
    """
    Format retrieved documents into a context string for the LLM prompt.

    This format is critical — Usama's training data must match this exactly.
    Format: [Doc: {source} | Page: {page} | Section: {section} | Lang: {lang}]
    """
    if not docs:
        return ""

    parts = []
    for idx, d in enumerate(docs, start=1):
        source = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", "?")
        section = d.metadata.get("section_title", "")
        lang = d.metadata.get("language", "unknown")

        header = f"[Doc: {source} | Page: {page} | Section: {section} | Lang: {lang}]"
        parts.append(f"{header}\n{d.page_content}")

    return "\n\n".join(parts)


def extract_sources(docs: List[Document]) -> List[Dict[str, Any]]:
    """Extract source metadata from documents for API response."""
    return [
        {
            "source": d.metadata.get("source"),
            "page": d.metadata.get("page"),
            "chunk_index": d.metadata.get("chunk_index"),
            "extraction_method": d.metadata.get("extraction_method"),
        }
        for d in docs
    ]
