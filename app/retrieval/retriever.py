"""
Retrieval pipeline: vector search → optional reranking → formatted context.

IMPROVEMENTS over original:
- Reranker integration (original had only raw similarity search)
- Context formatting extracted here (was mixed into utils)
- Structured source extraction
"""

import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

from langchain_core.documents import Document

from app.config import (
    VECTOR_TOP_K,
    RERANKER_ENABLED,
    GLOBAL_MIN_RELEVANCE_SCORE,
    STRONG_KEYWORD_CONFIDENCE_FLOOR,
)
from app.core.vectorstore import (
    similarity_search_with_scores,
    get_documents_by_source,
)
from app.retrieval.keyword_index import search_keyword_index, tokenize_text
from app.retrieval.reranker import rerank

logger = logging.getLogger("tilon.retriever")

RRF_K = 60


@dataclass
class RetrievalResult:
    docs: List[Document]
    confidence: float
    strong_keyword_hit: bool
    used_full_document: bool = False


def retrieve(
    query: str,
    source_filter: str = None,
    doc_id_filter: str = None,
    full_document: bool = False,
) -> RetrievalResult:
    """
    Retrieve relevant documents for a query.

    When source_filter is set (file uploaded): search only that file's chunks.
    When no filter (general chat): filter out low-relevance results so
    "hello" doesn't return random document chunks.
    """
    if (source_filter or doc_id_filter) and full_document:
        docs = get_documents_by_source(source=source_filter, doc_id=doc_id_filter)
        logger.info(
            "Loaded %d chunks for whole-document task from '%s'%s",
            len(docs),
            source_filter or "scoped document",
            f" ({doc_id_filter})" if doc_id_filter else "",
        )
        return RetrievalResult(
            docs=docs,
            confidence=1.0 if docs else 0.0,
            strong_keyword_hit=bool(docs),
            used_full_document=True,
        )

    fetch_k = VECTOR_TOP_K * 2 if RERANKER_ENABLED else VECTOR_TOP_K

    # Only apply score filtering when NOT scoped to a specific file
    min_score = None if (source_filter or doc_id_filter) else GLOBAL_MIN_RELEVANCE_SCORE

    vector_results = similarity_search_with_scores(
        query,
        k=fetch_k,
        filter_source=source_filter,
        filter_doc_id=doc_id_filter,
        min_score=min_score,
    )
    keyword_results = search_keyword_index(
        query,
        k=fetch_k,
        source_filter=source_filter,
        doc_id_filter=doc_id_filter,
    )

    merged = _fuse_results(vector_results, keyword_results, limit=fetch_k)
    docs = [entry["doc"] for entry in merged]

    scope_parts = []
    if source_filter:
        scope_parts.append(f"source='{source_filter}'")
    if doc_id_filter:
        scope_parts.append(f"doc_id='{doc_id_filter}'")
    scope = f" (scoped to {', '.join(scope_parts)})" if scope_parts else ""
    logger.info(
        "Hybrid retrieval: %d vector + %d keyword -> %d merged for '%s'%s",
        len(vector_results),
        len(keyword_results),
        len(docs),
        query[:80],
        scope,
    )

    if RERANKER_ENABLED and len(docs) > 1:
        docs = rerank(query, docs)
        logger.info("After reranking: %d documents", len(docs))
    elif docs:
        logger.info("Skipping reranker for %d retrieved document(s)", len(docs))

    strong_keyword_hit = _has_strong_keyword_hit(query, keyword_results, doc_id_filter=doc_id_filter)
    confidence = _estimate_confidence(vector_results, merged, strong_keyword_hit)

    return RetrievalResult(
        docs=docs,
        confidence=confidence,
        strong_keyword_hit=strong_keyword_hit,
        used_full_document=False,
    )


def _doc_key(doc: Document) -> str:
    meta = doc.metadata
    return str(
        meta.get("chunk_id")
        or f"{meta.get('doc_id') or meta.get('source')}::{meta.get('page')}::{meta.get('chunk_index')}"
    )


def _fuse_results(
    vector_results: List[Tuple[Document, float]],
    keyword_results: List[Tuple[Document, float]],
    limit: int,
) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}

    for rank, (doc, score) in enumerate(vector_results, start=1):
        key = _doc_key(doc)
        entry = merged.setdefault(
            key,
            {
                "doc": doc,
                "rrf_score": 0.0,
                "vector_score": 0.0,
                "keyword_score": 0.0,
                "in_vector": False,
                "in_keyword": False,
            },
        )
        entry["rrf_score"] += 1.0 / (RRF_K + rank)
        entry["vector_score"] = max(entry["vector_score"], score)
        entry["in_vector"] = True

    for rank, (doc, score) in enumerate(keyword_results, start=1):
        key = _doc_key(doc)
        entry = merged.setdefault(
            key,
            {
                "doc": doc,
                "rrf_score": 0.0,
                "vector_score": 0.0,
                "keyword_score": 0.0,
                "in_vector": False,
                "in_keyword": False,
            },
        )
        entry["rrf_score"] += 1.0 / (RRF_K + rank)
        entry["keyword_score"] = max(entry["keyword_score"], score)
        entry["in_keyword"] = True

    return sorted(
        merged.values(),
        key=lambda item: (
            item["rrf_score"],
            item["vector_score"],
            item["keyword_score"],
        ),
        reverse=True,
    )[:limit]


def _has_strong_keyword_hit(
    query: str,
    keyword_results: List[Tuple[Document, float]],
    doc_id_filter: str = None,
) -> bool:
    if not keyword_results:
        return False

    query_tokens = [token for token in tokenize_text(query) if len(token) >= 2]
    if not query_tokens:
        return False

    top_doc_tokens = set(tokenize_text(keyword_results[0][0].page_content))
    matches = [token for token in query_tokens if token in top_doc_tokens]
    if not matches:
        return False

    exact_technical = any(
        any(char.isdigit() for char in token) or "-" in token or "_" in token
        for token in matches
    )
    if exact_technical:
        return True

    if doc_id_filter and len(matches) >= 1:
        return True

    return len(matches) >= min(len(query_tokens), 2)


def _estimate_confidence(
    vector_results: List[Tuple[Document, float]],
    merged: List[Dict[str, Any]],
    strong_keyword_hit: bool,
) -> float:
    top_vector_score = vector_results[0][1] if vector_results else 0.0
    confidence = max(0.0, top_vector_score)

    if merged and merged[0]["in_vector"] and merged[0]["in_keyword"]:
        confidence = max(confidence, min(1.0, top_vector_score + 0.15))

    if strong_keyword_hit:
        confidence = max(confidence, STRONG_KEYWORD_CONFIDENCE_FLOOR)

    return min(confidence, 1.0)


def format_context(docs: List[Document]) -> str:
    """
    Format retrieved documents into a context string for the LLM prompt.

    Strips the enrichment header from chunk content (added by enricher.py
    for better embedding) since this function adds its own formatted header.

    Format: [Doc: {source} | Page: {page} | Section: {section} | Lang: {lang}]
    """
    if not docs:
        return ""

    import re
    # Pattern to match enrichment header at start of chunk
    _header_re = re.compile(r'^\[Document:.*?\]\n', re.DOTALL)

    parts = []
    for idx, d in enumerate(docs, start=1):
        source = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", "?")
        lang = d.metadata.get("language", "unknown")

        # Use breadcrumb if available (from semantic chunker)
        section = d.metadata.get("section_breadcrumb", "") or d.metadata.get("section_title", "")

        header = f"[Doc: {source} | Page: {page} | Section: {section} | Lang: {lang}]"

        # Strip enrichment header from content to avoid duplication
        content = _header_re.sub('', d.page_content).strip()

        parts.append(f"{header}\n{content}")

    return "\n\n".join(parts)


def extract_sources(docs: List[Document]) -> List[Dict[str, Any]]:
    """Extract source metadata from documents for API response."""
    return [
        {
            "doc_id": d.metadata.get("doc_id"),
            "source": d.metadata.get("source"),
            "source_type": d.metadata.get("source_type"),
            "source_path": d.metadata.get("source_path"),
            "page": d.metadata.get("page"),
            "section": d.metadata.get("section_breadcrumb", "") or d.metadata.get("section_title", ""),
            "chunk_index": d.metadata.get("chunk_index"),
            "chunk_type": d.metadata.get("chunk_type", ""),
            "extraction_method": d.metadata.get("extraction_method"),
        }
        for d in docs
    ]
