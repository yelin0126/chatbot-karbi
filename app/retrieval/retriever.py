"""
Retrieval pipeline: vector search → optional reranking → formatted context.

IMPROVEMENTS over original:
- Reranker integration (original had only raw similarity search)
- Context formatting extracted here (was mixed into utils)
- Structured source extraction
"""

import logging
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

from langchain_core.documents import Document

from app.config import (
    VECTOR_TOP_K,
    RERANKER_ENABLED,
    LIVE_RERANKER_ENABLED,
    GLOBAL_MIN_RELEVANCE_SCORE,
    STRONG_KEYWORD_CONFIDENCE_FLOOR,
)
from app.core.vectorstore import (
    similarity_search_with_scores,
    get_documents_by_source,
)
from app.core.parent_store import get_parent
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
    source_type_filter: str = None,
    full_document: bool = False,
    enable_rerank: bool | None = None,
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

    rerank_enabled = RERANKER_ENABLED and (
        LIVE_RERANKER_ENABLED if enable_rerank is None else enable_rerank
    )

    fetch_k = VECTOR_TOP_K * 2 if rerank_enabled else VECTOR_TOP_K

    # Only apply score filtering when NOT scoped to a specific file
    min_score = None if (source_filter or doc_id_filter or source_type_filter) else GLOBAL_MIN_RELEVANCE_SCORE

    vector_results = similarity_search_with_scores(
        query,
        k=fetch_k,
        filter_source=source_filter,
        filter_doc_id=doc_id_filter,
        filter_source_type=source_type_filter,
        min_score=min_score,
    )
    keyword_results = search_keyword_index(
        query,
        k=fetch_k,
        source_filter=source_filter,
        doc_id_filter=doc_id_filter,
        source_type_filter=source_type_filter,
    )

    merged = _fuse_results(vector_results, keyword_results, limit=fetch_k)
    docs = [entry["doc"] for entry in merged]

    scope_parts = []
    if source_filter:
        scope_parts.append(f"source='{source_filter}'")
    if doc_id_filter:
        scope_parts.append(f"doc_id='{doc_id_filter}'")
    if source_type_filter:
        scope_parts.append(f"source_type='{source_type_filter}'")
    scope = f" (scoped to {', '.join(scope_parts)})" if scope_parts else ""
    logger.info(
        "Hybrid retrieval: %d vector + %d keyword -> %d merged for '%s'%s",
        len(vector_results),
        len(keyword_results),
        len(docs),
        query[:80],
        scope,
    )

    if rerank_enabled and len(docs) > 1:
        docs = rerank(query, docs)
        logger.info("After reranking: %d documents", len(docs))
    elif docs and RERANKER_ENABLED and not rerank_enabled:
        logger.info(
            "Skipping reranker for %d retrieved document(s) because live reranking is disabled",
            len(docs),
        )
    elif docs:
        logger.info("Skipping reranker for %d retrieved document(s)", len(docs))

    # ── Parent-child expansion ──────────────────────────────────────────
    # Replace each child chunk with its full-context parent when available.
    # Multiple children sharing the same parent are collapsed to one entry.
    expanded: List[Document] = []
    seen_parent_ids: set = set()
    expanded_count = 0
    for doc in docs:
        parent_id = doc.metadata.get("parent_id")
        if parent_id:
            if parent_id in seen_parent_ids:
                continue  # deduplicate: same parent already added
            parent_text = get_parent(parent_id)
            if parent_text:
                seen_parent_ids.add(parent_id)
                expanded_count += 1
                expanded.append(
                    Document(
                        page_content=parent_text,
                        metadata={**doc.metadata, "expanded_to_parent": True},
                    )
                )
                continue
        expanded.append(doc)
    docs = expanded
    if expanded_count:
        logger.info(
            "Parent expansion: %d child(ren) → %d parent context(s) (%d total)",
            expanded_count, expanded_count, len(docs),
        )
    # ────────────────────────────────────────────────────────────────────

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


def format_context(docs: List[Document], max_chars_per_chunk: int = 900) -> str:
    """
    Format retrieved documents into a context string for the LLM prompt.

    Strips the enrichment header from chunk content (added by enricher.py
    for better embedding) since this function adds its own formatted header.

    Format: [Doc: {source} | Page: {page} | Section: {section} | Lang: {lang}]

    max_chars_per_chunk caps each chunk's text so 4-5 chunks don't blow the
    LLM's token budget before the prompt is even assembled. Truncation prefers
    a sentence boundary (Korean '다. ' / '. ') within the last 120 chars.
    """
    if not docs:
        return ""

    # Pattern to match enrichment header at start of chunk
    _header_re = re.compile(r'^\[Document:.*?\]\n', re.DOTALL)

    parts = []
    for idx, d in enumerate(docs, start=1):
        source = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", "?")
        lang = d.metadata.get("language", "unknown")

        # Use breadcrumb if available (from semantic chunker)
        section = d.metadata.get("section_breadcrumb", "") or d.metadata.get("section_title", "")

        header = f"[{idx}] [Doc: {source} | Page: {page} | Section: {section} | Lang: {lang}]"

        # Strip enrichment header from content to avoid duplication
        content = _header_re.sub('', d.page_content).strip()

        # Truncate to token budget — prefer clean sentence boundary
        if len(content) > max_chars_per_chunk:
            window = content[max_chars_per_chunk - 120 : max_chars_per_chunk]
            cut = -1
            for sep in ("다. ", "요. ", ". ", "? ", "! "):
                pos = window.rfind(sep)
                if pos != -1:
                    cut = max_chars_per_chunk - 120 + pos + len(sep)
                    break
            if cut == -1:
                cut = max_chars_per_chunk
            content = content[:cut].rstrip() + " ..."

        parts.append(f"{header}\n{content}")

    return "\n\n".join(parts)


def format_grouped_corpus_context(
    docs: List[Document],
    max_chunks_per_doc: int = 3,
    max_chars_per_chunk: int = 260,
) -> str:
    """
    Format a multi-document corpus for file-by-file summarization.

    This groups chunks by source and trims very long documents so a single file
    does not dominate the prompt when the user asks for a corpus-level summary.
    """
    if not docs:
        return ""

    header_re = re.compile(r'^\[Document:.*?\]\n', re.DOTALL)
    grouped: Dict[str, List[Document]] = {}
    order: List[str] = []
    for doc in docs:
        source = str(doc.metadata.get("source") or "unknown")
        if source not in grouped:
            grouped[source] = []
            order.append(source)
        grouped[source].append(doc)

    parts: List[str] = []
    for source in order:
        source_docs = grouped[source]
        lang = source_docs[0].metadata.get("language", "unknown")
        total_chunks = len(source_docs)

        if total_chunks > max_chunks_per_doc:
            visible_docs = source_docs[: max_chunks_per_doc - 1] + [source_docs[-1]]
        else:
            visible_docs = source_docs

        parts.append(
            f"[File: {source} | Lang: {lang} | Visible chunks: {len(visible_docs)}/{total_chunks}]"
        )
        seen_sections = []
        for doc in visible_docs:
            page = doc.metadata.get("page", "?")
            section = doc.metadata.get("section_breadcrumb", "") or doc.metadata.get("section_title", "")
            content = header_re.sub("", doc.page_content).strip()
            compact = re.sub(r"\s+", " ", content).strip()
            if len(compact) > max_chars_per_chunk:
                compact = compact[: max_chars_per_chunk].rstrip() + "..."

            if section and section not in seen_sections:
                seen_sections.append(section)

            parts.append(
                f"- Page: {page} | Section: {section or 'general'} | Evidence: {compact}"
            )

        if seen_sections:
            parts.append(
                f"[Sections observed] {', '.join(seen_sections[:5])}"
            )

        if total_chunks > len(visible_docs):
            parts.append(
                f"[Note] Additional chunks from '{source}' were omitted for brevity."
            )

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
