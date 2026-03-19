"""
Reranker using BGE-reranker-v2-m3.

NEW MODULE — not in the original code at all.

The original retrieved top-4 by vector similarity only.
A reranker re-scores results using cross-attention between the query
and each candidate, dramatically improving precision.

This is especially important for Korean+English mixed documents where
embedding similarity alone can miss semantic matches.
"""

import logging
from typing import List, Optional, Tuple

from langchain_core.documents import Document

from app.config import RERANKER_ENABLED, RERANKER_MODEL, RERANKER_TOP_N

logger = logging.getLogger("tilon.reranker")

_reranker = None


def _load_reranker():
    """Lazy-load the reranker model."""
    global _reranker

    if _reranker is not None:
        return _reranker

    if not RERANKER_ENABLED:
        return None

    try:
        from FlagEmbedding import FlagReranker

        logger.info("Loading reranker model '%s'...", RERANKER_MODEL)
        _reranker = FlagReranker(RERANKER_MODEL, use_fp16=True)
        logger.info("Reranker loaded successfully.")
        return _reranker
    except ImportError:
        logger.warning(
            "FlagEmbedding not installed — reranking disabled. "
            "Install with: pip install FlagEmbedding"
        )
        return None
    except Exception as e:
        logger.error("Failed to load reranker: %s", e)
        return None


def rerank(
    query: str,
    documents: List[Document],
    top_n: Optional[int] = None,
) -> List[Document]:
    """
    Re-score documents against the query and return top_n best matches.

    If reranker is disabled or unavailable, returns documents unchanged.
    """
    if not RERANKER_ENABLED or not documents:
        return documents

    reranker = _load_reranker()
    if reranker is None:
        return documents

    top_n = top_n or RERANKER_TOP_N

    # Build query-document pairs
    pairs = [[query, doc.page_content] for doc in documents]

    try:
        scores = reranker.compute_score(pairs, normalize=True)

        # Handle single result (returns float instead of list)
        if isinstance(scores, (float, int)):
            scores = [scores]

        # Sort by score descending
        scored_docs: List[Tuple[float, Document]] = list(zip(scores, documents))
        scored_docs.sort(key=lambda x: x[0], reverse=True)

        results = [doc for _, doc in scored_docs[:top_n]]
        logger.debug(
            "Reranked %d → %d docs (scores: %s)",
            len(documents),
            len(results),
            [f"{s:.3f}" for s, _ in scored_docs[:top_n]],
        )
        return results

    except Exception as e:
        logger.error("Reranking failed, returning original order: %s", e)
        return documents
