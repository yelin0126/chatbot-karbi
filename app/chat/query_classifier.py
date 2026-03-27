import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional

from app.chat.query_classifier_data import INTENT_EXEMPLARS, INTENT_MIN_CONFIDENCE
from app.config import (
    QUERY_CLASSIFIER_ENABLED,
    QUERY_CLASSIFIER_MIN_CONFIDENCE,
    QUERY_CLASSIFIER_PROVIDER,
)
from app.core.embeddings import get_embeddings

logger = logging.getLogger("tilon.query_classifier")


SUPPORTED_QUERY_INTENTS = {
    "smalltalk",
    "direct_extraction",
    "factual_lookup",
    "article_lookup",
    "summarization",
    "comparison",
    "general_knowledge",
    "default",
}


@dataclass(frozen=True)
class QueryClassification:
    intent: str
    confidence: float
    provider: str


def _normalize_intent(label: str) -> Optional[str]:
    normalized = (label or "").strip().lower()
    if normalized in SUPPORTED_QUERY_INTENTS:
        return normalized
    return None


def _dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _mean_vector(vectors: List[List[float]]) -> List[float]:
    if not vectors:
        return []
    length = len(vectors[0])
    return [
        sum(vec[i] for vec in vectors) / len(vectors)
        for i in range(length)
    ]


@lru_cache(maxsize=1)
def _label_prototypes() -> Dict[str, List[float]]:
    embedding_model = get_embeddings()
    prototypes: Dict[str, List[float]] = {}
    for intent, examples in INTENT_EXEMPLARS.items():
        vectors = embedding_model.embed_documents(examples)
        prototypes[intent] = _mean_vector(vectors)
    logger.info("Query classifier prototypes loaded for %d intents", len(prototypes))
    return prototypes


def _classify_query_embedding(text: str) -> Optional[QueryClassification]:
    try:
        prototypes = _label_prototypes()
        embedding_model = get_embeddings()
    except Exception as exc:
        logger.warning("Embedding query classifier unavailable; falling back to heuristics: %s", exc)
        return None

    if not prototypes:
        return None

    query_vec = embedding_model.embed_query(text)
    scores = sorted(
        ((intent, _dot(query_vec, prototype)) for intent, prototype in prototypes.items()),
        key=lambda item: item[1],
        reverse=True,
    )
    best_intent, best_score = scores[0]
    second_score = scores[1][1] if len(scores) > 1 else 0.0

    normalized_score = (best_score + 1.0) / 2.0
    margin = max(0.0, best_score - second_score)
    confidence = max(0.0, min(1.0, (normalized_score * 0.7) + (margin * 0.3)))
    min_confidence = max(
        QUERY_CLASSIFIER_MIN_CONFIDENCE,
        INTENT_MIN_CONFIDENCE.get(best_intent, QUERY_CLASSIFIER_MIN_CONFIDENCE),
    )

    if confidence < min_confidence:
        logger.debug(
            "Embedding query classifier below confidence threshold: intent=%s score=%.3f margin=%.3f confidence=%.3f threshold=%.3f",
            best_intent,
            best_score,
            margin,
            confidence,
            min_confidence,
        )
        return None

    return QueryClassification(
        intent=best_intent,
        confidence=confidence,
        provider="embedding",
    )


def classify_query(text: str) -> Optional[QueryClassification]:
    """
    Placeholder classifier hook for Phase 10B.

    Today this returns None unless a classifier backend is explicitly enabled.
    The policy layer will safely fall back to heuristics.  This gives the
    pipeline a stable insertion point for SetFit or another lightweight model
    without changing routing call sites again.
    """
    if not QUERY_CLASSIFIER_ENABLED:
        return None

    provider = (QUERY_CLASSIFIER_PROVIDER or "").strip().lower()
    if not provider or provider == "heuristic":
        return None

    if provider == "embedding":
        return _classify_query_embedding(text)

    # Future providers can be added here, for example:
    #   - setfit
    #   - transformers
    #   - fasttext
    return None
