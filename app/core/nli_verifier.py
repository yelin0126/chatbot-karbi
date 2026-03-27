"""
NLI-based answer faithfulness verifier.

Uses the already-loaded BGE reranker-v2-m3 cross-encoder to estimate how
well each sentence in the generated answer is supported by the retrieved
source chunks.  No additional model downloads required.

Method
------
For each sentence in the answer we compute
    max_score(sentence, source_chunks)
using the cross-encoder.  The per-sentence max scores are averaged to
produce a document-level faithfulness score in [0, 1].

A score > 0.35 means the answer is broadly grounded in the provided sources.
A score < 0.20 suggests the model may have hallucinated content.

Usage
-----
    from app.core.nli_verifier import check_faithfulness

    score = check_faithfulness(answer_text, [chunk.page_content for chunk in docs])
    if score < 0.25:
        logger.warning("Low faithfulness score: %.2f", score)
"""

import logging
import re
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger("tilon.nli_verifier")

# Split Korean/English sentences on common sentence endings.
_SENT_SPLIT_RE = re.compile(
    r'(?<=[다요았었겠]\.)\s+'           # Korean verb endings before period
    r'|(?<=[.!?])\s+'                  # Standard punctuation
    r'|\n+'                             # Newlines
)

_MIN_SENTENCE_CHARS = 15   # ignore very short fragments
_FAITHFULNESS_WARN_THRESHOLD = 0.25
_CITATION_RE = re.compile(r"\[(\d+)\]")
_SECTION_LABEL_RE = re.compile(
    r"^\s*[-*]?\s*"
    r"(핵심 답변|근거 요약|참고 문서|Key answer|Evidence summary|Final answer|Analysis|Key facts)"
    r"\s*:\s*",
    re.IGNORECASE,
)
_MARKDOWN_RE = re.compile(r"[*_`#>]+")


@dataclass
class FaithfulnessResult:
    score: float
    sentence_scores: List[float]
    unsupported_sentences: List[str]
    citation_mismatch_sentences: List[str]
    cited_sentence_count: int = 0


def _split_sentences(text: str) -> List[str]:
    """Rough sentence splitter for Korean/English mixed text."""
    parts = _SENT_SPLIT_RE.split(text.strip())
    return [p.strip() for p in parts if len(p.strip()) >= _MIN_SENTENCE_CHARS]


def _strip_citation_footer(answer: str) -> str:
    """Remove the auto-appended citation footer before verification."""
    if not answer:
        return ""
    footer_markers = [
        "\n---\n**출처:**",
        "\n---\n**Sources:**",
        "\n---\nSources:",
    ]
    for marker in footer_markers:
        if marker in answer:
            return answer.split(marker, 1)[0].strip()
    return answer.strip()


def _normalize_sentence_for_verification(sentence: str) -> str:
    """Remove markdown and citation noise before scoring."""
    cleaned = _strip_citation_footer(sentence)
    cleaned = _SECTION_LABEL_RE.sub("", cleaned)
    cleaned = _CITATION_RE.sub("", cleaned)
    cleaned = _MARKDOWN_RE.sub("", cleaned)
    cleaned = re.sub(r"^\s*[-•]+\s*", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _extract_valid_citation_indices(sentence: str, source_count: int) -> List[int]:
    indices: List[int] = []
    for match in _CITATION_RE.findall(sentence):
        try:
            idx = int(match)
        except ValueError:
            continue
        if 1 <= idx <= source_count:
            indices.append(idx - 1)
    return sorted(set(indices))


def analyze_faithfulness(
    answer: str,
    source_texts: List[str],
    max_sentences: int = 6,
) -> Optional[FaithfulnessResult]:
    """
    Return a detailed faithfulness analysis, or None if the verifier is unavailable.

    Sentences with inline citations like [1], [2] are scored primarily against
    the cited chunks. This makes the verifier more sensitive to citation quality
    and less sensitive to the automatically appended citation footer.
    """
    if not answer or not source_texts:
        return None

    try:
        from app.retrieval.reranker import _load_reranker

        reranker = _load_reranker()
        if reranker is None:
            return None
    except Exception as exc:
        logger.debug("Reranker unavailable for NLI: %s", exc)
        return None

    stripped_answer = _strip_citation_footer(answer)
    raw_sentences = _split_sentences(stripped_answer)
    normalized_sentences = []
    citation_indices: List[List[int]] = []
    for sent in raw_sentences:
        cleaned = _normalize_sentence_for_verification(sent)
        if len(cleaned) < _MIN_SENTENCE_CHARS:
            continue
        normalized_sentences.append(cleaned)
        citation_indices.append(_extract_valid_citation_indices(sent, len(source_texts)))
        if len(normalized_sentences) >= max_sentences:
            break

    if not normalized_sentences:
        return None

    pairs = [
        [sent, src]
        for sent in normalized_sentences
        for src in source_texts
    ]

    try:
        raw_scores = reranker.compute_score(pairs, normalize=True)
        if isinstance(raw_scores, (float, int)):
            raw_scores = [raw_scores]
    except Exception as exc:
        logger.warning("Faithfulness scoring failed: %s", exc)
        return None

    n_sources = len(source_texts)
    sentence_scores: List[float] = []
    unsupported_sentences: List[str] = []
    citation_mismatch_sentences: List[str] = []
    cited_sentence_count = 0

    for i, sent in enumerate(normalized_sentences):
        chunk_scores = raw_scores[i * n_sources : (i + 1) * n_sources]
        global_best = max(chunk_scores) if chunk_scores else 0.0
        cited = citation_indices[i]
        if cited:
            cited_sentence_count += 1
            cited_best = max((chunk_scores[j] for j in cited), default=0.0)
            score = cited_best
            if cited_best + 0.15 < global_best:
                citation_mismatch_sentences.append(sent)
        else:
            score = global_best
        sentence_scores.append(score)
        if score < _FAITHFULNESS_WARN_THRESHOLD:
            unsupported_sentences.append(sent)

    score = sum(sentence_scores) / len(sentence_scores)
    logger.debug(
        "Faithfulness: %.2f (per-sentence: %s, unsupported=%d, citation_mismatch=%d)",
        score,
        [f"{s:.2f}" for s in sentence_scores],
        len(unsupported_sentences),
        len(citation_mismatch_sentences),
    )

    return FaithfulnessResult(
        score=score,
        sentence_scores=sentence_scores,
        unsupported_sentences=unsupported_sentences,
        citation_mismatch_sentences=citation_mismatch_sentences,
        cited_sentence_count=cited_sentence_count,
    )


def check_faithfulness(
    answer: str,
    source_texts: List[str],
    max_sentences: int = 6,
) -> Optional[float]:
    """
    Return a faithfulness score in [0, 1], or None if the verifier is
    unavailable (e.g. reranker not loaded).

    Parameters
    ----------
    answer       : LLM-generated answer text
    source_texts : list of raw source chunk texts (no enrichment header)
    max_sentences: cap the number of answer sentences to score (performance)
    """
    result = analyze_faithfulness(
        answer=answer,
        source_texts=source_texts,
        max_sentences=max_sentences,
    )
    return result.score if result is not None else None


def check_context_relevance(
    question: str,
    source_texts: List[str],
    top_k: int = 3,
) -> Optional[float]:
    """
    Score how relevant the top retrieved chunks are to the user's question.

    Uses the same BGE reranker cross-encoder as faithfulness checking.
    Returns the max score among the top-k source texts, or None if the
    reranker is unavailable.

    A high score (> 0.4) means at least one chunk is clearly about the
    question.  A low score (< 0.25) means the retrieval is off-topic —
    the question asks about something the documents don't cover.
    """
    if not question or not source_texts:
        return None

    try:
        from app.retrieval.reranker import _load_reranker
        reranker = _load_reranker()
        if reranker is None:
            return None
    except Exception as exc:
        logger.debug("Reranker unavailable for context relevance: %s", exc)
        return None

    # Only check the top-k chunks (already ranked by retrieval)
    texts = source_texts[:top_k]
    pairs = [[question, src] for src in texts]

    try:
        raw_scores = reranker.compute_score(pairs, normalize=True)
        if isinstance(raw_scores, (float, int)):
            raw_scores = [raw_scores]
    except Exception as exc:
        logger.warning("Context relevance scoring failed: %s", exc)
        return None

    best = max(raw_scores) if raw_scores else 0.0
    logger.debug(
        "Context relevance: best=%.2f (scores: %s)",
        best,
        [f"{s:.2f}" for s in raw_scores],
    )
    return best
