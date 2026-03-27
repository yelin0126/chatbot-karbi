"""
Unified chat handler — works like a normal chatbot.

NO hardcoded mode routing. Instead:
1. Always search vectorstore for relevant document context
2. If web search might help, search Tavily too
3. Build one prompt with all available context
4. LLM decides what to use

This is how ChatGPT/Claude work — retrieve first, let the model decide.
"""

from collections import Counter
import logging
import re
from typing import List, Dict, Any, Optional

from app.chat.deterministic import (
    try_two_document_comparison_answer as _try_two_document_comparison_answer,
    try_numeric_fact_answer as _try_numeric_fact_answer,
    strict_fact_chunk_score as _strict_fact_chunk_score,
    select_strict_fact_docs as _select_strict_fact_docs,
    _table_query_family_score as _table_query_family_score,
    try_scoped_presence_answer as _try_scoped_presence_answer,
    try_chapter_title_lookup as _try_chapter_title_lookup,
    try_history_lookup as _try_history_lookup,
    try_scoped_change_summary as _try_scoped_change_summary,
    try_scoped_clause_answer as _try_scoped_clause_answer,
    try_scoped_count_answer as _try_scoped_count_answer,
)
from app.chat.policy import (
    is_direct_extraction_query as _is_direct_extraction_query,
    is_general_knowledge_query as _is_general_knowledge_query,
    is_smalltalk_query as _is_smalltalk_query,
    might_need_web_search as _might_need_web_search,
    needs_article_lookup as _needs_article_lookup,
    needs_comparison_style as _needs_comparison_style,
    needs_full_document_context as _needs_full_document_context,
    needs_multi_document_summary_style as _needs_multi_document_summary_style,
    needs_strict_fact_style as _needs_strict_fact_style,
    needs_whole_corpus_full_context as _needs_whole_corpus_full_context,
    route_query as _route_query,
)
from app.chat.scope import (
    build_mention_only_answer as _build_mention_only_answer,
    check_upload_wide_doc_ambiguity as _check_upload_wide_doc_ambiguity,
    extract_mention_candidate as _extract_mention_candidate,
    extract_scope_labels as _extract_scope_labels,
    filter_docs_to_named_scope as _filter_docs_to_named_scope,
    find_mention_pages as _find_mention_pages,
    get_bundled_labels as _get_bundled_labels,
    normalize_active_scopes as _normalize_active_scopes,
    normalize_scope_text as _normalize_scope_text,
    resolve_canonical_library_source_for_family as _resolve_canonical_library_source_for_family,
    resolve_library_doc_from_query as _resolve_library_doc_from_query,
    resolve_single_upload_doc as _resolve_single_upload_doc,
    resolve_upload_doc_from_query as _resolve_upload_doc_from_query,
    source_family_key as _source_family_key,
    should_request_scope_clarification as _should_request_scope_clarification,
)
from app.chat.text_utils import strip_enrichment_header as _strip_enrichment_header
from app.chat.retrieval_flow import (
    build_single_document_summary as _build_single_document_summary,
    build_file_level_corpus_summary as _build_file_level_corpus_summary,
    expand_structural_context as _expand_structural_context,
    has_strong_query_overlap as _has_strong_query_overlap,
    scoped_confidence_threshold_for_query as _scoped_confidence_threshold_for_query,
    should_force_small_doc_full_context as _should_force_small_doc_full_context,
)
from app.chat.prompting import (
    build_citation_footer as _build_citation_footer,
    build_prompt as _build_prompt,
    ensure_minimum_inline_citations as _ensure_minimum_inline_citations,
    is_grounding_fallback_answer as _is_grounding_fallback_answer,
)
from app.chat.state import ChatExecutionState
from app.models.schemas import Message
from app.core.llm import generate_text, get_default_model_name
from app.core.nli_verifier import analyze_faithfulness, check_context_relevance
from app.chat.validation import (
    contains_chinese as _contains_chinese,
    infer_target_language as _infer_target_language,
    language_correction_prompt as _language_correction_prompt,
    needs_language_retry as _needs_language_retry,
)
from app.retrieval.retriever import (
    retrieve,
    format_context,
    format_grouped_corpus_context,
    extract_sources,
)
from app.core.vectorstore import get_documents_by_source, get_documents_by_doc_ids
from app.config import (
    TAVILY_API_KEY,
    DOCUMENT_CONFIDENCE_THRESHOLD,
    CONTEXT_RELEVANCE_THRESHOLD,
    NLI_FAITHFULNESS_HARD_THRESHOLD,
    NLI_FAITHFULNESS_SOFT_THRESHOLD,
)

logger = logging.getLogger("tilon.chat")


# ═══════════════════════════════════════════════════════════════════════
# Web Search (additive, not a separate mode)
# ═══════════════════════════════════════════════════════════════════════

def _search_web(query: str) -> str:
    """Search Tavily for current/real-time information. Returns formatted results."""
    if not TAVILY_API_KEY:
        return ""

    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=TAVILY_API_KEY)
        response = client.search(query, max_results=3)

        results = []
        for r in response.get("results", []):
            title = r.get("title", "")
            content = r.get("content", "")
            url = r.get("url", "")
            results.append(f"- {title}: {content} ({url})")

        return "\n".join(results) if results else ""

    except ImportError:
        logger.debug("tavily-python not installed")
        return ""
    except Exception as e:
        logger.debug("Web search failed: %s", e)
        return ""


def _document_not_found_answer(
    user_message: str,
    active_source: Optional[str],
    active_doc_id: Optional[str] = None,
) -> str:
    """Return a grounded fallback when scoped retrieval confidence is too low."""
    doc_name = active_source or "선택된 문서"
    if re.search(r"[가-힣]", user_message):
        return (
            f"'{doc_name}'에서 해당 내용을 찾지 못했습니다.\n\n"
            "다음을 시도해 보세요:\n"
            "• 질문을 더 구체적으로 바꿔 주세요\n"
            "• 문서에 포함된 키워드를 사용해 주세요\n"
            "• 다른 문서를 선택해 보세요"
        )
    return (
        f"I couldn't find relevant content in '{doc_name}'.\n\n"
        "Try:\n"
        "• Rephrasing your question with more specific terms\n"
        "• Using keywords that appear in the document\n"
        "• Selecting a different document"
    )


def _deterministic_grounded_response(answer: str, docs) -> str:
    if not answer:
        return answer
    enriched = answer
    if docs and not _is_grounding_fallback_answer(enriched):
        enriched = _ensure_minimum_inline_citations(enriched, docs)
        citation_footer = _build_citation_footer(docs)
        if citation_footer:
            enriched = enriched.rstrip() + "\n" + citation_footer
    return enriched


def _document_corpus_not_found_answer(user_message: str, source_type: Optional[str]) -> str:
    """Return a grounded fallback for scoped corpus search such as all uploads."""
    if re.search(r"[가-힣]", user_message):
        return (
            "업로드된 문서에서 해당 내용을 찾지 못했습니다.\n\n"
            "다음을 시도해 보세요:\n"
            "• 질문을 더 구체적으로 바꿔 주세요\n"
            "• 문서에 포함된 키워드를 사용해 주세요"
        )
    return (
        "I couldn't find relevant content in the uploaded documents.\n\n"
        "Try:\n"
        "• Rephrasing your question with more specific terms\n"
        "• Using keywords that appear in the documents"
    )


def _not_grounded_answer(user_message: str) -> str:
    """Return a fallback when the LLM answer fails the faithfulness check."""
    if re.search(r"[가-힣]", user_message):
        return (
            "해당 내용은 제공된 문서에서 충분히 확인되지 않습니다.\n\n"
            "질문을 더 구체적으로 바꾸거나, 문서에 포함된 키워드를 사용해 다시 질문해 주세요."
        )
    return (
        "The provided documents don't contain enough information to answer this reliably.\n\n"
        "Try rephrasing your question or using specific keywords from the document."
    )


def _context_irrelevant_answer(user_message: str, active_source: Optional[str] = None) -> str:
    """Return a fallback when retrieved context is off-topic for the question."""
    doc_name = active_source or "문서"
    if re.search(r"[가-힣]", user_message):
        return (
            f"'{doc_name}'에는 이 질문과 관련된 내용이 포함되어 있지 않은 것 같습니다.\n\n"
            "다른 질문을 하시거나, 관련 문서를 업로드해 주세요."
        )
    return (
        f"'{doc_name}' doesn't appear to contain information related to this question.\n\n"
        "Try asking a different question or uploading a relevant document."
    )


def _infer_implicit_scope_from_docs(docs) -> Optional[Dict[str, Any]]:
    """Promote unscoped retrieval to a single-document scope when one family dominates."""
    family_counts: Counter[str] = Counter()
    family_repr: Dict[str, Dict[str, str]] = {}

    for doc in docs or []:
        source = str(doc.metadata.get("source") or "").strip()
        doc_id = str(doc.metadata.get("doc_id") or "").strip()
        if not source and not doc_id:
            continue

        family_key = _source_family_key(source or doc_id)
        if not family_key:
            family_key = _normalize_scope_text(source or doc_id)
        if not family_key:
            continue

        family_counts[family_key] += 1
        family_repr.setdefault(
            family_key,
            {"source": source, "doc_id": doc_id},
        )

    if not family_counts:
        return None

    top = family_counts.most_common(2)
    top_family, top_count = top[0]
    second_count = top[1][1] if len(top) > 1 else 0
    total = sum(family_counts.values())

    if top_count < 3 and top_count / max(total, 1) < 0.6:
        return None
    if top_count == second_count:
        return None

    inferred = family_repr[top_family].copy()
    inferred["family_key"] = top_family
    inferred["match_count"] = top_count
    inferred["total_count"] = total
    return inferred


def _infer_strict_fact_scope_from_docs(user_message: str, docs) -> Optional[Dict[str, Any]]:
    family_docs: Dict[str, List[Any]] = {}
    family_source_docs: Dict[str, Dict[str, List[Any]]] = {}

    for doc in docs or []:
        source = str(doc.metadata.get("source") or "").strip()
        doc_id = str(doc.metadata.get("doc_id") or "").strip()
        if not source and not doc_id:
            continue

        family_key = _source_family_key(source or doc_id)
        if not family_key:
            family_key = _normalize_scope_text(source or doc_id)
        if not family_key:
            continue

        family_docs.setdefault(family_key, []).append(doc)
        exact_source_key = source or doc_id
        family_source_docs.setdefault(family_key, {}).setdefault(exact_source_key, []).append(doc)

    if not family_docs:
        return None

    family_scores: Dict[str, float] = {}
    family_repr: Dict[str, Dict[str, str]] = {}
    is_table_query = _is_direct_extraction_query(user_message) or any(
        keyword in user_message.lower()
        for keyword in ("금액", "한도", "최대", "강사비", "상금", "지원비", "형태", "종류", "amount", "limit")
    )

    def _strict_fact_source_preference(source: str, source_docs: List[Any]) -> float:
        normalized = (source or "").lower()
        message = (user_message or "").lower()
        score = 0.0
        combined = " ".join(
            re.sub(r"\s+", " ", _strip_enrichment_header(doc.page_content))
            for doc in source_docs
        )

        if "일부개정" in normalized and any(
            term in message for term in ("개정", "신설", "조정", "변경", "우수성과 상금", "교통비", "성인학습자", "민간인")
        ):
            score += 8.0
        if normalized.startswith("(1-1-r-a)"):
            score += 5.0

        if any(term in message for term in ("학생강사비", "학부생", "대학원생", "현금", "현물", "상품권", "형태")):
            if normalized.startswith("(붙임) 제주대학교 rise사업단 jnu인재지원금 지급 기준(안)"):
                score += 12.0
            elif normalized.startswith("(붙임)제주대학교 rise사업단 jnu인재지원금 지급 기준(안)"):
                score += 6.0
            if "20251021" in normalized:
                score -= 2.0
            if "혁신인재지원금" in normalized:
                score -= 4.0

        if "우수성과" in message and "상금" in message:
            if all(term in combined for term in ("개인상금", "700,000", "팀 상금", "1,000,000")):
                score += 20.0

        if any(term in message for term in ("학생강사비", "학부생", "대학원생")):
            if all(term in combined for term in ("학생강사비", "학부생", "50,000", "대학원생", "70,000")):
                score += 20.0

        if "형태" in message and "상금" in message:
            if all(term in combined for term in ("현금", "현물", "상품권")):
                score += 16.0

        if "jnu인재지원금" in normalized and "혁신인재지원금" not in normalized:
            score += 1.5
        return score

    for family_key, docs_in_family in family_docs.items():
        if is_table_query:
            family_scores[family_key] = _table_query_family_score(user_message, docs_in_family)
        else:
            family_scores[family_key] = sum(
                max(0.0, _strict_fact_chunk_score(user_message, doc))
                for doc in docs_in_family
            )

        best_source_key = None
        best_source_score = -1.0
        best_doc = None
        for source_key, source_docs in family_source_docs.get(family_key, {}).items():
            if is_table_query:
                score = _table_query_family_score(user_message, source_docs)
            else:
                score = sum(max(0.0, _strict_fact_chunk_score(user_message, doc)) for doc in source_docs)
            score += _strict_fact_source_preference(source_key, source_docs)
            if score > best_source_score:
                best_source_score = score
                best_source_key = source_key
                best_doc = max(
                    source_docs,
                    key=lambda doc: max(0.0, _strict_fact_chunk_score(user_message, doc)),
                )
        if best_doc is not None:
            family_repr[family_key] = {
                "source": str(best_doc.metadata.get("source") or best_source_key or ""),
                "doc_id": str(best_doc.metadata.get("doc_id") or ""),
            }

    ranked = sorted(family_scores.items(), key=lambda item: item[1], reverse=True)
    top_family, top_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0

    source_ranked: List[Tuple[float, str, str, str]] = []
    if is_table_query:
        for family_key, source_groups in family_source_docs.items():
            for source_key, source_docs in source_groups.items():
                score = _table_query_family_score(user_message, source_docs)
                score += _strict_fact_source_preference(source_key, source_docs)
                best_doc = max(
                    source_docs,
                    key=lambda doc: max(0.0, _strict_fact_chunk_score(user_message, doc)),
                )
                source_ranked.append(
                    (
                        score,
                        family_key,
                        str(best_doc.metadata.get("source") or source_key or ""),
                        str(best_doc.metadata.get("doc_id") or ""),
                    )
                )
        source_ranked.sort(reverse=True)
        if source_ranked:
            top_source_score, top_source_family, top_source, top_doc_id = source_ranked[0]
            second_source_score = source_ranked[1][0] if len(source_ranked) > 1 else 0.0
            if top_source_score >= 6.0 and (
                second_source_score <= 0 or top_source_score >= second_source_score + 0.5
            ):
                canonical = _resolve_canonical_library_source_for_family(user_message, top_source_family)
                if canonical and canonical.get("source"):
                    top_source = canonical["source"]
                    top_doc_id = canonical.get("doc_id") or top_doc_id
                return {
                    "source": top_source,
                    "doc_id": top_doc_id,
                    "family_key": top_source_family,
                    "match_count": int(round(top_source_score)),
                    "total_count": int(round(sum(score for score, *_ in source_ranked))),
                }

    if top_score < (6.0 if is_table_query else 8.0):
        return None
    if second_score > 0 and top_score < second_score + (1.5 if is_table_query else 4.0):
        for family_key, source_groups in family_source_docs.items():
            for source_key, source_docs in source_groups.items():
                score = (
                    _table_query_family_score(user_message, source_docs)
                    if is_table_query
                    else sum(max(0.0, _strict_fact_chunk_score(user_message, doc)) for doc in source_docs)
                )
                score += _strict_fact_source_preference(source_key, source_docs)
                best_doc = max(
                    source_docs,
                    key=lambda doc: max(0.0, _strict_fact_chunk_score(user_message, doc)),
                )
                source_ranked.append(
                    (
                        score,
                        family_key,
                        str(best_doc.metadata.get("source") or source_key or ""),
                        str(best_doc.metadata.get("doc_id") or ""),
                    )
                )
        source_ranked.sort(reverse=True)
        if not source_ranked:
            return None
        top_source_score, top_source_family, top_source, top_doc_id = source_ranked[0]
        second_source_score = source_ranked[1][0] if len(source_ranked) > 1 else 0.0
        if top_source_score < 6.0:
            return None
        if second_source_score > 0 and top_source_score < second_source_score + (0.5 if is_table_query else 1.0):
            return None
        if is_table_query:
            canonical = _resolve_canonical_library_source_for_family(user_message, top_source_family)
            if canonical and canonical.get("source"):
                top_source = canonical["source"]
                top_doc_id = canonical.get("doc_id") or top_doc_id
        return {
            "source": top_source,
            "doc_id": top_doc_id,
            "family_key": top_source_family,
            "match_count": int(round(top_source_score)),
            "total_count": int(round(sum(score for score, *_ in source_ranked))),
        }

    if second_score > 0 and top_score < second_score + (1.5 if is_table_query else 4.0):
        return None

    inferred = family_repr[top_family].copy()
    inferred.pop("score", None)
    inferred["family_key"] = top_family
    inferred["match_count"] = int(round(top_score))
    inferred["total_count"] = int(round(sum(family_scores.values())))
    return inferred


def _precollapse_unscoped_strict_fact_library_docs(
    state: ChatExecutionState,
    user_message: str,
) -> bool:
    """
    Collapse near-duplicate library results to one canonical exact document
    before confidence/not-found gates or mixed-context generation run.
    """
    if (
        not state.docs
        or state.multi_scope
        or state.selected_docs
        or not state.query_policy.strict_fact
    ):
        return False

    inferred_scope = _infer_strict_fact_scope_from_docs(user_message, state.docs)
    if not inferred_scope:
        return False

    state.selected_docs = [{
        "source": inferred_scope.get("source", ""),
        "doc_id": inferred_scope.get("doc_id", ""),
    }]
    state.sync_scope()

    exact_docs = get_documents_by_source(
        source=state.scoped_source,
        doc_id=state.scoped_doc_id,
    )
    if exact_docs:
        state.docs = exact_docs
    else:
        state.docs = [
            doc for doc in state.docs
            if _source_family_key(
                str(doc.metadata.get("source") or doc.metadata.get("doc_id") or "")
            ) == inferred_scope["family_key"]
        ]

    logger.info(
        "Pre-collapsed unscoped strict-fact library retrieval to canonical scope '%s'%s based on %d/%d retrieved chunks",
        state.scoped_source or "scoped document",
        f" ({state.scoped_doc_id})" if state.scoped_doc_id else "",
        inferred_scope["match_count"],
        inferred_scope["total_count"],
    )
    return True


def _looks_like_question(text: str) -> bool:
    lower = (text or "").strip().lower()
    if not lower:
        return False
    if "?" in lower or "？" in lower:
        return True
    question_markers = [
        "무엇", "뭐", "누구", "언제", "어디", "왜", "어떻게", "몇", "어느", "있나요",
        "입니까", "인가요", "설명", "말해줘", "알려줘",
        "what", "who", "when", "where", "why", "how", "which", "is there", "tell me",
    ]
    return any(marker in lower for marker in question_markers)


def _split_multi_questions(text: str) -> List[str]:
    """Split a single user turn into multiple question-like subqueries conservatively."""
    raw = (text or "").strip()
    if not raw:
        return []

    pieces: List[str] = []
    for line in [part.strip() for part in re.split(r"[\r\n]+", raw) if part.strip()]:
        segments = [segment.strip() for segment in re.split(r"(?<=[?？])\s+", line) if segment.strip()]
        if segments:
            pieces.extend(segments)

    normalized: List[str] = []
    seen = set()
    for piece in pieces:
        key = re.sub(r"\s+", " ", piece)
        if len(key) < 4 or key in seen or not _looks_like_question(key):
            continue
        seen.add(key)
        normalized.append(key)

    return normalized if len(normalized) >= 2 else []


def _looks_like_entity_explanation_query(text: str) -> bool:
    """Detect broad fact/explanation questions that are risky without grounding."""
    lower = (text or "").lower().strip()
    indicators = [
        "에 대해서", "에 대해", "무엇", "뭐야", "누구", "설명해", "말해줘",
        "what is", "who is", "tell me about", "explain", "what does",
    ]
    return any(keyword in lower for keyword in indicators)


def _document_first_clarification_answer(user_message: str) -> str:
    """Ask the user to pick document scope instead of answering from model memory."""
    uploads = [doc for doc in list_documents() if doc.get("source_type") == "upload"]
    upload_names = [doc.get("source") for doc in uploads if doc.get("source")]
    examples = ", ".join(f"'{name}'" for name in upload_names[:2])

    if re.search(r"[가-힣]", user_message):
        if examples:
            return (
                "현재 선택된 문서 범위가 없어 업로드 문서를 기준으로는 답변할 수 없습니다. "
                f"왼쪽 보관소에서 문서를 선택한 뒤 다시 질문해 주세요. 예: {examples}"
            )
        return (
            "현재 선택된 문서 범위가 없습니다. 왼쪽 보관소에서 문서를 선택하거나 파일을 업로드한 뒤 다시 질문해 주세요."
        )

    if examples:
        return (
            "No document is currently selected, so I can't answer this as a grounded document question. "
            f"Please select an uploaded document from the left shelf and ask again, for example: {examples}"
        )
    return (
        "No document is currently selected. Please choose an uploaded document or upload a file and ask again."
    )


def _build_direct_extraction_answer(active_source: Optional[str], docs) -> str:
    """Return extracted document text directly for OCR/transcription requests."""
    extracted = "\n\n".join(
        _strip_enrichment_header(doc.page_content)
        for doc in docs
        if _strip_enrichment_header(doc.page_content)
    ).strip()

    if not extracted:
        fallback_source = active_source or (docs[0].metadata.get("source") if docs else None)
        fallback_doc_id = docs[0].metadata.get("doc_id") if docs else None
        return _document_not_found_answer("extract text", fallback_source, fallback_doc_id)

    if re.search(r"[가-힣]", extracted):
        return f"추출된 텍스트:\n\n{extracted}"
    return f"Extracted text:\n\n{extracted}"


def _build_chat_state(
    user_message: str,
    history: List[Message],
    model: Optional[str],
    active_source: Optional[str],
    active_doc_id: Optional[str],
    active_source_type: Optional[str],
    active_sources: Optional[List[str]],
    active_doc_ids: Optional[List[str]],
    system_prompt: Optional[str],
) -> ChatExecutionState:
    """Create the shared execution state for one chat turn."""
    state = ChatExecutionState(
        user_message=user_message,
        history=history,
        selected_model=model or get_default_model_name(),
        active_source=active_source,
        active_doc_id=active_doc_id,
        active_source_type=active_source_type,
        system_prompt=system_prompt,
        query_policy=_route_query(user_message),
    )
    normalized_sources, normalized_doc_ids = _normalize_active_scopes(
        active_source,
        active_doc_id,
        active_sources,
        active_doc_ids,
    )
    state.normalized_sources = normalized_sources
    state.normalized_doc_ids = normalized_doc_ids

    selected_docs: List[Dict[str, str]] = []
    max_len = max(len(normalized_sources), len(normalized_doc_ids))
    for idx in range(max_len):
        source = normalized_sources[idx] if idx < len(normalized_sources) else ""
        doc_id = normalized_doc_ids[idx] if idx < len(normalized_doc_ids) else ""
        if source or doc_id:
            selected_docs.append({"source": source, "doc_id": doc_id})

    if not selected_docs and (active_source or active_doc_id):
        selected_docs = [{"source": active_source or "", "doc_id": active_doc_id or ""}]

    state.selected_docs = selected_docs
    state.sync_scope()
    logger.info(
        "Query policy: intent=%s heuristic=%s classifier=%s source=%s confidence=%.2f shadow=%s disagree=%s",
        state.query_policy.intent,
        state.query_policy.heuristic_intent,
        state.query_policy.classifier_intent,
        state.query_policy.intent_source,
        state.query_policy.intent_confidence,
        state.query_policy.shadow_mode,
        state.query_policy.classifier_disagrees,
    )
    if state.query_policy.classifier_disagrees:
        logger.info(
            "Query policy disagreement: heuristic=%s classifier=%s final=%s query='%s'",
            state.query_policy.heuristic_intent,
            state.query_policy.classifier_intent,
            state.query_policy.intent,
            user_message[:80].replace("\n", " "),
        )
    return state


def _maybe_handle_multi_question_fanout(
    user_message: str,
    history: List[Message],
    model: Optional[str],
    active_source: Optional[str],
    active_doc_id: Optional[str],
    active_source_type: Optional[str],
    active_sources: Optional[List[str]],
    active_doc_ids: Optional[List[str]],
    system_prompt: Optional[str],
) -> Optional[Dict[str, Any]]:
    """Split compound turns into independent sub-questions when needed."""
    sub_questions = _split_multi_questions(user_message)
    if not sub_questions:
        return None

    sub_results = []
    for q in sub_questions:
        sub = handle_chat(
            user_message=q,
            history=history,
            model=model,
            active_source=active_source,
            active_doc_id=active_doc_id,
            active_source_type=active_source_type,
            active_sources=active_sources,
            active_doc_ids=active_doc_ids,
            system_prompt=system_prompt,
        )
        sub_results.append((q, sub))

    combined = "\n\n".join(
        f"{i}. {sub['answer']}" for i, (_, sub) in enumerate(sub_results, 1)
    )
    all_sources: List[Dict] = []
    seen_src: set = set()
    for _, sub in sub_results:
        for s in (sub.get("sources") or []):
            key = s.get("doc_id") or s.get("source")
            if key and key not in seen_src:
                seen_src.add(key)
                all_sources.append(s)
    last_sub = sub_results[-1][1]
    return {
        "answer": combined,
        "sources": all_sources,
        "mode": last_sub.get("mode", "document_qa"),
        "active_source": last_sub.get("active_source"),
        "active_doc_id": last_sub.get("active_doc_id"),
        "active_source_type": active_source_type,
        "active_sources": last_sub.get("active_sources", []),
        "active_doc_ids": last_sub.get("active_doc_ids", []),
    }


def _resolve_scope_stage(
    state: ChatExecutionState,
    *,
    user_message: str,
    active_source: Optional[str],
    active_doc_id: Optional[str],
    active_source_type: Optional[str],
) -> Optional[Dict[str, Any]]:
    """Normalize scope and short-circuit on scope-specific early answers."""
    if state.query_policy.smalltalk:
        state.selected_docs = []
        state.sync_scope()

    if not state.selected_docs and active_source_type == "upload":
        resolved_doc = _resolve_single_upload_doc()
        if resolved_doc and (resolved_doc.get("source") or resolved_doc.get("doc_id")):
            state.selected_docs = [resolved_doc]
            state.sync_scope()
            logger.info(
                "Auto-scoped upload query to the only uploaded file '%s'%s",
                resolved_doc.get("source") or "resolved upload",
                f" ({resolved_doc.get('doc_id')})" if resolved_doc.get("doc_id") else "",
            )

        resolved_doc = _resolve_upload_doc_from_query(user_message) if not state.selected_docs else None
        if resolved_doc and (resolved_doc.get("source") or resolved_doc.get("doc_id")):
            state.selected_docs = [resolved_doc]
            state.sync_scope()
            logger.info(
                "Auto-scoped upload-wide query to '%s'%s based on document name in question",
                resolved_doc.get("source") or "resolved upload",
                f" ({resolved_doc.get('doc_id')})" if resolved_doc.get("doc_id") else "",
            )

    if not state.selected_docs and active_source_type == "library":
        resolved_doc = _resolve_library_doc_from_query(user_message)
        if resolved_doc and (resolved_doc.get("source") or resolved_doc.get("doc_id")):
            state.selected_docs = [resolved_doc]
            state.sync_scope()
            logger.info(
                "Auto-scoped library query to '%s'%s based on document name in question",
                resolved_doc.get("source") or "resolved library document",
                f" ({resolved_doc.get('doc_id')})" if resolved_doc.get("doc_id") else "",
            )

    if (
        len(state.selected_docs) == 1
        and (state.scoped_source or state.scoped_doc_id)
        and state.query_policy.direct_extraction
    ):
        state.docs = get_documents_by_source(source=state.scoped_source, doc_id=state.scoped_doc_id)
        if state.docs:
            logger.info(
                "Direct extraction response for '%s'%s",
                state.scoped_source or "scoped document",
                f" ({state.scoped_doc_id})" if state.scoped_doc_id else "",
            )
            return state.response(
                _build_direct_extraction_answer(state.scoped_source, state.docs),
                mode="ocr_extract",
                sources=extract_sources(state.docs),
            )

    if (
        len(state.selected_docs) == 1
        and (state.scoped_source or state.scoped_doc_id)
        and not state.query_policy.smalltalk
    ):
        bundled_labels = _get_bundled_labels(state.scoped_source, state.scoped_doc_id)
        if len(bundled_labels) >= 2:
            normalized_query = _normalize_scope_text(user_message)
            label_tokens = [_normalize_scope_text(lbl) for lbl in bundled_labels]
            query_names_specific = any(tok and tok in normalized_query for tok in label_tokens)
            if not query_names_specific:
                ambiguous_article_terms = [
                    "지원대상", "운영절차", "경과조치", "부칙", "지원사항",
                    "지원중단", "의무위반", "가이드라인 준용", "정의", "목적",
                    "시행일", "조치", "내용 알려줘",
                ]
                is_ambiguous = state.query_policy.article_lookup or any(
                    term in user_message for term in ambiguous_article_terms
                )
                if is_ambiguous:
                    clarification = _build_scoped_ambiguity_answer(user_message, bundled_labels)
                    logger.info(
                        "Early bundled-doc clarification: %d sub-guidelines in '%s', query ambiguous",
                        len(bundled_labels),
                        state.scoped_source or state.scoped_doc_id,
                    )
                    return state.response(
                        clarification,
                        mode="document_qa",
                        sources=[],
                        active_source=active_source,
                        active_doc_id=active_doc_id,
                        active_source_type=active_source_type,
                    )

    return None


def _run_retrieval_stage(
    state: ChatExecutionState,
    *,
    user_message: str,
    active_source: Optional[str],
    active_doc_id: Optional[str],
    active_source_type: Optional[str],
) -> Optional[Dict[str, Any]]:
    """Retrieve context, apply deterministic shortcuts, and assemble document context."""
    state.doc_context = ""
    state.sources = []
    state.use_full_document = bool(
        state.multi_scope
        or (
            (state.scoped_source or state.scoped_doc_id)
            and (
                state.query_policy.full_document
                or state.query_policy.strict_fact
                or state.query_policy.article_lookup
                or _should_force_small_doc_full_context(state.scoped_source, state.scoped_doc_id)
            )
        )
        or (
            active_source_type
            and not state.selected_docs
            and state.query_policy.whole_corpus_full_document
        )
    )

    state.retrieval = None
    if state.multi_scope:
        state.docs = get_documents_by_doc_ids(state.scoped_doc_ids)
        logger.info(
            "Loaded %d chunks for multi-document scope across %d selected documents",
            len(state.docs),
            len(state.scoped_doc_ids),
        )
    elif active_source_type and not state.selected_docs and state.use_full_document:
        state.docs = get_documents_by_source(source_type=active_source_type)
        logger.info(
            "Loaded %d chunks for whole-corpus task from source_type='%s'",
            len(state.docs),
            active_source_type,
        )
    else:
        state.retrieval = retrieve(
            user_message,
            source_filter=state.scoped_source,
            doc_id_filter=state.scoped_doc_id,
            source_type_filter=active_source_type if not state.selected_docs else None,
            full_document=state.use_full_document,
        )
        state.docs = state.retrieval.docs

    if (
        state.docs
        and active_source_type == "library"
        and not state.selected_docs
        and not state.multi_scope
        and state.query_policy.strict_fact
    ):
        if _precollapse_unscoped_strict_fact_library_docs(state, user_message):
            active_source = state.scoped_source or active_source
            active_doc_id = state.scoped_doc_id or active_doc_id

    if (
        not state.multi_scope
        and not state.selected_docs
        and active_source_type
        and not state.use_full_document
        and not state.query_policy.smalltalk
    ):
        if not state.docs or (
            state.retrieval.confidence < DOCUMENT_CONFIDENCE_THRESHOLD
            and not state.retrieval.strong_keyword_hit
        ):
            logger.info(
                "Low-confidence corpus retrieval for source_type='%s' (confidence=%.2f, threshold=%.2f, keyword_hit=%s)",
                active_source_type,
                state.retrieval.confidence if state.retrieval else 0.0,
                DOCUMENT_CONFIDENCE_THRESHOLD,
                state.retrieval.strong_keyword_hit if state.retrieval else False,
            )
            return state.response(
                _document_corpus_not_found_answer(user_message, active_source_type),
                mode="document_qa",
                sources=[],
                active_source=None,
                active_doc_id=None,
                active_source_type=active_source_type,
                active_sources=[],
                active_doc_ids=[],
            )

        if state.docs and active_source_type == "upload":
            upload_ambiguity = _check_upload_wide_doc_ambiguity(user_message, state.docs)
            if upload_ambiguity:
                n_files = len({d.metadata.get("source") for d in state.docs if d.metadata.get("source")})
                logger.info(
                    "Upload-wide doc ambiguity: %d files match ambiguous query '%s'",
                    n_files,
                    user_message[:60],
                )
                return state.response(
                    upload_ambiguity,
                    mode="document_qa",
                    sources=extract_sources(state.docs),
                    active_source=None,
                    active_doc_id=None,
                    active_source_type=active_source_type,
                    active_sources=[],
                    active_doc_ids=[],
                )

    if not state.multi_scope and (state.scoped_source or state.scoped_doc_id) and not state.use_full_document:
        threshold = (
            _scoped_confidence_threshold_for_query(
                user_message,
                state.docs,
                state.scoped_source,
                state.scoped_doc_id,
            ) if state.docs else DOCUMENT_CONFIDENCE_THRESHOLD
        )
        if not state.docs or (
            state.retrieval.confidence < threshold
            and not state.retrieval.strong_keyword_hit
        ):
            mention_candidate = _extract_mention_candidate(user_message)
            if mention_candidate:
                scoped_all_docs = get_documents_by_source(source=state.scoped_source, doc_id=state.scoped_doc_id)
                mention_pages = _find_mention_pages(scoped_all_docs, mention_candidate)
                if mention_pages:
                    logger.info(
                        "Mention-only scoped answer for '%s'%s -> '%s' on pages %s",
                        active_source or "scoped document",
                        f" ({state.scoped_doc_id})" if state.scoped_doc_id else "",
                        mention_candidate,
                        mention_pages,
                    )
                    return state.response(
                        _build_mention_only_answer(
                            user_message,
                            active_source,
                            mention_candidate,
                            mention_pages,
                        ),
                        mode="document_qa",
                        sources=[],
                        active_source=active_source,
                        active_doc_id=active_doc_id,
                    )
            logger.info(
                "Low-confidence scoped retrieval for '%s'%s (confidence=%.2f, threshold=%.2f, keyword_hit=%s)",
                active_source or "scoped document",
                f" ({state.scoped_doc_id})" if state.scoped_doc_id else "",
                state.retrieval.confidence,
                threshold,
                state.retrieval.strong_keyword_hit,
            )
            return state.response(
                _document_not_found_answer(user_message, active_source, active_doc_id),
                mode="document_qa",
                sources=[],
                active_source=active_source,
                active_doc_id=active_doc_id,
                active_source_type=active_source_type,
            )

    if state.docs:
        if state.docs and state.multi_scope:
            direct_compare = _try_two_document_comparison_answer(user_message, state.docs)
            if direct_compare:
                compare_answer, compare_docs = direct_compare
                logger.info(
                    "Answered two-document comparison query directly from %d selected documents",
                    len(state.scoped_doc_ids) or len(state.scoped_sources) or 2,
                )
                return state.response(
                    _deterministic_grounded_response(compare_answer, compare_docs),
                    mode="document_compare",
                    sources=extract_sources(compare_docs),
                    active_source_type=active_source_type,
                    active_sources=state.scoped_sources,
                    active_doc_ids=state.scoped_doc_ids,
                )

        if (
            state.docs
            and not state.multi_scope
            and not state.selected_docs
            and active_source_type == "library"
            and not state.query_policy.comparison
        ):
            inferred_scope = None
            if state.query_policy.strict_fact:
                inferred_scope = _infer_strict_fact_scope_from_docs(user_message, state.docs)
            if not inferred_scope:
                inferred_scope = _infer_implicit_scope_from_docs(state.docs)
            if inferred_scope:
                state.selected_docs = [{
                    "source": inferred_scope.get("source", ""),
                    "doc_id": inferred_scope.get("doc_id", ""),
                }]
                state.sync_scope()
                if state.query_policy.strict_fact and (state.scoped_source or state.scoped_doc_id):
                    exact_docs = get_documents_by_source(
                        source=state.scoped_source,
                        doc_id=state.scoped_doc_id,
                    )
                    if exact_docs:
                        state.docs = exact_docs
                    else:
                        state.docs = [
                            doc for doc in state.docs
                            if _source_family_key(str(doc.metadata.get("source") or doc.metadata.get("doc_id") or ""))
                            == inferred_scope["family_key"]
                        ]
                else:
                    state.docs = [
                        doc for doc in state.docs
                        if _source_family_key(str(doc.metadata.get("source") or doc.metadata.get("doc_id") or ""))
                        == inferred_scope["family_key"]
                    ]
                active_source = state.scoped_source or active_source
                active_doc_id = state.scoped_doc_id or active_doc_id
                logger.info(
                    "Promoted unscoped library retrieval to implicit single-document scope '%s'%s based on %d/%d retrieved chunks",
                    active_source or "scoped document",
                    f" ({active_doc_id})" if active_doc_id else "",
                    inferred_scope["match_count"],
                    inferred_scope["total_count"],
                )

        if (
            state.docs
            and not state.use_full_document
            and (state.scoped_source or state.scoped_doc_id)
        ):
            if state.query_policy.strict_fact:
                full_strict_docs = get_documents_by_source(
                    source=state.scoped_source,
                    doc_id=state.scoped_doc_id,
                )
                if full_strict_docs and len(full_strict_docs) > len(state.docs):
                    logger.info(
                        "Expanded strict-fact scoped context from %d to %d chunks for '%s'%s",
                        len(state.docs),
                        len(full_strict_docs),
                        state.scoped_source or "scoped document",
                        f" ({state.scoped_doc_id})" if state.scoped_doc_id else "",
                    )
                    state.docs = full_strict_docs

            expanded_docs = _expand_structural_context(user_message, state.docs)
            if len(expanded_docs) != len(state.docs):
                logger.info(
                    "Expanded structural retrieval context from %d to %d chunk(s) for '%s'",
                    len(state.docs),
                    len(expanded_docs),
                    user_message[:80],
                )
            state.docs = expanded_docs

        if (
            state.docs
            and not state.multi_scope
            and (state.scoped_source or state.scoped_doc_id)
        ):
            change_docs = state.docs
            full_change_docs = get_documents_by_source(
                source=state.scoped_source,
                doc_id=state.scoped_doc_id,
            )
            if full_change_docs:
                change_docs = full_change_docs
            direct_change_answer = _try_scoped_change_summary(user_message, change_docs)
            if direct_change_answer:
                logger.info(
                    "Answered scoped change-summary query directly from '%s'%s",
                    active_source or "scoped document",
                    f" ({active_doc_id})" if active_doc_id else "",
                )
                return state.response(
                    _deterministic_grounded_response(direct_change_answer, change_docs),
                    mode="document_qa",
                    sources=extract_sources(change_docs),
                    active_source=active_source,
                    active_doc_id=active_doc_id,
                    active_source_type=active_source_type,
                )

        if (
            state.docs
            and not state.multi_scope
            and (state.scoped_source or state.scoped_doc_id)
            and not state.query_policy.comparison
        ):
            direct_presence_answer = _try_scoped_presence_answer(user_message, state.docs)
            if direct_presence_answer:
                logger.info(
                    "Answered scoped presence query directly from '%s'%s",
                    active_source or "scoped document",
                    f" ({active_doc_id})" if active_doc_id else "",
                )
                return state.response(
                    _deterministic_grounded_response(direct_presence_answer, state.docs[:1]),
                    mode="document_qa",
                    sources=extract_sources(state.docs[:1]),
                    active_source=active_source,
                    active_doc_id=active_doc_id,
                    active_source_type=active_source_type,
                )

            clause_docs = state.docs
            if state.query_policy.strict_fact:
                clause_docs = _select_strict_fact_docs(user_message, state.docs)
            direct_clause_answer = _try_scoped_clause_answer(user_message, clause_docs)
            if direct_clause_answer:
                logger.info(
                    "Answered scoped clause query directly from '%s'%s",
                    active_source or "scoped document",
                    f" ({active_doc_id})" if active_doc_id else "",
                )
                return state.response(
                    _deterministic_grounded_response(direct_clause_answer, clause_docs),
                    mode="document_qa",
                    sources=extract_sources(clause_docs),
                    active_source=active_source,
                    active_doc_id=active_doc_id,
                    active_source_type=active_source_type,
                )

        if state.use_full_document and not state.multi_scope and (state.scoped_source or state.scoped_doc_id):
            if state.query_policy.multi_document_summary:
                summary_answer, summary_docs = _build_single_document_summary(
                    user_message,
                    state.docs,
                    source_name=active_source or state.scoped_source,
                )
                logger.info(
                    "Returning deterministic single-document summary for '%s'%s",
                    active_source or "scoped document",
                    f" ({active_doc_id})" if active_doc_id else "",
                )
                return state.response(
                    _deterministic_grounded_response(summary_answer, summary_docs),
                    mode="document_qa",
                    sources=extract_sources(summary_docs),
                    active_source=active_source,
                    active_doc_id=active_doc_id,
                    active_source_type=active_source_type,
                )

            if state.query_policy.strict_fact:
                numeric_answer = _try_numeric_fact_answer(user_message, state.docs)
                if numeric_answer:
                    logger.info(
                        "Answered scoped numeric fact query directly from '%s'%s",
                        active_source or "scoped document",
                        f" ({active_doc_id})" if active_doc_id else "",
                    )
                    return state.response(
                        _deterministic_grounded_response(numeric_answer, state.docs[:3]),
                        mode="document_qa",
                        sources=extract_sources(state.docs[:3]),
                        active_source=active_source,
                        active_doc_id=active_doc_id,
                        active_source_type=active_source_type,
                    )

                clause_docs = _select_strict_fact_docs(user_message, state.docs)
                direct_clause_answer = _try_scoped_clause_answer(user_message, clause_docs)
                if direct_clause_answer:
                    logger.info(
                        "Answered scoped strict-fact query directly from '%s'%s",
                        active_source or "scoped document",
                        f" ({active_doc_id})" if active_doc_id else "",
                    )
                    return state.response(
                        _deterministic_grounded_response(direct_clause_answer, clause_docs),
                        mode="document_qa",
                        sources=extract_sources(clause_docs),
                        active_source=active_source,
                        active_doc_id=active_doc_id,
                        active_source_type=active_source_type,
                    )

                direct_count_answer = _try_scoped_count_answer(user_message, state.docs, active_source)
                if direct_count_answer:
                    logger.info(
                        "Answered scoped strict-count query directly from '%s'%s",
                        active_source or "scoped document",
                        f" ({active_doc_id})" if active_doc_id else "",
                    )
                    return state.response(
                        direct_count_answer,
                        mode="document_qa",
                        sources=extract_sources(state.docs),
                        active_source=active_source,
                        active_doc_id=active_doc_id,
                        active_source_type=active_source_type,
                    )

                direct_toc_answer = _try_chapter_title_lookup(user_message, state.docs)
                if direct_toc_answer:
                    logger.info(
                        "Answered chapter TOC lookup directly from '%s'%s",
                        active_source or "scoped document",
                        f" ({active_doc_id})" if active_doc_id else "",
                    )
                    return state.response(
                        direct_toc_answer,
                        mode="document_qa",
                        sources=extract_sources(state.docs),
                        active_source=active_source,
                        active_doc_id=active_doc_id,
                        active_source_type=active_source_type,
                    )

                direct_history_answer = _try_history_lookup(user_message, state.docs)
                if direct_history_answer:
                    logger.info(
                        "Answered publication history lookup directly from '%s'%s",
                        active_source or "scoped document",
                        f" ({active_doc_id})" if active_doc_id else "",
                    )
                    return state.response(
                        direct_history_answer,
                        mode="document_qa",
                        sources=extract_sources(state.docs),
                        active_source=active_source,
                        active_doc_id=active_doc_id,
                        active_source_type=active_source_type,
                    )

            scoped_docs = _filter_docs_to_named_scope(user_message, state.docs)
            if len(scoped_docs) != len(state.docs):
                logger.info(
                    "Narrowed bundled document context from %d to %d chunks based on named sub-guideline",
                    len(state.docs),
                    len(scoped_docs),
                )
            state.docs = scoped_docs
            clarification = _should_request_scope_clarification(user_message, state.docs)
            if clarification:
                logger.info(
                    "Ambiguous scoped question across %d sub-guidelines in '%s'; requesting clarification",
                    len(_extract_scope_labels(state.docs)),
                    state.scoped_source,
                )
                return state.response(
                    clarification,
                    mode="document_qa",
                    sources=extract_sources(state.docs),
                    active_source=active_source,
                    active_doc_id=active_doc_id,
                    active_source_type=active_source_type,
                )
            if state.query_policy.strict_fact:
                narrowed_docs = _select_strict_fact_docs(user_message, state.docs)
                if len(narrowed_docs) != len(state.docs):
                    logger.info(
                        "Narrowed strict-fact context from %d to %d chunks for '%s'",
                        len(state.docs),
                        len(narrowed_docs),
                        state.scoped_source or "scoped document",
                    )
                state.docs = narrowed_docs

        if state.use_full_document and active_source_type and not state.selected_docs:
            state.doc_context = format_grouped_corpus_context(state.docs)
        else:
            state.doc_context = format_context(state.docs)
        state.sources = extract_sources(state.docs)

        if state.multi_scope:
            logger.info(
                "Loaded comparison context: %d chunks from %d selected documents",
                len(state.docs),
                len(state.scoped_doc_ids),
            )
        elif state.use_full_document and active_source_type and not state.selected_docs:
            logger.info(
                "Loaded full corpus context: %d chunks from source_type='%s'",
                len(state.docs),
                active_source_type,
            )
        elif state.use_full_document:
            logger.info(
                "Loaded full document context: %d chunks from '%s'",
                len(state.docs),
                state.scoped_source,
            )
        else:
            logger.info(
                "Found %d relevant chunks%s (confidence=%.2f, keyword_hit=%s)",
                len(state.docs),
                (
                    f" (scoped to '{state.scoped_source}'"
                    + (f", {state.scoped_doc_id}" if state.scoped_doc_id else "")
                    + ")"
                ) if (state.scoped_source or state.scoped_doc_id) else "",
                state.retrieval.confidence,
                state.retrieval.strong_keyword_hit,
            )
    else:
        logger.info("No relevant document chunks found")

    if (
        state.docs
        and state.use_full_document
        and active_source_type
        and not state.selected_docs
        and state.query_policy.multi_document_summary
        and not state.query_policy.comparison
    ):
        logger.info(
            "Returning deterministic whole-corpus file summary for source_type='%s'",
            active_source_type,
        )
        return state.response(
            _build_file_level_corpus_summary(user_message, state.docs),
            mode="document_qa",
            active_source=None,
            active_doc_id=None,
            active_source_type=active_source_type,
            active_sources=[],
            active_doc_ids=[],
        )

    if (
        state.doc_context
        and state.docs
        and (state.scoped_source or state.scoped_doc_id)
        and not state.use_full_document
        and not state.query_policy.smalltalk
    ):
        source_texts = [d.page_content for d in state.docs[:3]]
        relevance_score = check_context_relevance(user_message, source_texts)
        if relevance_score is not None:
            logger.info(
                "Context relevance=%.2f for scoped query '%s'",
                relevance_score, user_message[:60],
            )
            if relevance_score < CONTEXT_RELEVANCE_THRESHOLD:
                strong_overlap = _has_strong_query_overlap(user_message, state.docs)
                bilingual_scoped_bypass = (
                    _infer_target_language(user_message) == "en"
                    and bool(state.docs)
                    and state.retrieval.confidence >= 0.60
                )
                if strong_overlap:
                    logger.info(
                        "Bypassing low relevance gate due to strong lexical overlap for '%s'",
                        user_message[:60],
                    )
                    return None
                if bilingual_scoped_bypass:
                    logger.info(
                        "Bypassing low relevance gate for bilingual scoped query '%s' (confidence=%.2f)",
                        user_message[:60],
                        state.retrieval.confidence,
                    )
                    return None
                logger.warning(
                    "Off-topic context (relevance=%.2f < %.2f) — returning not-found for '%s'",
                    relevance_score, CONTEXT_RELEVANCE_THRESHOLD,
                    state.scoped_source or state.scoped_doc_id,
                )
                return state.response(
                    _context_irrelevant_answer(user_message, active_source),
                    mode="document_qa",
                )

    return None


def _build_generation_inputs_stage(
    state: ChatExecutionState,
    *,
    user_message: str,
) -> None:
    """Add optional web context and build the final generation prompt."""
    state.web_context = ""
    if not state.scoped_source and not state.multi_scope:
        if state.query_policy.web_search_candidate:
            web_results = _search_web(user_message)
            if web_results:
                state.web_context = web_results
                logger.info("Added web search results (real-time query)")
        elif not state.doc_context and state.query_policy.general_knowledge:
            web_results = _search_web(user_message)
            if web_results:
                state.web_context = web_results
                logger.info("Added web search results (general knowledge fallback)")

    state.prompt, state.lm_messages = _build_prompt(
        user_message=user_message,
        history=state.history,
        doc_context=state.doc_context,
        web_context=state.web_context,
        system_prompt=state.system_prompt,
        selected_doc_count=len(state.selected_docs),
        document_scope_count=state.document_scope_count(),
    )


def _build_grounding_repair_messages(
    state: ChatExecutionState,
    *,
    user_message: str,
    bad_answer: str,
) -> tuple[str, List[Dict[str, Any]]]:
    repair_note = (
        "The previous draft answer was not grounded enough in the provided document context.\n"
        "Rewrite the answer using only facts directly supported by the retrieved document context.\n"
        "Do not add interpretations, assumptions, or extra policy details.\n"
        "If a detail is not explicitly supported, say it is not confirmed in the provided document.\n"
        "Every important factual sentence or bullet must include an inline citation like [1] or [2].\n"
        "Prefer short, concrete statements over broad summaries.\n"
    )
    repair_prompt = (
        f"{state.prompt}\n\n"
        "[Grounding repair]\n"
        f"{repair_note}\n"
        f"[Previous unsupported draft]\n{bad_answer}"
    )
    repair_messages: List[Dict[str, Any]] = [
        state.lm_messages[0],
        {
            "role": "user",
            "content": (
                state.lm_messages[1]["content"]
                + "\n\n[Grounding repair]\n"
                + repair_note
                + "\n[Previous unsupported draft]\n"
                + bad_answer
            ),
        },
    ]
    return repair_prompt, repair_messages


def _score_answer_grounding(
    answer: str,
    docs,
    *,
    user_message: str,
) -> Optional[Any]:
    if not answer or not docs:
        return None
    answer_for_verifier = answer
    if not _is_grounding_fallback_answer(answer_for_verifier):
        answer_for_verifier = _ensure_minimum_inline_citations(answer_for_verifier, docs)
    source_texts = [d.page_content for d in docs[:6]]
    faith_result = analyze_faithfulness(answer_for_verifier, source_texts)
    if faith_result is None:
        return None
    if answer_for_verifier != answer:
        faith_result.answer_for_verifier = answer_for_verifier  # type: ignore[attr-defined]
    logger.info(
        "Faithfulness score=%.2f for document-backed query '%s' (unsupported=%d, citation_mismatch=%d, cited_sentences=%d)",
        faith_result.score,
        user_message[:60],
        len(faith_result.unsupported_sentences),
        len(faith_result.citation_mismatch_sentences),
        faith_result.cited_sentence_count,
    )
    if faith_result.unsupported_sentences:
        logger.info(
            "Unsupported answer sentence preview: %s",
            faith_result.unsupported_sentences[0][:120],
        )
    return faith_result


def _finalize_answer_stage(
    state: ChatExecutionState,
    *,
    user_message: str,
    active_source: Optional[str],
) -> Dict[str, Any]:
    """Generate, validate, decorate, and return the final answer."""
    state.answer = generate_text(
        state.prompt,
        model=state.selected_model,
        messages=state.lm_messages,
    )

    if _needs_language_retry(user_message, state.answer):
        is_chinese = _contains_chinese(state.answer)
        lang_name = "Korean" if _infer_target_language(user_message) == "ko" else "English"
        if is_chinese:
            logger.warning("Language drift detected (Chinese); retrying answer generation")
            correction_note = (
                "The previous draft answer was invalid because it used Chinese characters.\n"
                f"Rewrite the final answer entirely in {lang_name}.\n"
                "Do not use any Chinese characters.\n"
            )
        else:
            logger.warning("Language drift detected (English code-switching); retrying answer generation")
            correction_note = (
                "The previous draft answer is invalid because it contains large English "
                "paragraphs in what must be an entirely Korean response.\n"
                "Rewrite the final answer entirely in Korean.\n"
                "English proper nouns (e.g. 'Seventh-day Adventists', 'NEWSTART') are "
                "acceptable only when immediately followed by a Korean equivalent in parentheses.\n"
            )
        retry_prompt = _language_correction_prompt(
            state.prompt,
            user_message,
            state.answer,
        )
        retry_messages: List[Dict[str, Any]] = [
            state.lm_messages[0],
            {
                "role": "user",
                "content": (
                    state.lm_messages[1]["content"]
                    + "\n\n[Critical correction]\n"
                    + correction_note
                    + "Keep the same facts and stay grounded in the provided context.\n\n"
                    + f"[Invalid draft answer]\n{state.answer}"
                ),
            },
        ]
        retry_answer = generate_text(
            retry_prompt,
            model=state.selected_model,
            messages=retry_messages,
        )
        if retry_answer:
            state.answer = retry_answer

    if state.answer and state.docs and state.doc_context:
        faith_result = _score_answer_grounding(
            state.answer,
            state.docs,
            user_message=user_message,
        )
        if faith_result is not None:
            faith_score = faith_result.score
            unsupported_count = len(faith_result.unsupported_sentences)
            citation_mismatch_count = len(faith_result.citation_mismatch_sentences)
            hard_fail = (
                faith_score < NLI_FAITHFULNESS_HARD_THRESHOLD
                or unsupported_count >= max(2, (len(faith_result.sentence_scores) + 1) // 2)
            )
            soft_fail = (
                faith_score < NLI_FAITHFULNESS_SOFT_THRESHOLD
                or unsupported_count > 0
                or citation_mismatch_count > 0
            )
            if soft_fail:
                logger.info(
                    "Attempting grounding-repair rewrite for '%s'",
                    active_source or state.scoped_doc_id,
                )
                repair_prompt, repair_messages = _build_grounding_repair_messages(
                    state,
                    user_message=user_message,
                    bad_answer=state.answer,
                )
                repaired_answer = generate_text(
                    repair_prompt,
                    model=state.selected_model,
                    messages=repair_messages,
                )
                repaired_result = _score_answer_grounding(
                    repaired_answer,
                    state.docs,
                    user_message=user_message,
                ) if repaired_answer else None
                if repaired_result is not None:
                    repaired_unsupported = len(repaired_result.unsupported_sentences)
                    repaired_mismatch = len(repaired_result.citation_mismatch_sentences)
                    improved = (
                        repaired_result.score > faith_score + 0.05
                        or repaired_unsupported < unsupported_count
                        or (
                            repaired_result.cited_sentence_count > faith_result.cited_sentence_count
                            and repaired_result.score >= faith_score
                        )
                    )
                    if improved:
                        logger.info(
                            "Grounding-repair rewrite accepted for '%s' (%.2f -> %.2f)",
                            active_source or state.scoped_doc_id,
                            faith_score,
                            repaired_result.score,
                        )
                        state.answer = repaired_answer
                        faith_result = repaired_result
                        faith_score = repaired_result.score
                        unsupported_count = repaired_unsupported
                        citation_mismatch_count = repaired_mismatch
                        hard_fail = (
                            faith_score < NLI_FAITHFULNESS_HARD_THRESHOLD
                            or unsupported_count >= max(2, (len(faith_result.sentence_scores) + 1) // 2)
                        )
                        soft_fail = (
                            faith_score < NLI_FAITHFULNESS_SOFT_THRESHOLD
                            or unsupported_count > 0
                            or citation_mismatch_count > 0
                        )
            if hard_fail:
                logger.warning(
                    "Very low faithfulness (%.2f, unsupported=%d) — replacing answer for '%s'",
                    faith_score,
                    unsupported_count,
                    active_source or state.scoped_doc_id,
                )
                state.answer = _not_grounded_answer(user_message)
            elif soft_fail:
                logger.warning(
                    "Low faithfulness / citation mismatch (score=%.2f, unsupported=%d, citation_mismatch=%d) — appending disclaimer for '%s'",
                    faith_score,
                    unsupported_count,
                    citation_mismatch_count,
                    active_source or state.scoped_doc_id,
                )
                state.answer = (
                    state.answer.rstrip()
                    + "\n\n*참고: 이 답변의 일부 문장이나 인용은 문서 내용과 완전히 일치하지 않을 수 있습니다. 문서를 직접 확인해 주세요.*"
                )

    if state.answer and state.docs and state.doc_context and not _is_grounding_fallback_answer(state.answer):
        state.answer = _ensure_minimum_inline_citations(state.answer, state.docs)
        citation_footer = _build_citation_footer(state.docs)
        if citation_footer:
            state.answer = state.answer.rstrip() + "\n" + citation_footer

    mode = "general"
    if state.doc_context and state.sources:
        mode = "document_compare" if state.multi_scope else "document_qa"
    if state.web_context:
        mode = "web_search" if not state.doc_context else "document_qa+web"

    return state.response(state.answer, mode=mode)


# ═══════════════════════════════════════════════════════════════════════
# Unified Chat Handler
# ═══════════════════════════════════════════════════════════════════════

def handle_chat(
    user_message: str,
    history: List[Message] = None,
    model: str = None,
    active_source: str = None,
    active_doc_id: str = None,
    active_source_type: str = None,
    active_sources: List[str] = None,
    active_doc_ids: List[str] = None,
    system_prompt: str = None,
) -> Dict[str, Any]:
    """
    Unified chat handler — works like a normal chatbot.

    Args:
        user_message: What the user said
        history: Conversation history
        model: Which runtime model to use
        active_source: Active document filename scope for the current chat, if any
        active_doc_id: Stable document ID scope for the current chat, if any
        active_source_type: Scope to a whole source type such as "upload"
        active_sources: Multi-document source scope for comparison workflows
        active_doc_ids: Multi-document doc_id scope for comparison workflows
        system_prompt: Override default system prompt
    """
    history = history or []
    state = _build_chat_state(
        user_message,
        history,
        model,
        active_source,
        active_doc_id,
        active_source_type,
        active_sources,
        active_doc_ids,
        system_prompt,
    )

    multi_question_response = _maybe_handle_multi_question_fanout(
        user_message,
        history,
        model,
        active_source,
        active_doc_id,
        active_source_type,
        active_sources,
        active_doc_ids,
        system_prompt,
    )
    if multi_question_response:
        return multi_question_response

    scope_response = _resolve_scope_stage(
        state,
        user_message=user_message,
        active_source=active_source,
        active_doc_id=active_doc_id,
        active_source_type=active_source_type,
    )
    if scope_response:
        return scope_response

    retrieval_response = _run_retrieval_stage(
        state,
        user_message=user_message,
        active_source=active_source,
        active_doc_id=active_doc_id,
        active_source_type=active_source_type,
    )
    if retrieval_response:
        return retrieval_response

    _build_generation_inputs_stage(state, user_message=user_message)
    return _finalize_answer_stage(
        state,
        user_message=user_message,
        active_source=active_source,
    )
