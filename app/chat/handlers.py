"""
Unified chat handler — works like a normal chatbot.

NO hardcoded mode routing. Instead:
1. Always search vectorstore for relevant document context
2. If web search might help, search Tavily too
3. Build one prompt with all available context
4. LLM decides what to use

This is how ChatGPT/Claude work — retrieve first, let the model decide.
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple

from app.models.schemas import Message
from app.core.llm import call_ollama, get_response_text
from app.retrieval.retriever import retrieve, format_context, extract_sources
from app.core.vectorstore import (
    get_documents_by_source,
    get_documents_by_doc_ids,
    get_document_chunk_count,
)
from app.config import (
    OLLAMA_MODEL,
    TAVILY_API_KEY,
    DOCUMENT_CONFIDENCE_THRESHOLD,
    SCOPED_SINGLE_CHUNK_CONFIDENCE_THRESHOLD,
    SCOPED_SMALL_DOC_FULL_CONTEXT_MAX_CHUNKS,
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


def _might_need_web_search(text: str) -> bool:
    """Light heuristic: does this query likely need real-time info?"""
    indicators = [
        "오늘", "현재", "최신", "최근", "실시간", "지금", "올해",
        "날씨", "뉴스", "환율", "주가", "시세", "속보",
        "today", "current", "latest", "recent", "now", "weather",
        "news", "stock", "price", "score", "live",
        "검색", "search", "look up", "find out",
    ]
    lower = text.lower()
    return any(kw in lower for kw in indicators)


def _needs_full_document_context(text: str) -> bool:
    """
    Detect requests that need the whole uploaded document, not just top-k chunks.

    This is used only for file-scoped chat after upload.
    """
    lower = text.lower().strip()
    indicators = [
        "요약", "요약해", "정리", "정리해", "전체", "전반", "구조", "목차",
        "분석", "분석해", "핵심", "주요 내용", "전체 내용", "섹션", "section",
        "structure", "outline", "overview", "summarize", "summary",
        "analyze", "analysis", "key points", "main points", "extract key",
        "extract data", "important information", "important info",
        "부칙", "경과조치", "지원대상", "운영절차", "시행일", "준용", "위원장",
        "article", "clause", "section title",
    ]
    if any(keyword in lower for keyword in indicators):
        return True
    return bool(re.search(r"제\s*\d+\s*조", text))


def _is_smalltalk_query(text: str) -> bool:
    """Detect greetings/acknowledgements that should bypass document scoping."""
    lower = text.lower().strip()
    indicators = [
        "안녕", "고마워", "감사", "오케이", "알겠", "응", "네",
        "hello", "hi", "thanks", "thank you", "okay", "ok", "got it",
    ]
    return any(keyword in lower for keyword in indicators)


def _document_not_found_answer(
    user_message: str,
    active_source: Optional[str],
    active_doc_id: Optional[str] = None,
) -> str:
    """Return a grounded fallback when scoped retrieval confidence is too low."""
    source_name = active_source or active_doc_id or "the uploaded document"
    if re.search(r"[가-힣]", user_message):
        return (
            f"업로드된 문서 '{source_name}'에서 질문과 관련된 정보를 찾지 못했습니다. "
            "질문을 조금 더 구체적으로 해주세요."
        )
    return (
        f"I couldn't find relevant information in the uploaded document '{source_name}'. "
        "Please try a more specific question."
    )


def _normalize_active_scopes(
    active_source: Optional[str],
    active_doc_id: Optional[str],
    active_sources: Optional[List[str]],
    active_doc_ids: Optional[List[str]],
) -> Tuple[List[str], List[str]]:
    sources = [source for source in (active_sources or []) if source]
    doc_ids = [doc_id for doc_id in (active_doc_ids or []) if doc_id]

    if active_doc_id and active_doc_id not in doc_ids:
        doc_ids.insert(0, active_doc_id)
    if active_source and active_source not in sources:
        sources.insert(0, active_source)

    return sources, doc_ids


def _infer_target_language(text: str) -> str:
    """Infer the user's requested answer language from the current message."""
    if re.search(r"[가-힣]", text or ""):
        return "ko"
    if re.search(r"[A-Za-z]", text or ""):
        return "en"
    return "same"


def _contains_chinese(text: str) -> bool:
    """Detect Chinese Han characters that should not appear in normal answers."""
    return bool(re.search(r"[\u4e00-\u9fff]", text or ""))


def _needs_language_retry(user_message: str, answer: str) -> bool:
    """Retry if the model drifted into Chinese despite explicit instructions."""
    target_lang = _infer_target_language(user_message)
    if target_lang not in {"ko", "en"}:
        return False
    return _contains_chinese(answer)


def _language_correction_prompt(
    original_prompt: str,
    user_message: str,
    bad_answer: str,
) -> str:
    target_lang = _infer_target_language(user_message)
    lang_name = "Korean" if target_lang == "ko" else "English"
    return (
        f"{original_prompt}\n\n"
        "[Critical correction]\n"
        f"The previous draft answer was invalid because it used Chinese characters.\n"
        f"Rewrite the final answer entirely in {lang_name}.\n"
        "Do not use any Chinese characters.\n"
        "Keep the same facts and stay grounded in the provided context.\n\n"
        f"[Invalid draft answer]\n{bad_answer}"
    )


def _is_direct_extraction_query(text: str) -> bool:
    """Detect requests that want raw OCR/text output from the uploaded file."""
    lower = text.lower().strip()
    indicators = [
        "텍스트 추출", "문자 추출", "글자 추출", "읽어줘", "텍스트만", "원문", "ocr",
        "내용 추출", "내용 보여", "전문 보여", "본문 보여", "여기 안에 있는 내용 추출",
        "what does this image say", "what does the image say",
        "give me the text", "extract the text", "read the text",
        "read this image", "text in the image", "transcribe",
    ]
    return any(keyword in lower for keyword in indicators)


def _normalize_scope_text(text: str) -> str:
    return re.sub(r"[^0-9a-z가-힣]", "", (text or "").lower())


def _extract_scope_labels(docs) -> List[str]:
    """Extract distinct sub-guideline labels from a combined uploaded document."""
    labels = []
    seen = set()
    for doc in docs:
        raw_title = str(doc.metadata.get("section_title") or "").strip()
        if not raw_title:
            continue

        cleaned = raw_title
        cleaned = re.sub(r"^제주대학교\s*RISE사업단\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*운영\s*지침.*$", "", cleaned)
        cleaned = re.sub(r"\s*규정.*$", "", cleaned)
        cleaned = cleaned.strip(" -:")
        if not cleaned:
            continue

        key = _normalize_scope_text(cleaned)
        if key in seen:
            continue
        seen.add(key)
        labels.append(cleaned)
    return labels


def _chunk_scope_label(doc, active_label: Optional[str]) -> Optional[str]:
    """Infer which bundled sub-guideline a chunk belongs to."""
    raw_title = str(doc.metadata.get("section_title") or "").strip()
    if raw_title:
        labels = _extract_scope_labels([doc])
        if labels:
            return labels[0]
    return active_label


def _filter_docs_to_named_scope(user_message: str, docs):
    """
    If the user explicitly names one bundled sub-guideline, keep only those chunks.

    Continuation pages without their own section title inherit the previous label.
    """
    labels = _extract_scope_labels(docs)
    if len(labels) < 2:
        return docs

    normalized_query = _normalize_scope_text(user_message)
    matched_label = None
    for label in labels:
        token = _normalize_scope_text(label)
        if token and token in normalized_query:
            matched_label = label
            break

    if not matched_label:
        return docs

    filtered = []
    current_label = None
    for doc in docs:
        current_label = _chunk_scope_label(doc, current_label)
        if current_label == matched_label:
            filtered.append(doc)

    return filtered or docs


def _build_scoped_ambiguity_answer(user_message: str, labels: List[str]) -> str:
    examples = ", ".join(f"'{label}'" for label in labels[:3])
    if re.search(r"[가-힣]", user_message):
        return (
            "이 업로드 파일에는 여러 운영지침이 함께 들어 있어 질문이 모호합니다. "
            f"현재 확인되는 지침은 {examples} 입니다. "
            "어느 지침을 기준으로 답변할지 함께 지정해 주세요. "
            "예: '프로젝트Lab 지원대상 알려줘', '대학원 인턴십 제4조 알려줘'"
        )
    return (
        "This uploaded file contains multiple sub-guidelines, so the question is ambiguous. "
        f"I found these guideline scopes: {examples}. "
        "Please name which one you mean, for example: "
        "'Tell me the support target for ProjectLab' or 'Explain Article 4 of the Graduate Internship guideline.'"
    )


def _should_request_scope_clarification(user_message: str, docs) -> Optional[str]:
    """
    Ask for clarification when a single uploaded PDF bundles multiple sub-guidelines
    that reuse overlapping article numbers or headings.
    """
    labels = _extract_scope_labels(docs)
    if len(labels) < 2:
        return None

    normalized_query = _normalize_scope_text(user_message)
    label_tokens = [_normalize_scope_text(label) for label in labels]
    if any(token and token in normalized_query for token in label_tokens):
        return None

    ambiguous_article = bool(re.search(r"제\s*\d+\s*조", user_message))
    ambiguous_heading_terms = [
        "지원대상", "운영절차", "경과조치", "부칙", "지원사항",
        "지원중단", "의무위반", "가이드라인 준용", "정의", "목적",
        "시행일", "조치", "내용 알려줘",
    ]
    if ambiguous_article or any(term in user_message for term in ambiguous_heading_terms):
        return _build_scoped_ambiguity_answer(user_message, labels)

    return None


def _needs_section_understanding_style(text: str) -> bool:
    """Detect prompts asking for the role/meaning of a section rather than raw lookup."""
    lower = text.lower().strip()
    indicators = [
        "어떤 역할", "무슨 역할", "어떤 의미", "무슨 의미", "설명해줘", "설명해 줘",
        "차이", "구분", "성격", "의미를", "역할을", "what role", "what does this section do",
        "what does it mean", "explain the section", "difference between", "how is it different",
    ]
    return any(keyword in lower for keyword in indicators)


def _needs_comparison_style(text: str) -> bool:
    lower = text.lower().strip()
    indicators = [
        "비교", "차이", "다른 점", "공통점", "구분", "대조",
        "compare", "comparison", "difference", "differences", "similarity",
    ]
    return any(keyword in lower for keyword in indicators)


def _strip_enrichment_header(text: str) -> str:
    """Remove enrichment header prepended before embedding/retrieval."""
    return re.sub(r'^\[Document:.*?\]\n', '', text or '', flags=re.DOTALL).strip()


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


def _scoped_confidence_threshold(docs) -> float:
    """
    Use a slightly lower threshold for tiny uploaded docs where one chunk is
    effectively the whole document (e.g. a screenshot or one-page upload).
    """
    if len(docs) != 1:
        return DOCUMENT_CONFIDENCE_THRESHOLD

    meta = docs[0].metadata
    if meta.get("source_type") == "upload":
        return SCOPED_SINGLE_CHUNK_CONFIDENCE_THRESHOLD

    return DOCUMENT_CONFIDENCE_THRESHOLD


def _scoped_confidence_threshold_for_query(
    user_message: str,
    docs,
    active_source: Optional[str],
    active_doc_id: Optional[str],
) -> float:
    """
    Lower the gate slightly for cross-language document QA where the right doc was found
    but semantic similarity can be lower (e.g. English query over Korean policy text).
    """
    base_threshold = _scoped_confidence_threshold(docs)
    if not docs:
        return base_threshold

    query_lang = _infer_target_language(user_message)
    doc_langs = {
        str(doc.metadata.get("language", "")).lower()
        for doc in docs
        if doc.metadata.get("language")
    }

    is_cross_language = (
        query_lang == "en" and "ko" in doc_langs
    ) or (
        query_lang == "ko" and "en" in doc_langs
    )

    if is_cross_language and (active_source or active_doc_id):
        return max(0.25, base_threshold - 0.12)

    return base_threshold


def _should_force_small_doc_full_context(
    active_source: Optional[str],
    active_doc_id: Optional[str],
) -> bool:
    """Use full-document context for tiny scoped docs where top-k retrieval is brittle."""
    if not active_source and not active_doc_id:
        return False

    try:
        chunk_count = get_document_chunk_count(source=active_source, doc_id=active_doc_id)
    except Exception as e:
        logger.debug("Could not inspect scoped document chunk count: %s", e)
        return False

    if chunk_count <= 0:
        return False

    return chunk_count <= SCOPED_SMALL_DOC_FULL_CONTEXT_MAX_CHUNKS


# ═══════════════════════════════════════════════════════════════════════
# Prompt Builder
# ═══════════════════════════════════════════════════════════════════════

def _format_history(history: List[Message], max_turns: int = 8) -> str:
    if not history:
        return ""
    return "\n\n".join(
        f"[{msg.role}]\n{msg.content}" for msg in history[-max_turns:]
    )


_SYSTEM_PROMPT = """You are Tilon AI, a helpful document-based chatbot.

CRITICAL RULES:
1. Respond in the SAME language the user is using. Korean → Korean. English → English. NEVER respond in Chinese (中文).
2. If document context is provided and relevant to the question, answer based on that context and cite the source (document name, page number).
3. If document context is provided but NOT relevant to the question, ignore it and answer normally.
4. If no document context is available, answer from your general knowledge.
5. If web search results are provided, use them for current/real-time information.
6. If you don't know, say so honestly. Never make up information.
7. Be concise and direct. Answer the question first, then explain if needed.
8. When citing documents, mention the source naturally (e.g., "문서 3페이지에 따르면..." or "According to page 3...")."""


def _build_prompt(
    user_message: str,
    history: List[Message],
    doc_context: str = "",
    web_context: str = "",
    system_prompt: str = "",
    selected_doc_count: int = 0,
) -> str:
    """Build a single unified prompt with all available context."""
    parts = [f"[System]\n{system_prompt or _SYSTEM_PROMPT}"]
    target_lang = _infer_target_language(user_message)
    if target_lang == "ko":
        parts.append("[Required response language]\nKorean only. Do not use Chinese characters.")
    elif target_lang == "en":
        parts.append("[Required response language]\nEnglish only. Do not use Chinese characters.")
        parts.append(
            "[Cross-language instruction]\n"
            "If the retrieved document evidence is in Korean, translate and explain it in English. "
            "Do not refuse only because the source document is in another language."
        )
    if _needs_section_understanding_style(user_message):
        parts.append(
            "[Task style]\n"
            "Answer as a section-understanding question. "
            "Explain the role, purpose, differences, or kinds of support described in the document. "
            "Name concrete items from the evidence instead of giving only a generic summary."
        )
    if selected_doc_count > 1 or _needs_comparison_style(user_message):
        parts.append(
            "[Comparison instruction]\n"
            "You are answering a document-comparison question.\n"
            "Compare only the selected documents in the retrieved document context.\n"
            "Do not merge multiple documents into one policy or invent shared rules.\n"
            "For each comparison point, explicitly say which document it belongs to.\n"
            "If a point is supported by only one document, say that clearly.\n"
            "If the evidence is insufficient for a comparison point, say it is not confirmed in the provided documents.\n"
            "Focus only on the dimension asked by the user, such as purpose, eligibility, procedure, support, or termination.\n"
            "Prefer a compact side-by-side structure.\n"
            "If the user is writing in Korean, use this structure:\n"
            "- 핵심 비교:\n"
            "- 문서별 요약:\n"
            "- 공통점:\n"
            "- 차이점:\n"
            "If the user is writing in English, use this structure:\n"
            "- Key comparison:\n"
            "- By document:\n"
            "- Similarities:\n"
            "- Differences:\n"
            "Every section must stay grounded in the retrieved evidence."
        )

    history_text = _format_history(history)
    if history_text:
        parts.append(f"[Conversation history]\n{history_text}")

    if doc_context:
        parts.append(f"[Retrieved document context]\n{doc_context}")

    if web_context:
        parts.append(f"[Web search results]\n{web_context}")

    parts.append(f"[User message]\n{user_message}")

    return "\n\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════
# Unified Chat Handler
# ═══════════════════════════════════════════════════════════════════════

def handle_chat(
    user_message: str,
    history: List[Message] = None,
    model: str = None,
    active_source: str = None,
    active_doc_id: str = None,
    active_sources: List[str] = None,
    active_doc_ids: List[str] = None,
    system_prompt: str = None,
) -> Dict[str, Any]:
    """
    Unified chat handler — works like a normal chatbot.

    Args:
        user_message: What the user said
        history: Conversation history
        model: Which Ollama model to use
        active_source: Active document filename scope for the current chat, if any
        active_doc_id: Stable document ID scope for the current chat, if any
        active_sources: Multi-document source scope for comparison workflows
        active_doc_ids: Multi-document doc_id scope for comparison workflows
        system_prompt: Override default system prompt
    """
    history = history or []
    selected_model = model or OLLAMA_MODEL
    normalized_sources, normalized_doc_ids = _normalize_active_scopes(
        active_source,
        active_doc_id,
        active_sources,
        active_doc_ids,
    )
    selected_docs: List[Dict[str, str]] = []
    max_len = max(len(normalized_sources), len(normalized_doc_ids))
    for idx in range(max_len):
        source = normalized_sources[idx] if idx < len(normalized_sources) else ""
        doc_id = normalized_doc_ids[idx] if idx < len(normalized_doc_ids) else ""
        if source or doc_id:
            selected_docs.append({"source": source, "doc_id": doc_id})

    if not selected_docs and (active_source or active_doc_id):
        selected_docs = [{"source": active_source or "", "doc_id": active_doc_id or ""}]

    if _is_smalltalk_query(user_message):
        selected_docs = []

    multi_scope = len(selected_docs) > 1
    scoped_source = selected_docs[0]["source"] if len(selected_docs) == 1 else None
    scoped_doc_id = selected_docs[0]["doc_id"] if len(selected_docs) == 1 else None
    scoped_sources = [doc["source"] for doc in selected_docs if doc.get("source")]
    scoped_doc_ids = [doc["doc_id"] for doc in selected_docs if doc.get("doc_id")]

    if len(selected_docs) == 1 and (scoped_source or scoped_doc_id) and _is_direct_extraction_query(user_message):
        docs = get_documents_by_source(source=scoped_source, doc_id=scoped_doc_id)
        if docs:
            logger.info(
                "Direct extraction response for '%s'%s",
                scoped_source or "scoped document",
                f" ({scoped_doc_id})" if scoped_doc_id else "",
            )
            return {
                "answer": _build_direct_extraction_answer(scoped_source, docs),
                "sources": extract_sources(docs),
                "mode": "ocr_extract",
                "active_source": scoped_source,
                "active_doc_id": scoped_doc_id,
                "active_sources": scoped_sources,
                "active_doc_ids": scoped_doc_ids,
            }

    # ── Step 1: Always search for relevant document context ──
    doc_context = ""
    sources = []

    use_full_document = bool(
        multi_scope or (
            (scoped_source or scoped_doc_id)
            and (
                _needs_full_document_context(user_message)
                or _should_force_small_doc_full_context(scoped_source, scoped_doc_id)
            )
        )
    )

    retrieval = None
    if multi_scope:
        docs = get_documents_by_doc_ids(scoped_doc_ids)
        logger.info(
            "Loaded %d chunks for multi-document scope across %d selected documents",
            len(docs),
            len(scoped_doc_ids),
        )
    else:
        retrieval = retrieve(
            user_message,
            source_filter=scoped_source,
            doc_id_filter=scoped_doc_id,
            full_document=use_full_document,
        )
        docs = retrieval.docs

    if not multi_scope and (scoped_source or scoped_doc_id) and not use_full_document:
        threshold = (
            _scoped_confidence_threshold_for_query(
                user_message,
                docs,
                scoped_source,
                scoped_doc_id,
            ) if docs else DOCUMENT_CONFIDENCE_THRESHOLD
        )
        if not docs or (
            retrieval.confidence < threshold
            and not retrieval.strong_keyword_hit
        ):
            logger.info(
                "Low-confidence scoped retrieval for '%s'%s (confidence=%.2f, threshold=%.2f, keyword_hit=%s)",
                active_source or "scoped document",
                f" ({scoped_doc_id})" if scoped_doc_id else "",
                retrieval.confidence,
                threshold,
                retrieval.strong_keyword_hit,
            )
            return {
                "answer": _document_not_found_answer(user_message, active_source, active_doc_id),
                "sources": [],
                "mode": "document_qa",
                "active_source": active_source,
                "active_doc_id": active_doc_id,
                "active_sources": scoped_sources,
                "active_doc_ids": scoped_doc_ids,
            }

    if docs:
        if use_full_document and not multi_scope:
            scoped_docs = _filter_docs_to_named_scope(user_message, docs)
            if len(scoped_docs) != len(docs):
                logger.info(
                    "Narrowed bundled document context from %d to %d chunks based on named sub-guideline",
                    len(docs),
                    len(scoped_docs),
                )
            docs = scoped_docs
            clarification = _should_request_scope_clarification(user_message, docs)
            if clarification:
                logger.info(
                    "Ambiguous scoped question across %d sub-guidelines in '%s'; requesting clarification",
                    len(_extract_scope_labels(docs)),
                    scoped_source,
                )
                return {
                    "answer": clarification,
                    "sources": extract_sources(docs),
                    "mode": "document_qa",
                    "active_source": active_source,
                    "active_doc_id": active_doc_id,
                    "active_sources": scoped_sources,
                    "active_doc_ids": scoped_doc_ids,
                }
        doc_context = format_context(docs)
        sources = extract_sources(docs)
        if multi_scope:
            logger.info(
                "Loaded comparison context: %d chunks from %d selected documents",
                len(docs),
                len(scoped_doc_ids),
            )
        elif use_full_document:
            logger.info(
                "Loaded full document context: %d chunks from '%s'",
                len(docs),
                scoped_source,
            )
        else:
            logger.info(
                "Found %d relevant chunks%s (confidence=%.2f, keyword_hit=%s)",
                len(docs),
                (
                    f" (scoped to '{scoped_source}'"
                    + (f", {scoped_doc_id}" if scoped_doc_id else "")
                    + ")"
                ) if (scoped_source or scoped_doc_id) else "",
                retrieval.confidence,
                retrieval.strong_keyword_hit,
            )
    else:
        logger.info("No relevant document chunks found")

    # ── Step 2: Check if web search might help ──
    web_context = ""
    if not scoped_source and not multi_scope and _might_need_web_search(user_message):
        web_results = _search_web(user_message)
        if web_results:
            web_context = web_results
            logger.info("Added web search results")

    # ── Step 3: Build prompt with all context and let LLM decide ──
    prompt = _build_prompt(
        user_message=user_message,
        history=history,
        doc_context=doc_context,
        web_context=web_context,
        system_prompt=system_prompt,
        selected_doc_count=len(selected_docs),
    )

    result = call_ollama(prompt, model=selected_model)
    answer = get_response_text(result)

    if _needs_language_retry(user_message, answer):
        logger.warning("Language drift detected; retrying answer generation without Chinese output")
        retry_prompt = _language_correction_prompt(prompt, user_message, answer)
        retry_result = call_ollama(retry_prompt, model=selected_model)
        retry_answer = get_response_text(retry_result)
        if retry_answer:
            answer = retry_answer

    # Determine what was used (for UI display)
    mode = "general"
    if doc_context and sources:
        mode = "document_compare" if multi_scope else "document_qa"
    if web_context:
        mode = "web_search" if not doc_context else "document_qa+web"

    return {
        "answer": answer,
        "sources": sources,
        "mode": mode,
        "active_source": scoped_source,
        "active_doc_id": scoped_doc_id,
        "active_sources": scoped_sources,
        "active_doc_ids": scoped_doc_ids,
    }
