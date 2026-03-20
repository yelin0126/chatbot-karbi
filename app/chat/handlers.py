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
from typing import List, Dict, Any, Optional

from app.models.schemas import Message
from app.core.llm import call_ollama, get_response_text
from app.retrieval.retriever import retrieve, format_context, extract_sources
from app.core.vectorstore import get_documents_by_source, get_document_chunk_count
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
    ]
    return any(keyword in lower for keyword in indicators)


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
        "what does this image say", "what does the image say",
        "give me the text", "extract the text", "read the text",
        "read this image", "text in the image", "transcribe",
    ]
    return any(keyword in lower for keyword in indicators)


def _needs_section_understanding_style(text: str) -> bool:
    """Detect prompts asking for the role/meaning of a section rather than raw lookup."""
    lower = text.lower().strip()
    indicators = [
        "어떤 역할", "무슨 역할", "어떤 의미", "무슨 의미", "설명해줘", "설명해 줘",
        "차이", "구분", "성격", "의미를", "역할을", "what role", "what does this section do",
        "what does it mean", "explain the section", "difference between", "how is it different",
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
        system_prompt: Override default system prompt
    """
    history = history or []
    selected_model = model or OLLAMA_MODEL
    scoped_source = None if _is_smalltalk_query(user_message) else active_source
    scoped_doc_id = None if _is_smalltalk_query(user_message) else active_doc_id

    if (scoped_source or scoped_doc_id) and _is_direct_extraction_query(user_message):
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
            }

    # ── Step 1: Always search for relevant document context ──
    doc_context = ""
    sources = []

    use_full_document = bool(
        (scoped_source or scoped_doc_id)
        and (
            _needs_full_document_context(user_message)
            or _should_force_small_doc_full_context(scoped_source, scoped_doc_id)
        )
    )
    retrieval = retrieve(
        user_message,
        source_filter=scoped_source,
        doc_id_filter=scoped_doc_id,
        full_document=use_full_document,
    )
    docs = retrieval.docs

    if (scoped_source or scoped_doc_id) and not use_full_document:
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
            }

    if docs:
        doc_context = format_context(docs)
        sources = extract_sources(docs)
        if use_full_document:
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
    if not scoped_source and _might_need_web_search(user_message):
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
        mode = "document_qa"
    if web_context:
        mode = "web_search" if not doc_context else "document_qa+web"

    return {
        "answer": answer,
        "sources": sources,
        "mode": mode,
        "active_source": active_source,
        "active_doc_id": active_doc_id,
    }
