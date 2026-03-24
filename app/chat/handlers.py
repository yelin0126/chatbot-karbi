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
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from app.models.schemas import Message
from app.core.llm import generate_text, get_default_model_name
from app.retrieval.retriever import (
    retrieve,
    format_context,
    format_grouped_corpus_context,
    extract_sources,
)
from app.retrieval.keyword_index import tokenize_text
from app.core.vectorstore import (
    get_documents_by_source,
    get_documents_by_doc_ids,
    get_document_chunk_count,
    get_section_titles,
)
from app.core.document_registry import list_documents
from app.config import (
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
    # Count/enumeration queries need the intro/TOC which top-k misses
    if re.search(r"총\s*몇|몇\s*가지|몇\s*항목|몇\s*개|몇\s*장|how\s+many", lower):
        return True
    return bool(re.search(r"제\s*\d+\s*조", text))


def _needs_multi_document_summary_style(text: str) -> bool:
    """Detect summary-style requests over multiple documents."""
    lower = text.lower().strip()
    indicators = [
        "요약", "정리", "전체적으로", "전반적으로", "한번에", "묶어서", "파일별",
        "문서별", "각 문서", "각 파일",
        "summarize", "summary", "overview", "overall", "together",
        "by file", "each file", "each document",
    ]
    return any(keyword in lower for keyword in indicators)


def _needs_whole_corpus_full_context(text: str) -> bool:
    """
    Be much stricter for "all uploaded documents" mode.

    Whole-corpus loading is useful for file-by-file summary style requests, but it is
    too expensive for ordinary lookup/comparison questions and can cause GPU OOM with
    the local HF backend.
    """
    lower = text.lower().strip()
    if _needs_multi_document_summary_style(text):
        return True

    indicators = [
        "업로드된 문서 전체", "업로드된 전체", "모든 업로드", "전체 업로드",
        "all uploaded", "entire upload set", "whole upload corpus",
    ]
    if any(keyword in lower for keyword in indicators):
        return True
    # Count/enumeration questions need the intro section, not random top-k chunks
    return bool(re.search(r"총\s*몇|몇\s*가지|몇\s*항목|몇\s*개|몇\s*장|how\s+many", lower))


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
    if re.search(r"[가-힣]", user_message):
        return "해당 질문에는 답변할 수 없습니다."
    return "I can't answer that question based on the provided documents."


def _document_corpus_not_found_answer(user_message: str, source_type: Optional[str]) -> str:
    """Return a grounded fallback for scoped corpus search such as all uploads."""
    if re.search(r"[가-힣]", user_message):
        return "해당 질문에는 답변할 수 없습니다."
    return "I can't answer that question based on the provided documents."


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


def _extract_mention_candidate(text: str) -> Optional[str]:
    """Extract the entity/topic being asked about from a broad explanation query."""
    stripped = (text or "").strip()
    ko_patterns = [
        r"(.+?)에\s*대해서\s*(?:말해줘|설명해줘|알려줘)?$",
        r"(.+?)에\s*대해\s*(?:말해줘|설명해줘|알려줘)?$",
    ]
    en_patterns = [
        r"(?:who is|what is|tell me about|explain)\s+(.+?)\??$",
    ]

    for pattern in ko_patterns + en_patterns:
        match = re.search(pattern, stripped, flags=re.IGNORECASE)
        if match:
            candidate = re.sub(r"\s+", " ", match.group(1)).strip(" '\"?.,")
            if candidate:
                return candidate
    return None


def _find_mention_pages(docs, mention: str) -> List[int]:
    """Return pages where the requested mention appears in the document text."""
    pages = []
    needle = (mention or "").strip().lower()
    if not needle:
        return pages

    for doc in docs:
        haystack = _strip_enrichment_header(doc.page_content).lower()
        if needle in haystack:
            page = doc.metadata.get("page")
            if isinstance(page, int) and page not in pages:
                pages.append(page)
    return pages


def _build_mention_only_answer(
    user_message: str,
    active_source: Optional[str],
    mention: str,
    pages: List[int],
) -> str:
    """Return a safer answer when a document only mentions a term without explaining it."""
    source_name = active_source or "the uploaded document"
    page_text = ""
    if pages:
        page_text = f" (page {pages[0]})" if len(pages) == 1 else f" (pages {', '.join(str(page) for page in pages[:3])})"

    if re.search(r"[가-힣]", user_message):
        return (
            f"문서 '{source_name}'{page_text}에서 '{mention}'이 언급되지만, "
            "해당 대상에 대한 별도의 설명이나 정의는 제공되지 않습니다."
        )
    return (
        f"The document '{source_name}'{page_text} mentions '{mention}', "
        "but it does not provide a further explanation or definition."
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


def _extract_source_match_tokens(source: str) -> List[str]:
    stem = Path(source or "").stem
    raw_parts = re.split(r"[_\-\s()]+", stem)
    tokens: List[str] = []
    for part in raw_parts:
        normalized = _normalize_scope_text(part)
        if len(normalized) >= 2:
            tokens.append(normalized)
    return tokens


def _resolve_upload_doc_from_query(user_message: str) -> Optional[Dict[str, str]]:
    """
    If the user naturally names one uploaded file in the question, auto-scope to it.

    Example:
      "제칠일안식일예수재림교의 기본교리는 총 몇 개입니까?"
    should resolve to:
      "제칠일안식일예수재림교_기본교리.pdf"

    This makes upload-wide mode behave more like a user-scoped query when the
    document identity is already present in the text.
    """
    normalized_query = _normalize_scope_text(user_message)
    if len(normalized_query) < 4:
        return None

    candidates: List[Tuple[Tuple[int, int, int], Dict[str, Any]]] = []
    for doc in list_documents():
        if doc.get("source_type") != "upload":
            continue

        source = str(doc.get("source") or "").strip()
        if not source:
            continue

        source_stem = _normalize_scope_text(Path(source).stem)
        match_tokens = [
            token for token in _extract_source_match_tokens(source)
            if token and token in normalized_query
        ]

        full_stem_match = bool(source_stem and source_stem in normalized_query)
        if not full_stem_match and len(match_tokens) < 2:
            continue

        score = (
            1 if full_stem_match else 0,
            len(match_tokens),
            max((len(token) for token in match_tokens), default=0),
        )
        candidates.append((score, doc))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0], reverse=True)
    best_score, best_doc = candidates[0]
    if len(candidates) > 1 and candidates[1][0] == best_score:
        return None

    return {
        "source": str(best_doc.get("source") or ""),
        "doc_id": str(best_doc.get("doc_id") or ""),
    }


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


def _contains_excessive_english(text: str) -> bool:
    """True when the response contains ≥ 10 consecutive ASCII-only words.

    This detects English sentences/paragraphs (code-switching) while tolerating
    legitimate English proper nouns like 'Seventh-day Adventists', 'NEWSTART',
    or 'General Conference' that naturally appear in Korean theological text.
    Hyphens, slashes, and common punctuation are treated as separators so
    hyphenated terms like 'Seventh-day' count as two words, not one long run.
    """
    run = 0
    for token in re.split(r"[\s\-_/,.;:!?()\[\]\"\']+", text or ""):
        if token and re.fullmatch(r"[A-Za-z]+", token):
            run += 1
            if run >= 10:
                return True
        elif token:  # non-empty non-ASCII token resets the consecutive run
            run = 0
    return False


def _needs_language_retry(user_message: str, answer: str) -> bool:
    """Retry when the model drifted into Chinese OR into excessive English for a Korean query."""
    target_lang = _infer_target_language(user_message)
    if target_lang not in {"ko", "en"}:
        return False
    if _contains_chinese(answer):
        return True
    if target_lang == "ko" and _contains_excessive_english(answer):
        return True
    return False


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
        "가사 추출", "가사 보여", "가사 읽어", "lyrics", "lyric",
        "what does this image say", "what does the image say",
        "give me the text", "extract the text", "read the text",
        "read this image", "text in the image", "transcribe",
    ]
    return any(keyword in lower for keyword in indicators)


def _normalize_scope_text(text: str) -> str:
    return re.sub(r"[^0-9a-z가-힣]", "", (text or "").lower())


_GUIDELINE_TITLE_PATTERN = re.compile(
    r"운영\s*지침|규정|지침\(안\)|지침안|운영\s*규정",
    re.IGNORECASE,
)


def _extract_scope_labels(docs) -> List[str]:
    """Extract distinct sub-guideline labels from a combined uploaded document.

    Returns an empty list when none of the section titles look like bundled
    sub-guideline titles (i.e. regular chapter-structured documents like books
    or single-subject PDFs are not treated as multi-guideline bundles).
    """
    # First pass: collect raw titles and check if any look like guideline titles.
    raw_titles = [
        str(doc.metadata.get("section_title") or "").strip()
        for doc in docs
        if doc.metadata.get("section_title")
    ]
    if not any(_GUIDELINE_TITLE_PATTERN.search(t) for t in raw_titles):
        return []

    labels = []
    seen = set()
    for raw_title in raw_titles:
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


def _needs_strict_fact_style(text: str) -> bool:
    """Detect exact-lookup questions where numbers/titles/dates must stay literal."""
    lower = text.lower().strip()
    indicators = [
        "몇 개", "몇장", "몇 장", "총 몇", "몇 항", "몇 조", "언제", "연도", "년도", "제목",
        "명칭", "이름", "처음", "최초", "어디", "장소", "추가된", "추가", "몇 명", "몇인",
        "what year", "when", "how many", "how much", "which title", "title", "name",
        "first", "where", "added", "count", "number of",
    ]
    return any(keyword in lower for keyword in indicators)


_STRICT_FACT_STOP_TOKENS = {
    "몇", "개", "장", "조", "항", "명", "인", "총", "언제", "연도", "년도", "제목",
    "명칭", "이름", "처음", "최초", "어디", "장소", "추가", "added", "count",
    "number", "of", "how", "many", "what", "year", "when", "where", "title", "name",
    "is", "are", "the",
}


def _normalize_fact_query_tokens(text: str) -> List[str]:
    tokens: List[str] = []
    seen = set()
    for token in tokenize_text(text):
        candidates = {token}
        if len(token) >= 3 and re.match(r"[가-힣]+$", token):
            candidates.add(re.sub(r"[은는이가의를을와과도만로으로에에서께]|입니다|입니까|인가요|인가|인가\??$", "", token))
        for candidate in candidates:
            candidate = candidate.strip()
            if len(candidate) < 2 or candidate in _STRICT_FACT_STOP_TOKENS:
                continue
            if candidate not in seen:
                seen.add(candidate)
                tokens.append(candidate)
    return tokens


def _is_count_question(text: str) -> bool:
    """Return True only for document-total count questions (not historical counts).

    A query containing a specific year (e.g. "1872년에는 몇 개") is asking
    about a historical fact, not the document's own total — skip deterministic
    count extraction and let the LLM answer from retrieved context.
    """
    lower = (text or "").lower()
    indicators = ["몇 개", "총 몇", "number of", "how many", "count"]
    if not any(keyword in lower for keyword in indicators):
        return False
    # Specific year present → historical/contextual question, not a total count
    # \b fails on Korean (년 is treated as \w in Python Unicode mode) — use (?<!\d) instead
    if re.search(r"(?<!\d)(1[0-9]{3}|20[0-9]{2})년", text):
        return False
    return True


def _looks_like_doctrine_count_query(user_message: str, active_source: Optional[str]) -> bool:
    lower = (user_message or "").lower()
    source_lower = (active_source or "").lower()
    indicators = ["기본교리", "교리", "belief", "beliefs", "doctrine", "doctrines"]
    return any(term in lower for term in indicators) or any(term in source_lower for term in indicators)


def _try_scoped_chapter_count_answer(
    user_message: str,
    docs,
    active_source: Optional[str],
) -> Optional[str]:
    """
    Deterministically answer count questions from explicit chapter numbering.

    Example:
      제1장 ... 제28장
    in a table of contents or chapter headings should allow answering
    "기본교리는 총 몇 개입니까?" without relying on free-form generation.
    """
    if not _is_count_question(user_message):
        return None
    if not _looks_like_doctrine_count_query(user_message, active_source):
        return None

    chapter_pages: Dict[int, int] = {}
    toc_signal = False
    for doc in docs:
        text = _strip_enrichment_header(doc.page_content)
        if not text:
            continue
        page = int(doc.metadata.get("page") or 0)
        compact = re.sub(r"\s+", " ", text)
        lowered = compact.lower()
        if any(term in lowered for term in ("contents", "차례")):
            toc_signal = True
        for match in re.finditer(r"제\s*(\d+)\s*장", compact):
            chapter_no = int(match.group(1))
            if 0 < chapter_no <= 300:
                chapter_pages[chapter_no] = min(page, chapter_pages.get(chapter_no, page))
        for match in re.finditer(r"\bchapter\s+(\d+)\b", compact, flags=re.IGNORECASE):
            chapter_no = int(match.group(1))
            if 0 < chapter_no <= 300:
                chapter_pages[chapter_no] = min(page, chapter_pages.get(chapter_no, page))

    if not chapter_pages:
        return None

    max_chapter = max(chapter_pages)
    consecutive = sum(1 for i in range(1, max_chapter + 1) if i in chapter_pages)
    coverage = consecutive / max(max_chapter, 1)

    if max_chapter < 5:
        return None
    if 1 not in chapter_pages:
        return None

    # OCR-heavy contents pages often miss some chapter labels in the middle,
    # but if the doc clearly shows a table of contents and reaches a high
    # chapter number, that is still stronger evidence than a stray "기본교리 27"
    # title mention on an earlier page.
    if toc_signal:
        if max_chapter < 20 or len(chapter_pages) < 8:
            return None
    elif coverage < 0.75:
        return None

    if re.search(r"[가-힣]", user_message):
        page_note = f" 차례 기준으로 문서 {chapter_pages[max_chapter]}페이지에서 제{max_chapter}장까지 확인됩니다." if chapter_pages.get(max_chapter) else ""
        return f"문서에 따르면 기본교리는 총 {max_chapter}개입니다.{page_note}"

    page_note = (
        f" The table of contents/chapter structure shows up to Chapter {max_chapter} around page {chapter_pages[max_chapter]}."
        if chapter_pages.get(max_chapter) else ""
    )
    return f"According to the document, there are {max_chapter} core items in total.{page_note}"


def _strict_fact_chunk_score(user_message: str, doc) -> float:
    text = _strip_enrichment_header(doc.page_content)
    normalized = re.sub(r"\s+", " ", text).strip().lower()
    if not normalized:
        return 0.0

    tokens = _normalize_fact_query_tokens(user_message)
    score = 0.0
    for token in tokens:
        if token in normalized:
            score += 3.0 if len(token) >= 4 else 1.5

    if _is_count_question(user_message):
        if re.search(r"\b\d+\s*개\s*항목", normalized):
            score += 8.0
        elif re.search(r"\b총\s*\d+\s*개", normalized):
            score += 7.0
        elif re.search(r"\b\d+\s*개", normalized) and any(
            term in normalized for term in ("기본교리", "신조", "belief", "beliefs")
        ):
            score += 5.0

    if any(term in user_message for term in ("언제", "연도", "년도", "처음", "최초")):
        if re.search(r"\b(17|18|19|20)\d{2}\b", normalized):
            score += 4.0

    if any(term in user_message for term in ("제목", "명칭", "이름", "title", "name")):
        if "제목" in normalized or "title" in normalized or "“" in text or '"' in text:
            score += 3.0

    page = int(doc.metadata.get("page") or 0)
    if 0 < page <= 20:
        score += 1.0

    return score


def _select_strict_fact_docs(user_message: str, docs, front_chunks: int = 2, top_chunks: int = 4):
    """
    For huge single-document strict-fact queries, keep front-matter plus the most
    query-relevant fact-bearing chunks instead of sending the entire document and
    losing evidence to later context trimming.
    """
    # History/date/place queries need deeper front-matter coverage (preface, TOC, intro).
    # front_chunks=8 covers the first ~10 pages which include publication history (page 8-9).
    if any(kw in user_message for kw in ("언제", "연도", "처음", "최초", "어디", "장소", "결의", "채택", "공식", "대총회")):
        front_chunks = max(front_chunks, 8)
    # Explicit year in query → publication-history chunk is on page 8 (index 5 in sorted order)
    if re.search(r"(?<!\d)(1[0-9]{3}|20[0-9]{2})년", user_message):
        front_chunks = max(front_chunks, 8)

    if len(docs) <= (front_chunks + top_chunks):
        return docs

    ordered_docs = sorted(
        docs,
        key=lambda d: (
            int(d.metadata.get("page") or 0),
            int(d.metadata.get("chunk_index") or 0),
        ),
    )

    selected: List[Any] = []
    seen_keys = set()

    def _doc_key(doc) -> Tuple[Any, Any, Any]:
        return (
            doc.metadata.get("doc_id") or doc.metadata.get("source"),
            int(doc.metadata.get("page") or 0),
            int(doc.metadata.get("chunk_index") or 0),
        )

    for doc in ordered_docs[:front_chunks]:
        key = _doc_key(doc)
        if key not in seen_keys:
            seen_keys.add(key)
            selected.append(doc)

    scored_docs = sorted(
        ordered_docs,
        key=lambda d: (
            _strict_fact_chunk_score(user_message, d),
            -int(d.metadata.get("page") or 0),
        ),
        reverse=True,
    )

    for doc in scored_docs:
        if len(selected) >= front_chunks + top_chunks:
            break
        key = _doc_key(doc)
        if key in seen_keys:
            continue
        if _strict_fact_chunk_score(user_message, doc) <= 0:
            continue
        seen_keys.add(key)
        selected.append(doc)

    return sorted(
        selected,
        key=lambda d: (
            int(d.metadata.get("page") or 0),
            int(d.metadata.get("chunk_index") or 0),
        ),
    ) or ordered_docs[: front_chunks + top_chunks]


def _try_scoped_count_answer(user_message: str, docs, active_source: Optional[str]) -> Optional[str]:
    """
    Deterministically answer simple "총 몇 개" style questions when a strong
    count phrase is present in the scoped document.
    """
    if not _is_count_question(user_message):
        return None

    chapter_answer = _try_scoped_chapter_count_answer(user_message, docs, active_source)
    if chapter_answer:
        return chapter_answer

    query_terms = _normalize_fact_query_tokens(user_message)
    candidates: List[Tuple[float, int, int, str]] = []
    patterns = [
        re.compile(r"총\s*(\d+)\s*개"),
        re.compile(r"(\d+)\s*개\s*항목"),
        re.compile(r"(\d+)\s*개"),
    ]

    for doc in docs:
        text = _strip_enrichment_header(doc.page_content)
        if not text:
            continue
        compact = re.sub(r"\s+", " ", text)
        lowered = compact.lower()
        for pattern in patterns:
            for match in pattern.finditer(compact):
                number = int(match.group(1))
                if number <= 0 or number > 500:
                    continue
                start = max(0, match.start() - 120)
                end = min(len(compact), match.end() + 120)
                window = compact[start:end]
                window_lower = lowered[start:end]

                score = 0.0
                if "항목" in window:
                    score += 6.0
                if any(term in window_lower for term in ("기본교리", "신조", "belief", "beliefs")):
                    score += 5.0
                score += sum(2.0 for token in query_terms if token in window_lower)

                page = int(doc.metadata.get("page") or 0)
                if 0 < page <= 20:
                    score += 1.0

                if score >= 8.0:
                    candidates.append((score, number, page, window.strip()))

    if not candidates:
        return None

    best_score, best_number, best_page, _ = sorted(
        candidates,
        key=lambda item: (item[0], item[1], -item[2]),
        reverse=True,
    )[0]
    if best_score < 8.0:
        return None

    if re.search(r"[가-힣]", user_message):
        page_note = f" 문서 {best_page}페이지 기준입니다." if best_page > 0 else ""
        return f"문서에 따르면 기본교리는 총 {best_number}개입니다.{page_note}"

    page_note = f" This is stated on page {best_page} of the document." if best_page > 0 else ""
    return f"According to the document, there are {best_number} core items in total.{page_note}"


def _parse_toc_from_docs(docs) -> Dict[int, str]:
    """Extract {chapter_no: title} from TOC or chapter-heading chunks.

    Handles two OCR-parsed formats:
      TOC line:  제 1 장  성  경 ·····  5
      Heading:   제1장\\n성경
    Uses setdefault so the first occurrence (TOC) wins over later headings.
    """
    mapping: Dict[int, str] = {}
    for doc in docs:
        text = _strip_enrichment_header(doc.page_content)
        # Collapse tabs/ideographic spaces but keep newlines for heading pattern
        compact = re.sub(r"[ \t　]+", " ", text)

        # Shared dot/noise terminator class (covers ., ·, … U+2026, ‥ U+2025)
        _DOT_TERM = r"[·\.…‥]"

        # Pattern A: 제N장 <Korean title> then dots / ASCII noise / newline / end
        # Notes:
        #   • \s* not \s+ — OCR sometimes drops the space between 장 and title
        #   • [\[\(]? handles OCR bracket artifacts like 제[23장
        #   • \s*[a-z] lookahead catches ASCII OCR noise after title ("말씀wee eee...")
        #   • ‥ (U+2025) added alongside … (U+2026) in dot terminator
        for m in re.finditer(
            r"(?:^|\n)\s*제\s*[\[\(]?\s*(\d{1,3})\s*[\]\)]?\s*장\s*([가-힣][가-힣 ]{1,30}?)"
            r"(?=\s*" + _DOT_TERM + r"+|\s*\d{1,3}\s*\n|\s*\n|\s*[a-z]|\s*$)",
            compact,
        ):
            no = int(m.group(1))
            title = re.sub(r"[· \t\.…‥\u3131-\u318E]+$", "", m.group(2)).strip()
            if title and 2 <= len(title) <= 30 and re.search(r"[가-힣]", title):
                mapping.setdefault(no, title)

        # Pattern B: 제N장 immediately followed by newline then title
        for m in re.finditer(
            r"제\s*[\[\(]?\s*(\d{1,3})\s*[\]\)]?\s*장\s*\n\s*([가-힣][가-힣 ]{1,30}?)(?:\s*\n|$)",
            compact,
        ):
            no = int(m.group(1))
            title = m.group(2).strip()
            if title and re.search(r"[가-힣]", title):
                mapping.setdefault(no, title)

        # Pattern C: body running header "N. 제목" at line start (page footer format)
        # e.g. "171. 하나님의 말씀" → chapter 1, title 하나님의 말씀
        # Only applied for small chapter numbers (1-30) to avoid false positives
        for m in re.finditer(
            r"(?:^|\n)\s*(\d{1,2})\.\s+([가-힣][가-힣 ]{2,25}?)(?:\s*\n|$)",
            compact,
        ):
            no = int(m.group(1))
            title = m.group(2).strip()
            # Require at least 3 Korean syllables to filter out footnote refs like "참조"
            korean_chars = re.sub(r"[^가-힣]", "", title)
            if 1 <= no <= 30 and len(korean_chars) >= 3:
                mapping.setdefault(no, title)

    return mapping


def _try_chapter_title_lookup(user_message: str, docs) -> Optional[str]:
    """Deterministically answer chapter-title and chapter-number queries.

    Handles:
      "제28장의 제목은?" → look up chapter 28 → return its title
      "안식일은 몇 장?" / "새 땅은 몇 장?" → reverse-lookup keyword → return chapter number
    """
    korean = bool(re.search(r"[가-힣]", user_message))

    # Type 1: chapter number → title  ("제28장의 제목/이름은?", "제25장은 무엇을 다룹니까?")
    # (?:[은는이]?\s*)? handles the Korean topic particle between 장 and the question word
    m = re.search(
        r"제\s*(\d{1,3})\s*장\s*(?:[은는이]\s*)?(?:의\s*)?(?:제목|이름|명칭|무엇|뭐|what)",
        user_message,
        re.IGNORECASE,
    )
    if m:
        target = int(m.group(1))
        toc = _merge_chapter_maps(_parse_toc_from_docs(docs), _parse_body_chapter_map_from_docs(docs))
        if target in toc:
            title = toc[target]
            return f"제{target}장의 제목은 「{title}」입니다." if korean else f"Chapter {target} is titled '{title}'."
        return None

    # Type 2: keyword → chapter number  ("안식일은 몇 장에 나옵니까?", "새 땅은 몇 장?")
    # Use lookahead to stop at the trailing particle so "안식일은" captures "안식일",
    # and allow internal spaces so "새 땅" (two syllables with space) is captured correctly.
    m = re.search(
        r"([가-힣][가-힣 ]*?[가-힣])(?=(?:은|는|이|가)?\s*몇\s*장)",
        user_message,
    )
    if not m:
        return None
    keyword = m.group(1).strip()
    toc = _merge_chapter_maps(_parse_toc_from_docs(docs), _parse_body_chapter_map_from_docs(docs))
    hits = [(no, title) for no, title in sorted(toc.items()) if keyword in title]
    if len(hits) == 1:
        no, title = hits[0]
        return f"「{title}」은(는) 제{no}장에 나옵니다."
    if len(hits) > 1:
        items = ", ".join(f"제{no}장 {title}" for no, title in hits)
        return f"'{keyword}'와 관련된 장: {items}입니다."
    return None


# ── Bucket 2: body chapter map (supplements garbled TOC entries) ────────────

def _parse_body_chapter_map_from_docs(docs) -> Dict[int, str]:
    """Find chapter titles from section_title metadata for chapters missing in the TOC.

    Steps:
      1. Extract chapter_no → start_page from TOC lines (even garbled ones have the page number).
      2. Build page → first_section_title from the actual chunk metadata.
      3. For each chapter's start page, normalize the section_title (remove OCR spaces).
    """
    # Step 1: chapter_no -> start_page from TOC
    toc_chapter_pages: Dict[int, int] = {}
    for doc in docs:
        pg = int(doc.metadata.get("page") or 0)
        if pg > 20:
            continue
        text = _strip_enrichment_header(doc.page_content)
        compact = re.sub(r"[ \t　]+", " ", text)
        for m in re.finditer(
            r"(?:^|\n)\s*제\s*[\[\(]?\s*(\d{1,3})\s*장[^\n]*?(\d{2,3})\s*(?:\n|$)",
            compact,
        ):
            no, start_pg = int(m.group(1)), int(m.group(2))
            if 0 < no <= 30 and 10 <= start_pg <= 600:
                toc_chapter_pages.setdefault(no, start_pg)

    def _is_likely_chapter_title(title: str) -> bool:
        korean = re.sub(r"[^가-힣]", "", title or "")
        if len(korean) < 2 or len(korean) > 25:
            return False
        noise_terms = ("참고문헌", "서문", "차례", "contents")
        return not any(term in title.lower() for term in noise_terms)

    def _page_title_score(title: str, page_text: str) -> Tuple[int, int]:
        korean = re.sub(r"[^가-힣]", "", title or "")
        starts_page = 0 if page_text.startswith(title) else 1
        length_penalty = abs(len(korean) - 4)
        return (starts_page, length_penalty)

    # Step 2: page -> best clean section_title candidate
    ordered = sorted(
        docs,
        key=lambda d: (int(d.metadata.get("page") or 0), int(d.metadata.get("chunk_index") or 0)),
    )
    page_titles: Dict[int, List[str]] = {}
    page_texts: Dict[int, str] = {}
    for doc in ordered:
        pg = int(doc.metadata.get("page") or 0)
        page_texts.setdefault(pg, _strip_enrichment_header(doc.page_content).strip())
        st = str(doc.metadata.get("section_title") or "").strip()
        if st:
            normalized = re.sub(r"\s+", "", st)  # "침 례" → "침례"
            if _is_likely_chapter_title(normalized):
                page_titles.setdefault(pg, [])
                if normalized not in page_titles[pg]:
                    page_titles[pg].append(normalized)

    page_to_first_title: Dict[int, str] = {}
    for pg, titles in page_titles.items():
        page_text = re.sub(r"\s+", "", page_texts.get(pg, ""))
        ranked = sorted(titles, key=lambda title: _page_title_score(title, page_text))
        if ranked:
            page_to_first_title[pg] = ranked[0]

    # Step 3: map chapter_no -> section_title via exact/nearby start page
    result: Dict[int, str] = {}
    for no, start_pg in toc_chapter_pages.items():
        for offset in range(4):
            title = page_to_first_title.get(start_pg + offset)
            if title and _is_likely_chapter_title(title):
                result.setdefault(no, title)
                break

    # Step 4: infer missing chapter titles from gaps between known chapter starts.
    # This recovers OCR-missed TOC entries like chapter 12 "교회" and chapter 15 "침례"
    # by picking the best title-bearing body page between neighboring known starts.
    known = sorted((no, pg) for no, pg in toc_chapter_pages.items())
    # Never reuse a page already identified as a chapter start in the TOC page map.
    used_pages = set(toc_chapter_pages.values())
    candidate_pages = [
        (pg, title)
        for pg, title in sorted(page_to_first_title.items())
        if _is_likely_chapter_title(title)
    ]
    for (prev_no, prev_pg), (next_no, next_pg) in zip(known, known[1:]):
        missing = next_no - prev_no - 1
        if missing <= 0:
            continue
        between = [
            (pg, title)
            for pg, title in candidate_pages
            if prev_pg < pg < next_pg and pg not in used_pages
        ]
        if len(between) < missing:
            continue
        chosen: List[Tuple[int, str]] = []
        remaining = list(between)
        span = next_pg - prev_pg
        for offset in range(1, missing + 1):
            expected_pg = prev_pg + (span * offset / (missing + 1))
            remaining.sort(
                key=lambda item: (
                    abs(item[0] - expected_pg),
                    len(re.sub(r"[^가-힣]", "", item[1])),
                ),
            )
            pg, title = remaining.pop(0)
            chosen.append((pg, title))
        chosen.sort(key=lambda item: item[0])
        for offset, (pg, title) in enumerate(chosen, start=1):
            result.setdefault(prev_no + offset, title)
            used_pages.add(pg)

    return result


def _merge_chapter_maps(toc_map: Dict[int, str], body_map: Dict[int, str]) -> Dict[int, str]:
    """Merge TOC-extracted and body-extracted chapter maps. TOC entries take precedence."""
    merged = dict(body_map)
    merged.update(toc_map)
    return merged


# ── Bucket 1: publication history lookup ────────────────────────────────────

_HISTORY_INDICATORS = [
    "추가", "결의", "출간", "출판", "한국어",
    "번역", "개정", "항목", "몇 개", "제목", "명칭",
]


def _front_matter_text(docs, max_page: int = 15) -> str:
    parts: List[str] = []
    for doc in sorted(
        docs,
        key=lambda d: (int(d.metadata.get("page") or 0), int(d.metadata.get("chunk_index") or 0)),
    ):
        if int(doc.metadata.get("page") or 0) > max_page:
            continue
        parts.append(_strip_enrichment_header(doc.page_content))
    return re.sub(r"\s+", " ", " ".join(parts)).strip()


def _parse_publication_history_from_docs(docs) -> Dict[str, List[str]]:
    """Extract year → [sentences] from front-matter chunks (pages 1–15).

    Captures sentences that mention years in the range 1800–2050 (avoids
    false positives from biblical verse references like 창 1:1).
    """
    result: Dict[str, List[str]] = {}
    for doc in docs:
        if int(doc.metadata.get("page") or 0) > 15:
            continue
        text = _strip_enrichment_header(doc.page_content)
        # Split into rough sentences at Korean full stops and newlines
        for sent in re.split(r"(?<=[다.！？\n])\s*", text):
            sent = re.sub(r"\s+", " ", sent).strip()
            if len(sent) < 10:
                continue
            for m in re.finditer(r"(?<!\d)(1[89]\d{2}|20\d{2})(?!\d)", sent):
                year = m.group(1)
                result.setdefault(year, [])
                if sent not in result[year]:
                    result[year].append(sent)
    return result


def _try_history_lookup(user_message: str, docs) -> Optional[str]:
    """Deterministically answer publication-history questions.

    Fires when the query contains a year (1800–2050) AND a history keyword.
    Returns the most relevant sentence(s) from the front-matter verbatim.
    """
    year_m = re.search(r"(?<!\d)(1[89]\d{2}|20\d{2})(?!\d)", user_message)
    if not year_m:
        return None
    lower = user_message.lower()
    if not any(kw in lower for kw in _HISTORY_INDICATORS):
        return None

    year = year_m.group(1)
    history = _parse_publication_history_from_docs(docs)
    sentences = history.get(year, [])
    if not sentences:
        return None

    query_tokens = _normalize_fact_query_tokens(user_message)
    ranked = sorted(
        sentences,
        key=lambda s: sum(1 for t in query_tokens if t in s.lower()),
        reverse=True,
    )
    best = " ".join(ranked[:2])
    korean = bool(re.search(r"[가-힣]", user_message))
    return f"문서에 따르면, {best}" if korean else f"According to the document: {best}"



_OUT_OF_SCOPE_TERMS_RE = re.compile(
    r"\bRAG\b|파인튜닝|벡터\s*(?:DB|데이터베이스)?|임베딩\s*모델?|(?<![가-힣])LLM(?![가-힣])|ChatGPT|GPT[-\s]?\d|랭체인|langchain|llama\b",
    re.IGNORECASE,
)


def _is_clearly_out_of_scope(user_message: str, docs) -> bool:
    """True when the query contains modern tech/ML terms clearly absent from the document.

    Samples the first 50 chunks (front-matter + early body) for the same terms.
    If none appear in the document, the question cannot be answered from it.
    """
    if not _OUT_OF_SCOPE_TERMS_RE.search(user_message):
        return False
    sample = " ".join(_strip_enrichment_header(d.page_content)[:300] for d in docs[:50])
    return not _OUT_OF_SCOPE_TERMS_RE.search(sample)


def _strip_enrichment_header(text: str) -> str:
    """Remove enrichment header prepended before embedding/retrieval."""
    return re.sub(r'^\[Document:.*?\]\n', '', text or '', flags=re.DOTALL).strip()


def _split_sentences(text: str) -> List[str]:
    normalized = re.sub(r"\s+", " ", (text or "")).strip()
    if not normalized:
        return []
    parts = re.split(r"(?<=[.!?。])\s+|(?<=다\.)\s+|(?<=요\.)\s+|\n+", normalized)
    return [part.strip() for part in parts if part.strip()]


def _sentence_quality_score(text: str) -> float:
    normalized = re.sub(r"\s+", " ", (text or "")).strip()
    if not normalized:
        return 0.0

    good_chars = sum(
        1
        for ch in normalized
        if ch.isalnum() or ch.isspace() or ch in ".,:;!?()[]-/%'\"·"
    )
    hangul_or_alpha = sum(
        1
        for ch in normalized
        if re.match(r"[A-Za-z가-힣]", ch)
    )
    return (good_chars / max(len(normalized), 1)) * 0.5 + (hangul_or_alpha / max(len(normalized), 1)) * 0.5


def _clean_ocr_sentence(text: str) -> str:
    """
    Clean obvious OCR/document-header noise conservatively.

    Important: if cleaning becomes too destructive, keep the original content.
    """
    original = re.sub(r"\s+", " ", (text or "")).strip()
    if not original:
        return ""

    cleaned = original
    cleaned = re.sub(r"^[\s<>\[\]\(\)\-_=+*#~|\\/.,:;]+", "", cleaned)
    cleaned = re.sub(r"^\d+(?:\s*[.)>-]\s*|\s+)", "", cleaned)
    cleaned = re.sub(r"^[0-9.\s]+<[^>]{0,80}>\s*", "", cleaned)
    cleaned = re.sub(r"^(?:제정|개정|시행)\s*\d{4}[./-]\d{1,2}[./-]\d{1,2}\.?\s*", "", cleaned)
    cleaned = re.sub(r"^(?:규칙\s*제\d+호|제\d+호)\s*", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -:;,.")

    if len(cleaned) < 20:
        return original

    if _sentence_quality_score(cleaned) + 0.08 < _sentence_quality_score(original):
        return original

    return cleaned


def _collect_sections_for_docs(docs, limit: int = 5) -> List[str]:
    sections: List[str] = []
    seen = set()
    for doc in docs:
        section = str(doc.metadata.get("section_breadcrumb") or doc.metadata.get("section_title") or "").strip()
        if not section:
            continue
        if section in seen:
            continue
        seen.add(section)
        sections.append(section)
        if len(sections) >= limit:
            break
    return sections


def _collect_representative_sentences(docs, limit: int = 2) -> List[str]:
    sentences: List[str] = []
    seen = set()
    fallback_sentences: List[str] = []
    for doc in docs:
        text = _strip_enrichment_header(doc.page_content)
        for sentence in _split_sentences(text):
            original = sentence.strip()
            if len(original) < 20:
                continue
            cleaned = _clean_ocr_sentence(original)
            candidate = cleaned or original
            key = re.sub(r"\s+", " ", candidate)
            if key in seen:
                continue
            seen.add(key)
            quality = _sentence_quality_score(candidate)
            if quality >= 0.55:
                sentences.append(candidate)
            else:
                fallback_sentences.append(candidate)
            if len(sentences) >= limit:
                return sentences

    if len(sentences) < limit:
        for candidate in fallback_sentences:
            if candidate not in sentences:
                sentences.append(candidate)
            if len(sentences) >= limit:
                break

    return sentences


def _build_file_level_corpus_summary(user_message: str, docs) -> str:
    grouped: Dict[str, List[Any]] = {}
    order: List[str] = []
    for doc in docs:
        source = str(doc.metadata.get("source") or "unknown")
        if source not in grouped:
            grouped[source] = []
            order.append(source)
        grouped[source].append(doc)

    is_korean = bool(re.search(r"[가-힣]", user_message))
    lines: List[str] = []
    if is_korean:
        lines.append("업로드된 문서별 요약")
    else:
        lines.append("Uploaded file summaries")

    for source in order:
        source_docs = grouped[source]
        sections = _collect_sections_for_docs(source_docs)
        sentences = _collect_representative_sentences(source_docs)

        if is_korean:
            lines.append(f"\n### 파일명: {source}")
            if sentences:
                lines.append(f"- 요약: {sentences[0]}")
            if sections:
                lines.append(f"- 주요 섹션: {', '.join(sections)}")
            if len(sentences) > 1:
                lines.append(f"- 대표 내용: {sentences[1]}")
        else:
            lines.append(f"\n### File: {source}")
            if sentences:
                lines.append(f"- Summary: {sentences[0]}")
            if sections:
                lines.append(f"- Main sections: {', '.join(sections)}")
            if len(sentences) > 1:
                lines.append(f"- Representative content: {sentences[1]}")

    if is_korean:
        lines.append(
            "\n### 전체 요약(선택 사항)\n"
            "업로드 문서에는 여러 운영지침·규정·참고자료가 함께 포함되어 있을 수 있으므로, 서로 다른 성격의 문서는 구분해서 해석하는 것이 좋습니다."
        )
    else:
        lines.append(
            "\n### Overall summary (optional)\n"
            "The uploaded set may contain a mix of guidelines, regulations, and reference materials, so unrelated files should be interpreted separately."
        )

    return "\n".join(lines).strip()


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

# Compact system prompt for local_hf messages — matches finetuning/train.py
# DEFAULT_SYSTEM_PROMPT + build_document_system_message rules exactly.
# Keeps the messages system token count close to training (~200 tokens) so
# retrieved doc evidence is not evicted by the head/tail trim in llm.py.
_COMPACT_SYSTEM_PROMPT = (
    "너는 한국어로 답하는 AI 챗봇이다. "
    "짧은 질문에는 짧고 자연스럽게 답한다. "
    "문서 질문은 문서 근거로만 답하고, 근거가 없으면 모른다고 말한다.\n\n"
    "답변 규칙:\n"
    "1. CRITICAL: Respond in the SAME language the user is using. "
    "Korean→Korean, English→English. NEVER respond in Chinese (中文).\n"
    "2. 제공된 문서 문맥만 근거로 답한다.\n"
    "3. 문서에 없는 내용은 추측하지 않는다.\n"
    '4. 문맥이 부족하면 "해당 내용은 제공된 문서에서 확인되지 않습니다."라고 답한다.\n'
    "5. 핵심 답변을 먼저 말한다.\n"
    "6. 가능하면 페이지와 문서를 근거로 설명한다.\n"
    "7. 이미지에서 추출된 텍스트가 제공되면 해당 텍스트를 기반으로 답한다."
)

# Character budget for doc context in local_hf messages.
# 3072 token limit (12 GB GPU) − ~115 compact sys − ~100 question/format − ~200 history
# = ~2657 tokens for doc evidence.  At ~3.2 chars/token for Korean ≈ 8500 chars.
# This fits ~7 full 1200-char chunks, covering all VECTOR_TOP_K=6 results with headroom.
_LOCAL_HF_DOC_CONTEXT_CHAR_LIMIT = 8500


def _trim_doc_context_for_local_hf(doc_context: str) -> str:
    """Trim retrieved doc context by dropping trailing chunk blocks to stay within
    the local_hf token budget.

    Preserves complete [Doc: ...] header + content blocks so the LLM never sees a
    half-chunk.  Drops the lowest-priority (last-ranked) chunks rather than cutting
    through a chunk at an arbitrary byte boundary.
    """
    if not doc_context or len(doc_context) <= _LOCAL_HF_DOC_CONTEXT_CHAR_LIMIT:
        return doc_context

    blocks = re.split(r'(?=\[Doc: )', doc_context)
    kept: List[str] = []
    total = 0
    for block in blocks:
        stripped = block.strip()
        if not stripped:
            continue
        if total + len(stripped) > _LOCAL_HF_DOC_CONTEXT_CHAR_LIMIT and kept:
            break
        kept.append(stripped)
        total += len(stripped)

    trimmed = "\n\n".join(kept)
    n_dropped = sum(1 for b in blocks if b.strip()) - len(kept)
    if n_dropped > 0:
        trimmed += f"\n\n[Note: {n_dropped} lower-ranked chunk(s) omitted to fit context window]"
    return trimmed


def _build_prompt(
    user_message: str,
    history: List[Message],
    doc_context: str = "",
    web_context: str = "",
    system_prompt: str = "",
    selected_doc_count: int = 0,
    document_scope_count: int = 0,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Build a unified prompt and matching structured messages.

    Returns:
        (prompt_string, messages) — prompt_string is the flat string used by
        Ollama; messages is the [system, user] list fed to the local_hf backend
        via apply_chat_template, matching the system+user split used in
        finetuning/train.py (render_document_chat).
    """
    # ── System-role content (instructions only) ──────────────────────────
    sys_parts: List[str] = [f"[System]\n{system_prompt or _SYSTEM_PROMPT}"]
    target_lang = _infer_target_language(user_message)
    if target_lang == "ko":
        sys_parts.append("[Required response language]\nKorean only. Do not use Chinese characters.")
    elif target_lang == "en":
        sys_parts.append(
            "[Required response language]\nEnglish only. Do not use Chinese characters.\n"
            "[Cross-language instruction]\n"
            "If the retrieved document evidence is in Korean, translate and explain it in English. "
            "Do not refuse only because the source document is in another language."
        )
    if _needs_section_understanding_style(user_message):
        sys_parts.append(
            "[Task style]\n"
            "Answer as a section-understanding question. "
            "Explain the role, purpose, differences, or kinds of support described in the document. "
            "Name concrete items from the evidence instead of giving only a generic summary."
        )
    if document_scope_count > 1 and _needs_multi_document_summary_style(user_message) and not _needs_comparison_style(user_message):
        sys_parts.append(
            "[Multi-document summary instruction]\n"
            "You are answering a multi-document summary question.\n"
            "Summarize the retrieved documents file by file before giving any overall theme.\n"
            "You MUST create exactly one summary block per distinct file in the retrieved context.\n"
            "You MUST cover every distinct file that appears in the retrieved context.\n"
            "Use the actual filename or title as the heading for each block.\n"
            "Do not split one file into multiple headings just because it has several sections or clauses.\n"
            "Do not repeat the same file under slightly different names.\n"
            "Keep each file summary short and concrete.\n"
            "For each file, mention only:\n"
            "1. what the document is about,\n"
            "2. the most important points,\n"
            "3. anything notably different or special.\n"
            "Do not merge unrelated documents into one topic.\n"
            "If some documents are unrelated, say that clearly.\n"
            "Do not invent broad common themes unless multiple documents clearly support them.\n"
            "Only include a final overall summary if it is genuinely useful.\n"
            "Prefer this structure.\n"
            "If the user is writing in Korean:\n"
            "- 업로드된 문서별 요약:\n"
            "- 파일명: ...\n"
            "  - 요약:\n"
            "  - 핵심 내용:\n"
            "  - 특이사항:\n"
            "- 전체 요약(선택 사항):\n"
            "If the user is writing in English:\n"
            "- Uploaded file summaries:\n"
            "- File: ...\n"
            "  - Summary:\n"
            "  - Key points:\n"
            "  - Notable notes:\n"
            "- Overall summary (optional):\n"
            "Every point must stay grounded in the retrieved evidence."
        )
    elif selected_doc_count > 1 or _needs_comparison_style(user_message):
        sys_parts.append(
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

    # ── User-role content (evidence + question) ───────────────────────────
    usr_parts: List[str] = []
    history_text = _format_history(history)
    if history_text:
        usr_parts.append(f"[Conversation history]\n{history_text}")
    if doc_context:
        usr_parts.append(f"[Retrieved document context]\n{doc_context}")
    if web_context:
        usr_parts.append(f"[Web search results]\n{web_context}")
    usr_parts.append(f"[User message]\n{user_message}")

    system_content = "\n\n".join(sys_parts)
    user_content = "\n\n".join(usr_parts)

    # Flat string for Ollama — full, unchanged
    prompt = system_content + "\n\n" + user_content

    # ── Compact messages for local_hf ────────────────────────────────────
    # Use the training-aligned compact system so the model sees the same
    # instruction format it was fine-tuned on, saving ~400-600 tokens.
    # Doc context is trimmed to the char budget so evidence chunks are never
    # silently evicted by the head/tail byte-trim inside _encode_local_prompt.
    compact_sys_parts: List[str] = [_COMPACT_SYSTEM_PROMPT]
    if target_lang == "en":
        compact_sys_parts.append(
            "If document evidence is in Korean and the question is in English, "
            "translate and explain the answer in English."
        )
    if _needs_section_understanding_style(user_message):
        compact_sys_parts.append(
            "Task: Explain the role, purpose, or differences of the described section "
            "with concrete items from the evidence."
        )
    if document_scope_count > 1 and _needs_multi_document_summary_style(user_message) and not _needs_comparison_style(user_message):
        compact_sys_parts.append(
            "Task: 파일별로 요약하되 각 파일을 독립적으로 다루고 파일명을 헤딩으로 써라."
        )
    elif selected_doc_count > 1 or _needs_comparison_style(user_message):
        compact_sys_parts.append(
            "Task: 문서별로 비교하고 각 포인트가 어느 문서에서 나왔는지 명시해라."
        )

    compact_usr_parts: List[str] = []
    if history_text:
        compact_usr_parts.append(f"[Conversation history]\n{history_text}")
    if doc_context:
        compact_usr_parts.append(
            f"[Retrieved document context]\n{_trim_doc_context_for_local_hf(doc_context)}"
        )
    if web_context:
        compact_usr_parts.append(f"[Web search results]\n{web_context}")
    compact_usr_parts.append(f"[User message]\n{user_message}")

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": "\n".join(compact_sys_parts)},
        {"role": "user",   "content": "\n\n".join(compact_usr_parts)},
    ]
    return prompt, messages


def _check_upload_wide_doc_ambiguity(user_message: str, docs) -> Optional[str]:
    """Return a clarification message when upload-wide retrieval hits 2+ distinct files
    for an ambiguous article/section query.

    When a user asks '제2조 알려줘' or '지원대상이 어떻게 돼?' without scoping to one
    document, every uploaded guideline file that has a 제2조 or a 지원대상 clause will
    surface chunks — causing the LLM to mix content from multiple unrelated guidelines.
    This check fires before the prompt is built and asks the user to name the file.
    """
    distinct_sources: List[str] = []
    seen: set = set()
    for doc in docs:
        src = str(doc.metadata.get("source") or "").strip()
        if src and src not in seen:
            seen.add(src)
            distinct_sources.append(src)

    if len(distinct_sources) < 2:
        return None

    _AMBIGUOUS_UPLOAD_TERMS = [
        "지원대상", "운영절차", "경과조치", "부칙", "지원사항",
        "지원조건", "목적", "정의", "시행일", "신청방법", "신청",
        "지원중단", "의무사항", "의무위반", "성과", "평가",
    ]
    is_ambiguous = _needs_article_lookup(user_message) or any(
        term in user_message for term in _AMBIGUOUS_UPLOAD_TERMS
    )
    if not is_ambiguous:
        return None

    file_items = "\n".join(f"  • {src}" for src in distinct_sources[:4])
    overflow = f"\n  • … 외 {len(distinct_sources) - 4}개" if len(distinct_sources) > 4 else ""

    if re.search(r"[가-힣]", user_message):
        return (
            "업로드된 여러 문서에서 관련 내용이 검색됩니다:\n"
            f"{file_items}{overflow}\n\n"
            "어느 문서를 기준으로 답변할지 왼쪽 보관소에서 파일을 선택하거나, "
            "파일명을 포함해서 다시 질문해 주세요.\n"
            "예: '런케이션 제2조 알려줘' 또는 '가족회사 지원대상이 어떻게 돼?'"
        )
    file_list_en = ", ".join(f"'{s}'" for s in distinct_sources[:3])
    return (
        f"Multiple uploaded files match this query: {file_list_en}.\n"
        "Please select a specific document from the left shelf "
        "or include the filename in your question.\n"
        "Example: 'What is Article 2 of the runkeation guideline?'"
    )


def _needs_article_lookup(text: str) -> bool:
    """Detect specific article/clause lookup queries (제N조, Article N, Section N)."""
    if re.search(r"제\s*\d+\s*조", text):
        return True
    if re.search(r"\b(?:article|section|clause)\s*\d+\b", text, re.IGNORECASE):
        return True
    return False


def _get_bundled_labels(source: Optional[str], doc_id: Optional[str]) -> List[str]:
    """Return sub-guideline labels for a scoped document using metadata only (cheap).

    Reuses _extract_scope_labels logic on a lightweight list of fake Documents
    built from section titles fetched from Chroma without loading chunk content.
    """
    if not source and not doc_id:
        return []
    try:
        from langchain_core.documents import Document as _Doc
        titles = get_section_titles(source=source, doc_id=doc_id)
        # Wrap as minimal Document objects so _extract_scope_labels can process them
        mock_docs = [_Doc(page_content="", metadata={"section_title": t}) for t in titles]
        return _extract_scope_labels(mock_docs)
    except Exception as exc:
        logger.debug("Bundled-label check failed: %s", exc)
        return []


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
    selected_model = model or get_default_model_name()
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

    # ── Multi-question fan-out ──────────────────────────────────────────────
    # When the user packs multiple distinct questions into one message
    # (newline-separated or ?-separated), answer each one individually so
    # none get silently dropped.  Each sub-question re-enters handle_chat
    # with the same scope; _split_multi_questions returns [] for single
    # questions so there is no recursion.
    sub_questions = _split_multi_questions(user_message)
    if sub_questions:
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

    if _is_smalltalk_query(user_message):
        selected_docs = []

    if not selected_docs and active_source_type == "upload":
        resolved_doc = _resolve_upload_doc_from_query(user_message)
        if resolved_doc and (resolved_doc.get("source") or resolved_doc.get("doc_id")):
            selected_docs = [resolved_doc]
            logger.info(
                "Auto-scoped upload-wide query to '%s'%s based on document name in question",
                resolved_doc.get("source") or "resolved upload",
                f" ({resolved_doc.get('doc_id')})" if resolved_doc.get("doc_id") else "",
            )

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

    # ── Early bundled-document ambiguity check ──────────────────────────
    # When exactly one document is scoped and it bundles multiple sub-guidelines
    # (e.g. a PDF with캡스톤디자인 + 프로젝트Lab + 대학원인턴십 guidelines), an
    # ambiguous query (제N조, 지원대상, 운영절차, …) without naming a specific
    # sub-guideline will produce unreliable mixed retrieval results.
    # We detect this cheaply with a metadata-only scan and ask for clarification
    # before running any retrieval.
    if (
        len(selected_docs) == 1
        and (scoped_source or scoped_doc_id)
        and not _is_smalltalk_query(user_message)
    ):
        bundled_labels = _get_bundled_labels(scoped_source, scoped_doc_id)
        if len(bundled_labels) >= 2:
            normalized_query = _normalize_scope_text(user_message)
            label_tokens = [_normalize_scope_text(lbl) for lbl in bundled_labels]
            query_names_specific = any(tok and tok in normalized_query for tok in label_tokens)
            if not query_names_specific:
                _AMBIGUOUS_ARTICLE_TERMS = [
                    "지원대상", "운영절차", "경과조치", "부칙", "지원사항",
                    "지원중단", "의무위반", "가이드라인 준용", "정의", "목적",
                    "시행일", "조치", "내용 알려줘",
                ]
                is_ambiguous = _needs_article_lookup(user_message) or any(
                    term in user_message for term in _AMBIGUOUS_ARTICLE_TERMS
                )
                if is_ambiguous:
                    clarification = _build_scoped_ambiguity_answer(user_message, bundled_labels)
                    logger.info(
                        "Early bundled-doc clarification: %d sub-guidelines in '%s', query ambiguous",
                        len(bundled_labels),
                        scoped_source or scoped_doc_id,
                    )
                    return {
                        "answer": clarification,
                        "sources": [],
                        "mode": "document_qa",
                        "active_source": active_source,
                        "active_doc_id": active_doc_id,
                        "active_source_type": active_source_type,
                        "active_sources": scoped_sources,
                        "active_doc_ids": scoped_doc_ids,
                    }

    # ── Step 1: Always search for relevant document context ──
    doc_context = ""
    sources = []

    use_full_document = bool(
        multi_scope
        or (
            (scoped_source or scoped_doc_id)
            and (
                _needs_full_document_context(user_message)
                or _needs_strict_fact_style(user_message)
                # Article/clause lookups (제N조, Article N) must load the full
                # document so the exact article chunk is never missed by top-k.
                or _needs_article_lookup(user_message)
                or _should_force_small_doc_full_context(scoped_source, scoped_doc_id)
            )
        )
        or (
            active_source_type
            and not selected_docs
            and _needs_whole_corpus_full_context(user_message)
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
    elif active_source_type and not selected_docs and use_full_document:
        docs = get_documents_by_source(source_type=active_source_type)
        logger.info(
            "Loaded %d chunks for whole-corpus task from source_type='%s'",
            len(docs),
            active_source_type,
        )
    else:
        retrieval = retrieve(
            user_message,
            source_filter=scoped_source,
            doc_id_filter=scoped_doc_id,
            source_type_filter=active_source_type if not selected_docs else None,
            full_document=use_full_document,
        )
        docs = retrieval.docs

    if (
        not multi_scope
        and not selected_docs
        and active_source_type
        and not use_full_document
    ):
        if not docs or (
            retrieval.confidence < DOCUMENT_CONFIDENCE_THRESHOLD
            and not retrieval.strong_keyword_hit
        ):
            logger.info(
                "Low-confidence corpus retrieval for source_type='%s' (confidence=%.2f, threshold=%.2f, keyword_hit=%s)",
                active_source_type,
                retrieval.confidence if retrieval else 0.0,
                DOCUMENT_CONFIDENCE_THRESHOLD,
                retrieval.strong_keyword_hit if retrieval else False,
            )
            return {
                "answer": _document_corpus_not_found_answer(user_message, active_source_type),
                "sources": [],
                "mode": "document_qa",
                "active_source": None,
                "active_doc_id": None,
                "active_source_type": active_source_type,
                "active_sources": [],
                "active_doc_ids": [],
            }

        # Upload-wide multi-file ambiguity: chunks retrieved from 2+ different files
        # for an article/section query that doesn't name a specific file.
        # Asking now prevents the LLM from mixing content across unrelated guidelines.
        if docs and active_source_type == "upload":
            upload_ambiguity = _check_upload_wide_doc_ambiguity(user_message, docs)
            if upload_ambiguity:
                n_files = len({d.metadata.get("source") for d in docs if d.metadata.get("source")})
                logger.info(
                    "Upload-wide doc ambiguity: %d files match ambiguous query '%s'",
                    n_files,
                    user_message[:60],
                )
                return {
                    "answer": upload_ambiguity,
                    "sources": extract_sources(docs),
                    "mode": "document_qa",
                    "active_source": None,
                    "active_doc_id": None,
                    "active_source_type": active_source_type,
                    "active_sources": [],
                    "active_doc_ids": [],
                }

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
            mention_candidate = _extract_mention_candidate(user_message)
            if mention_candidate:
                scoped_all_docs = get_documents_by_source(source=scoped_source, doc_id=scoped_doc_id)
                mention_pages = _find_mention_pages(scoped_all_docs, mention_candidate)
                if mention_pages:
                    logger.info(
                        "Mention-only scoped answer for '%s'%s -> '%s' on pages %s",
                        active_source or "scoped document",
                        f" ({scoped_doc_id})" if scoped_doc_id else "",
                        mention_candidate,
                        mention_pages,
                    )
                    return {
                        "answer": _build_mention_only_answer(
                            user_message,
                            active_source,
                            mention_candidate,
                            mention_pages,
                        ),
                        "sources": [],
                        "mode": "document_qa",
                        "active_source": active_source,
                        "active_doc_id": active_doc_id,
                        "active_sources": scoped_sources,
                        "active_doc_ids": scoped_doc_ids,
                    }
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
                "active_source_type": active_source_type,
                "active_sources": scoped_sources,
                "active_doc_ids": scoped_doc_ids,
            }

    if docs:
        if use_full_document and not multi_scope and (scoped_source or scoped_doc_id):
            # Early exit: query asks about tech/ML topics clearly absent from this document
            if _is_clearly_out_of_scope(user_message, docs):
                logger.info(
                    "Out-of-scope query for '%s'; returning not-found",
                    scoped_source or scoped_doc_id,
                )
                return {
                    "answer": "해당 질문에는 답변할 수 없습니다.",
                    "sources": [],
                    "mode": "document_qa",
                    "active_source": active_source,
                    "active_doc_id": active_doc_id,
                    "active_source_type": active_source_type,
                    "active_sources": scoped_sources,
                    "active_doc_ids": scoped_doc_ids,
                }

            if _needs_strict_fact_style(user_message):
                direct_count_answer = _try_scoped_count_answer(
                    user_message,
                    docs,
                    active_source,
                )
                if direct_count_answer:
                    logger.info(
                        "Answered scoped strict-count query directly from '%s'%s",
                        active_source or "scoped document",
                        f" ({active_doc_id})" if active_doc_id else "",
                    )
                    return {
                        "answer": direct_count_answer,
                        "sources": extract_sources(docs),
                        "mode": "document_qa",
                        "active_source": active_source,
                        "active_doc_id": active_doc_id,
                        "active_source_type": active_source_type,
                        "active_sources": scoped_sources,
                        "active_doc_ids": scoped_doc_ids,
                    }

                direct_toc_answer = _try_chapter_title_lookup(user_message, docs)
                if direct_toc_answer:
                    logger.info(
                        "Answered chapter TOC lookup directly from '%s'%s",
                        active_source or "scoped document",
                        f" ({active_doc_id})" if active_doc_id else "",
                    )
                    return {
                        "answer": direct_toc_answer,
                        "sources": extract_sources(docs),
                        "mode": "document_qa",
                        "active_source": active_source,
                        "active_doc_id": active_doc_id,
                        "active_source_type": active_source_type,
                        "active_sources": scoped_sources,
                        "active_doc_ids": scoped_doc_ids,
                    }

                direct_history_answer = _try_history_lookup(user_message, docs)
                if direct_history_answer:
                    logger.info(
                        "Answered publication history lookup directly from '%s'%s",
                        active_source or "scoped document",
                        f" ({active_doc_id})" if active_doc_id else "",
                    )
                    return {
                        "answer": direct_history_answer,
                        "sources": extract_sources(docs),
                        "mode": "document_qa",
                        "active_source": active_source,
                        "active_doc_id": active_doc_id,
                        "active_source_type": active_source_type,
                        "active_sources": scoped_sources,
                        "active_doc_ids": scoped_doc_ids,
                    }



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
                    "active_source_type": active_source_type,
                    "active_sources": scoped_sources,
                    "active_doc_ids": scoped_doc_ids,
                }
            if _needs_strict_fact_style(user_message):
                narrowed_docs = _select_strict_fact_docs(user_message, docs)
                if len(narrowed_docs) != len(docs):
                    logger.info(
                        "Narrowed strict-fact context from %d to %d chunks for '%s'",
                        len(docs),
                        len(narrowed_docs),
                        scoped_source or "scoped document",
                    )
                docs = narrowed_docs
        if use_full_document and active_source_type and not selected_docs:
            doc_context = format_grouped_corpus_context(docs)
        else:
            doc_context = format_context(docs)
        sources = extract_sources(docs)
        if multi_scope:
            logger.info(
                "Loaded comparison context: %d chunks from %d selected documents",
                len(docs),
                len(scoped_doc_ids),
            )
        elif use_full_document and active_source_type and not selected_docs:
            logger.info(
                "Loaded full corpus context: %d chunks from source_type='%s'",
                len(docs),
                active_source_type,
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
        uploads_exist = any(doc.get("source_type") == "upload" for doc in list_documents())
        if (
            not selected_docs
            and not active_source_type
            and uploads_exist
            and not _might_need_web_search(user_message)
            and _looks_like_entity_explanation_query(user_message)
        ):
            logger.info("Document-first clarification triggered for unscoped query with existing uploads")
            return {
                "answer": _document_first_clarification_answer(user_message),
                "sources": [],
                "mode": "document_qa",
                "active_source": None,
                "active_doc_id": None,
                "active_source_type": active_source_type,
                "active_sources": [],
                "active_doc_ids": [],
            }

    if (
        docs
        and use_full_document
        and active_source_type
        and not selected_docs
        and _needs_multi_document_summary_style(user_message)
        and not _needs_comparison_style(user_message)
    ):
        logger.info(
            "Returning deterministic whole-corpus file summary for source_type='%s'",
            active_source_type,
        )
        return {
            "answer": _build_file_level_corpus_summary(user_message, docs),
            "sources": sources,
            "mode": "document_qa",
            "active_source": None,
            "active_doc_id": None,
            "active_source_type": active_source_type,
            "active_sources": [],
            "active_doc_ids": [],
        }

    # ── Step 2: Check if web search might help ──
    web_context = ""
    if not scoped_source and not multi_scope and _might_need_web_search(user_message):
        web_results = _search_web(user_message)
        if web_results:
            web_context = web_results
            logger.info("Added web search results")

    # ── Step 3: Build prompt with all context and let LLM decide ──
    prompt, lm_messages = _build_prompt(
        user_message=user_message,
        history=history,
        doc_context=doc_context,
        web_context=web_context,
        system_prompt=system_prompt,
        selected_doc_count=len(selected_docs),
        document_scope_count=max(
            len(selected_docs),
            len({
                source.get("doc_id") or source.get("source")
                for source in sources
                if source.get("doc_id") or source.get("source")
            }),
        ),
    )

    # Pass structured messages to local_hf so apply_chat_template sees the
    # correct system/user split (matching training format).  Ollama ignores
    # the messages kwarg and uses the flat prompt string as before.
    answer = generate_text(prompt, model=selected_model, messages=lm_messages)

    if _needs_language_retry(user_message, answer):
        _is_chinese = _contains_chinese(answer)
        lang_name = "Korean" if _infer_target_language(user_message) == "ko" else "English"
        if _is_chinese:
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
        retry_prompt = _language_correction_prompt(prompt, user_message, answer)
        retry_messages: List[Dict[str, Any]] = [
            lm_messages[0],
            {
                "role": "user",
                "content": (
                    lm_messages[1]["content"]
                    + "\n\n[Critical correction]\n"
                    + correction_note
                    + "Keep the same facts and stay grounded in the provided context.\n\n"
                    + f"[Invalid draft answer]\n{answer}"
                ),
            },
        ]
        retry_answer = generate_text(retry_prompt, model=selected_model, messages=retry_messages)
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
        "active_source_type": active_source_type,
        "active_sources": scoped_sources,
        "active_doc_ids": scoped_doc_ids,
    }
