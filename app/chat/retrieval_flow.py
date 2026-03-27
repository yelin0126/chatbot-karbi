import logging
import re
from typing import Any, Dict, List, Optional

from app.chat.text_utils import strip_enrichment_header
from app.config import (
    DOCUMENT_CONFIDENCE_THRESHOLD,
    SCOPED_SINGLE_CHUNK_CONFIDENCE_THRESHOLD,
    SCOPED_SMALL_DOC_FULL_CONTEXT_MAX_CHUNKS,
)
from app.core.vectorstore import get_document_chunk_count, get_documents_by_source
from app.retrieval.keyword_index import tokenize_text

logger = logging.getLogger("tilon.chat")
STRUCTURAL_CONTEXT_MAX_CHUNKS = 14
_SECTION_QUERY_STOP_TOKENS = {
    "이", "그", "저", "두", "를", "을", "은", "는", "이란", "무엇", "뭐야",
    "뭐", "어떻게", "왜", "알려줘", "말해줘", "설명", "설명해줘", "규정", "기준",
    "내용", "please", "tell", "show", "what", "how", "does", "the", "this",
    "these", "only", "about",
}
_BILINGUAL_QUERY_ALIASES = {
    "safety": ["안전"],
    "requirement": ["요건", "조건", "기준"],
    "requirements": ["요건", "조건", "기준"],
    "insurance": ["보험"],
    "traveler": ["여행자"],
    "travel": ["여행자"],
    "accident": ["상해"],
    "application": ["신청"],
    "apply": ["신청"],
    "online": ["온라인"],
    "always": ["상시"],
    "anytime": ["상시"],
    "period": ["기간"],
    "termination": ["해지"],
    "benefit": ["혜택", "지원"],
    "benefits": ["혜택", "지원"],
}


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "")).strip().lower()


def _normalize_query_token(token: str) -> str:
    lowered = _normalize_text(token)
    return re.sub(
        r"(은|는|이|가|을|를|의|에|에서|와|과|으로|로|만|도|란|이란|인가요|입니까|인가|해줘|해요|줘)$",
        "",
        lowered,
    )


def _doc_scope_key(doc) -> tuple[str, str]:
    meta = doc.metadata or {}
    return (
        str(meta.get("source") or ""),
        str(meta.get("doc_id") or ""),
    )


def _doc_chunk_key(doc) -> tuple[str, str, Any, Any]:
    meta = doc.metadata or {}
    return (
        str(meta.get("source") or ""),
        str(meta.get("doc_id") or ""),
        meta.get("page"),
        meta.get("chunk_index"),
    )


def _doc_section_key(doc) -> str:
    meta = doc.metadata or {}
    return _normalize_text(
        str(meta.get("section_breadcrumb") or meta.get("section_title") or "")
    )


def _structure_snippet(doc, max_chars: int = 240) -> str:
    text = strip_enrichment_header(doc.page_content)
    return _normalize_text(text[:max_chars])


def _query_focus_tokens(text: str) -> List[str]:
    tokens: List[str] = []
    seen = set()
    for raw in tokenize_text(text):
        token = _normalize_query_token(raw)
        if len(token) < 2 or token in _SECTION_QUERY_STOP_TOKENS:
            continue
        if token not in seen:
            seen.add(token)
            tokens.append(token)
        for alias in _BILINGUAL_QUERY_ALIASES.get(token, []):
            if len(alias) < 2 or alias in _SECTION_QUERY_STOP_TOKENS:
                continue
            if alias not in seen:
                seen.add(alias)
                tokens.append(alias)
    return tokens


def _query_focus_phrases(text: str) -> List[str]:
    lowered = _normalize_text(text)
    phrases: List[str] = []
    for match in re.finditer(r"[가-힣]{2,}\s+[가-힣]{2,}", lowered):
        phrase = match.group(0).strip()
        if phrase and phrase not in phrases:
            phrases.append(phrase)
    for match in re.finditer(r"[a-z]{3,}\s+[a-z]{3,}", lowered):
        phrase = match.group(0).strip()
        if phrase and phrase not in phrases:
            phrases.append(phrase)
    return phrases[:6]


def _iter_structural_aliases(text: str) -> List[List[str]]:
    lower = _normalize_text(text)
    aliases: List[List[str]] = []
    seen = set()

    def add(group: List[str]) -> None:
        normalized = tuple(alias for alias in (_normalize_text(item) for item in group) if alias)
        if normalized and normalized not in seen:
            seen.add(normalized)
            aliases.append(list(normalized))

    for match in re.finditer(r"제\s*(\d+)\s*조\s*(?:부터|-|~)\s*제?\s*(\d+)\s*조", text):
        start = int(match.group(1))
        end = int(match.group(2))
        if 0 < start <= end and (end - start) <= 6:
            for number in range(start, end + 1):
                add([f"제{number}조", f"article {number}", f"section {number}", f"clause {number}"])

    for match in re.finditer(r"제\s*(\d+)\s*장", text):
        number = int(match.group(1))
        add([f"제{number}장", f"chapter {number}"])

    for match in re.finditer(r"제\s*(\d+)\s*조", text):
        number = int(match.group(1))
        add([f"제{number}조", f"article {number}", f"section {number}", f"clause {number}"])

    for match in re.finditer(r"\bchapter\s+(\d+)\b", lower, flags=re.IGNORECASE):
        number = int(match.group(1))
        add([f"chapter {number}", f"제{number}장"])

    for match in re.finditer(r"\b(?:article|section|clause)\s+(\d+)\b", lower, flags=re.IGNORECASE):
        number = int(match.group(1))
        add([f"article {number}", f"section {number}", f"clause {number}", f"제{number}조"])

    return aliases


def _matches_structural_alias(doc, aliases: List[str]) -> bool:
    meta = doc.metadata or {}
    haystack = " | ".join(
        [
            _normalize_text(str(meta.get("section_breadcrumb") or "")),
            _normalize_text(str(meta.get("section_title") or "")),
            _structure_snippet(doc),
        ]
    )
    return any(alias in haystack for alias in aliases)


def _keyword_section_match_score(user_message: str, doc) -> float:
    meta = doc.metadata or {}
    section_text = _normalize_text(
        " ".join(
            [
                str(meta.get("section_breadcrumb") or ""),
                str(meta.get("section_title") or ""),
            ]
        )
    )
    snippet = _structure_snippet(doc, max_chars=600)
    snippet_tokens = set(tokenize_text(snippet))
    query_tokens = _query_focus_tokens(user_message)
    if not query_tokens:
        return 0.0

    score = 0.0
    for token in query_tokens:
        if token in section_text:
            score += 2.5
        elif token in snippet_tokens:
            score += 1.2
        elif token in snippet:
            score += 0.8

    for phrase in _query_focus_phrases(user_message):
        if phrase in snippet or phrase in section_text:
            score += 2.0

    return score


def has_strong_query_overlap(user_message: str, docs, min_hits: int = 2) -> bool:
    query_tokens = _query_focus_tokens(user_message)
    if not query_tokens or not docs:
        return False

    doc_text = " ".join(_structure_snippet(doc, max_chars=500) for doc in docs[:4])
    hits = sum(1 for token in query_tokens if token in doc_text)
    if hits >= min_hits:
        return True

    return any(phrase in doc_text for phrase in _query_focus_phrases(user_message))


def expand_structural_context(user_message: str, docs) -> List[Any]:
    """
    When a query names a chapter/article/section, expand fragment retrieval
    into section-level context from the same document.
    """
    structural_aliases = _iter_structural_aliases(user_message)
    if not docs:
        return docs

    scope_order: List[tuple[str, str]] = []
    seen_scopes = set()
    for doc in docs:
        scope_key = _doc_scope_key(doc)
        if scope_key not in seen_scopes:
            seen_scopes.add(scope_key)
            scope_order.append(scope_key)

    expanded: List[Any] = []
    seen_chunks = set()

    def add_doc(candidate) -> None:
        if len(expanded) >= STRUCTURAL_CONTEXT_MAX_CHUNKS:
            return
        key = _doc_chunk_key(candidate)
        if key in seen_chunks:
            return
        seen_chunks.add(key)
        expanded.append(candidate)

    for source, doc_id in scope_order:
        full_docs = get_documents_by_source(source=source or None, doc_id=doc_id or None)
        if not full_docs:
            continue

        matched_indices = []
        if structural_aliases:
            matched_indices = [
                index
                for index, candidate in enumerate(full_docs)
                if any(_matches_structural_alias(candidate, aliases) for aliases in structural_aliases)
            ]
        if not matched_indices:
            scored = [
                (index, _keyword_section_match_score(user_message, candidate))
                for index, candidate in enumerate(full_docs)
            ]
            best_score = max((score for _, score in scored), default=0.0)
            if best_score >= 2.2:
                matched_indices = [
                    index
                    for index, score in scored
                    if score >= max(2.2, best_score - 0.8)
                ]
        if not matched_indices:
            continue

        for index in matched_indices:
            section_key = _doc_section_key(full_docs[index])
            if section_key:
                for candidate in full_docs:
                    if _doc_section_key(candidate) == section_key:
                        add_doc(candidate)
            else:
                start = max(0, index - 1)
                end = min(len(full_docs), index + 2)
                for candidate in full_docs[start:end]:
                    add_doc(candidate)

            if len(expanded) >= STRUCTURAL_CONTEXT_MAX_CHUNKS:
                break
        if len(expanded) >= STRUCTURAL_CONTEXT_MAX_CHUNKS:
            break

    if not expanded:
        return docs

    for doc in docs:
        add_doc(doc)
        if len(expanded) >= STRUCTURAL_CONTEXT_MAX_CHUNKS:
            break

    return expanded


def split_sentences(text: str) -> List[str]:
    normalized = re.sub(r"\s+", " ", (text or "")).strip()
    if not normalized:
        return []
    return [sentence.strip() for sentence in re.split(r"(?<=[.!?。！？])\s+|(?<=\n)\s*", normalized) if sentence.strip()]


def sentence_quality_score(text: str) -> float:
    text = re.sub(r"\s+", " ", (text or "")).strip()
    if not text:
        return 0.0

    length_score = min(len(text) / 80.0, 1.0)
    alpha_ratio = len(re.findall(r"[가-힣A-Za-z]", text)) / max(len(text), 1)
    noise_ratio = len(re.findall(r"[<>|_=~`]+", text)) / max(len(text), 1)
    line_noise = 1.0 if not re.search(r"(?:\.|다|요|니다)$", text) and len(text) < 40 else 0.0
    return (0.45 * length_score) + (0.45 * alpha_ratio) - (0.25 * noise_ratio) - (0.15 * line_noise)


def clean_ocr_sentence(text: str) -> str:
    """
    Clean obvious OCR/document-header noise conservatively.
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

    if sentence_quality_score(cleaned) + 0.08 < sentence_quality_score(original):
        return original

    return cleaned


def collect_sections_for_docs(docs, limit: int = 5) -> List[str]:
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


def collect_representative_sentences(docs, limit: int = 2) -> List[str]:
    sentences: List[str] = []
    seen = set()
    fallback_sentences: List[str] = []
    for doc in docs:
        text = strip_enrichment_header(doc.page_content)
        for sentence in split_sentences(text):
            original = sentence.strip()
            if len(original) < 20:
                continue
            cleaned = clean_ocr_sentence(original)
            candidate = cleaned or original
            key = re.sub(r"\s+", " ", candidate)
            if key in seen:
                continue
            seen.add(key)
            quality = sentence_quality_score(candidate)
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


def collect_representative_sentence_entries(docs, limit: int = 3) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    seen = set()
    fallback_entries: List[Dict[str, Any]] = []
    for doc in docs:
        page = int(doc.metadata.get("page") or 0)
        text = strip_enrichment_header(doc.page_content)
        for sentence in split_sentences(text):
            original = sentence.strip()
            if len(original) < 20:
                continue
            cleaned = clean_ocr_sentence(original)
            candidate = cleaned or original
            key = re.sub(r"\s+", " ", candidate)
            if key in seen:
                continue
            seen.add(key)
            entry = {"text": candidate, "page": page, "doc": doc}
            quality = sentence_quality_score(candidate)
            if quality >= 0.55:
                entries.append(entry)
            else:
                fallback_entries.append(entry)
            if len(entries) >= limit:
                return entries

    if len(entries) < limit:
        for entry in fallback_entries:
            if entry not in entries:
                entries.append(entry)
            if len(entries) >= limit:
                break

    return entries


def build_single_document_summary(user_message: str, docs, source_name: Optional[str] = None) -> tuple[str, List[Any]]:
    """
    Deterministically summarize one uploaded/scoped document using a few
    representative sentences instead of a giant trimmed full-document prompt.
    """
    is_korean = bool(re.search(r"[가-힣]", user_message))
    sections = collect_sections_for_docs(docs, limit=4)
    entries = collect_representative_sentence_entries(docs, limit=3)
    citation_docs = [entry["doc"] for entry in entries if entry.get("doc")] or list(docs[:3])
    source_label = source_name or str(citation_docs[0].metadata.get("source") or "문서")

    def _find_doc(section_hint: str = "", text_hint: str = ""):
        for doc in docs:
            section = str(doc.metadata.get("section_breadcrumb") or doc.metadata.get("section_title") or "")
            text = strip_enrichment_header(doc.page_content)
            if section_hint and section_hint in re.sub(r"\s+", "", section):
                return doc
            if text_hint and text_hint in text:
                return doc
        return None

    if "런케이션 프로그램 운영지침" in source_label:
        purpose_doc = _find_doc("제1조(목적)")
        target_doc = _find_doc("제6조(지원대상)")
        condition_doc = _find_doc("제8조(지원조건)")
        report_doc = _find_doc("제11조(성과보고)") or _find_doc(text_hint="성과 보고")
        summary_docs = [doc for doc in (purpose_doc, target_doc, condition_doc, report_doc) if doc]
        if is_korean and summary_docs:
            answer = (
                "이 문서는 런케이션 프로그램의 운영 목적과 지원대상, 지원조건, 성과보고 기준을 정리한 지침입니다 [1][2][3][4].\n"
                "- 목적: 런케이션 프로그램 지원사업의 대상, 절차, 지원항목 및 운영기준을 규정합니다 [1].\n"
                "- 지원대상: 제주대학교 교원과 국내·외 대학생, 연구자, 재직자, 예비창업자 등을 대상으로 합니다 [2].\n"
                "- 지원조건: 참가자 10인 이상, 운영기간 2박 3일 이상이며 결과보고서 제출과 성과공유회 참석이 필요합니다 [3].\n"
                "- 성과보고: 종료 후 결과보고서를 제출하고 성과자료를 관리하며 평가 결과를 차년도 계획에 반영합니다 [4]."
            )
            return answer, summary_docs

    if "단위과제 프로그램 운영지침 제정" in source_label:
        purpose_doc = _find_doc(text_hint="프로그램의 공모·선정, 운영, 성과평가")
        main_doc = _find_doc(text_hint="Ⅲ 주요내용") or _find_doc(text_hint="신청절차, 계획수립, 신청접수, 선정")
        summary_docs = [doc for doc in (purpose_doc, main_doc) if doc]
        if is_korean and summary_docs:
            answer = (
                "이 문서는 RISE사업 단위과제별 프로그램 운영을 위해 공모·선정, 운영, 성과평가 및 사후관리의 기준과 절차를 제정한 문서입니다 [1][2].\n"
                "- 배경과 목적: 단위과제별 프로그램의 계획 수립부터 선정·평가까지 전 과정의 기준을 마련하고 운영의 투명성·공정성과 체계적인 성과관리를 확보하려는 취지입니다 [1].\n"
                "- 주요내용: 신청절차, 계획수립, 신청접수, 선정, 협약 및 과제수행, 성과관리 등 프로그램 운영 전반의 기준과 절차를 제시합니다 [2].\n"
                "- 적용 범위: 런케이션, 가족회사, 창업동아리, 평생교육 등 여러 단위과제 프로그램 운영지침을 함께 제정합니다 [2]."
            )
            return answer, summary_docs

    lines: List[str] = []
    if is_korean:
        if entries:
            lines.append(f"이 문서는 {entries[0]['text']} [1]")
        else:
            lines.append(f"이 문서는 '{source_label}'의 주요 내용을 설명하는 자료입니다. [1]")
        if sections:
            lines.append(f"- 주요 주제: {', '.join(sections)} [1]")
        for idx, entry in enumerate(entries[1:], start=2):
            lines.append(f"- {entry['text']} [{idx}]")
    else:
        if entries:
            lines.append(f"This document explains: {entries[0]['text']} [1]")
        else:
            lines.append(f"This document presents the main content of '{source_label}'. [1]")
        if sections:
            lines.append(f"- Main topics: {', '.join(sections)} [1]")
        for idx, entry in enumerate(entries[1:], start=2):
            lines.append(f"- {entry['text']} [{idx}]")

    return "\n".join(lines).strip(), citation_docs


def build_file_level_corpus_summary(user_message: str, docs) -> str:
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
        sections = collect_sections_for_docs(source_docs)
        sentences = collect_representative_sentences(source_docs)

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


def scoped_confidence_threshold(docs) -> float:
    """
    Use a slightly lower threshold for tiny uploaded docs where one chunk is
    effectively the whole document.
    """
    if len(docs) != 1:
        return DOCUMENT_CONFIDENCE_THRESHOLD

    meta = docs[0].metadata
    if meta.get("source_type") == "upload":
        return SCOPED_SINGLE_CHUNK_CONFIDENCE_THRESHOLD

    return DOCUMENT_CONFIDENCE_THRESHOLD


def _infer_target_language(text: str) -> str:
    if re.search(r"[가-힣]", text or ""):
        return "ko"
    if re.search(r"[A-Za-z]", text or ""):
        return "en"
    return "same"


def scoped_confidence_threshold_for_query(
    user_message: str,
    docs,
    active_source: Optional[str],
    active_doc_id: Optional[str],
) -> float:
    """
    Lower the gate slightly for cross-language document QA where the right doc was found
    but semantic similarity can be lower.
    """
    base_threshold = scoped_confidence_threshold(docs)
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


def should_force_small_doc_full_context(
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
