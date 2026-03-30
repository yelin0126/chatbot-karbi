from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.chat.text_utils import strip_enrichment_header
from app.config import MEDIA_BRIEF_MAX_CHARS, MEDIA_BRIEF_USE_LLM
from app.config import LIBRARY_DIR, UPLOADS_DIR
from app.core.document_registry import get_document, list_documents
from app.core.llm import generate_text
from app.retrieval.retriever import extract_sources, format_context, retrieve

logger = logging.getLogger("tilon.media.router")

_IMAGE_GEN_RE = re.compile(
    r"(이미지|그림|일러스트|포스터|썸네일|렌더|배경화면|image).*(생성|만들|그려|render|generate)|"
    r"(생성|만들|그려).*(이미지|그림|일러스트|포스터|썸네일|image)",
    re.IGNORECASE,
)
_VIDEO_GEN_RE = re.compile(
    r"(영상|비디오|동영상|애니메이션|clip|video).*(생성|만들|제작|render|generate)|"
    r"(생성|만들|제작).*(영상|비디오|동영상|애니메이션|clip|video)",
    re.IGNORECASE,
)
_IMAGE_ANALYSIS_RE = re.compile(
    r"(이 이미지|이미지|사진|photo|picture).*(설명|분석|해석|읽어|보이|무엇|뭐)|"
    r"(설명|분석|해석|읽어|보이|무엇|뭐).*(이미지|사진|photo|picture)",
    re.IGNORECASE,
)

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}


def is_image_path(path_like: Optional[str]) -> bool:
    if not path_like:
        return False
    return Path(path_like).suffix.lower() in _IMAGE_EXTENSIONS


def detect_media_intent(message: str, *, active_source: Optional[str] = None) -> Optional[str]:
    text = (message or "").strip()
    if not text:
        return None
    if _VIDEO_GEN_RE.search(text):
        return "video_generation"
    if _IMAGE_GEN_RE.search(text):
        return "image_generation"
    if active_source and is_image_path(active_source) and _IMAGE_ANALYSIS_RE.search(text):
        return "image_understanding"
    return None


def resolve_source_record(active_source: Optional[str], active_doc_id: Optional[str]) -> Optional[Dict[str, Any]]:
    if active_doc_id:
        record = get_document(active_doc_id)
        if record:
            return record

    if not active_source:
        return None

    for record in list_documents():
        if record.get("source") == active_source:
            return record
    return None


def resolve_image_target_path(
    active_source: Optional[str],
    active_doc_id: Optional[str],
    explicit_path: Optional[str] = None,
) -> Optional[Path]:
    if explicit_path:
        file_path = Path(explicit_path)
        if file_path.exists() and is_image_path(str(file_path)):
            return file_path

    record = resolve_source_record(active_source, active_doc_id)
    if not record:
        if active_source:
            for base_dir in (UPLOADS_DIR, LIBRARY_DIR):
                candidate = base_dir / Path(active_source).name
                if candidate.exists() and is_image_path(str(candidate)):
                    return candidate
        return None

    file_path = Path(record.get("source_path") or "")
    if file_path.exists() and is_image_path(str(file_path)):
        return file_path
    return None


def _fallback_brief(user_message: str, docs: List[Any]) -> str:
    snippets: List[str] = []
    for doc in docs[:3]:
        cleaned = strip_enrichment_header(doc.page_content or "").strip()
        if not cleaned:
            continue
        compact = re.sub(r"\s+", " ", cleaned)
        snippets.append(compact[:220].rstrip())
    if not snippets:
        return user_message.strip()
    detail = " / ".join(snippets)
    return f"{user_message.strip()}\n문서 기반 핵심 시각 정보: {detail[:MEDIA_BRIEF_MAX_CHARS].rstrip()}"


def build_grounded_media_brief(
    *,
    user_message: str,
    active_source: Optional[str] = None,
    active_doc_id: Optional[str] = None,
    active_source_type: Optional[str] = None,
) -> Dict[str, Any]:
    if not any([active_source, active_doc_id, active_source_type]):
        return {
            "brief": user_message.strip(),
            "sources": [],
        }

    retrieval = retrieve(
        user_message,
        source_filter=active_source,
        doc_id_filter=active_doc_id,
        source_type_filter=active_source_type,
        full_document=False,
        enable_rerank=True,
    )
    docs = retrieval.docs[:3]
    source_refs = extract_sources(docs)
    if not docs:
        return {
            "brief": user_message.strip(),
            "sources": [],
        }

    context = format_context(docs, max_chars_per_chunk=320)
    if not MEDIA_BRIEF_USE_LLM:
        return {"brief": _fallback_brief(user_message, docs), "sources": source_refs}

    prompt = (
        "다음 문서 근거를 바탕으로 이미지/영상 생성용 짧은 시각 브리프를 작성하세요.\n"
        "- 1~3문장\n"
        "- 장면, 객체, 관계, 표기해야 할 핵심 요소만 남기기\n"
        "- 추측 금지\n"
        "- 문서에 없는 내용 추가 금지\n"
        "- 스타일은 사용자가 요청한 경우만 유지\n\n"
        f"[사용자 요청]\n{user_message.strip()}\n\n"
        f"[문서 근거]\n{context}\n\n"
        "[출력]\n"
    )
    try:
        brief = generate_text(prompt=prompt, temperature=0.0, max_tokens=180).strip()
        if not brief:
            brief = _fallback_brief(user_message, docs)
    except Exception as exc:
        logger.warning("Falling back to heuristic media brief: %s", exc)
        brief = _fallback_brief(user_message, docs)

    return {
        "brief": brief[:MEDIA_BRIEF_MAX_CHARS].strip(),
        "sources": source_refs,
    }
