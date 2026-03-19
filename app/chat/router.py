"""
Chat mode detection / routing.

IMPROVEMENTS over original:
- Logging which mode was selected and why
- Easier to extend with new modes
- Separated from handler logic
"""

import logging
from typing import Literal

logger = logging.getLogger("tilon.router")

ModeType = Literal["general", "document_qa", "web_search", "ocr_extract"]

# ── Keyword lists ──────────────────────────────────────────────────────
# NOTE: This is simple keyword matching. A future improvement would be
# to use the LLM itself to classify intent (more accurate but slower).

OCR_KEYWORDS = [
    "ocr", "텍스트만", "텍스트 추출", "원문", "글자 추출",
    "읽어줘", "뽑아줘", "추출해줘", "문자 인식", "extract text",
    "read the text", "show the text", "only text",
]

DOCUMENT_KEYWORDS = [
    "pdf", "문서", "파일", "업로드", "페이지", "요약",
    "첨부", "첨부파일", "이 문서", "이 파일", "이 pdf",
    "summarize this pdf", "read the uploaded pdf", "document",
]

WEB_KEYWORDS = [
    "오늘", "현재", "최신", "최근", "실시간", "주가", "날씨",
    "뉴스", "환율", "지금", "현재가", "today", "current",
    "latest", "recent", "stock price", "weather", "news",
]


def detect_mode(user_message: str) -> ModeType:
    """Detect the appropriate response mode from the user's message."""
    text = user_message.lower().strip()

    # Priority order: OCR > Web > Document > General
    if any(kw in text for kw in OCR_KEYWORDS):
        mode = "ocr_extract"
    elif any(kw in text for kw in WEB_KEYWORDS):
        mode = "web_search"
    elif any(kw in text for kw in DOCUMENT_KEYWORDS):
        mode = "document_qa"
    else:
        mode = "general"

    logger.info("Mode detected: %s (message: '%s')", mode, text[:60])
    return mode
