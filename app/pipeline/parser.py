"""
Document parsing — PDF and images.

Extraction strategy (in order):
1. marker_single → best for digital/text-heavy PDFs
2. PyMuPDF text extraction → fast fallback
3. VLM extraction (Qwen2.5-VL via Ollama) → for image-heavy PDFs
4. Tesseract OCR → last resort

The parser auto-detects when text extraction is poor (low chars/page)
and escalates to VLM or OCR automatically.
"""

import io
import re
import base64
import hashlib
import logging
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import fitz  # pymupdf
import requests
from PIL import Image
import pytesseract
from pdf2image import convert_from_path
from langchain_core.documents import Document

try:
    import pdfplumber as _pdfplumber  # optional — column-aware + table extraction
    _PDFPLUMBER_AVAILABLE = True
except ImportError:
    _pdfplumber = None  # type: ignore
    _PDFPLUMBER_AVAILABLE = False

from app.config import (
    ENABLE_OCR,
    PADDLE_OCR_ENABLED,
    MARKER_OUTPUT_DIR,
    OLLAMA_BASE_URL,
    VLM_EXTRACTION_ENABLED,
    VLM_SCANNED_PDF_ENABLED,
    VLM_HYBRID_PDF_ENABLED,
    VLM_EXTRACTION_MODEL,
)
from app.core.paddle_ocr import ocr_page_image as _paddle_ocr_page

logger = logging.getLogger("tilon.parser")


_PAGE_MIN_REAL_CHARS = 80
_HYBRID_MIN_REAL_CHARS = 150
_GIBBERISH_THRESHOLD = 0.35
_OCR_RENDER_DPI = 300
_VLM_RENDER_DPI = 220
_HEADING_MAX_CHARS = 120
_HEADING_MAX_LINES = 3
_TABLE_LINE_BREAK_THRESHOLD = 3
_MARKER_FALLBACK_RATIO = 1.75
_SLOW_PAGE_LOG_MS = 1500


# ═══════════════════════════════════════════════════════════════════════
# Artifact Contract Helpers
# ═══════════════════════════════════════════════════════════════════════

def _compute_checksum(file_path: Path) -> str:
    """Compute a stable SHA-256 checksum for a file."""
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_artifact_meta(file_path: Path, page_total: int, input_type: str) -> dict:
    """Build shared metadata for all pages/chunks extracted from one file."""
    checksum = _compute_checksum(file_path)
    return {
        "source": file_path.name,
        "source_path": str(file_path),
        "doc_id": f"{file_path.stem}-{checksum[:12]}",
        "doc_checksum": checksum,
        "page_total": page_total,
        "input_type": input_type,
    }


def _gibberish_ratio(text: str) -> float:
    """Estimate how much of the text looks like broken OCR/encoding noise."""
    stripped = re.sub(r"\s+", "", text or "")
    if not stripped:
        return 1.0

    valid_chars = re.findall(
        r"[A-Za-z0-9\uAC00-\uD7AF\u1100-\u11FF\u3130-\u318F.,:;!?()\[\]{}%/\-_=+\'\"@#&*]",
        stripped,
    )
    ratio = 1.0 - (len(valid_chars) / max(len(stripped), 1))
    return max(0.0, min(1.0, ratio))


def _quality_flags(real_chars: int, gibberish_ratio: float, language: str) -> str:
    """Serialize quality warnings into a compact metadata string."""
    flags = []
    if real_chars < _PAGE_MIN_REAL_CHARS:
        flags.append("low_text_yield")
    if gibberish_ratio > 0.35:
        flags.append("garbled_text")
    if language == "unknown" and real_chars >= 40:
        flags.append("language_uncertain")
    return ",".join(flags)


def _estimate_confidence(method: str, real_chars: int, gibberish_ratio: float) -> float:
    """Estimate extraction confidence for routing/debugging purposes."""
    base = {
        "marker_pdf": 0.95,
        "text": 0.92,
        "vlm": 0.78,
        "paddle_ocr": 0.82,
        "ocr": 0.72,
        "ocr_image": 0.72,
    }.get(method, 0.7)

    if real_chars < 120:
        base -= 0.08
    if real_chars < 40:
        base -= 0.15
    base -= min(gibberish_ratio, 0.5) * 0.6

    return round(max(0.05, min(0.99, base)), 2)


def _make_page_document(
    text: str,
    base_meta: dict,
    page: int,
    extraction_method: str,
    page_kind: str,
    chunk_type: str = "page",
    section_title: str = "",
    extractors_used: str = "",
    **extra_meta,
) -> Document:
    """Build a normalized page-level Document with quality metadata."""
    normalized_text = (text or "").strip()
    language = detect_language(normalized_text[:300])
    real_chars = len(re.sub(r"\s+", "", normalized_text))
    gib_ratio = round(_gibberish_ratio(normalized_text), 3)

    return Document(
        page_content=normalized_text,
        metadata={
            **base_meta,
            "page": page,
            "section_title": section_title,
            "language": language,
            "chunk_type": chunk_type,
            "extraction_method": extraction_method,
            "page_kind": page_kind,
            "extractors_used": extractors_used or extraction_method,
            "text_yield_chars": len(normalized_text),
            "real_char_count": real_chars,
            "gibberish_ratio": gib_ratio,
            "extraction_confidence": _estimate_confidence(extraction_method, real_chars, gib_ratio),
            "quality_flags": _quality_flags(real_chars, gib_ratio, language),
            **extra_meta,
        },
    )


# ═══════════════════════════════════════════════════════════════════════
# Language Detection
# ═══════════════════════════════════════════════════════════════════════

def detect_language(text: str) -> str:
    """Detect language. Checks for Korean characters first."""
    text = text.strip()
    if len(text) < 10:
        return "unknown"

    korean_chars = len(re.findall(r'[\uAC00-\uD7AF\u1100-\u11FF\u3130-\u318F]', text))
    total_alpha = len(re.findall(r'[a-zA-Z\uAC00-\uD7AF]', text))

    if total_alpha > 0 and korean_chars / max(total_alpha, 1) > 0.3:
        return "ko"

    try:
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = 0
        return detect(text)
    except Exception:
        return "unknown"


# ═══════════════════════════════════════════════════════════════════════
# Method 1: Marker PDF
# ═══════════════════════════════════════════════════════════════════════

def _extract_with_marker(pdf_path: str) -> str:
    """Use marker_single for markdown extraction."""
    pdf_file = Path(pdf_path)
    MARKER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        result = subprocess.run(
            ["marker_single", str(pdf_file),
             "--output_format", "markdown",
             "--output_dir", str(MARKER_OUTPUT_DIR)],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=120,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        logger.debug("marker_single unavailable or failed: %s", e)
        return ""
    except Exception as e:
        logger.debug("marker_single error: %s", e)
        return ""

    result_dir = MARKER_OUTPUT_DIR / pdf_file.stem
    if not result_dir.exists():
        return ""

    md_files = list(result_dir.glob("*.md"))
    if not md_files:
        return ""

    try:
        text = md_files[0].read_text(encoding="utf-8").strip()
        import shutil
        shutil.rmtree(result_dir, ignore_errors=True)
        return text
    except Exception:
        return ""


def _clean_marker_text(text: str) -> str:
    """Remove markdown image references to get actual text content."""
    cleaned = re.sub(r'!\[[^\]]*\]\([^)]*\)', '', text)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned.strip()


def _count_real_chars(text: str) -> int:
    """Count meaningful characters (not whitespace or markdown)."""
    cleaned = _clean_marker_text(text)
    return len(re.sub(r'\s+', '', cleaned))


def _serialize_bbox(bbox: Optional[List[float]]) -> str:
    """Serialize bbox values compactly for metadata-safe storage."""
    if not bbox:
        return ""
    try:
        return ",".join(f"{float(value):.1f}" for value in bbox)
    except Exception:
        return ""


def _normalize_extracted_text(text: str) -> str:
    """Normalize extracted text while preserving paragraph boundaries."""
    if not text:
        return ""

    lines = [line.rstrip() for line in text.splitlines()]
    normalized_lines = []
    blank_run = 0

    for line in lines:
        stripped = re.sub(r"\s+", " ", line).strip()
        if not stripped:
            blank_run += 1
            if blank_run <= 1:
                normalized_lines.append("")
            continue

        blank_run = 0
        normalized_lines.append(stripped)

    return "\n".join(normalized_lines).strip()


def _looks_like_heading(
    text: str,
    line_count: int,
    max_font_size: float,
    page_max_font_size: float,
    page_avg_font_size: float,
) -> bool:
    """Heuristic heading detector for PDF text blocks."""
    stripped = text.strip()
    if not stripped:
        return False
    if len(stripped) > _HEADING_MAX_CHARS or line_count > _HEADING_MAX_LINES:
        return False
    if stripped.endswith((".", "?", "!", "다", "요", ";", ":")):
        return False

    effective_avg = max(page_avg_font_size, 1.0)
    font_gap = max(page_max_font_size - effective_avg, 0.0)
    if font_gap < 0.5 and max_font_size < 16:
        return False

    large_font = (
        max_font_size >= 16
        or max_font_size >= effective_avg * 1.4
        or (font_gap >= 0.8 and max_font_size >= effective_avg * 1.18)
    )

    return large_font


def _guess_block_type(text: str, line_count: int, is_heading: bool) -> str:
    """Guess a layout block type from extracted text."""
    stripped = text.strip()
    if not stripped:
        return "empty"
    if is_heading:
        return "heading"
    if re.match(r"^([-*•]|[0-9]+[.)])\s+", stripped):
        return "list"
    if stripped.count("\n") >= _TABLE_LINE_BREAK_THRESHOLD and re.search(r"\S\s{3,}\S", stripped):
        return "table"
    return "paragraph"


def _extract_heading_candidates_from_text(text: str) -> List[str]:
    """Lightweight heading guesser for OCR/VLM text without layout data."""
    candidates: List[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if len(stripped) > _HEADING_MAX_CHARS:
            continue
        if stripped.endswith((".", "?", "!", "다", "요", ";")):
            continue
        candidates.append(stripped)
        if len(candidates) >= 3:
            break
    return candidates


def _analyze_pymupdf_page(page: fitz.Page) -> Dict[str, Any]:
    """
    Extract sorted text and layout signals from a single PDF page.

    This gives Stage 1C richer metadata without changing the rest of the
    ingestion pipeline shape.
    """
    page_dict = page.get_text("dict", sort=True)
    text_blocks: List[Dict[str, Any]] = []
    font_sizes: List[float] = []
    image_block_count = 0

    for block in page_dict.get("blocks", []):
        block_type = block.get("type", 0)
        if block_type == 1:
            image_block_count += 1
            continue
        if block_type != 0:
            continue

        lines = block.get("lines", []) or []
        line_texts = []
        span_sizes = []
        for line in lines:
            spans = line.get("spans", []) or []
            span_text = "".join(span.get("text", "") for span in spans).strip()
            if span_text:
                line_texts.append(span_text)
            span_sizes.extend(float(span.get("size", 0.0)) for span in spans if span.get("size"))

        block_text = _normalize_extracted_text("\n".join(line_texts))
        if not block_text:
            continue

        max_font = max(span_sizes) if span_sizes else 0.0
        avg_font = sum(span_sizes) / len(span_sizes) if span_sizes else 0.0
        font_sizes.extend(span_sizes)
        text_blocks.append(
            {
                "text": block_text,
                "bbox": block.get("bbox") or [],
                "line_count": max(len(line_texts), 1),
                "max_font": max_font,
                "avg_font": avg_font,
            }
        )

    page_avg_font = sum(font_sizes) / len(font_sizes) if font_sizes else 0.0
    page_max_font = max(font_sizes) if font_sizes else 0.0

    parts: List[str] = []
    block_types: List[str] = []
    heading_candidates: List[str] = []
    heading_bboxes: List[str] = []
    table_like_count = 0

    heading_flags = [
        _looks_like_heading(
            block["text"],
            block["line_count"],
            block["max_font"],
            page_max_font,
            page_avg_font,
        )
        for block in text_blocks
    ]

    if (
        len(text_blocks) >= 3
        and sum(heading_flags) >= max(3, int(len(text_blocks) * 0.6))
    ):
        first_heading_idx = next((idx for idx, flag in enumerate(heading_flags) if flag), None)
        heading_flags = [
            idx == first_heading_idx
            for idx, _ in enumerate(heading_flags)
        ]

    for idx, block in enumerate(text_blocks):
        is_heading = _looks_like_heading(
            block["text"],
            block["line_count"],
            block["max_font"],
            page_max_font,
            page_avg_font,
        )
        if idx < len(heading_flags):
            is_heading = heading_flags[idx]
        block_type = _guess_block_type(block["text"], block["line_count"], is_heading)
        block_types.append(block_type)

        if block_type == "heading":
            parts.append(f"## {block['text']}")
            heading_candidates.append(block["text"])
            heading_bboxes.append(_serialize_bbox(block["bbox"]))
        else:
            parts.append(block["text"])
            if block_type == "table":
                table_like_count += 1

    combined_text = "\n\n".join(parts).strip()
    real_chars = len(re.sub(r"\s+", "", combined_text))
    gib_ratio = round(_gibberish_ratio(combined_text), 3)

    if real_chars == 0 and image_block_count > 0:
        page_kind = "scanned"
    elif image_block_count > 0 and real_chars < _HYBRID_MIN_REAL_CHARS:
        page_kind = "hybrid"
    else:
        page_kind = "digital"

    return {
        "text": combined_text,
        "real_chars": real_chars,
        "gibberish_ratio": gib_ratio,
        "page_kind": page_kind,
        "layout_block_count": len(text_blocks) + image_block_count,
        "layout_text_block_count": len(text_blocks),
        "layout_image_block_count": image_block_count,
        "layout_heading_count": len(heading_candidates),
        "layout_table_like_count": table_like_count,
        "layout_block_types": ",".join(sorted(set(block_types))) if block_types else "",
        "primary_heading": heading_candidates[0] if heading_candidates else "",
        "heading_candidates": " | ".join(heading_candidates[:3]),
        "primary_heading_bbox": heading_bboxes[0] if heading_bboxes else "",
        "page_bbox": _serialize_bbox(list(page.rect)),
        "reading_order_mode": "pymupdf_sort",
        "has_text_layer": bool(real_chars),
        "page_font_avg": round(page_avg_font, 2),
        "page_font_max": round(page_max_font, 2),
    }


def _render_pdf_page_image(pdf_path: str, page_number: int, dpi: int) -> Optional[Image.Image]:
    """Render a single PDF page to a PIL image."""
    try:
        images = convert_from_path(
            pdf_path,
            first_page=page_number,
            last_page=page_number,
            dpi=dpi,
        )
    except Exception as e:
        logger.warning("Failed to render page %d at %d DPI: %s", page_number, dpi, e)
        return None

    return images[0] if images else None


def _image_to_base64(image: Image.Image) -> str:
    """Convert a PIL image to base64 PNG."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _score_extraction_candidate(
    text: str,
    method: str,
    page_kind_hint: str,
) -> float:
    """Score competing extraction candidates for one page."""
    real_chars = len(re.sub(r"\s+", "", text or ""))
    gib_ratio = _gibberish_ratio(text or "")
    confidence = _estimate_confidence(method, real_chars, gib_ratio)

    score = real_chars * max(0.2, 1.0 - min(gib_ratio, 0.7))
    score += confidence * 120

    if method == "text":
        score += 25
        if page_kind_hint == "digital":
            score += 20
    elif method == "vlm" and page_kind_hint in {"scanned", "hybrid"}:
        score += 20
    elif method.startswith("ocr") and page_kind_hint == "scanned":
        score += 10

    return round(score, 2)


def _needs_page_fallback(page_analysis: Dict[str, Any]) -> bool:
    """Decide if a page should escalate beyond PyMuPDF text extraction."""
    real_chars = page_analysis["real_chars"]
    gib_ratio = page_analysis["gibberish_ratio"]
    page_kind = page_analysis["page_kind"]

    if page_kind == "scanned":
        return True
    if gib_ratio > _GIBBERISH_THRESHOLD:
        return True
    if page_kind == "hybrid":
        return real_chars < _HYBRID_MIN_REAL_CHARS
    return real_chars < _PAGE_MIN_REAL_CHARS


# ═══════════════════════════════════════════════════════════════════════
# Method 2: PyMuPDF Text Extraction
# ═══════════════════════════════════════════════════════════════════════

def _extract_pymupdf(pdf_path: str, artifact_meta: dict) -> List[Document]:
    """Extract text page-by-page using PyMuPDF (text layer only)."""
    docs = []
    pdf_file = Path(pdf_path)

    try:
        pdf_doc = fitz.open(pdf_path)
    except Exception as e:
        logger.error("Cannot open PDF %s: %s", pdf_file.name, e)
        return docs

    for i, page in enumerate(pdf_doc):
        page_analysis = _analyze_pymupdf_page(page)
        if not page_analysis["text"]:
            continue
        page_meta = {
            key: value
            for key, value in page_analysis.items()
            if key not in {"text", "page_kind", "real_chars", "gibberish_ratio"}
        }

        docs.append(_make_page_document(
            text=page_analysis["text"],
            base_meta=artifact_meta,
            page=i + 1,
            extraction_method="text",
            page_kind=page_analysis["page_kind"],
            chunk_type="page",
            extractors_used="pymupdf",
            section_title=page_analysis["primary_heading"],
            routing_reason="fast_text_layer",
            fallback_chain="pymupdf",
            **page_meta,
        ))

    pdf_doc.close()
    return docs


# ═══════════════════════════════════════════════════════════════════════
# Method 3: VLM Extraction (Qwen2.5-VL via Ollama) — NEW
# ═══════════════════════════════════════════════════════════════════════

def _render_page_to_base64(pdf_path: str, page_number: int, dpi: int = 200) -> str:
    """Render a single PDF page to a base64-encoded PNG image."""
    image = _render_pdf_page_image(pdf_path, page_number, dpi=dpi)
    if image is None:
        return ""
    return _image_to_base64(image)


def _vlm_extract_page(image_base64: str, page_num: int, timeout: int = 60) -> str:
    """
    Send a page image to Qwen2.5-VL via Ollama and get extracted text.

    This uses Ollama's multimodal API — the vision model "reads" the page
    image and returns all visible text with structure preserved.
    """
    prompt = (
        "이 이미지에서 보이는 모든 텍스트를 정확히 추출해주세요. "
        "텍스트만 출력하고, 설명이나 해석은 하지 마세요. "
        "한국어와 영어 모두 포함해주세요. "
        "줄바꿈과 구조를 유지해주세요."
    )

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/generate",
            json={
                "model": VLM_EXTRACTION_MODEL,
                "prompt": prompt,
                "images": [image_base64],
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 2048,
                },
            },
            timeout=timeout,
        )

        if response.status_code != 200:
            logger.warning("VLM extraction failed for page %d: HTTP %d", page_num, response.status_code)
            return ""

        data = response.json()
        text = (data.get("response") or "").strip()
        return text

    except requests.exceptions.ConnectionError:
        logger.warning("Cannot connect to Ollama for VLM extraction")
        return ""
    except requests.exceptions.Timeout:
        logger.warning("VLM extraction timed out for page %d (%ds)", page_num, timeout)
        return ""
    except Exception as e:
        logger.warning("VLM extraction error for page %d: %s", page_num, e)
        return ""


# Max consecutive VLM failures before aborting (avoids 28min stall)
_VLM_MAX_CONSECUTIVE_FAILURES = 2
# First page gets extra time for cold model loading
_VLM_FIRST_PAGE_TIMEOUT = 90
_VLM_PAGE_TIMEOUT = 30


def _extract_with_vlm(pdf_path: str, artifact_meta: dict) -> List[Document]:
    """
    Extract text from each PDF page using a Vision Language Model.

    Renders each page as an image, sends to Qwen2.5-VL via Ollama,
    and gets back the text content. Far more accurate than tesseract
    for image-heavy documents, Korean text in illustrations, etc.

    Safety: aborts early if VLM fails on consecutive pages (model
    likely unavailable — no point waiting 120s × N pages).
    """
    if not VLM_EXTRACTION_ENABLED:
        logger.info("  → VLM extraction disabled")
        return []

    docs = []

    try:
        pdf_doc = fitz.open(pdf_path)
        total_pages = len(pdf_doc)
    except Exception as e:
        logger.error("Cannot open PDF for VLM extraction: %s", e)
        return docs

    logger.info("  → Running VLM extraction (%s) on %d pages...", VLM_EXTRACTION_MODEL, total_pages)

    consecutive_failures = 0

    for page_num in range(1, total_pages + 1):
        # Early exit if VLM keeps failing
        if consecutive_failures >= _VLM_MAX_CONSECUTIVE_FAILURES:
            logger.warning(
                "  → VLM failed %d consecutive pages — aborting (model likely unavailable)",
                consecutive_failures,
            )
            break

        try:
            page_text_layer = pdf_doc[page_num - 1].get_text("text").strip()
            page_kind = "hybrid" if page_text_layer else "scanned"
            img_b64 = _render_page_to_base64(pdf_path, page_num)
            if not img_b64:
                consecutive_failures += 1
                continue

            # First page gets extra time for cold model loading
            timeout = _VLM_FIRST_PAGE_TIMEOUT if page_num == 1 else _VLM_PAGE_TIMEOUT

            text = _vlm_extract_page(img_b64, page_num, timeout=timeout)
            if not text or len(text.strip()) < 10:
                logger.debug("  → Page %d: no text from VLM", page_num)
                consecutive_failures += 1
                continue

            # Success — reset failure counter
            consecutive_failures = 0
            logger.debug("  → Page %d: %d chars from VLM", page_num, len(text))

            docs.append(_make_page_document(
                text=text,
                base_meta=artifact_meta,
                page=page_num,
                extraction_method="vlm",
                page_kind=page_kind,
                chunk_type="page",
                extractors_used="qwen2.5vl",
            ))
        except Exception as e:
            logger.warning("  → Page %d VLM error: %s", page_num, e)
            consecutive_failures += 1

    pdf_doc.close()
    return docs


# ═══════════════════════════════════════════════════════════════════════
# Method 4: Tesseract OCR (last resort)
# ═══════════════════════════════════════════════════════════════════════

def _ocr_pdf_page(pdf_path: str, page_number: int) -> str:
    """OCR a single page using tesseract."""
    if not ENABLE_OCR:
        return ""
    try:
        image = _render_pdf_page_image(pdf_path, page_number, dpi=_OCR_RENDER_DPI)
        if image is None:
            return ""
        text = pytesseract.image_to_string(image, lang="kor+eng")
        return (text or "").strip()
    except Exception as e:
        logger.warning("OCR failed (page %d): %s", page_number, e)
        return ""


def _extract_with_ocr(pdf_path: str, artifact_meta: dict) -> List[Document]:
    """OCR every page with tesseract. Last resort."""
    if not ENABLE_OCR:
        return []

    docs = []

    try:
        pdf_doc = fitz.open(pdf_path)
        total_pages = len(pdf_doc)
    except Exception:
        return docs

    logger.info("  → Running tesseract OCR on %d pages...", total_pages)

    for page_num in range(1, total_pages + 1):
        page_text_layer = pdf_doc[page_num - 1].get_text("text").strip()
        page_kind = "hybrid" if page_text_layer else "scanned"
        text = _ocr_pdf_page(pdf_path, page_num)
        if not text or len(text.strip()) < 10:
            continue

        docs.append(_make_page_document(
            text=text,
            base_meta=artifact_meta,
            page=page_num,
            extraction_method="ocr",
            page_kind=page_kind,
            chunk_type="page",
            extractors_used="tesseract",
        ))

    pdf_doc.close()
    return docs


def _build_page_candidate_document(
    text: str,
    artifact_meta: dict,
    page_num: int,
    method: str,
    page_analysis: Dict[str, Any],
    routing_reason: str,
    fallback_chain: str,
) -> Optional[Document]:
    """Create a page document candidate with consistent Stage 1 metadata."""
    normalized_text = _normalize_extracted_text(text)
    if not normalized_text:
        return None

    heading_candidates = page_analysis.get("heading_candidates", "")
    primary_heading = page_analysis.get("primary_heading", "")
    if not primary_heading and heading_candidates:
        primary_heading = heading_candidates.split(" | ")[0]

    if method in {"vlm", "ocr", "ocr_image"} and not primary_heading:
        guessed = _extract_heading_candidates_from_text(normalized_text)
        if guessed:
            primary_heading = guessed[0]
            heading_candidates = " | ".join(guessed)

    extractors_used = {
        "text": "pymupdf",
        "vlm": "qwen2.5vl",
        "ocr": "tesseract",
        "ocr_image": "tesseract",
        "marker_pdf": "marker",
    }.get(method, method)

    page_kind = page_analysis.get("page_kind", "digital")
    if method in {"vlm", "ocr", "ocr_image"} and page_kind == "digital":
        page_kind = "hybrid" if page_analysis.get("has_text_layer") else "scanned"

    extra_meta = {
        **{
            key: value
            for key, value in page_analysis.items()
            if key not in {"text", "page_kind", "real_chars", "gibberish_ratio"}
        },
        "primary_heading": primary_heading,
        "heading_candidates": heading_candidates,
        "routing_reason": routing_reason,
        "fallback_chain": fallback_chain,
    }

    if method == "vlm":
        extra_meta["vlm_render_dpi"] = _VLM_RENDER_DPI
    if method in {"ocr", "ocr_image"}:
        extra_meta["ocr_render_dpi"] = _OCR_RENDER_DPI

    return _make_page_document(
        text=normalized_text,
        base_meta=artifact_meta,
        page=page_num,
        extraction_method=method,
        page_kind=page_kind,
        chunk_type="page",
        section_title=primary_heading,
        extractors_used=extractors_used,
        **extra_meta,
    )


def _select_best_page_candidate(
    candidates: List[Document],
    page_kind_hint: str,
) -> Optional[Document]:
    """Choose the best extraction result for a single page."""
    if not candidates:
        return None

    scored = [
        (
            _score_extraction_candidate(
                candidate.page_content,
                candidate.metadata.get("extraction_method", ""),
                page_kind_hint,
            ),
            candidate,
        )
        for candidate in candidates
    ]
    scored.sort(key=lambda item: item[0], reverse=True)
    best_score, best_doc = scored[0]

    text_doc = next(
        (doc for score, doc in scored if doc.metadata.get("extraction_method") == "text"),
        None,
    )
    ocr_doc = next(
        (doc for score, doc in scored if doc.metadata.get("extraction_method") == "ocr"),
        None,
    )

    # For hybrid/scanned pages, prefer OCR when it clearly captures much more
    # text than the thin text layer. This is common for lyric sheets, posters,
    # and image-heavy pages where PyMuPDF only sees headings or fragments.
    if page_kind_hint in {"hybrid", "scanned"} and text_doc is not None and ocr_doc is not None:
        text_real_chars = int(text_doc.metadata.get("real_char_count") or 0)
        ocr_real_chars = int(ocr_doc.metadata.get("real_char_count") or 0)
        ocr_gib_ratio = float(ocr_doc.metadata.get("gibberish_ratio") or 1.0)
        text_flags = set((text_doc.metadata.get("quality_flags") or "").split(",")) - {""}
        if (
            ocr_gib_ratio <= _GIBBERISH_THRESHOLD
            and (
                ocr_real_chars >= max(40, text_real_chars * 2)
                or ("low_text_yield" in text_flags and ocr_real_chars >= text_real_chars + 20)
            )
        ):
            return ocr_doc

    # Prefer the original text layer when it is close in quality. It tends to
    # preserve exact wording better than OCR/VLM for born-digital PDFs.
    if text_doc is not None:
        text_score = next(
            score
            for score, doc in scored
            if doc.metadata.get("extraction_method") == "text"
        )
        text_flags = set((text_doc.metadata.get("quality_flags") or "").split(",")) - {""}
        if (
            page_kind_hint == "digital"
            and "garbled_text" not in text_flags
            and text_score >= best_score - 18
        ):
            return text_doc

    return best_doc


def _candidate_is_good_enough(candidate: Optional[Document], page_kind_hint: str) -> bool:
    """Decide whether a fallback candidate is good enough to skip slower VLM."""
    if candidate is None:
        return False

    method = str(candidate.metadata.get("extraction_method") or "")
    real_chars = int(candidate.metadata.get("real_char_count") or 0)
    gib_ratio = float(candidate.metadata.get("gibberish_ratio") or 1.0)
    if method == "ocr":
        min_real_chars = 30 if page_kind_hint in {"scanned", "hybrid"} else _PAGE_MIN_REAL_CHARS
    else:
        min_real_chars = 30 if page_kind_hint == "scanned" else _PAGE_MIN_REAL_CHARS

    return real_chars >= min_real_chars and gib_ratio <= _GIBBERISH_THRESHOLD


def _parse_pdf_page(
    page: fitz.Page,
    pdf_path: str,
    artifact_meta: Dict[str, Any],
) -> Optional[Document]:
    """Parse one PDF page with quality-gated routing."""
    started_at = time.perf_counter()
    page_num = page.number + 1
    page_analysis = _analyze_pymupdf_page(page)
    candidates: List[Document] = []

    if page_analysis["text"]:
        candidates.append(
            _build_page_candidate_document(
                text=page_analysis["text"],
                artifact_meta=artifact_meta,
                page_num=page_num,
                method="text",
                page_analysis=page_analysis,
                routing_reason="fast_text_layer",
                fallback_chain="pymupdf",
            )
        )

    if not _needs_page_fallback(page_analysis):
        return candidates[0] if candidates else None

    route_reasons = []
    if page_analysis["page_kind"] in {"scanned", "hybrid"}:
        route_reasons.append(page_analysis["page_kind"])
    if page_analysis["real_chars"] < _PAGE_MIN_REAL_CHARS:
        route_reasons.append("low_text_yield")
    if page_analysis["gibberish_ratio"] > _GIBBERISH_THRESHOLD:
        route_reasons.append("garbled_text")
    routing_reason = "+".join(route_reasons) or "quality_gate"

    fallback_chain = ["pymupdf"]
    rendered_page = _render_pdf_page_image(pdf_path, page_num, dpi=max(_OCR_RENDER_DPI, _VLM_RENDER_DPI))
    ocr_candidate: Optional[Document] = None

    if rendered_page is not None and ENABLE_OCR:
        ocr_text = pytesseract.image_to_string(rendered_page, lang="kor+eng")
        if ocr_text and len(ocr_text.strip()) >= 10:
            ocr_chain = fallback_chain + ["tesseract"]
            ocr_candidate = _build_page_candidate_document(
                text=ocr_text,
                artifact_meta=artifact_meta,
                page_num=page_num,
                method="ocr",
                page_analysis=page_analysis,
                routing_reason=routing_reason,
                fallback_chain=">".join(ocr_chain),
            )
            if ocr_candidate is not None:
                candidates.append(ocr_candidate)

    # PaddleOCR v5 — higher-accuracy Korean OCR with PyKoSpacing post-processing.
    # Scores 0.82 vs Tesseract's 0.72, so _select_best_page_candidate prefers it.
    if rendered_page is not None and ENABLE_OCR and PADDLE_OCR_ENABLED:
        paddle_text = _paddle_ocr_page(rendered_page)
        if paddle_text and len(paddle_text.strip()) >= 10:
            paddle_chain = fallback_chain + ["paddleocr"]
            paddle_candidate = _build_page_candidate_document(
                text=paddle_text,
                artifact_meta=artifact_meta,
                page_num=page_num,
                method="paddle_ocr",
                page_analysis=page_analysis,
                routing_reason=routing_reason,
                fallback_chain=">".join(paddle_chain),
            )
            if paddle_candidate is not None:
                candidates.append(paddle_candidate)

    text_candidate = next(
        (candidate for candidate in candidates if candidate.metadata.get("extraction_method") == "text"),
        None,
    )
    text_good_enough = _candidate_is_good_enough(text_candidate, page_analysis["page_kind"])
    ocr_good_enough = _candidate_is_good_enough(ocr_candidate, page_analysis["page_kind"])

    should_try_vlm = False
    if rendered_page is not None and VLM_EXTRACTION_ENABLED:
        if page_analysis["page_kind"] == "scanned":
            should_try_vlm = VLM_SCANNED_PDF_ENABLED and not ocr_good_enough
        elif page_analysis["page_kind"] == "hybrid":
            should_try_vlm = (
                VLM_HYBRID_PDF_ENABLED
                and not text_good_enough
                and not ocr_good_enough
            )
        else:
            should_try_vlm = (
                page_analysis["gibberish_ratio"] > _GIBBERISH_THRESHOLD
                and not text_good_enough
                and not ocr_good_enough
            )

    if should_try_vlm:
        timeout = _VLM_FIRST_PAGE_TIMEOUT if page_num == 1 else _VLM_PAGE_TIMEOUT
        vlm_text = _vlm_extract_page(_image_to_base64(rendered_page), page_num, timeout=timeout)
        if vlm_text and len(vlm_text.strip()) >= 10:
            fallback_chain.append("qwen2.5vl")
            candidate = _build_page_candidate_document(
                text=vlm_text,
                artifact_meta=artifact_meta,
                page_num=page_num,
                method="vlm",
                page_analysis=page_analysis,
                routing_reason=routing_reason,
                fallback_chain=">".join(fallback_chain),
            )
            if candidate is not None:
                candidates.append(candidate)

    selected = _select_best_page_candidate(
        [candidate for candidate in candidates if candidate is not None],
        page_analysis["page_kind"],
    )

    if selected is not None:
        elapsed_ms = round((time.perf_counter() - started_at) * 1000, 1)
        log_fn = logger.info if elapsed_ms >= _SLOW_PAGE_LOG_MS else logger.debug
        log_fn(
            "  → Page %d: %s selected (kind=%s, reason=%s, %.1fms, vlm=%s, ocr=%s)",
            page_num,
            selected.metadata.get("extractors_used"),
            selected.metadata.get("page_kind"),
            selected.metadata.get("routing_reason"),
            elapsed_ms,
            "yes" if should_try_vlm else "no",
            "yes" if ocr_candidate is not None else "no",
        )

    return selected


# ═══════════════════════════════════════════════════════════════════════
# Main Parser — per-page routing with document-level fallback
# ═══════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════
# Method 6: pdfplumber — column-aware + table extraction
# ═══════════════════════════════════════════════════════════════════════

_PDFPLUMBER_COLUMN_MIN_WORDS = 6   # minimum words in each column to declare 2-col layout
_PDFPLUMBER_COLUMN_MARGIN_PX = 15  # pixels of dead-zone around page midpoint


def _pdfplumber_page_text(page: Any) -> str:
    """
    Extract text from one pdfplumber page with automatic column detection.

    If the page has roughly equal word distributions on left and right halves,
    we treat it as 2-column and extract each half separately so reading order
    is preserved (left col first, right col after).
    """
    words = page.extract_words() or []
    if not words:
        return ""

    mid = page.width / 2
    left_words = [w for w in words if w["x1"] <= mid - _PDFPLUMBER_COLUMN_MARGIN_PX]
    right_words = [w for w in words if w["x0"] >= mid + _PDFPLUMBER_COLUMN_MARGIN_PX]

    is_two_column = (
        len(left_words) >= _PDFPLUMBER_COLUMN_MIN_WORDS
        and len(right_words) >= _PDFPLUMBER_COLUMN_MIN_WORDS
        and len(right_words) >= len(left_words) * 0.25  # right col has at least 25% of left
    )

    if is_two_column:
        left_col = page.crop((0, 0, mid, page.height))
        right_col = page.crop((mid, 0, page.width, page.height))
        left_text = (left_col.extract_text() or "").strip()
        right_text = (right_col.extract_text() or "").strip()
        combined = "\n\n".join(p for p in [left_text, right_text] if p)
        return combined

    return (page.extract_text() or "").strip()


def _table_to_text(table: List[List[Any]]) -> str:
    """Convert a pdfplumber table (list of rows) to readable pipe-delimited text."""
    if not table:
        return ""
    lines = []
    for row in table:
        cells = [str(c).strip() if c is not None else "" for c in row]
        lines.append(" | ".join(cells))
    return "\n".join(lines)


def _extract_pdfplumber(pdf_path: str, artifact_meta: dict) -> List[Document]:
    """
    Extract Documents from PDF using pdfplumber with column-aware text ordering
    and structured table extraction.

    Returns one Document per page.  Table items on a page are appended as
    extra Documents with chunk_type='table'.
    """
    if not _PDFPLUMBER_AVAILABLE:
        return []

    docs: List[Document] = []
    try:
        with _pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                page_num = i + 1

                # Column-aware text
                text = _pdfplumber_page_text(page)
                if text:
                    normalized = _normalize_extracted_text(text)
                    real_chars = len(re.sub(r"\s+", "", normalized))
                    gib_ratio = round(_gibberish_ratio(normalized), 3)
                    language = detect_language(normalized[:300])
                    docs.append(Document(
                        page_content=normalized,
                        metadata={
                            **artifact_meta,
                            "page": page_num,
                            "chunk_type": "page",
                            "extraction_method": "pdfplumber",
                            "page_kind": "digital",
                            "extractors_used": "pdfplumber",
                            "text_yield_chars": len(normalized),
                            "real_char_count": real_chars,
                            "gibberish_ratio": gib_ratio,
                            "extraction_confidence": _estimate_confidence("text", real_chars, gib_ratio),
                            "quality_flags": _quality_flags(real_chars, gib_ratio, language),
                            "language": language,
                            "routing_reason": "pdfplumber_column_aware",
                            "fallback_chain": "pdfplumber",
                            "section_title": "",
                            "reading_order_mode": "pdfplumber_column",
                        },
                    ))

                # Tables as separate Documents
                tables = page.extract_tables() or []
                for t_idx, table in enumerate(tables):
                    table_text = _table_to_text(table).strip()
                    if len(table_text) < 10:
                        continue
                    real_chars_t = len(re.sub(r"\s+", "", table_text))
                    gib_ratio_t = round(_gibberish_ratio(table_text), 3)
                    language_t = detect_language(table_text[:300])
                    docs.append(Document(
                        page_content=table_text,
                        metadata={
                            **artifact_meta,
                            "page": page_num,
                            "chunk_type": "table",
                            "extraction_method": "pdfplumber_table",
                            "page_kind": "digital",
                            "extractors_used": "pdfplumber",
                            "text_yield_chars": len(table_text),
                            "real_char_count": real_chars_t,
                            "gibberish_ratio": gib_ratio_t,
                            "extraction_confidence": _estimate_confidence("text", real_chars_t, gib_ratio_t),
                            "quality_flags": _quality_flags(real_chars_t, gib_ratio_t, language_t),
                            "language": language_t,
                            "routing_reason": "pdfplumber_table",
                            "fallback_chain": "pdfplumber",
                            "section_title": f"표 (page {page_num}, table {t_idx + 1})",
                            "table_index_on_page": t_idx,
                            "reading_order_mode": "pdfplumber_table",
                        },
                    ))
    except Exception as exc:
        logger.warning("pdfplumber extraction failed for %s: %s", pdf_path, exc)

    return docs


def _should_prefer_pdfplumber(pymupdf_doc: Optional[Document], pdfplumber_doc: Optional[Document]) -> bool:
    """
    Return True if pdfplumber produced meaningfully better text for a page.

    Criteria:
    - pdfplumber detected a 2-column layout → always prefer (reading order fix)
    - pdfplumber has noticeably more real chars AND less gibberish
    - PyMuPDF is garbled and pdfplumber is clean
    """
    if pdfplumber_doc is None:
        return False
    if pymupdf_doc is None:
        return True

    py_chars = int(pymupdf_doc.metadata.get("real_char_count") or 0)
    py_gib = float(pymupdf_doc.metadata.get("gibberish_ratio") or 1.0)
    pl_chars = int(pdfplumber_doc.metadata.get("real_char_count") or 0)
    pl_gib = float(pdfplumber_doc.metadata.get("gibberish_ratio") or 1.0)

    py_flags = set((pymupdf_doc.metadata.get("quality_flags") or "").split(",")) - {""}

    # Column-aware extraction — always prefer if pdfplumber is reasonably clean
    # and has at least 70% of PyMuPDF char count (small gaps expected due to
    # whitespace normalisation differences between extractors)
    if (pdfplumber_doc.metadata.get("reading_order_mode") == "pdfplumber_column"
            and pl_gib <= _GIBBERISH_THRESHOLD
            and pl_chars >= max(50, py_chars * 0.70)):
        return True

    # PyMuPDF is garbled — pdfplumber is always preferred when clean
    if "garbled_text" in py_flags and pl_gib <= _GIBBERISH_THRESHOLD:
        return True

    # pdfplumber has 25%+ more real chars with equal or better gibberish
    if pl_chars >= py_chars * 1.25 and pl_gib <= py_gib + 0.05:
        return True

    return False


def _apply_pdfplumber_enhancements(
    page_docs: List[Document],
    pdf_path: str,
    artifact_meta: dict,
) -> List[Document]:
    """
    Post-process page_docs: replace any page whose text is worse than what
    pdfplumber can produce, and append standalone table Documents.
    """
    if not _PDFPLUMBER_AVAILABLE:
        return page_docs

    pl_docs = _extract_pdfplumber(pdf_path, artifact_meta)
    if not pl_docs:
        return page_docs

    # Separate text pages from table chunks
    pl_pages: dict[int, Document] = {}
    pl_tables: List[Document] = []
    for doc in pl_docs:
        if doc.metadata.get("chunk_type") == "table":
            pl_tables.append(doc)
        else:
            pnum = doc.metadata.get("page", 0)
            pl_pages[pnum] = doc

    # Possibly replace PyMuPDF pages with pdfplumber versions
    enhanced: List[Document] = []
    replaced = 0
    for doc in page_docs:
        pnum = doc.metadata.get("page", 0)
        pl_doc = pl_pages.get(pnum)
        if _should_prefer_pdfplumber(doc, pl_doc):
            enhanced.append(pl_doc)
            replaced += 1
        else:
            enhanced.append(doc)

    # Add table Documents (deduplicated by content length — avoid tiny tables)
    added_tables = 0
    for table_doc in pl_tables:
        enhanced.append(table_doc)
        added_tables += 1

    if replaced or added_tables:
        logger.info(
            "  → pdfplumber: replaced %d page(s), added %d table chunk(s)",
            replaced, added_tables,
        )

    return enhanced


def parse_pdf(pdf_path: str) -> List[Document]:
    """
    Parse a PDF with per-page routing and quality gates.

    Strategy:
    1. Analyze each page via PyMuPDF for text + layout signals
    2. Keep clean digital pages on the fast text path
    3. Escalate low-yield / hybrid / scanned / garbled pages to VLM and OCR
    4. Choose the best extractor for each page
    5. Only use marker as a document-level fallback when page routing fails badly
    """
    pdf_file = Path(pdf_path)
    logger.info("Parsing PDF: %s", pdf_file.name)

    try:
        pdf_doc = fitz.open(pdf_path)
        total_pages = len(pdf_doc)
    except Exception as e:
        logger.error("Cannot open PDF %s: %s", pdf_file.name, e)
        return []

    artifact_meta = _build_artifact_meta(pdf_file, page_total=total_pages, input_type="pdf")
    marker_text = _extract_with_marker(pdf_path)
    marker_real_chars = _count_real_chars(marker_text) if marker_text else 0

    page_docs: List[Document] = []
    method_counts: Dict[str, int] = {}

    for page in pdf_doc:
        selected = _parse_pdf_page(page, pdf_path, artifact_meta)
        if selected is None:
            continue
        page_docs.append(selected)
        method = selected.metadata.get("extractors_used", selected.metadata.get("extraction_method", "unknown"))
        method_counts[method] = method_counts.get(method, 0) + 1

    pdf_doc.close()

    page_chars = sum(len(doc.page_content) for doc in page_docs)
    avg_chars_per_page = page_chars / max(total_pages, 1)
    page_kind_counts: Dict[str, int] = {}
    for doc in page_docs:
        kind = doc.metadata.get("page_kind", "unknown")
        page_kind_counts[kind] = page_kind_counts.get(kind, 0) + 1

    logger.info(
        "  → Marker: %d real chars | Routed pages: %d chars (%d/%d pages) | Methods: %s | Page kinds: %s",
        marker_real_chars,
        page_chars,
        len(page_docs),
        total_pages,
        method_counts or {},
        page_kind_counts or {},
    )

    if page_docs:
        low_quality_pages = [
            doc for doc in page_docs
            if "garbled_text" in (doc.metadata.get("quality_flags") or "")
        ]
        if marker_text and (
            marker_real_chars > page_chars * _MARKER_FALLBACK_RATIO
            and len(low_quality_pages) >= max(1, total_pages // 3)
        ):
            cleaned = _clean_marker_text(marker_text)
            marker_doc = _make_page_document(
                text=cleaned,
                base_meta=artifact_meta,
                page=1,
                extraction_method="marker_pdf",
                page_kind="digital",
                chunk_type="marker_markdown",
                extractors_used="marker",
                routing_reason="marker_structure_fallback",
                fallback_chain="marker",
                reading_order_mode="marker_markdown",
                layout_block_count=1,
                layout_text_block_count=1,
                layout_image_block_count=0,
                layout_heading_count=len(_HEADING_RE.findall(cleaned)),
                layout_table_like_count=0,
                layout_block_types="heading,paragraph" if _HEADING_RE.search(cleaned) else "paragraph",
                primary_heading="",
                heading_candidates="",
                primary_heading_bbox="",
                page_bbox="",
                has_text_layer=True,
                page_font_avg=0.0,
                page_font_max=0.0,
            )
            logger.info("  → Using marker fallback (structured text significantly better)")
            return [marker_doc]

        logger.info(
            "  → Using routed per-page extraction (avg %.0f chars/page)",
            avg_chars_per_page,
        )
        return _apply_pdfplumber_enhancements(page_docs, pdf_path, artifact_meta)

    if marker_text:
        cleaned = _clean_marker_text(marker_text)
        marker_doc = _make_page_document(
            text=cleaned,
            base_meta=artifact_meta,
            page=1,
            extraction_method="marker_pdf",
            page_kind="digital",
            chunk_type="marker_markdown",
            extractors_used="marker",
            routing_reason="marker_only_result",
            fallback_chain="marker",
            reading_order_mode="marker_markdown",
            layout_block_count=1,
            layout_text_block_count=1,
            layout_image_block_count=0,
            layout_heading_count=len(_HEADING_RE.findall(cleaned)),
            layout_table_like_count=0,
            layout_block_types="heading,paragraph" if _HEADING_RE.search(cleaned) else "paragraph",
            primary_heading="",
            heading_candidates="",
            primary_heading_bbox="",
            page_bbox="",
            has_text_layer=True,
            page_font_avg=0.0,
            page_font_max=0.0,
        )
        logger.info("  → Using marker only (page routing found no usable content)")
        return [marker_doc]

    logger.warning("  → No text extracted from %s", pdf_file.name)
    return []


# ═══════════════════════════════════════════════════════════════════════
# Image Parsing
# ═══════════════════════════════════════════════════════════════════════

def extract_text_from_image(image_path: str) -> tuple:
    """
    OCR an image file. Uses VLM if available, else tesseract.
    Returns (text, method) tuple so caller knows which method succeeded.
    """
    if VLM_EXTRACTION_ENABLED:
        try:
            image = Image.open(image_path)
            img_b64 = _image_to_base64(image)

            text = _vlm_extract_page(img_b64, 1)
            if text and len(text.strip()) > 10:
                return text, "vlm"
        except Exception as e:
            logger.debug("VLM image extraction failed, falling back to tesseract: %s", e)

    # Fallback to tesseract
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image, lang="kor+eng")
        return (text or "").strip(), "ocr_image"
    except Exception as e:
        logger.warning("Image OCR failed for %s: %s", image_path, e)
        return "", "none"


def parse_image(image_path: str) -> List[Document]:
    """Parse an image file into a Document."""
    text, method = extract_text_from_image(image_path)
    if not text:
        return []

    image_file = Path(image_path)
    artifact_meta = _build_artifact_meta(image_file, page_total=1, input_type="image")
    extractors_used = "qwen2.5vl" if method == "vlm" else "tesseract"
    heading_candidates = _extract_heading_candidates_from_text(text)

    try:
        with Image.open(image_path) as image:
            width, height = image.size
    except Exception:
        width, height = 0, 0

    return [_make_page_document(
        text=text,
        base_meta=artifact_meta,
        page=1,
        extraction_method=method,
        page_kind="scanned",
        chunk_type="image",
        extractors_used=extractors_used,
        routing_reason="image_input",
        fallback_chain=extractors_used,
        reading_order_mode="ocr_flat" if method != "vlm" else "vlm_flat",
        layout_block_count=1,
        layout_text_block_count=0,
        layout_image_block_count=1,
        layout_heading_count=len(heading_candidates),
        layout_table_like_count=0,
        layout_block_types="image",
        primary_heading=heading_candidates[0] if heading_candidates else "",
        heading_candidates=" | ".join(heading_candidates),
        primary_heading_bbox="",
        page_bbox="",
        has_text_layer=False,
        page_font_avg=0.0,
        page_font_max=0.0,
        image_width=width,
        image_height=height,
    )]


def extract_full_text(file_path: str) -> str:
    """Extract all text from a PDF or image."""
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        docs = parse_pdf(file_path)
    elif ext in (".png", ".jpg", ".jpeg", ".webp"):
        docs = parse_image(file_path)
    else:
        return ""
    return "\n".join(d.page_content for d in docs if d.page_content)
