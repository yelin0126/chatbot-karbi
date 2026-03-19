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
import logging
import subprocess
from pathlib import Path
from typing import List

import fitz  # pymupdf
import requests
from PIL import Image
import pytesseract
from pdf2image import convert_from_path
from langchain_core.documents import Document

from app.config import (
    ENABLE_OCR,
    MARKER_OUTPUT_DIR,
    OLLAMA_BASE_URL,
    VLM_EXTRACTION_ENABLED,
    VLM_EXTRACTION_MODEL,
)

logger = logging.getLogger("tilon.parser")


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


# ═══════════════════════════════════════════════════════════════════════
# Method 2: PyMuPDF Text Extraction
# ═══════════════════════════════════════════════════════════════════════

def _extract_pymupdf(pdf_path: str) -> List[Document]:
    """Extract text page-by-page using PyMuPDF (text layer only)."""
    docs = []
    pdf_file = Path(pdf_path)

    try:
        pdf_doc = fitz.open(pdf_path)
    except Exception as e:
        logger.error("Cannot open PDF %s: %s", pdf_file.name, e)
        return docs

    for i, page in enumerate(pdf_doc):
        text = page.get_text("text").strip()
        if not text:
            continue

        docs.append(Document(
            page_content=text,
            metadata={
                "source": pdf_file.name,
                "source_path": str(pdf_file),
                "page": i + 1,
                "section_title": "",
                "language": detect_language(text[:300]),
                "chunk_type": "page",
                "extraction_method": "text",
            },
        ))

    pdf_doc.close()
    return docs


# ═══════════════════════════════════════════════════════════════════════
# Method 3: VLM Extraction (Qwen2.5-VL via Ollama) — NEW
# ═══════════════════════════════════════════════════════════════════════

def _render_page_to_base64(pdf_path: str, page_number: int, dpi: int = 200) -> str:
    """Render a single PDF page to a base64-encoded PNG image."""
    images = convert_from_path(
        pdf_path,
        first_page=page_number,
        last_page=page_number,
        dpi=dpi,
    )
    if not images:
        return ""

    buffer = io.BytesIO()
    images[0].save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


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
_VLM_PAGE_TIMEOUT = 60


def _extract_with_vlm(pdf_path: str) -> List[Document]:
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
    pdf_file = Path(pdf_path)

    try:
        pdf_doc = fitz.open(pdf_path)
        total_pages = len(pdf_doc)
        pdf_doc.close()
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

            docs.append(Document(
                page_content=text,
                metadata={
                    "source": pdf_file.name,
                    "source_path": str(pdf_file),
                    "page": page_num,
                    "section_title": "",
                    "language": detect_language(text[:300]),
                    "chunk_type": "page",
                    "extraction_method": "vlm",
                },
            ))
        except Exception as e:
            logger.warning("  → Page %d VLM error: %s", page_num, e)
            consecutive_failures += 1

    return docs


# ═══════════════════════════════════════════════════════════════════════
# Method 4: Tesseract OCR (last resort)
# ═══════════════════════════════════════════════════════════════════════

def _ocr_pdf_page(pdf_path: str, page_number: int) -> str:
    """OCR a single page using tesseract."""
    if not ENABLE_OCR:
        return ""
    try:
        images = convert_from_path(pdf_path, first_page=page_number, last_page=page_number, dpi=200)
        if not images:
            return ""
        text = pytesseract.image_to_string(images[0], lang="kor+eng")
        return (text or "").strip()
    except Exception as e:
        logger.warning("OCR failed (page %d): %s", page_number, e)
        return ""


def _extract_with_ocr(pdf_path: str) -> List[Document]:
    """OCR every page with tesseract. Last resort."""
    if not ENABLE_OCR:
        return []

    docs = []
    pdf_file = Path(pdf_path)

    try:
        pdf_doc = fitz.open(pdf_path)
        total_pages = len(pdf_doc)
        pdf_doc.close()
    except Exception:
        return docs

    logger.info("  → Running tesseract OCR on %d pages...", total_pages)

    for page_num in range(1, total_pages + 1):
        text = _ocr_pdf_page(pdf_path, page_num)
        if not text or len(text.strip()) < 10:
            continue

        docs.append(Document(
            page_content=text,
            metadata={
                "source": pdf_file.name,
                "source_path": str(pdf_file),
                "page": page_num,
                "section_title": "",
                "language": detect_language(text[:300]),
                "chunk_type": "page",
                "extraction_method": "ocr",
            },
        ))

    return docs


# ═══════════════════════════════════════════════════════════════════════
# Main Parser — picks the best extraction method
# ═══════════════════════════════════════════════════════════════════════

MIN_CHARS_PER_PAGE = 200  # Below this = image-heavy, needs VLM/OCR


def parse_pdf(pdf_path: str) -> List[Document]:
    """
    Parse a PDF. Strategy:

    1. Try marker + PyMuPDF (fast, text-based)
    2. If text density is good (>200 chars/page avg) → done
    3. If low density → try VLM (Qwen2.5-VL) — best for image-heavy docs
    4. If VLM unavailable → fallback to tesseract OCR
    5. Use whichever method got the most content
    """
    pdf_file = Path(pdf_path)
    logger.info("Parsing PDF: %s", pdf_file.name)

    # Count pages
    try:
        pdf_doc = fitz.open(pdf_path)
        total_pages = len(pdf_doc)
        pdf_doc.close()
    except Exception:
        total_pages = 1

    # ── Step 1: Fast text extraction ──
    marker_text = _extract_with_marker(pdf_path)
    marker_real_chars = _count_real_chars(marker_text) if marker_text else 0

    pymupdf_docs = _extract_pymupdf(pdf_path)
    pymupdf_chars = sum(len(d.page_content) for d in pymupdf_docs)

    logger.info(
        "  → marker: %d real chars | PyMuPDF: %d chars (%d pages) | Total: %d pages",
        marker_real_chars, pymupdf_chars, len(pymupdf_docs), total_pages,
    )

    best_text_chars = max(marker_real_chars, pymupdf_chars)
    avg_chars_per_page = best_text_chars / max(total_pages, 1)

    # ── Step 2: If text extraction is good enough, use it ──
    if avg_chars_per_page >= MIN_CHARS_PER_PAGE:
        if marker_real_chars >= pymupdf_chars and marker_text:
            cleaned = _clean_marker_text(marker_text)
            lang = detect_language(cleaned[:500])
            logger.info("  → Using marker (good text density), lang=%s", lang)
            return [Document(
                page_content=cleaned,
                metadata={
                    "source": pdf_file.name, "source_path": str(pdf_file),
                    "page": 1, "section_title": "", "language": lang,
                    "chunk_type": "marker_markdown", "extraction_method": "marker_pdf",
                },
            )]
        else:
            logger.info("  → Using PyMuPDF (good text density), %d pages", len(pymupdf_docs))
            return pymupdf_docs

    # ── Step 3: Low text density — try VLM extraction ──
    logger.info(
        "  → Low text density (%.0f chars/page) — escalating...",
        avg_chars_per_page,
    )

    vlm_docs = _extract_with_vlm(pdf_path)
    vlm_chars = sum(len(d.page_content) for d in vlm_docs)
    logger.info("  → VLM: %d chars (%d pages)", vlm_chars, len(vlm_docs))

    # ── Step 4: If VLM didn't work, try tesseract ──
    ocr_docs = []
    ocr_chars = 0
    if vlm_chars <= best_text_chars:
        ocr_docs = _extract_with_ocr(pdf_path)
        ocr_chars = sum(len(d.page_content) for d in ocr_docs)
        logger.info("  → Tesseract: %d chars (%d pages)", ocr_chars, len(ocr_docs))

    # ── Step 5: Pick the best result ──
    candidates = [
        ("vlm", vlm_chars, vlm_docs),
        ("ocr", ocr_chars, ocr_docs),
        ("pymupdf", pymupdf_chars, pymupdf_docs),
    ]

    if marker_real_chars > 0 and marker_text:
        cleaned = _clean_marker_text(marker_text)
        lang = detect_language(cleaned[:500])
        marker_docs = [Document(
            page_content=cleaned,
            metadata={
                "source": pdf_file.name, "source_path": str(pdf_file),
                "page": 1, "section_title": "", "language": lang,
                "chunk_type": "marker_markdown", "extraction_method": "marker_pdf",
            },
        )]
        candidates.append(("marker", marker_real_chars, marker_docs))

    # Sort by total chars, pick best
    candidates.sort(key=lambda x: x[1], reverse=True)
    best_name, best_chars, best_docs = candidates[0]

    if best_docs:
        logger.info("  → Using %s (best: %d chars)", best_name, best_chars)
        return best_docs

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
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

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
    return [Document(
        page_content=text,
        metadata={
            "source": image_file.name, "source_path": str(image_path),
            "page": 1, "section_title": "",
            "language": detect_language(text[:300]),
            "chunk_type": "image",
            "extraction_method": method,
        },
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