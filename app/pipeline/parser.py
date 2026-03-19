"""
Document parsing — PDF (marker → PyMuPDF → OCR fallback) and images.

IMPROVEMENTS over original:
- Language detection added (original set "unknown" for every doc)
- Marker output cleanup after extraction (original left temp files)
- Better error handling with logging instead of silent failures
- Separated from chunking logic
"""

import logging
import subprocess
from pathlib import Path
from typing import List, Optional

import fitz  # pymupdf
from PIL import Image
import pytesseract
from pdf2image import convert_from_path
from langchain_core.documents import Document

from app.config import ENABLE_OCR, MARKER_OUTPUT_DIR

logger = logging.getLogger("tilron.parser")


# ── Language Detection (FIXED) ────────────────────────────────────────

def detect_language(text: str) -> str:
    """
    Detect language of text. Returns ISO code or 'unknown'.

    FIXED: langdetect often misidentifies Korean as French/other languages.
    We check for Korean characters (Hangul) first before falling back
    to langdetect.
    """
    import re

    text = text.strip()
    if len(text) < 10:
        return "unknown"

    # Check for Korean characters (Hangul block: AC00-D7AF, Jamo: 1100-11FF, 3130-318F)
    korean_chars = len(re.findall(r'[\uAC00-\uD7AF\u1100-\u11FF\u3130-\u318F]', text))
    total_alpha = len(re.findall(r'[a-zA-Z\uAC00-\uD7AF]', text))

    if total_alpha > 0 and korean_chars / max(total_alpha, 1) > 0.3:
        return "ko"

    try:
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = 0  # Deterministic results
        return detect(text)
    except Exception:
        return "unknown"


# ── Image OCR ─────────────────────────────────────────────────────────

def extract_text_from_image(image_path: str) -> str:
    """OCR an image file and return extracted text."""
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image, lang="kor+eng")
        return (text or "").strip()
    except Exception as e:
        logger.warning("Image OCR failed for %s: %s", image_path, e)
        return ""


# ── PDF Page OCR ──────────────────────────────────────────────────────

def _ocr_pdf_page(pdf_path: str, page_number: int) -> str:
    """OCR a single page from a PDF (1-based index)."""
    if not ENABLE_OCR:
        return ""
    try:
        images = convert_from_path(
            pdf_path,
            first_page=page_number,
            last_page=page_number,
            dpi=200,
        )
        if not images:
            return ""
        text = pytesseract.image_to_string(images[0], lang="kor+eng")
        return (text or "").strip()
    except Exception as e:
        logger.warning("PDF page OCR failed (page %d): %s", page_number, e)
        return ""


# ── Marker PDF ────────────────────────────────────────────────────────

def _extract_with_marker(pdf_path: str) -> str:
    """Use marker_single to extract markdown from a PDF."""
    pdf_file = Path(pdf_path)
    MARKER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        result = subprocess.run(
            [
                "marker_single",
                str(pdf_file),
                "--output_format", "markdown",
                "--output_dir", str(MARKER_OUTPUT_DIR),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=120,  # IMPROVEMENT: timeout to prevent hanging
        )
        if result.stderr:
            logger.debug("[marker stderr] %s", result.stderr[:200])
    except FileNotFoundError:
        logger.warning("marker_single not found — falling back to PyMuPDF.")
        return ""
    except subprocess.CalledProcessError as e:
        logger.warning("marker_single failed: %s", e.stderr[:200] if e.stderr else "unknown")
        return ""
    except subprocess.TimeoutExpired:
        logger.warning("marker_single timed out for %s", pdf_file.name)
        return ""
    except Exception as e:
        logger.warning("marker_single unexpected error: %s — falling back to PyMuPDF.", e)
        return ""

    result_dir = MARKER_OUTPUT_DIR / pdf_file.stem
    if not result_dir.exists():
        return ""

    md_files = list(result_dir.glob("*.md"))
    if not md_files:
        return ""

    try:
        text = md_files[0].read_text(encoding="utf-8").strip()
        # IMPROVEMENT: clean up marker output after reading
        import shutil
        shutil.rmtree(result_dir, ignore_errors=True)
        return text
    except Exception as e:
        logger.warning("Failed to read marker output: %s", e)
        return ""


# ── Helpers ───────────────────────────────────────────────────────────

def _clean_marker_text(text: str) -> str:
    """
    Remove markdown image references from marker output to get actual text.
    Marker outputs ![](_page_X_Picture_Y.jpeg) for images — these aren't
    real content and inflate the character count.
    """
    import re
    # Remove ![](...) image references
    cleaned = re.sub(r'!\[[^\]]*\]\([^)]*\)', '', text)
    # Remove excessive whitespace
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned.strip()


def _count_real_chars(text: str) -> int:
    """Count meaningful characters (not whitespace or markdown syntax)."""
    import re
    # Remove markdown image refs, then count non-whitespace
    cleaned = _clean_marker_text(text)
    return len(re.sub(r'\s+', '', cleaned))


# ── Main PDF Loader ───────────────────────────────────────────────────

def _extract_pymupdf_text_only(pdf_path: str) -> List[Document]:
    """Extract text page-by-page using PyMuPDF (no OCR)."""
    docs: List[Document] = []
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

        lang = detect_language(text[:300])
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": pdf_file.name,
                    "source_path": str(pdf_file),
                    "page": i + 1,
                    "section_title": "",
                    "language": lang,
                    "chunk_type": "page",
                    "extraction_method": "text",
                },
            )
        )

    pdf_doc.close()
    return docs


def _extract_with_ocr(pdf_path: str) -> List[Document]:
    """
    OCR every page of the PDF. Used for image-heavy documents where
    text extraction yields very little content.
    """
    if not ENABLE_OCR:
        logger.info("  → OCR disabled, skipping")
        return []

    docs: List[Document] = []
    pdf_file = Path(pdf_path)

    try:
        pdf_doc = fitz.open(pdf_path)
        total_pages = len(pdf_doc)
        pdf_doc.close()
    except Exception as e:
        logger.error("Cannot open PDF for OCR %s: %s", pdf_file.name, e)
        return docs

    logger.info("  → Running OCR on all %d pages...", total_pages)

    for page_num in range(1, total_pages + 1):
        text = _ocr_pdf_page(pdf_path, page_num)
        if not text or len(text.strip()) < 10:
            continue

        lang = detect_language(text[:300])
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": pdf_file.name,
                    "source_path": str(pdf_file),
                    "page": page_num,
                    "section_title": "",
                    "language": lang,
                    "chunk_type": "page",
                    "extraction_method": "ocr",
                },
            )
        )

    return docs


# Minimum real chars per page to consider extraction "good enough"
# Below this threshold, the PDF is likely image-heavy and needs OCR
MIN_CHARS_PER_PAGE = 200


def parse_pdf(pdf_path: str) -> List[Document]:
    """
    Parse a PDF into Documents. Strategy:

    1. Try marker + PyMuPDF text extraction
    2. Pick whichever has more REAL text (after stripping image refs)
    3. If the best result averages < 200 chars/page → image-heavy PDF
       → OCR every page and use that if it's better

    This handles:
    - Normal text PDFs → marker or PyMuPDF wins quickly
    - Image-heavy PDFs (like 가사-한국을빛낸100명) → OCR kicks in
    - Scanned PDFs → OCR kicks in
    """
    pdf_file = Path(pdf_path)
    logger.info("Parsing PDF: %s", pdf_file.name)

    # Count total pages
    try:
        pdf_doc = fitz.open(pdf_path)
        total_pages = len(pdf_doc)
        pdf_doc.close()
    except Exception:
        total_pages = 1

    # ── Step 1: Try marker ──
    marker_text = _extract_with_marker(pdf_path)
    marker_real_chars = _count_real_chars(marker_text) if marker_text else 0

    # ── Step 2: Try PyMuPDF text extraction ──
    pymupdf_docs = _extract_pymupdf_text_only(pdf_path)
    pymupdf_chars = sum(len(d.page_content) for d in pymupdf_docs)

    logger.info(
        "  → marker: %d real chars | PyMuPDF: %d chars (%d pages) | Total pages: %d",
        marker_real_chars, pymupdf_chars, len(pymupdf_docs), total_pages,
    )

    # ── Step 3: Pick the best text extraction ──
    best_text_chars = max(marker_real_chars, pymupdf_chars)
    avg_chars_per_page = best_text_chars / max(total_pages, 1)

    # ── Step 4: If extraction quality is poor, try OCR ──
    if avg_chars_per_page < MIN_CHARS_PER_PAGE and ENABLE_OCR:
        logger.info(
            "  → Low text density (%.0f chars/page avg) — trying OCR...",
            avg_chars_per_page,
        )
        ocr_docs = _extract_with_ocr(pdf_path)
        ocr_chars = sum(len(d.page_content) for d in ocr_docs)

        logger.info("  → OCR: %d chars (%d pages)", ocr_chars, len(ocr_docs))

        # Use OCR if it got more content
        if ocr_chars > best_text_chars:
            logger.info("  → Using OCR (best extraction)")
            return ocr_docs
        else:
            logger.info("  → OCR didn't improve — falling back to text extraction")

    # ── Step 5: Return the best text extraction ──
    if marker_real_chars > 0 and marker_real_chars >= pymupdf_chars:
        # Use marker but with cleaned text (no image refs)
        cleaned = _clean_marker_text(marker_text)
        lang = detect_language(cleaned[:500])
        logger.info("  → Using marker (cleaned), lang=%s", lang)
        return [
            Document(
                page_content=cleaned,
                metadata={
                    "source": pdf_file.name,
                    "source_path": str(pdf_file),
                    "page": 1,
                    "section_title": "",
                    "language": lang,
                    "chunk_type": "marker_markdown",
                    "extraction_method": "marker_pdf",
                },
            )
        ]
    elif pymupdf_docs:
        logger.info("  → Using PyMuPDF, %d pages", len(pymupdf_docs))
        return pymupdf_docs
    elif marker_text:
        cleaned = _clean_marker_text(marker_text)
        lang = detect_language(cleaned[:500])
        logger.info("  → Using marker (only option)")
        return [
            Document(
                page_content=cleaned,
                metadata={
                    "source": pdf_file.name,
                    "source_path": str(pdf_file),
                    "page": 1,
                    "section_title": "",
                    "language": lang,
                    "chunk_type": "marker_markdown",
                    "extraction_method": "marker_pdf",
                },
            )
        ]
    else:
        logger.warning("  → No text extracted from %s", pdf_file.name)
        return []


# ── Image Loader ──────────────────────────────────────────────────────

def parse_image(image_path: str) -> List[Document]:
    """OCR an image file into a Document."""
    text = extract_text_from_image(image_path)
    if not text:
        return []

    image_file = Path(image_path)
    lang = detect_language(text[:300])

    return [
        Document(
            page_content=text,
            metadata={
                "source": image_file.name,
                "source_path": str(image_path),
                "page": 1,
                "section_title": "",
                "language": lang,
                "chunk_type": "image",
                "extraction_method": "ocr_image",
            },
        )
    ]


# ── Full Text Helper ─────────────────────────────────────────────────

def extract_full_text(file_path: str) -> str:
    """Extract all text from a PDF or image (used by keyword count)."""
    ext = Path(file_path).suffix.lower()

    if ext == ".pdf":
        docs = parse_pdf(file_path)
    elif ext in (".png", ".jpg", ".jpeg", ".webp"):
        docs = parse_image(file_path)
    else:
        return ""

    return "\n".join(d.page_content for d in docs if d.page_content)