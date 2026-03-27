"""
PaddleOCR v5 Korean OCR engine with PyKoSpacing post-processing.

Replaces Tesseract as the primary OCR candidate for scanned/hybrid Korean PDFs.
PaddleOCR PPOCRv5 Korean model delivers ~13 percentage-point accuracy gain over v4,
and substantially better than Tesseract on Korean text without spacing.

PyKoSpacing corrects word-spacing errors that OCR models frequently introduce into
Korean text — critical for semantic integrity in downstream embeddings and retrieval.

Usage:
    from app.core.paddle_ocr import ocr_page_image
    text = ocr_page_image(pil_image)   # returns str or None

Environment:
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True  (set at module level)
    CUDA_VISIBLE_DEVICES=""  (set before TensorFlow/PyKoSpacing import)
"""

import os
import logging
import re
from pathlib import Path

# Prevent PaddleOCR from doing an outbound connectivity check on every import
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
# Suppress verbose paddle/paddlepaddle log spam
os.environ.setdefault("GLOG_minloglevel", "2")

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PADDLE_CACHE_DIR = _PROJECT_ROOT / "data" / "temp" / "paddle_cache"
_PADDLE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
# Keep Paddle/PaddleX caches in the project workspace so model downloads and
# cache reuse do not depend on external home-directory cache paths.
os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(_PADDLE_CACHE_DIR / "paddlex_home"))
os.environ.setdefault("PADDLE_HOME", str(_PADDLE_CACHE_DIR / "paddle_home"))
os.environ.setdefault("XDG_CACHE_HOME", str(_PADDLE_CACHE_DIR / "xdg_cache"))

from typing import Optional
from PIL import Image

logger = logging.getLogger("tilon.paddle_ocr")

_ocr_engine = None
_ocr_load_failed = False

_spacing_engine = None
_spacing_load_failed = False


def _get_engine():
    """Lazy-load PaddleOCR Korean engine (CPU). Returns None on failure."""
    global _ocr_engine, _ocr_load_failed
    if _ocr_engine is not None:
        return _ocr_engine
    if _ocr_load_failed:
        return None
    try:
        from paddleocr import PaddleOCR
        # CPU-only: OCR runs during document ingest, not real-time chat.
        # PaddleOCR 3.4.0 rejects older init kwargs like `use_gpu` and
        # `show_log`, so keep the constructor minimal and version-safe.
        _ocr_engine = PaddleOCR(lang="korean")
        logger.info("PaddleOCR Korean engine loaded (CPU)")
        return _ocr_engine
    except Exception as exc:
        _ocr_load_failed = True
        logger.warning("PaddleOCR unavailable — falling back to Tesseract: %s", exc)
        return None


def _get_spacing():
    """Lazy-load PyKoSpacing (CPU-only via TF_FORCE env vars). Returns None on failure."""
    global _spacing_engine, _spacing_load_failed
    if _spacing_engine is not None:
        return _spacing_engine
    if _spacing_load_failed:
        return None
    # TensorFlow (used internally by PyKoSpacing) fails with the libdevice JIT
    # issue when the GPU is visible. Force CPU before TF initializes by
    # temporarily hiding CUDA devices. PyTorch (already initialized for the LLM
    # and reranker) is unaffected because it caches its device list at startup.
    _prev_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    try:
        from pykospacing import Spacing
        _spacing_engine = Spacing()
        logger.info("PyKoSpacing loaded (CPU)")
        return _spacing_engine
    except Exception as exc:
        _spacing_load_failed = True
        logger.debug("PyKoSpacing unavailable (spacing correction disabled): %s", exc)
        return None
    finally:
        # Restore CUDA_VISIBLE_DEVICES for any later torch/paddle operations
        if _prev_cuda is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = _prev_cuda


_KO_CHAR = re.compile(r"[\uAC00-\uD7AF]")


def _apply_spacing(text: str) -> str:
    """Run PyKoSpacing on Korean lines. Leaves non-Korean lines untouched."""
    spacing = _get_spacing()
    if spacing is None or not text:
        return text
    try:
        lines = text.split("\n")
        corrected = []
        for line in lines:
            stripped = line.strip()
            # Only apply to lines with meaningful Korean content
            if stripped and len(stripped) >= 5 and _KO_CHAR.search(stripped):
                corrected.append(spacing(stripped))
            else:
                corrected.append(line)
        return "\n".join(corrected)
    except Exception as exc:
        logger.debug("PyKoSpacing correction failed: %s", exc)
        return text


def ocr_page_image(image: Image.Image) -> Optional[str]:
    """
    Run PaddleOCR v5 on a PIL Image and return extracted text.

    Post-processes Korean output with PyKoSpacing for better word boundaries.
    Returns None if the engine is unavailable or no text is detected.
    """
    engine = _get_engine()
    if engine is None:
        return None
    try:
        import numpy as np
        img_array = np.array(image.convert("RGB"))
        # PaddleOCR 3.x API: predict() replaces the deprecated ocr()
        results = engine.predict(img_array)
        if not results:
            return None
        result = results[0]
        texts = result.get("rec_texts") if hasattr(result, "get") else result["rec_texts"]
        if not texts:
            return None
        text = "\n".join(t for t in texts if t and t.strip())
        if not text.strip():
            return None
        return _apply_spacing(text)
    except Exception as exc:
        logger.warning("PaddleOCR page extraction failed: %s", exc)
        return None
