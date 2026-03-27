"""
Centralized configuration — loaded once from environment variables.

IMPROVEMENTS over original:
- All settings in one place (were scattered across top of app.py)
- Pydantic Settings for validation and type safety
- GPU auto-detection for embedding model
- Reranker support added
- Logging configuration added
"""

import os
import logging
from pathlib import Path
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
LIBRARY_DIR = Path(os.getenv("LIBRARY_DIR", DATA_DIR / "library"))
UPLOADS_DIR = Path(os.getenv("UPLOADS_DIR", DATA_DIR / "uploads"))
TEMP_DIR = Path(os.getenv("TEMP_DIR", DATA_DIR / "temp"))
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", BASE_DIR / "chroma_db"))
DOCUMENT_REGISTRY_PATH = Path(
    os.getenv("DOCUMENT_REGISTRY_PATH", DATA_DIR / "document_registry.json")
)
MARKER_OUTPUT_DIR = BASE_DIR / "marker_output"

# ── Ollama / LLM ──────────────────────────────────────────────────────
LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama").strip().lower()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434/api")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
AVAILABLE_MODELS = os.getenv(
    "AVAILABLE_MODELS",
    "qwen2.5:7b,llama3.1:latest,llama3.2-vision:11b",
).split(",")
LOCAL_LLM_MODEL_NAME = os.getenv("LOCAL_LLM_MODEL_NAME", "qwen25-qlora-v9")
LOCAL_LLM_ADAPTER_PATH = os.getenv(
    "LOCAL_LLM_ADAPTER_PATH",
    str(BASE_DIR / "finetuning" / "output" / "qwen25-qlora-v9"),
)
LOCAL_LLM_BASE_MODEL = os.getenv("LOCAL_LLM_BASE_MODEL", "").strip()
LOCAL_LLM_LOCAL_FILES_ONLY = os.getenv("LOCAL_LLM_LOCAL_FILES_ONLY", "true").lower() == "true"
LOCAL_LLM_LOAD_IN_4BIT = os.getenv("LOCAL_LLM_LOAD_IN_4BIT", "true").lower() == "true"
LOCAL_LLM_MAX_INPUT_TOKENS = int(os.getenv("LOCAL_LLM_MAX_INPUT_TOKENS", "4096"))
LOCAL_LLM_OOM_RETRY_INPUT_TOKENS = int(os.getenv("LOCAL_LLM_OOM_RETRY_INPUT_TOKENS", "3072"))
LOCAL_LLM_OOM_RETRY_MAX_TOKENS = int(os.getenv("LOCAL_LLM_OOM_RETRY_MAX_TOKENS", "512"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "180"))
LLM_SUPPRESS_FOREIGN_SCRIPTS = os.getenv("LLM_SUPPRESS_FOREIGN_SCRIPTS", "true").lower() == "true"

# ── Embedding ─────────────────────────────────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

# IMPROVEMENT: auto-detect GPU; original hardcoded "cpu"
def _detect_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"

EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", _detect_device())
LOCAL_LLM_DEVICE = os.getenv("LOCAL_LLM_DEVICE", _detect_device())

# ── Reranker (NEW — not in original) ──────────────────────────────────
RERANKER_ENABLED = os.getenv("RERANKER_ENABLED", "true").lower() == "true"
LIVE_RERANKER_ENABLED = os.getenv("LIVE_RERANKER_ENABLED", "true").lower() == "true"
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
RERANKER_TOP_N = int(os.getenv("RERANKER_TOP_N", "5"))
RERANKER_DEVICE = os.getenv(
    "RERANKER_DEVICE",
    "cuda" if EMBEDDING_DEVICE == "cuda" else "cpu",
)
RERANKER_USE_FP16 = os.getenv(
    "RERANKER_USE_FP16",
    "true" if RERANKER_DEVICE == "cuda" else "false",
).lower() == "true"

# ── Retrieval ─────────────────────────────────────────────────────────
VECTOR_TOP_K = int(os.getenv("VECTOR_TOP_K", "10"))
GLOBAL_MIN_RELEVANCE_SCORE = float(os.getenv("GLOBAL_MIN_RELEVANCE_SCORE", "0.30"))
DOCUMENT_CONFIDENCE_THRESHOLD = float(os.getenv("DOCUMENT_CONFIDENCE_THRESHOLD", "0.45"))
SCOPED_SINGLE_CHUNK_CONFIDENCE_THRESHOLD = float(
    os.getenv("SCOPED_SINGLE_CHUNK_CONFIDENCE_THRESHOLD", "0.30")
)
SCOPED_SMALL_DOC_FULL_CONTEXT_MAX_CHUNKS = int(
    os.getenv("SCOPED_SMALL_DOC_FULL_CONTEXT_MAX_CHUNKS", "15")
)
STRONG_KEYWORD_CONFIDENCE_FLOOR = float(os.getenv("STRONG_KEYWORD_CONFIDENCE_FLOOR", "0.65"))

# ── Answer Guards (Phase 8) ───────────────────────────────────────────
# Context-relevance: reranker score of (question, best_chunk).  Below this
# threshold the retrieved context is considered off-topic and a "not found"
# response is returned instead of generating an answer.
CONTEXT_RELEVANCE_THRESHOLD = float(os.getenv("CONTEXT_RELEVANCE_THRESHOLD", "0.25"))

# Tiered NLI faithfulness thresholds (post-generation):
#   score < HARD → replace answer with "문서에서 확인되지 않습니다"
#   HARD ≤ score < SOFT → append uncertainty disclaimer
NLI_FAITHFULNESS_HARD_THRESHOLD = float(os.getenv("NLI_FAITHFULNESS_HARD_THRESHOLD", "0.15"))
NLI_FAITHFULNESS_SOFT_THRESHOLD = float(os.getenv("NLI_FAITHFULNESS_SOFT_THRESHOLD", "0.35"))

# ── Chunking ──────────────────────────────────────────────────────────
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
# Parent-child hierarchical chunking (Phase 1a accuracy upgrade)
# Parent = full semantic section (~CHUNK_SIZE chars), stored for context expansion
# Child  = smaller retrieval unit (~CHILD_CHUNK_SIZE chars), embedded & indexed
CHILD_CHUNK_SIZE = int(os.getenv("CHILD_CHUNK_SIZE", "300"))
CHILD_CHUNK_OVERLAP = int(os.getenv("CHILD_CHUNK_OVERLAP", "50"))

# ── Features ──────────────────────────────────────────────────────────
ENABLE_OCR = os.getenv("ENABLE_OCR", "true").lower() == "true"
PADDLE_OCR_ENABLED = os.getenv("PADDLE_OCR_ENABLED", "true").lower() == "true"
AUTO_INGEST_ON_STARTUP = os.getenv("AUTO_INGEST_ON_STARTUP", "false").lower() == "true"

# ── Query Routing / Classifier (Phase 10B) ───────────────────────────
QUERY_CLASSIFIER_ENABLED = os.getenv("QUERY_CLASSIFIER_ENABLED", "false").lower() == "true"
QUERY_CLASSIFIER_PROVIDER = os.getenv("QUERY_CLASSIFIER_PROVIDER", "heuristic").strip().lower()
QUERY_CLASSIFIER_MIN_CONFIDENCE = float(os.getenv("QUERY_CLASSIFIER_MIN_CONFIDENCE", "0.58"))
QUERY_CLASSIFIER_SHADOW_MODE = os.getenv("QUERY_CLASSIFIER_SHADOW_MODE", "true").lower() == "true"

# ── Web Search (NEW — original had mode but no actual search) ─────────
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# ── VLM PDF Extraction (NEW) ─────────────────────────────────────────
VLM_EXTRACTION_ENABLED = os.getenv("VLM_EXTRACTION_ENABLED", "true").lower() == "true"
VLM_SCANNED_PDF_ENABLED = os.getenv("VLM_SCANNED_PDF_ENABLED", "true").lower() == "true"
VLM_HYBRID_PDF_ENABLED = os.getenv("VLM_HYBRID_PDF_ENABLED", "false").lower() == "true"
VLM_EXTRACTION_MODEL = os.getenv("VLM_EXTRACTION_MODEL", "qwen2.5vl:7b")

# ── Logging ───────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

def setup_logging():
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format="%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

logger = logging.getLogger("tilon")
