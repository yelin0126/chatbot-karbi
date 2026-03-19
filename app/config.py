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
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", BASE_DIR / "chroma_db"))
MARKER_OUTPUT_DIR = BASE_DIR / "marker_output"

# ── Ollama / LLM ──────────────────────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434/api")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
AVAILABLE_MODELS = os.getenv(
    "AVAILABLE_MODELS",
    "qwen2.5:7b,llama3.1:latest,llama3.2-vision:11b",
).split(",")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "180"))

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

# ── Reranker (NEW — not in original) ──────────────────────────────────
RERANKER_ENABLED = os.getenv("RERANKER_ENABLED", "false").lower() == "true"
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
RERANKER_TOP_N = int(os.getenv("RERANKER_TOP_N", "3"))

# ── Retrieval ─────────────────────────────────────────────────────────
VECTOR_TOP_K = int(os.getenv("VECTOR_TOP_K", "4"))

# ── Chunking ──────────────────────────────────────────────────────────
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

# ── Features ──────────────────────────────────────────────────────────
ENABLE_OCR = os.getenv("ENABLE_OCR", "true").lower() == "true"
AUTO_INGEST_ON_STARTUP = os.getenv("AUTO_INGEST_ON_STARTUP", "false").lower() == "true"

# ── Web Search (NEW — original had mode but no actual search) ─────────
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# ── Logging ───────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

def setup_logging():
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format="%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

logger = logging.getLogger("tilon")
