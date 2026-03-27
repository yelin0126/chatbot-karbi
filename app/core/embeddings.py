"""
Embedding model singleton.

IMPROVEMENTS over original:
- GPU auto-detection (original hardcoded CPU — very slow for bge-m3)
- Lazy loading with proper logging
- Isolated from other concerns
"""

import logging

from app.config import EMBEDDING_MODEL, EMBEDDING_DEVICE

logger = logging.getLogger("tilon.embeddings")

_embedding_model = None


def get_embeddings():
    global _embedding_model

    if _embedding_model is None:
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError as exc:
            raise RuntimeError(
                "langchain_huggingface is not installed; embedding features are unavailable"
            ) from exc

        logger.info(
            "Loading embedding model '%s' on device '%s'...",
            EMBEDDING_MODEL,
            EMBEDDING_DEVICE,
        )
        _embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": EMBEDDING_DEVICE},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info("Embedding model loaded successfully.")

    return _embedding_model
