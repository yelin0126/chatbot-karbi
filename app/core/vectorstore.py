"""
ChromaDB vector store management.

IMPROVEMENTS over original:
- Duplicate document detection (original re-ingested same files)
- Collection stats helper
- Isolated from other concerns
"""

import logging
import shutil
from typing import List, Optional, Dict, Any

from langchain_core.documents import Document
from langchain_chroma import Chroma

from app.config import CHROMA_DIR, VECTOR_TOP_K
from app.core.embeddings import get_embeddings

logger = logging.getLogger("tilon.vectorstore")

_vectorstore: Optional[Chroma] = None


def get_vectorstore() -> Chroma:
    global _vectorstore

    if _vectorstore is None:
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _vectorstore = Chroma(
            collection_name="rag_docs",
            embedding_function=get_embeddings(),
            persist_directory=str(CHROMA_DIR),
        )
        count = _vectorstore._collection.count()
        logger.info("Vectorstore ready — %d chunks loaded.", count)

    return _vectorstore


def add_documents(docs: List[Document]) -> int:
    """Add documents to the vectorstore. Returns number added."""
    vs = get_vectorstore()
    vs.add_documents(docs)
    logger.info("Added %d chunks to vectorstore.", len(docs))
    return len(docs)


def similarity_search(query: str, k: Optional[int] = None) -> List[Document]:
    """Search for similar documents."""
    vs = get_vectorstore()
    return vs.similarity_search(query, k=k or VECTOR_TOP_K)


def get_collection_stats() -> Dict[str, Any]:
    """Return basic stats about the vectorstore."""
    vs = get_vectorstore()
    count = vs._collection.count()
    return {"total_chunks": count}


def get_all_metadata() -> List[Dict]:
    """Return metadata for all stored documents."""
    vs = get_vectorstore()
    store = vs.get(include=["metadatas"])
    return store.get("metadatas", [])


def get_ingested_sources() -> set:
    """
    IMPROVEMENT: Return set of already-ingested filenames.
    Prevents duplicate ingestion of the same file.
    """
    metadata_list = get_all_metadata()
    return {m.get("source") for m in metadata_list if m and m.get("source")}


def reset() -> None:
    """Wipe and re-initialize the vectorstore."""
    global _vectorstore
    _vectorstore = None

    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
        logger.info("Deleted vectorstore at %s", CHROMA_DIR)

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    get_vectorstore()
    logger.info("Vectorstore reset complete.")
