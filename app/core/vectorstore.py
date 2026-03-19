"""
ChromaDB vector store management.

IMPROVEMENTS over original:
- Duplicate document detection (original re-ingested same files)
- Collection stats helper
- Isolated from other concerns
"""

import logging
import shutil
from typing import List, Optional, Dict, Any, Tuple

from langchain_core.documents import Document
from langchain_chroma import Chroma

from app.config import CHROMA_DIR, VECTOR_TOP_K
from app.core.embeddings import get_embeddings
from app.retrieval.keyword_index import (
    add_keyword_documents,
    clear_keyword_index,
    rebuild_keyword_index,
)

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
        rebuild_keyword_index(get_all_documents())
        logger.info("Vectorstore ready — %d chunks loaded.", count)

    return _vectorstore


def add_documents(docs: List[Document]) -> int:
    """Add documents to the vectorstore. Returns number added."""
    vs = get_vectorstore()
    vs.add_documents(docs)
    add_keyword_documents(docs)
    logger.info("Added %d chunks to vectorstore.", len(docs))
    return len(docs)


def similarity_search_with_scores(
    query: str,
    k: Optional[int] = None,
    filter_source: Optional[str] = None,
    min_score: Optional[float] = None,
) -> List[Tuple[Document, float]]:
    """Return similarity search results with relevance scores when available."""
    vs = get_vectorstore()
    kwargs = {"k": k or VECTOR_TOP_K}

    if filter_source:
        kwargs["filter"] = {"source": filter_source}

    try:
        results = vs.similarity_search_with_relevance_scores(query, **kwargs)
    except Exception:
        docs = vs.similarity_search(query, **kwargs)
        results = [(doc, 0.0) for doc in docs]

    if min_score is not None:
        results = [(doc, score) for doc, score in results if score >= min_score]

    return results


def similarity_search(
    query: str,
    k: Optional[int] = None,
    filter_source: Optional[str] = None,
    min_score: Optional[float] = None,
) -> List[Document]:
    """
    Search for similar documents.
    If filter_source is provided, only return chunks from that file.
    If min_score is provided, filter out results below that relevance score.
    """
    results = similarity_search_with_scores(
        query=query,
        k=k,
        filter_source=filter_source,
        min_score=min_score,
    )
    if min_score is not None:
        logger.debug(
            "Score filter: %d results above %.2f",
            len(results),
            min_score,
        )
    return [doc for doc, _ in results]


def get_documents_by_source(source: str) -> List[Document]:
    """
    Return all chunks for a single ingested document, ordered by page/chunk.

    Used for whole-document tasks like summarization, structure analysis,
    and key data extraction after a user uploads a file.
    """
    vs = get_vectorstore()
    store = vs.get(
        where={"source": source},
        include=["documents", "metadatas"],
    )

    documents = store.get("documents", []) or []
    metadatas = store.get("metadatas", []) or []

    docs = [
        Document(page_content=content, metadata=metadata or {})
        for content, metadata in zip(documents, metadatas)
    ]

    def _sort_value(value: Any) -> Any:
        if value is None:
            return float("inf")
        if isinstance(value, (int, float)):
            return value
        try:
            return int(value)
        except (TypeError, ValueError):
            return str(value)

    docs.sort(
        key=lambda doc: (
            _sort_value(doc.metadata.get("page")),
            _sort_value(doc.metadata.get("chunk_index")),
        )
    )
    return docs


def get_all_documents() -> List[Document]:
    """Return all stored documents from Chroma with their metadata."""
    vs = get_vectorstore()
    store = vs.get(include=["documents", "metadatas"])

    documents = store.get("documents", []) or []
    metadatas = store.get("metadatas", []) or []
    return [
        Document(page_content=content, metadata=metadata or {})
        for content, metadata in zip(documents, metadatas)
    ]


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
    clear_keyword_index()

    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
        logger.info("Deleted vectorstore at %s", CHROMA_DIR)

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    get_vectorstore()
    logger.info("Vectorstore reset complete.")
