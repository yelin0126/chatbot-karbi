"""
ChromaDB vector store management.

IMPROVEMENTS over original:
- Duplicate document detection (original re-ingested same files)
- Collection stats helper
- Isolated from other concerns
"""

import logging
from typing import List, Optional, Dict, Any, Tuple

from langchain_core.documents import Document
from langchain_chroma import Chroma

from app.config import CHROMA_DIR, VECTOR_TOP_K
from app.core.embeddings import (
    get_embeddings,
    get_embedding_device,
    switch_embeddings_device,
)
from app.retrieval.keyword_index import (
    add_keyword_documents,
    clear_keyword_index,
    rebuild_keyword_index,
)

logger = logging.getLogger("tilon.vectorstore")

_vectorstore: Optional[Chroma] = None


def _is_cuda_oom(error: Exception) -> bool:
    msg = str(error).lower()
    return "out of memory" in msg and "cuda" in msg


def _recreate_vectorstore_with_current_embeddings() -> Chroma:
    """Recreate Chroma client with the currently active embedding backend."""
    global _vectorstore

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    _vectorstore = Chroma(
        collection_name="rag_docs",
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )
    return _vectorstore


def _switch_to_cpu_embeddings_due_to_oom(error: Exception) -> bool:
    """Switch embedding backend to CPU when CUDA OOM is detected."""
    if not _is_cuda_oom(error):
        return False

    if get_embedding_device() == "cpu":
        return False

    logger.warning(
        "Embedding CUDA OOM detected; switching embeddings to CPU and retrying."
    )

    try:
        switch_embeddings_device("cpu")
        _recreate_vectorstore_with_current_embeddings()
        return True
    except Exception as switch_error:
        logger.error(
            "Failed to switch embeddings to CPU after CUDA OOM: %s",
            switch_error,
        )
        return False


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

    try:
        vs.add_documents(docs)
    except Exception as e:
        if _switch_to_cpu_embeddings_due_to_oom(e):
            vs = get_vectorstore()
            vs.add_documents(docs)
        else:
            raise

    add_keyword_documents(docs)
    logger.info(
        "Added %d chunks to vectorstore (embedding_device=%s).",
        len(docs),
        get_embedding_device(),
    )
    return len(docs)


def _build_where(
    filter_source: Optional[str] = None,
    filter_doc_id: Optional[str] = None,
    filter_source_type: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    clauses = []
    if filter_source:
        clauses.append({"source": filter_source})
    if filter_doc_id:
        clauses.append({"doc_id": filter_doc_id})
    if filter_source_type:
        clauses.append({"source_type": filter_source_type})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def similarity_search_with_scores(
    query: str,
    k: Optional[int] = None,
    filter_source: Optional[str] = None,
    filter_doc_id: Optional[str] = None,
    filter_source_type: Optional[str] = None,
    min_score: Optional[float] = None,
) -> List[Tuple[Document, float]]:
    """Return similarity search results with relevance scores when available."""
    vs = get_vectorstore()
    kwargs = {"k": k or VECTOR_TOP_K}

    where = _build_where(
        filter_source=filter_source,
        filter_doc_id=filter_doc_id,
        filter_source_type=filter_source_type,
    )
    if where:
        kwargs["filter"] = where

    try:
        results = vs.similarity_search_with_relevance_scores(query, **kwargs)
    except Exception as e:
        if _switch_to_cpu_embeddings_due_to_oom(e):
            vs = get_vectorstore()
            try:
                results = vs.similarity_search_with_relevance_scores(query, **kwargs)
            except Exception:
                docs = vs.similarity_search(query, **kwargs)
                results = [(doc, 0.0) for doc in docs]
        else:
            docs = vs.similarity_search(query, **kwargs)
            results = [(doc, 0.0) for doc in docs]

    if min_score is not None:
        results = [(doc, score) for doc, score in results if score >= min_score]

    return results


def similarity_search(
    query: str,
    k: Optional[int] = None,
    filter_source: Optional[str] = None,
    filter_doc_id: Optional[str] = None,
    filter_source_type: Optional[str] = None,
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
        filter_doc_id=filter_doc_id,
        filter_source_type=filter_source_type,
        min_score=min_score,
    )
    if min_score is not None:
        logger.debug(
            "Score filter: %d results above %.2f",
            len(results),
            min_score,
        )
    return [doc for doc, _ in results]


def get_documents_by_source(
    source: Optional[str] = None,
    doc_id: Optional[str] = None,
    source_type: Optional[str] = None,
) -> List[Document]:
    """
    Return all chunks for a single ingested document, ordered by page/chunk.

    Used for whole-document tasks like summarization, structure analysis,
    and key data extraction after a user uploads a file.
    """
    vs = get_vectorstore()
    where = _build_where(
        filter_source=source,
        filter_doc_id=doc_id,
        filter_source_type=source_type,
    )
    kwargs = {"include": ["documents", "metadatas"]}
    if where:
        kwargs["where"] = where
    store = vs.get(**kwargs)

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


def get_documents_by_doc_id(doc_id: str) -> List[Document]:
    """Return all chunks for a single registered document by doc_id."""
    return get_documents_by_source(doc_id=doc_id)


def get_document_chunk_count(
    source: Optional[str] = None,
    doc_id: Optional[str] = None,
    source_type: Optional[str] = None,
) -> int:
    """Return how many chunks exist for a scoped document filter."""
    vs = get_vectorstore()
    where = _build_where(
        filter_source=source,
        filter_doc_id=doc_id,
        filter_source_type=source_type,
    )
    kwargs = {"include": ["metadatas"]}
    if where:
        kwargs["where"] = where
    store = vs.get(**kwargs)
    return len(store.get("metadatas", []) or [])


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


def get_ingested_doc_ids() -> set:
    """Return set of already-ingested stable document IDs."""
    metadata_list = get_all_metadata()
    return {m.get("doc_id") for m in metadata_list if m and m.get("doc_id")}


def delete_documents(
    source: Optional[str] = None,
    doc_id: Optional[str] = None,
    source_type: Optional[str] = None,
) -> int:
    """Delete chunks that match the given filters. Returns deleted chunk count."""
    where = _build_where(
        filter_source=source,
        filter_doc_id=doc_id,
        filter_source_type=source_type,
    )
    if not where:
        return 0

    deleted_count = get_document_chunk_count(
        source=source,
        doc_id=doc_id,
        source_type=source_type,
    )
    if deleted_count <= 0:
        return 0

    vs = get_vectorstore()
    vs._collection.delete(where=where)
    rebuild_keyword_index(get_all_documents())

    logger.info(
        "Deleted %d chunks from vectorstore (source=%s, doc_id=%s, source_type=%s)",
        deleted_count,
        source,
        doc_id,
        source_type,
    )
    return deleted_count


def reset() -> None:
    """Wipe and re-initialize the vectorstore."""
    global _vectorstore
    clear_keyword_index()
    vs = get_vectorstore()

    try:
        vs.reset_collection()
        logger.info("Reset Chroma collection in place.")
    except Exception as e:
        logger.warning("Collection reset failed, recreating collection: %s", e)
        try:
            vs.delete_collection()
        except Exception as delete_error:
            logger.warning("Collection delete fallback failed: %s", delete_error)
        _vectorstore = None
        get_vectorstore()

    rebuild_keyword_index([])
    logger.info("Vectorstore reset complete.")
