"""
Document ingestion orchestrator — scan folder, parse, chunk, store.

IMPROVEMENTS over original:
- Skips already-ingested files (original re-ingested duplicates every time)
- Returns detailed per-file status
- Separated orchestration from parsing/chunking
- NEW: Single-file ingestion for upload endpoint
"""

import glob
import logging
from pathlib import Path
from typing import Dict, Any, List

from langchain_core.documents import Document

from app.config import LIBRARY_DIR
from app.core.document_registry import infer_source_type, upsert_document
from app.pipeline.parser import parse_pdf, parse_image
from app.pipeline.chunker import chunk_documents
from app.pipeline.enricher import enrich_chunks
from app.core.vectorstore import add_documents, get_ingested_sources

logger = logging.getLogger("tilon.ingest")


def _annotate_source_identity(docs: List[Document], file_path: Path, owner_id: str = None) -> List[Document]:
    """Add source-type metadata before chunking so identity survives downstream."""
    source_type = infer_source_type(file_path)
    doc_scope = "persistent" if source_type == "library" else "chat_upload"
    annotated = []
    for doc in docs:
        annotated.append(
            Document(
                page_content=doc.page_content,
                metadata={
                    **doc.metadata,
                    "source_type": source_type,
                    "doc_scope": doc_scope,
                    "owner_id": owner_id if source_type == "upload" else None,
                },
            )
        )
    return annotated


# ═══════════════════════════════════════════════════════════════════════
# Single File Ingestion (NEW — for /upload endpoint)
# ═══════════════════════════════════════════════════════════════════════

def ingest_single_file(file_path: Path, owner_id: str = None) -> Dict[str, Any]:
    """
    Parse, chunk, and store a single file into the vectorstore.
    Used by the /upload endpoint when users attach files in chat.
    """
    if not file_path.exists():
        return {"message": f"File not found: {file_path.name}", "count": 0}

    ext = file_path.suffix.lower()
    name = file_path.name

    logger.info("Ingesting single file: %s", name)

    # Parse based on file type
    if ext == ".pdf":
        docs = parse_pdf(str(file_path))
    elif ext in (".png", ".jpg", ".jpeg", ".webp"):
        docs = parse_image(str(file_path))
    else:
        return {
            "message": f"Unsupported file type: {ext}",
            "count": 0,
            "file": name,
        }

    docs = _annotate_source_identity(docs, file_path, owner_id=owner_id)

    if not docs:
        return {
            "message": f"Could not extract text from {name}. "
                       "The file may be a scanned image — check if OCR is enabled.",
            "count": 0,
            "file": name,
        }

    # Chunk, enrich, and store
    chunks = chunk_documents(docs)
    chunks = enrich_chunks(chunks)
    add_documents(chunks)
    registry_entry = upsert_document(file_path, docs, len(chunks), owner_id=owner_id)

    logger.info("Ingested %s → %d chunks", name, len(chunks))

    return {
        "message": f"Successfully ingested {name}: {len(chunks)} chunks stored.",
        "count": len(chunks),
        "file": name,
        "doc_id": registry_entry.get("doc_id") if registry_entry else None,
        "source_type": registry_entry.get("source_type") if registry_entry else None,
    }


def ingest_folder(folder_path: Path = None) -> Dict[str, Any]:
    """
    Scan a folder for PDFs and images, parse, chunk, and store.
    Skips files that are already in the vectorstore.
    """
    folder = folder_path or LIBRARY_DIR
    folder.mkdir(parents=True, exist_ok=True)

    # IMPROVEMENT: check which files are already ingested
    already_ingested = get_ingested_sources()
    logger.info(
        "Ingesting from %s — %d files already in DB",
        folder, len(already_ingested),
    )

    # Collect files
    pdf_files = sorted(glob.glob(str(folder / "*.pdf")))
    image_files = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
        image_files.extend(glob.glob(str(folder / ext)))
    image_files = sorted(image_files)

    processed_files = []
    skipped_files = []
    total_chunks = 0

    for file_path_str in pdf_files + image_files:
        path = Path(file_path_str)
        name = path.name
        if name in already_ingested:
            skipped_files.append(name)
            continue

        result = ingest_single_file(path)
        if result.get("count", 0) > 0:
            processed_files.append(name)
            total_chunks += int(result["count"])

    if skipped_files:
        logger.info("Skipped %d already-ingested files: %s", len(skipped_files), skipped_files)

    if total_chunks == 0:
        return {
            "message": "No new documents to ingest.",
            "count": 0,
            "files": [],
            "skipped": skipped_files,
        }

    return {
        "message": f"Ingested {total_chunks} chunks from {len(processed_files)} files.",
        "count": total_chunks,
        "files": processed_files,
        "skipped": skipped_files,
    }
