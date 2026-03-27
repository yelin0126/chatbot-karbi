"""
Parent text store for hierarchical chunking.

Maps parent_id (uuid str) → {text, doc_id, source, page, section}.
Stored as a compact JSON file in DATA_DIR/parent_store.json.

Usage:
  save_parents(parent_docs)         — called during ingestion
  get_parent(parent_id) -> str|None — called during retrieval context expansion
  clear_parents_for_doc(doc_id)     — called before re-ingesting a document
"""

import json
import logging
import threading
from typing import Dict, List, Optional

from langchain_core.documents import Document

from app.config import DATA_DIR

logger = logging.getLogger("tilon.parent_store")

_STORE_PATH = DATA_DIR / "parent_store.json"
_lock = threading.Lock()
_cache: Optional[Dict[str, dict]] = None


def _load() -> Dict[str, dict]:
    global _cache
    if _cache is not None:
        return _cache
    if _STORE_PATH.exists():
        with open(_STORE_PATH, "r", encoding="utf-8") as f:
            _cache = json.load(f)
    else:
        _cache = {}
    return _cache


def _flush(store: Dict[str, dict]) -> None:
    _STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_STORE_PATH, "w", encoding="utf-8") as f:
        json.dump(store, f, ensure_ascii=False, separators=(",", ":"))


def save_parents(parents: List[Document]) -> None:
    """Persist parent chunks to disk. Called during ingestion before add_documents()."""
    if not parents:
        return
    with _lock:
        store = _load()
        for p in parents:
            pid = p.metadata.get("parent_id")
            if not pid:
                continue
            store[pid] = {
                "text": p.page_content,
                "doc_id": str(p.metadata.get("doc_id", "")),
                "source": str(p.metadata.get("source", "")),
                "page": str(p.metadata.get("page", "")),
                "section": str(p.metadata.get("section_breadcrumb", "")),
            }
        _flush(store)
    logger.info("Saved %d parent chunks to parent store", len(parents))


def get_parent(parent_id: str) -> Optional[str]:
    """Return parent text for the given parent_id, or None if not found."""
    with _lock:
        store = _load()
    entry = store.get(parent_id)
    return entry["text"] if entry else None


def clear_parents_for_doc(doc_id: str) -> int:
    """Remove all parent entries belonging to a doc_id. Returns number removed."""
    with _lock:
        store = _load()
        to_remove = [
            pid for pid, v in store.items()
            if v.get("doc_id") == str(doc_id)
        ]
        for pid in to_remove:
            del store[pid]
        if to_remove:
            _flush(store)
    if to_remove:
        logger.info(
            "Removed %d parent entries for doc_id=%s", len(to_remove), doc_id
        )
    return len(to_remove)
