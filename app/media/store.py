from __future__ import annotations

import json
import logging
import mimetypes
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

from app.config import MEDIA_JOBS_PATH, MEDIA_OUTPUT_DIR, MEDIA_REGISTRY_PATH

logger = logging.getLogger("tilon.media.store")

_store_lock = Lock()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def ensure_media_dirs() -> None:
    MEDIA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _ensure_parent(MEDIA_REGISTRY_PATH)
    _ensure_parent(MEDIA_JOBS_PATH)


def _load_json(path: Path, empty_key: str) -> Dict[str, Any]:
    ensure_media_dirs()
    if not path.exists():
        return {empty_key: []}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("Resetting unreadable media store file: %s", path)
        return {empty_key: []}


def _save_json(path: Path, data: Dict[str, Any]) -> None:
    ensure_media_dirs()
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def ensure_chat_dir(chat_id: str) -> Path:
    ensure_media_dirs()
    chat_dir = MEDIA_OUTPUT_DIR / chat_id
    chat_dir.mkdir(parents=True, exist_ok=True)
    return chat_dir


def build_asset_output_path(chat_id: str, asset_type: str, extension: str) -> Path:
    chat_dir = ensure_chat_dir(chat_id)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    suffix = extension if extension.startswith(".") else f".{extension}"
    return chat_dir / f"{asset_type}_{stamp}_{uuid.uuid4().hex[:8]}{suffix}"


def create_job(
    *,
    chat_id: str,
    job_type: str,
    prompt: str,
    grounded_brief: Optional[str],
    source_refs: List[Dict[str, Any]],
    model_name: str,
    payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    job_id = f"job_{uuid.uuid4().hex}"
    now = utc_now()
    entry = {
        "job_id": job_id,
        "chat_id": chat_id,
        "job_type": job_type,
        "status": "queued",
        "prompt": prompt,
        "grounded_brief": grounded_brief,
        "source_refs": source_refs or [],
        "model_name": model_name,
        "asset_id": None,
        "error": None,
        "payload": payload or {},
        "created_at": now,
        "updated_at": now,
    }
    with _store_lock:
        registry = _load_json(MEDIA_JOBS_PATH, "jobs")
        registry.setdefault("jobs", []).append(entry)
        _save_json(MEDIA_JOBS_PATH, registry)
    return entry


def update_job(job_id: str, **updates: Any) -> Optional[Dict[str, Any]]:
    with _store_lock:
        registry = _load_json(MEDIA_JOBS_PATH, "jobs")
        jobs = registry.setdefault("jobs", [])
        for job in jobs:
            if job.get("job_id") != job_id:
                continue
            job.update(updates)
            job["updated_at"] = utc_now()
            _save_json(MEDIA_JOBS_PATH, registry)
            return dict(job)
    return None


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    with _store_lock:
        registry = _load_json(MEDIA_JOBS_PATH, "jobs")
    return next((job for job in registry.get("jobs", []) if job.get("job_id") == job_id), None)


def list_jobs_for_chat(chat_id: str) -> List[Dict[str, Any]]:
    with _store_lock:
        registry = _load_json(MEDIA_JOBS_PATH, "jobs")
    jobs = [job for job in registry.get("jobs", []) if job.get("chat_id") == chat_id]
    return sorted(jobs, key=lambda item: item.get("created_at", ""), reverse=True)


def create_asset(
    *,
    chat_id: str,
    job_id: Optional[str],
    asset_type: str,
    model_name: str,
    prompt: str,
    grounded_brief: Optional[str],
    source_refs: List[Dict[str, Any]],
    file_path: Path,
    thumbnail_path: Optional[Path] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    asset_id = f"asset_{uuid.uuid4().hex}"
    mime_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
    created_at = utc_now()
    entry = {
        "asset_id": asset_id,
        "chat_id": chat_id,
        "job_id": job_id,
        "asset_type": asset_type,
        "status": "ready",
        "model_name": model_name,
        "prompt": prompt,
        "grounded_brief": grounded_brief,
        "source_refs": source_refs or [],
        "file_path": str(file_path),
        "file_url": f"/media/assets/{asset_id}",
        "thumbnail_path": str(thumbnail_path) if thumbnail_path else None,
        "thumbnail_url": f"/media/assets/{asset_id}?thumbnail=1" if thumbnail_path else None,
        "mime_type": mime_type,
        "metadata": metadata or {},
        "created_at": created_at,
    }
    with _store_lock:
        registry = _load_json(MEDIA_REGISTRY_PATH, "assets")
        registry.setdefault("assets", []).append(entry)
        _save_json(MEDIA_REGISTRY_PATH, registry)
    return entry


def get_asset(asset_id: str) -> Optional[Dict[str, Any]]:
    with _store_lock:
        registry = _load_json(MEDIA_REGISTRY_PATH, "assets")
    return next((asset for asset in registry.get("assets", []) if asset.get("asset_id") == asset_id), None)


def list_assets_for_chat(chat_id: str) -> List[Dict[str, Any]]:
    with _store_lock:
        registry = _load_json(MEDIA_REGISTRY_PATH, "assets")
    assets = [asset for asset in registry.get("assets", []) if asset.get("chat_id") == chat_id]
    return sorted(assets, key=lambda item: item.get("created_at", ""), reverse=True)
