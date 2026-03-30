from __future__ import annotations

import itertools
import logging
import queue
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from app.config import (
    MEDIA_ENABLED,
    MEDIA_IMAGE_GENERATION_MODEL,
    MEDIA_JOB_TIMEOUT_SECONDS,
    MEDIA_VIDEO_GENERATION_MODEL,
)
from app.media.runtime import MediaRuntime
from app.media.store import (
    build_asset_output_path,
    create_asset,
    create_job,
    ensure_media_dirs,
    get_job,
    list_assets_for_chat,
    list_jobs_for_chat,
    update_job,
)

logger = logging.getLogger("tilon.media.jobs")

_JOB_PRIORITY = {
    "image_generation": 30,
    "video_generation": 40,
}


class MediaJobManager:
    def __init__(self) -> None:
        self._queue: "queue.PriorityQueue[tuple[int, int, str]]" = queue.PriorityQueue()
        self._seq = itertools.count()
        self._stop_event = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._gpu_lock = threading.RLock()
        self._runtime = MediaRuntime()

    def start(self) -> None:
        if not MEDIA_ENABLED:
            logger.info("Media subsystem disabled; skipping worker startup.")
            return
        if self._worker and self._worker.is_alive():
            return
        ensure_media_dirs()
        self._stop_event.clear()
        self._worker = threading.Thread(target=self._run, name="media-job-worker", daemon=True)
        self._worker.start()
        logger.info("Media job worker started.")

    def stop(self) -> None:
        self._stop_event.set()
        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=3)
        logger.info("Media job worker stopped.")

    def submit_image_job(
        self,
        *,
        chat_id: str,
        prompt: str,
        grounded_brief: Optional[str],
        source_refs: list[dict[str, Any]],
        payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self._submit_job(
            chat_id=chat_id,
            job_type="image_generation",
            prompt=prompt,
            grounded_brief=grounded_brief,
            source_refs=source_refs,
            model_name=MEDIA_IMAGE_GENERATION_MODEL,
            payload=payload,
        )

    def submit_video_job(
        self,
        *,
        chat_id: str,
        prompt: str,
        grounded_brief: Optional[str],
        source_refs: list[dict[str, Any]],
        payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self._submit_job(
            chat_id=chat_id,
            job_type="video_generation",
            prompt=prompt,
            grounded_brief=grounded_brief,
            source_refs=source_refs,
            model_name=MEDIA_VIDEO_GENERATION_MODEL,
            payload=payload,
        )

    def _submit_job(
        self,
        *,
        chat_id: str,
        job_type: str,
        prompt: str,
        grounded_brief: Optional[str],
        source_refs: list[dict[str, Any]],
        model_name: str,
        payload: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        job = create_job(
            chat_id=chat_id,
            job_type=job_type,
            prompt=prompt,
            grounded_brief=grounded_brief,
            source_refs=source_refs,
            model_name=model_name,
            payload=payload or {},
        )
        self._queue.put((_JOB_PRIORITY.get(job_type, 99), next(self._seq), job["job_id"]))
        return job

    def run_with_gpu_lock(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        with self._gpu_lock:
            return func(*args, **kwargs)

    def analyze_image(self, *, image_path: Path, prompt: str) -> Dict[str, Any]:
        with self._gpu_lock:
            return self._runtime.analyze_image(image_path=image_path, prompt=prompt)

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        return get_job(job_id)

    def list_jobs_for_chat(self, chat_id: str) -> list[dict[str, Any]]:
        return list_jobs_for_chat(chat_id)

    def list_assets_for_chat(self, chat_id: str) -> list[dict[str, Any]]:
        return list_assets_for_chat(chat_id)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                _, _, job_id = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._process_job(job_id)
            except Exception:
                logger.exception("Unhandled media job failure: %s", job_id)
            finally:
                self._queue.task_done()

    def _process_job(self, job_id: str) -> None:
        job = get_job(job_id)
        if not job:
            return
        if job.get("status") == "cancelled":
            return

        update_job(job_id, status="running", error=None)
        prompt = str(job.get("grounded_brief") or job.get("prompt") or "").strip()
        payload = dict(job.get("payload") or {})
        started_at = time.monotonic()

        try:
            with self._gpu_lock:
                if job["job_type"] == "image_generation":
                    output_path = build_asset_output_path(job["chat_id"], "image", ".png")
                    metadata = self._runtime.generate_image(
                        prompt=prompt,
                        output_path=output_path,
                        width=payload.get("width"),
                        height=payload.get("height"),
                        num_inference_steps=payload.get("num_inference_steps"),
                        guidance_scale=payload.get("guidance_scale"),
                    )
                    asset = create_asset(
                        chat_id=job["chat_id"],
                        job_id=job_id,
                        asset_type="image",
                        model_name=metadata["model_name"],
                        prompt=job["prompt"],
                        grounded_brief=job.get("grounded_brief"),
                        source_refs=job.get("source_refs") or [],
                        file_path=output_path,
                        metadata=metadata,
                    )
                elif job["job_type"] == "video_generation":
                    output_path = build_asset_output_path(job["chat_id"], "video", ".mp4")
                    metadata = self._runtime.generate_video(
                        prompt=prompt,
                        output_path=output_path,
                        width=payload.get("width"),
                        height=payload.get("height"),
                        num_frames=payload.get("num_frames"),
                        fps=payload.get("fps"),
                    )
                    asset = create_asset(
                        chat_id=job["chat_id"],
                        job_id=job_id,
                        asset_type="video",
                        model_name=metadata["model_name"],
                        prompt=job["prompt"],
                        grounded_brief=job.get("grounded_brief"),
                        source_refs=job.get("source_refs") or [],
                        file_path=output_path,
                        metadata=metadata,
                    )
                else:
                    raise RuntimeError(f"Unsupported media job type: {job['job_type']}")

            elapsed = time.monotonic() - started_at
            if elapsed > MEDIA_JOB_TIMEOUT_SECONDS:
                raise TimeoutError(f"Media job exceeded timeout of {MEDIA_JOB_TIMEOUT_SECONDS}s")

            update_job(job_id, status="completed", asset_id=asset["asset_id"])
        except Exception as exc:
            logger.exception("Media job failed: %s", job_id)
            update_job(job_id, status="failed", error=str(exc))


_MEDIA_JOB_MANAGER: Optional[MediaJobManager] = None


def get_media_job_manager() -> MediaJobManager:
    global _MEDIA_JOB_MANAGER
    if _MEDIA_JOB_MANAGER is None:
        _MEDIA_JOB_MANAGER = MediaJobManager()
    return _MEDIA_JOB_MANAGER


def start_media_job_manager() -> MediaJobManager:
    manager = get_media_job_manager()
    manager.start()
    return manager


def stop_media_job_manager() -> None:
    global _MEDIA_JOB_MANAGER
    if _MEDIA_JOB_MANAGER is not None:
        _MEDIA_JOB_MANAGER.stop()
