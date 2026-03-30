"""
Core API routes — /chat, /ingest, /health, /docs-list, etc.

IMPROVEMENTS over original:
- Routes separated from business logic
- Response models for type safety
- Better error messages
- Health check includes more diagnostics
"""

import logging
import time
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Query
from fastapi.responses import FileResponse

from app.config import (
    DATA_DIR,
    LIBRARY_DIR,
    UPLOADS_DIR,
    CHROMA_DIR,
    ENABLE_OCR,
    MEDIA_ENABLED,
)
from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    GenerateImageRequest,
    GenerateVideoRequest,
    IngestRequest,
    CountKeywordRequest,
    GeneratedAssetInfo,
    MediaJobInfo,
    SourceInfo,
)
from app.core.llm import check_llm_health, get_available_models, get_default_model_name
from app.core.document_registry import clear_document_registry, list_documents, remove_documents
from app.core.watcher import suppress_watcher_for
from app.core.vectorstore import (
    get_vectorstore,
    get_collection_stats,
    get_all_metadata,
    delete_documents,
    reset as reset_vectorstore,
)
from app.chat.handlers import handle_chat
from app.media.jobs import get_media_job_manager
from app.media.router import (
    build_grounded_media_brief,
    detect_media_intent,
    resolve_image_target_path,
)
from app.media.store import get_asset
from app.pipeline.ingest import ingest_folder, ingest_single_file
from app.pipeline.parser import extract_full_text

logger = logging.getLogger("tilon.api")

router = APIRouter()


def _ensure_chat_id(chat_id: Optional[str]) -> str:
    if chat_id and chat_id.strip():
        return chat_id.strip()
    return f"chat_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}"


def _build_chat_response(
    *,
    model: str,
    answer: str,
    mode: str,
    chat_id: Optional[str],
    result: Optional[dict] = None,
    sources: Optional[list] = None,
    media: Optional[list] = None,
    job: Optional[dict] = None,
    active_source: Optional[str] = None,
    active_doc_id: Optional[str] = None,
    active_source_type: Optional[str] = None,
    active_sources: Optional[list] = None,
    active_doc_ids: Optional[list] = None,
) -> ChatResponse:
    payload = result or {}
    raw_sources = payload.get("sources", []) if sources is None else sources
    return ChatResponse(
        model=model,
        answer=answer,
        sources=[SourceInfo(**s) for s in raw_sources],
        mode=mode,
        active_source=payload.get("active_source", active_source),
        active_doc_id=payload.get("active_doc_id", active_doc_id),
        active_source_type=payload.get("active_source_type", active_source_type),
        active_sources=payload.get("active_sources", active_sources or []),
        active_doc_ids=payload.get("active_doc_ids", active_doc_ids or []),
        chat_id=chat_id,
        media=[GeneratedAssetInfo(**item) for item in (media or [])],
        job=MediaJobInfo(**job) if job else None,
        done=True,
    )


# ── Root ───────────────────────────────────────────────────────────────

@router.get("/")
def root():
    return {
        "message": "Tilon AI Chatbot API is running",
        "version": "7.0.0",
        "model": get_default_model_name(),
        "data_dir": str(DATA_DIR),
        "library_dir": str(LIBRARY_DIR),
        "uploads_dir": str(UPLOADS_DIR),
        "chroma_dir": str(CHROMA_DIR),
        "ocr_enabled": ENABLE_OCR,
    }


# ── Health ─────────────────────────────────────────────────────────────

@router.get("/health")
def health():
    try:
        llm_status = check_llm_health()
        stats = get_collection_stats()

        return {
            "status": "ok",
            "ollama": llm_status["status"],
            "llm_backend": llm_status.get("backend"),
            "llm_status": llm_status["status"],
            "model": get_default_model_name(),
            "available_models": get_available_models(),
            "documents_in_vectorstore": stats["total_chunks"],
            "ocr_enabled": ENABLE_OCR,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {e}")


# ── Available Models ──────────────────────────────────────────────────

@router.get("/models")
def list_models():
    """Return available runtime models for the UI model selector."""
    return {
        "default": get_default_model_name(),
        "available": get_available_models(),
    }


# ── Chat ───────────────────────────────────────────────────────────────

@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    Main chat endpoint. No hardcoded modes — always searches for context,
    LLM decides how to respond. Works like a normal chatbot.
    """
    try:
        selected_model = req.model or get_default_model_name()
        chat_id = _ensure_chat_id(req.chat_id)

        if MEDIA_ENABLED:
            intent = detect_media_intent(req.message, active_source=req.active_source)
            manager = get_media_job_manager()

            if intent == "image_understanding":
                image_path = resolve_image_target_path(req.active_source, req.active_doc_id)
                if not image_path:
                    raise HTTPException(
                        status_code=400,
                        detail="Image understanding requires an active uploaded image in the current chat scope.",
                    )
                analysis = manager.analyze_image(image_path=image_path, prompt=req.message)
                return _build_chat_response(
                    model=selected_model,
                    answer=analysis["answer"],
                    mode="image_understanding",
                    chat_id=chat_id,
                    sources=[],
                    media=[],
                    active_source=req.active_source,
                    active_doc_id=req.active_doc_id,
                    active_source_type=req.active_source_type,
                    active_sources=req.active_sources,
                    active_doc_ids=req.active_doc_ids,
                )

            if intent in {"image_generation", "video_generation"}:
                brief = build_grounded_media_brief(
                    user_message=req.message,
                    active_source=req.active_source,
                    active_doc_id=req.active_doc_id,
                    active_source_type=req.active_source_type,
                )
                if intent == "image_generation":
                    job = manager.submit_image_job(
                        chat_id=chat_id,
                        prompt=req.message,
                        grounded_brief=brief["brief"],
                        source_refs=brief["sources"],
                    )
                    answer = "이미지 생성을 시작했습니다. 완료되면 이 채팅에 결과를 연결해 드릴게요."
                    mode = "image_generation"
                else:
                    job = manager.submit_video_job(
                        chat_id=chat_id,
                        prompt=req.message,
                        grounded_brief=brief["brief"],
                        source_refs=brief["sources"],
                    )
                    answer = "영상 생성을 시작했습니다. 완료되면 이 채팅에 결과를 연결해 드릴게요."
                    mode = "video_generation"

                return _build_chat_response(
                    model=selected_model,
                    answer=answer,
                    mode=mode,
                    chat_id=chat_id,
                    sources=brief["sources"],
                    media=[],
                    job=job,
                    active_source=req.active_source,
                    active_doc_id=req.active_doc_id,
                    active_source_type=req.active_source_type,
                    active_sources=req.active_sources,
                    active_doc_ids=req.active_doc_ids,
                )

        result = handle_chat(
            user_message=req.message,
            history=req.history,
            model=selected_model,
            active_source=req.active_source,
            active_doc_id=req.active_doc_id,
            active_source_type=req.active_source_type,
            active_sources=req.active_sources,
            active_doc_ids=req.active_doc_ids,
            system_prompt=req.system_prompt,
        )

        return _build_chat_response(
            model=selected_model,
            answer=result["answer"],
            mode=result.get("mode", "general"),
            chat_id=chat_id,
            result=result,
            active_source=req.active_source,
            active_doc_id=req.active_doc_id,
            active_source_type=req.active_source_type,
            active_sources=req.active_sources,
            active_doc_ids=req.active_doc_ids,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Chat failed")
        raise HTTPException(status_code=500, detail=f"Chat failed: {e}")


# ── Chat with File Upload (NEW — the missing piece) ──────────────────

@router.post("/chat-with-file")
async def chat_with_file(
    file: UploadFile = File(...),
    message: str = Form(default="이 문서의 내용을 요약해줘"),
    model: str = Form(default=None),
    chat_id: str = Form(default=None),
):
    """
    Upload a file AND ask a question about it in one request.
    The file is saved, parsed, chunked, stored, then the question is answered.

    Usage:
      curl -X POST http://localhost:8000/chat-with-file \
        -F "file=@document.pdf" \
        -F "message=이 문서의 주요 내용은?"
    """
    allowed_extensions = {".pdf", ".png", ".jpg", ".jpeg", ".webp"}
    ext = Path(file.filename).suffix.lower()

    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}",
        )

    # Step 1: Save the file to chat uploads
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = UPLOADS_DIR / file.filename

    try:
        with open(save_path, "wb") as f:
            content = await file.read()
            f.write(content)
        suppress_watcher_for(save_path)
        logger.info("Saved uploaded file: %s (%d bytes)", file.filename, len(content))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    selected_model = model or get_default_model_name()
    resolved_chat_id = _ensure_chat_id(chat_id)

    if MEDIA_ENABLED and ext in {".png", ".jpg", ".jpeg", ".webp"}:
        intent = detect_media_intent(message or "이 이미지를 설명해줘", active_source=file.filename)
        if intent == "image_understanding" or not message.strip():
            try:
                analysis = get_media_job_manager().analyze_image(
                    image_path=save_path,
                    prompt=message or "이 이미지를 설명해줘",
                )
                return {
                    "model": selected_model,
                    "answer": analysis["answer"],
                    "sources": [],
                    "mode": "image_understanding",
                    "chat_id": resolved_chat_id,
                    "active_source": file.filename,
                    "active_doc_id": None,
                    "active_sources": [file.filename],
                    "active_doc_ids": [],
                    "media": [],
                    "done": True,
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Image understanding failed: {e}")

    # Step 2: Ingest into ChromaDB
    ingest_result = ingest_single_file(save_path)

    if ingest_result["count"] == 0:
        return {
            "model": get_default_model_name(),
            "answer": f"파일 '{file.filename}'에서 텍스트를 추출하지 못했습니다. "
                      "스캔된 이미지 PDF일 수 있습니다. OCR 설정을 확인해주세요.",
            "sources": [],
            "mode": "document_qa",
            "ingest": ingest_result,
            "done": True,
        }

    # Step 3: Answer using the unified handler, scoped to this file
    result = handle_chat(
        user_message=message,
        model=selected_model,
        active_source=file.filename,
        active_doc_id=ingest_result.get("doc_id"),
    )

    return {
        "model": selected_model,
        "answer": result["answer"],
        "sources": result.get("sources", []),
        "mode": result.get("mode", "document_qa"),
        "active_source": file.filename,
        "active_doc_id": ingest_result.get("doc_id"),
        "active_sources": [file.filename],
        "active_doc_ids": [ingest_result.get("doc_id")] if ingest_result.get("doc_id") else [],
        "chat_id": resolved_chat_id,
        "ingest": ingest_result,
        "done": True,
    }


# ── Ingest ─────────────────────────────────────────────────────────────

@router.post("/ingest")
def ingest(req: IngestRequest):
    folder = Path(req.folder_path) if req.folder_path else LIBRARY_DIR

    try:
        result = ingest_folder(folder)
        return result
    except Exception as e:
        logger.exception("Ingest failed")
        raise HTTPException(status_code=500, detail=f"Ingest failed: {e}")


# ── Upload (NEW) ──────────────────────────────────────────────────────

ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".webp"}

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a file, parse it, chunk it, and store it in the vectorstore.

    This is the MISSING PIECE from the original code:
    The chat UI lets users attach files, but the backend had no way
    to receive and process them. Now it does.

    Usage:
        curl -X POST http://localhost:8000/upload -F "file=@document.pdf"
    """
    # Validate file type
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {ALLOWED_EXTENSIONS}",
        )

    # Save to chat uploads directory
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = UPLOADS_DIR / file.filename

    try:
        with open(save_path, "wb") as f:
            content = await file.read()
            f.write(content)
        suppress_watcher_for(save_path)
        logger.info("Saved uploaded file: %s (%d bytes)", file.filename, len(content))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    # Parse, chunk, and store
    try:
        result = ingest_single_file(save_path)

        if result["count"] == 0:
            raise HTTPException(
                status_code=422,
                detail=result.get("message", "Could not extract text from file."),
            )

        return {
            "message": result["message"],
            "filename": file.filename,
            "chunks_stored": result["count"],
            "doc_id": result.get("doc_id"),
            "source_type": result.get("source_type"),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Upload processing failed")
        raise HTTPException(status_code=500, detail=f"Upload processing failed: {e}")


@router.post("/upload-multiple")
async def upload_multiple_files(files: List[UploadFile] = File(...)):
    """Upload and ingest multiple files at once."""
    results = []

    for file in files:
        ext = Path(file.filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            results.append({"file": file.filename, "status": "skipped", "reason": f"Unsupported: {ext}"})
            continue

        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        save_path = UPLOADS_DIR / file.filename

        try:
            with open(save_path, "wb") as f:
                content = await file.read()
                f.write(content)
            suppress_watcher_for(save_path)

            result = ingest_single_file(save_path)
            results.append({
                "file": file.filename,
                "status": "success" if result["count"] > 0 else "failed",
                "chunks": result["count"],
                "message": result["message"],
                "doc_id": result.get("doc_id"),
                "source_type": result.get("source_type"),
            })
        except Exception as e:
            results.append({"file": file.filename, "status": "error", "reason": str(e)})

    total_chunks = sum(r.get("chunks", 0) for r in results)
    return {
        "message": f"Processed {len(results)} files, {total_chunks} total chunks stored.",
        "results": results,
    }


# ── Reset DB ───────────────────────────────────────────────────────────

@router.delete("/reset-db")
def reset_db():
    try:
        reset_vectorstore()
        clear_document_registry()
        return {"message": "벡터 DB 초기화 완료"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"reset-db failed: {e}")


# ── Document List ──────────────────────────────────────────────────────

@router.get("/docs-list")
def docs_list():
    try:
        metadata_list = get_all_metadata()

        unique_docs = {}
        for meta in metadata_list:
            if not meta:
                continue
            key = (
                meta.get("doc_id"),
                meta.get("source"),
                meta.get("page"),
                meta.get("chunk_index"),
            )
            unique_docs[key] = {
                "doc_id": meta.get("doc_id"),
                "source": meta.get("source"),
                "source_type": meta.get("source_type"),
                "page_total": meta.get("page_total"),
                "page": meta.get("page"),
                "chunk_index": meta.get("chunk_index"),
                "source_path": meta.get("source_path"),
                "extraction_method": meta.get("extraction_method"),
            }

        return {
            "count": len(unique_docs),
            "documents": list(unique_docs.values()),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"docs-list failed: {e}")


@router.get("/uploaded-docs")
def uploaded_docs():
    """Return remembered chat-upload documents from the document registry."""
    try:
        uploads = [
            {
                "doc_id": doc.get("doc_id"),
                "source": doc.get("source"),
                "source_type": doc.get("source_type"),
                "source_path": doc.get("source_path"),
                "page_total": doc.get("page_total"),
                "chunk_count": doc.get("chunk_count"),
                "languages": doc.get("languages", []),
                "updated_at": doc.get("updated_at"),
                "uploaded_at": doc.get("uploaded_at"),
            }
            for doc in list_documents()
            if doc.get("source_type") == "upload"
        ]
        uploads.sort(
            key=lambda doc: doc.get("updated_at") or doc.get("uploaded_at") or "",
            reverse=True,
        )
        return {
            "count": len(uploads),
            "documents": uploads,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"uploaded-docs failed: {e}")


@router.delete("/upload-document")
def delete_upload_document(
    source: Optional[str] = None,
    doc_id: Optional[str] = None,
):
    """Delete one remembered upload from vectorstore, registry, and local uploads."""
    if not source and not doc_id:
        raise HTTPException(status_code=400, detail="source 또는 doc_id 중 하나는 필요합니다.")

    try:
        deleted_chunks = delete_documents(
            source=source,
            doc_id=doc_id,
            source_type="upload",
        )
        removed_registry = remove_documents(
            source=source,
            doc_id=doc_id,
            source_type="upload",
        )

        file_deleted = False
        if source:
            upload_path = UPLOADS_DIR / Path(source).name
            if upload_path.exists() and upload_path.is_file():
                upload_path.unlink()
                file_deleted = True

        if deleted_chunks == 0 and removed_registry == 0 and not file_deleted:
            raise HTTPException(status_code=404, detail="삭제할 업로드 문서를 찾지 못했습니다.")

        return {
            "message": "업로드 파일이 삭제되었습니다.",
            "deleted_chunks": deleted_chunks,
            "removed_registry": removed_registry,
            "file_deleted": file_deleted,
            "source": source,
            "doc_id": doc_id,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"upload-document delete failed: {e}")


# ── Keyword Count ──────────────────────────────────────────────────────

@router.post("/count-keyword")
def count_keyword(req: CountKeywordRequest):
    try:
        candidate_paths = [
            UPLOADS_DIR / req.filename,
            LIBRARY_DIR / req.filename,
            DATA_DIR / req.filename,
        ]
        target_path = next((path for path in candidate_paths if path.exists()), None)

        if target_path is None:
            raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다.")

        text = extract_full_text(str(target_path))
        if not text:
            return {
                "filename": req.filename,
                "keyword": req.keyword,
                "count": 0,
                "message": "추출된 텍스트가 없습니다.",
            }

        count = text.lower().count(req.keyword.lower())

        return {
            "filename": req.filename,
            "keyword": req.keyword,
            "count": count,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"count-keyword failed: {e}")


# ── Media Generation / Assets ────────────────────────────────────────

@router.post("/generate-image")
def generate_image(req: GenerateImageRequest):
    if not MEDIA_ENABLED:
        raise HTTPException(status_code=503, detail="Media generation is disabled.")

    chat_id = _ensure_chat_id(req.chat_id)
    brief = build_grounded_media_brief(
        user_message=req.prompt,
        active_source=req.active_source,
        active_doc_id=req.active_doc_id,
        active_source_type=req.active_source_type,
    )
    job = get_media_job_manager().submit_image_job(
        chat_id=chat_id,
        prompt=req.prompt,
        grounded_brief=brief["brief"],
        source_refs=brief["sources"],
        payload={
            "width": req.width,
            "height": req.height,
            "num_inference_steps": req.num_inference_steps,
            "guidance_scale": req.guidance_scale,
        },
    )
    return {
        "chat_id": chat_id,
        "job": job,
        "media": [],
        "sources": brief["sources"],
        "answer": "이미지 생성을 시작했습니다.",
    }


@router.post("/generate-video")
def generate_video(req: GenerateVideoRequest):
    if not MEDIA_ENABLED:
        raise HTTPException(status_code=503, detail="Media generation is disabled.")

    chat_id = _ensure_chat_id(req.chat_id)
    brief = build_grounded_media_brief(
        user_message=req.prompt,
        active_source=req.active_source,
        active_doc_id=req.active_doc_id,
        active_source_type=req.active_source_type,
    )
    job = get_media_job_manager().submit_video_job(
        chat_id=chat_id,
        prompt=req.prompt,
        grounded_brief=brief["brief"],
        source_refs=brief["sources"],
        payload={
            "width": req.width,
            "height": req.height,
            "num_frames": req.num_frames,
            "fps": req.fps,
        },
    )
    return {
        "chat_id": chat_id,
        "job": job,
        "media": [],
        "sources": brief["sources"],
        "answer": "영상 생성을 시작했습니다.",
    }


@router.get("/media/jobs/{job_id}")
def get_media_job(job_id: str):
    job = get_media_job_manager().get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Media job not found.")
    asset = get_asset(job["asset_id"]) if job.get("asset_id") else None
    return {
        "job": job,
        "asset": asset,
    }


@router.get("/media/assets/{asset_id}")
def get_media_asset(asset_id: str, thumbnail: int = Query(default=0)):
    asset = get_asset(asset_id)
    if not asset:
        raise HTTPException(status_code=404, detail="Media asset not found.")

    key = "thumbnail_path" if thumbnail else "file_path"
    file_path = Path(asset.get(key) or asset.get("file_path") or "")
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Media asset file is missing.")
    return FileResponse(file_path)


@router.get("/media/chats/{chat_id}")
def get_chat_media(chat_id: str):
    manager = get_media_job_manager()
    return {
        "chat_id": chat_id,
        "jobs": manager.list_jobs_for_chat(chat_id),
        "media": manager.list_assets_for_chat(chat_id),
    }
