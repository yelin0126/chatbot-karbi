"""
Core API routes — /chat, /ingest, /health, /docs-list, etc.

IMPROVEMENTS over original:
- Routes separated from business logic
- Response models for type safety
- Better error messages
- Health check includes more diagnostics
"""

import logging
from pathlib import Path
from typing import List

from fastapi import APIRouter, HTTPException, UploadFile, File, Form

from app.config import (
    OLLAMA_MODEL,
    AVAILABLE_MODELS,
    DATA_DIR,
    LIBRARY_DIR,
    UPLOADS_DIR,
    CHROMA_DIR,
    ENABLE_OCR,
)
from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    IngestRequest,
    CountKeywordRequest,
    SourceInfo,
)
from app.core.llm import check_ollama_health
from app.core.watcher import suppress_watcher_for
from app.core.vectorstore import (
    get_vectorstore,
    get_collection_stats,
    get_all_metadata,
    reset as reset_vectorstore,
)
from app.chat.handlers import handle_chat
from app.pipeline.ingest import ingest_folder, ingest_single_file
from app.pipeline.parser import extract_full_text

logger = logging.getLogger("tilon.api")

router = APIRouter()


# ── Root ───────────────────────────────────────────────────────────────

@router.get("/")
def root():
    return {
        "message": "Tilon AI Chatbot API is running",
        "version": "7.0.0",
        "model": OLLAMA_MODEL,
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
        ollama_status = check_ollama_health()
        stats = get_collection_stats()

        return {
            "status": "ok",
            "ollama": ollama_status["status"],
            "model": OLLAMA_MODEL,
            "available_models": AVAILABLE_MODELS,
            "documents_in_vectorstore": stats["total_chunks"],
            "ocr_enabled": ENABLE_OCR,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {e}")


# ── Available Models ──────────────────────────────────────────────────

@router.get("/models")
def list_models():
    """Return available Ollama models for the UI model selector."""
    return {
        "default": OLLAMA_MODEL,
        "available": AVAILABLE_MODELS,
    }


# ── Chat ───────────────────────────────────────────────────────────────

@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    Main chat endpoint. No hardcoded modes — always searches for context,
    LLM decides how to respond. Works like a normal chatbot.
    """
    try:
        result = handle_chat(
            user_message=req.message,
            history=req.history,
            model=req.model or OLLAMA_MODEL,
            active_source=req.active_source,
            system_prompt=req.system_prompt,
        )

        return ChatResponse(
            model=req.model or OLLAMA_MODEL,
            answer=result["answer"],
            sources=[SourceInfo(**s) for s in result.get("sources", [])],
            mode=result.get("mode", "general"),
            active_source=result.get("active_source", req.active_source),
            done=True,
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

    # Step 2: Ingest into ChromaDB
    ingest_result = ingest_single_file(save_path)

    if ingest_result["count"] == 0:
        return {
            "model": OLLAMA_MODEL,
            "answer": f"파일 '{file.filename}'에서 텍스트를 추출하지 못했습니다. "
                      "스캔된 이미지 PDF일 수 있습니다. OCR 설정을 확인해주세요.",
            "sources": [],
            "mode": "document_qa",
            "ingest": ingest_result,
            "done": True,
        }

    # Step 3: Answer using the unified handler, scoped to this file
    selected_model = model or OLLAMA_MODEL

    result = handle_chat(
        user_message=message,
        model=selected_model,
        active_source=file.filename,
    )

    return {
        "model": selected_model,
        "answer": result["answer"],
        "sources": result.get("sources", []),
        "mode": result.get("mode", "document_qa"),
        "active_source": file.filename,
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
                meta.get("source"),
                meta.get("page"),
                meta.get("chunk_index"),
            )
            unique_docs[key] = {
                "source": meta.get("source"),
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
