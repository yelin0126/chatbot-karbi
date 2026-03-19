"""
ChatBot API Server - main.py
Ubuntu 22.04 LTS / Ollama + marker-pdf
DB 없음 - PDF 업로드시 텍스트 직접 추출 후 프롬프트에 삽입
"""

import os
import sys
import uuid
import time
import shutil
import subprocess
from pathlib import Path

import httpx
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# -- 환경 설정 -------------------------------------------------------
OLLAMA_URL        = os.getenv("OLLAMA_URL",        "http://localhost:11434")
LLAMA_MODEL       = os.getenv("LLAMA_MODEL",       "qwen2.5:7b")
STATIC_DIR        = os.getenv("STATIC_DIR",        "./static")
UPLOAD_DIR        = os.getenv("UPLOAD_DIR",        "./data")
MARKER_OUTPUT_DIR = os.getenv("MARKER_OUTPUT_DIR", "./marker_output")

# -- FastAPI 초기화 --------------------------------------------------
app = FastAPI(title="ChatBot API", version="4.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(STATIC_DIR,        exist_ok=True)
os.makedirs(UPLOAD_DIR,        exist_ok=True)
os.makedirs(MARKER_OUTPUT_DIR, exist_ok=True)

# -- marker_single 절대경로 ------------------------------------------
MARKER_BIN = os.path.join(os.path.dirname(sys.executable), "marker_single")

# -- 세션별 PDF 텍스트 저장 (메모리, DB 없음) ------------------------
# { session_id: extracted_text }
_session_pdf: dict[str, str] = {}


# -- 요청 스키마 -----------------------------------------------------
class OpenAIMessage(BaseModel):
    role: str
    content: str

class OpenAIChatRequest(BaseModel):
    model: str
    messages: list[OpenAIMessage]
    temperature: float | None = 0.7
    stream: bool | None = False


# -- marker-pdf 텍스트 추출 ------------------------------------------
def extract_text_with_marker(pdf_path: str) -> str:
    pdf_file    = Path(pdf_path)
    output_root = Path(MARKER_OUTPUT_DIR)
    output_root.mkdir(parents=True, exist_ok=True)

    # 이전 출력 폴더 삭제 후 재추출 (덮어쓰기 보장)
    result_dir = output_root / pdf_file.stem
    if result_dir.exists():
        shutil.rmtree(result_dir)

    try:
        result = subprocess.run(
            [MARKER_BIN, str(pdf_file), "--output_format", "markdown", "--output_dir", str(output_root)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=120,
        )
        print("[marker] OK:", pdf_file.name)
    except subprocess.CalledProcessError as e:
        print("[marker error]", e.stderr[:300])
        return ""
    except subprocess.TimeoutExpired:
        print("[marker error] timeout")
        return ""
    except FileNotFoundError:
        print("[marker error] not found:", MARKER_BIN)
        return ""

    md_files = list(result_dir.glob("*.md"))
    if not md_files:
        print("[marker error] no .md output")
        return ""

    try:
        return md_files[0].read_text(encoding="utf-8").strip()
    except Exception as e:
        print("[marker read error]", e)
        return ""


def extract_text_fallback(pdf_path: str) -> str:
    """marker 실패 시 pypdf fallback"""
    try:
        from pypdf import PdfReader
        reader = PdfReader(pdf_path)
        pages  = [p.extract_text() or "" for p in reader.pages]
        text   = "\n\n".join(pages).strip()
        print("[fallback] pypdf OK, chars:", len(text))
        return text
    except Exception as e:
        print("[fallback error]", e)
        return ""


# -- 헬스 체크 -------------------------------------------------------
@app.get("/health")
async def health():
    return {
        "status":        "ok",
        "model":         LLAMA_MODEL,
        "marker_bin":    MARKER_BIN,
        "marker_exists": os.path.exists(MARKER_BIN),
        "active_sessions": len(_session_pdf),
    }


# -- PDF 업로드 -> 텍스트 추출 -> 세션 저장 -------------------------
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF only")

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Empty file")

    save_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(save_path, "wb") as f:
        f.write(content)

    # marker로 추출, 실패 시 pypdf fallback
    text = extract_text_with_marker(save_path)
    if not text:
        text = extract_text_fallback(save_path)
    if not text:
        raise HTTPException(status_code=422, detail="Cannot extract text from PDF")

    # 세션 ID 발급 후 메모리에 저장
    session_id = uuid.uuid4().hex
    _session_pdf[session_id] = text

    print("[upload] session:", session_id, "chars:", len(text))

    return {
        "session_id": session_id,
        "filename":   file.filename,
        "chars":      len(text),
        "message":    "PDF 추출 완료. session_id를 채팅에 포함하세요.",
    }


# -- 세션 초기화 -----------------------------------------------------
@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    _session_pdf.pop(session_id, None)
    return {"message": "session cleared"}

@app.delete("/sessions")
async def clear_all_sessions():
    count = len(_session_pdf)
    _session_pdf.clear()
    return {"message": str(count) + " sessions cleared"}


# -- 프론트엔드 서빙 -------------------------------------------------
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
async def root():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "ChatBot API is running"}


# -- OpenAI 호환 API -------------------------------------------------
@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{"id": LLAMA_MODEL, "object": "model", "created": int(time.time()), "owned_by": "ollama"}],
    }


@app.post("/v1/chat/completions")
async def openai_chat(req: OpenAIChatRequest):
    system_prompt = ""
    user_message  = ""
    session_id    = ""

    for msg in req.messages:
        if msg.role == "system":
            system_prompt = msg.content
        elif msg.role == "user":
            user_message = msg.content

    # 메시지에서 session_id 추출 (형식: "[session:xxxx] 질문내용")
    if user_message.startswith("[session:"):
        end = user_message.find("]")
        if end != -1:
            session_id   = user_message[9:end]
            user_message = user_message[end+1:].strip()

    # 세션에 PDF 텍스트가 있으면 프롬프트에 삽입
    pdf_text = _session_pdf.get(session_id, "")

    base_system = system_prompt or "You are a helpful assistant. Always answer in Korean."
    if pdf_text:
        # 토큰 제한 고려해서 앞 8000자만 사용
        truncated = pdf_text[:8000]
        base_system = (
            "You are a helpful assistant. Always answer in Korean.\n\n"
            "Answer based on the document below only. "
            "If the answer is not in the document, say so.\n\n"
            "[Document]\n" + truncated
        )

    payload = {
        "model": req.model or LLAMA_MODEL,
        "messages": [
            {"role": "system", "content": base_system},
            {"role": "user",   "content": user_message},
        ],
        "stream": False,
        "options": {"temperature": req.temperature or 0.7, "num_ctx": 8192},
    }

    async with httpx.AsyncClient(timeout=180) as client:
        resp = await client.post(OLLAMA_URL + "/api/chat", json=payload)

    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)

    answer = resp.json()["message"]["content"]

    return {
        "id":      "chatcmpl-" + uuid.uuid4().hex,
        "object":  "chat.completion",
        "created": int(time.time()),
        "model":   req.model or LLAMA_MODEL,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}],
        "usage":   {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


# -- PDF 세션 기반 채팅 엔드포인트 ----------------------------------
class PdfChatRequest(BaseModel):
    session_id: str
    message: str
    system_prompt: str = "당신은 사내 AI 어시스턴트입니다. 한국어로 친절하고 명확하게 답변하세요."

@app.post("/chat-pdf")
async def chat_pdf(req: PdfChatRequest):
    pdf_text = _session_pdf.get(req.session_id, "")
    if not pdf_text:
        raise HTTPException(status_code=404, detail="PDF 세션이 없습니다. 먼저 PDF를 업로드하세요.")

    # PDF 텍스트를 프롬프트에 직접 삽입 (토큰 제한 고려 앞 8000자)
    truncated   = pdf_text[:8000]
    base_system = (
        req.system_prompt + "\n\n"
        "아래 문서를 기반으로만 답변하세요. "
        "문서에 없는 내용은 '문서에서 찾을 수 없습니다'라고 하세요. "
        "반드시 한국어로 답변하세요.\n\n"
        "[문서 내용]\n" + truncated
    )

    payload = {
        "model": LLAMA_MODEL,
        "messages": [
            {"role": "system", "content": base_system},
            {"role": "user",   "content": req.message},
        ],
        "stream": False,
        "options": {"temperature": 0.5, "num_ctx": 8192},
    }

    async with httpx.AsyncClient(timeout=180) as client:
        try:
            resp = await client.post(OLLAMA_URL + "/api/chat", json=payload)
        except httpx.ConnectError:
            raise HTTPException(status_code=503, detail="Ollama 서버에 연결할 수 없습니다.")

    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)

    reply = resp.json()["message"]["content"]
    return {
        "reply":      reply,
        "session_id": req.session_id,
        "rag_used":   True,
    }
