"""
ChatBot API Server — main.py
Ubuntu 22.04 LTS / Ollama + Qwen/Llama + ChromaDB RAG + marker-pdf
"""

import os
import glob
import shutil
import subprocess
from pathlib import Path

import httpx
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

# ── 환경 설정 ────────────────────────────────────────────
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
LLAMA_MODEL = os.getenv("LLAMA_MODEL", "qwen2.5:7b")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
STATIC_DIR = os.getenv("STATIC_DIR", "./static")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./data")
MARKER_OUTPUT_DIR = os.getenv("MARKER_OUTPUT_DIR", "./marker_output")

# ── FastAPI 초기화 ───────────────────────────────────────
app = FastAPI(title="ChatBot API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 디렉토리 생성 ────────────────────────────────────────
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MARKER_OUTPUT_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)

# ── 임베딩 / 벡터스토어 ─────────────────────────────────
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
vector_store = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings,
)

# ── 요청 스키마 ──────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    use_rag: bool = False
    system_prompt: str = "You are a helpful assistant. Answer in the same language as the user's question."


# ── 유틸: marker-pdf 추출 ───────────────────────────────
# venv 안의 marker_single 절대경로 자동 탐색
MARKER_BIN = os.path.join(os.path.dirname(sys.executable), "marker_single")

def extract_text_with_marker(pdf_path: str) -> str:
    pdf_file = Path(pdf_path)
    output_root = Path(MARKER_OUTPUT_DIR)
    output_root.mkdir(parents=True, exist_ok=True)

    try:
        result = subprocess.run(
            [
                MARKER_BIN,
                str(pdf_file),
                "--output_format", "markdown",
                "--output_dir", str(output_root),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.stdout:
            print("[marker stdout]", result.stdout)
        if result.stderr:
            print("[marker stderr]", result.stderr)
    except subprocess.CalledProcessError as e:
        print("[marker error]", e.stderr)
        return ""
    except FileNotFoundError:
        print("[marker error] marker_single not found")
        return ""

    result_dir = output_root / pdf_file.stem
    if not result_dir.exists():
        return ""

    md_files = list(result_dir.glob("*.md"))
    if not md_files:
        return ""

    try:
        return md_files[0].read_text(encoding="utf-8").strip()
    except Exception as e:
        print("[marker read error]", e)
        return ""


def load_pdf_documents(pdf_path: str):
    """
    1차: marker-pdf로 구조적 추출
    2차 fallback: PyPDFLoader
    """
    pdf_file = Path(pdf_path)

    marker_text = extract_text_with_marker(pdf_path)
    if marker_text:
        return [
            Document(
                page_content=marker_text,
                metadata={
                    "source": str(pdf_file),
                    "filename": pdf_file.name,
                    "page": 0,
                    "loader": "marker_pdf",
                },
            )
        ]

    try:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        for d in docs:
            d.metadata["filename"] = pdf_file.name
            d.metadata["loader"] = "pypdfloader"
        return docs
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"PDF 파싱 실패: {e}")


def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=120,
        separators=["\n\n", "\n", ".", "。", " ", ""],
    )
    return splitter.split_documents(docs)


def delete_existing_file_chunks(filename: str):
    try:
        existing = vector_store.get(include=["metadatas"])
        ids_to_delete = [
            existing["ids"][i]
            for i, m in enumerate(existing["metadatas"])
            if os.path.basename(m.get("source", "")) == filename
            or m.get("filename", "") == filename
        ]
        if ids_to_delete:
            vector_store.delete(ids=ids_to_delete)
    except Exception:
        pass


# ── 헬스 체크 ────────────────────────────────────────────
@app.get("/health")
async def health():
    try:
        count = vector_store._collection.count()
    except Exception:
        count = 0
    return {
        "status": "ok",
        "model": LLAMA_MODEL,
        "embed_model": EMBED_MODEL,
        "documents_in_vectorstore": count,
    }


# ── 업로드된 PDF 목록 ────────────────────────────────────
@app.get("/documents")
async def list_documents():
    try:
        results = vector_store.get(include=["metadatas"])
        sources = list({
            os.path.basename(m.get("source", m.get("filename", "unknown")))
            for m in results["metadatas"]
        })
        return {"documents": sorted(sources)}
    except Exception as e:
        return {"documents": [], "error": str(e)}


# ── PDF 업로드 → 즉시 파싱 → ChromaDB 저장 ───────────────
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드 가능합니다.")

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="빈 파일입니다.")

    save_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(save_path, "wb") as f:
        f.write(content)

    delete_existing_file_chunks(file.filename)

    docs = load_pdf_documents(save_path)
    if not docs:
        raise HTTPException(status_code=422, detail="PDF에서 텍스트를 추출할 수 없습니다.")

    chunks = split_documents(docs)
    vector_store.add_documents(chunks)
    vector_store.persist()

    return {
        "message": f"{len(chunks)}개 청크가 저장되었습니다.",
        "file": file.filename,
        "pages": len(docs),
        "chunks": len(chunks),
        "loader": docs[0].metadata.get("loader", "unknown"),
    }


# ── data 폴더 전체 재적재 ────────────────────────────────
@app.post("/ingest")
async def ingest_all():
    pdf_files = sorted(glob.glob(os.path.join(UPLOAD_DIR, "*.pdf")))
    total_chunks = 0
    processed = []

    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        delete_existing_file_chunks(filename)
        docs = load_pdf_documents(pdf_path)
        if not docs:
            continue
        chunks = split_documents(docs)
        vector_store.add_documents(chunks)
        total_chunks += len(chunks)
        processed.append(filename)

    vector_store.persist()
    return {
        "message": "재적재 완료",
        "files": processed,
        "chunks": total_chunks,
    }


# ── 전체 문서 삭제 ───────────────────────────────────────
@app.delete("/documents")
async def delete_all_documents():
    try:
        existing = vector_store.get()
        if existing["ids"]:
            vector_store.delete(ids=existing["ids"])
        return {"message": f"{len(existing['ids'])}개 청크 삭제 완료"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Chroma DB 폴더 초기화 ───────────────────────────────
@app.delete("/reset-db")
async def reset_db():
    global vector_store
    try:
        if os.path.exists(CHROMA_DIR):
            shutil.rmtree(CHROMA_DIR)
        os.makedirs(CHROMA_DIR, exist_ok=True)

        vector_store = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings,
        )
        return {"message": "벡터 DB 초기화 완료"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── 챗봇 응답 ────────────────────────────────────────────
@app.post("/chat")
async def chat(req: ChatRequest):
    context = ""
    filtered = []

    if req.use_rag:
        results = vector_store.similarity_search_with_score(req.message, k=5)
        filtered = [(doc, score) for doc, score in results if score < 1.3]

        if filtered:
            context_parts = []
            for doc, score in filtered:
                source = os.path.basename(doc.metadata.get("source", "unknown"))
                page = doc.metadata.get("page", "?")
                context_parts.append(
                    f"[출처: {source}, {page if isinstance(page, int) else page}페이지]\n{doc.page_content}"
                )
            context = "\n\n---\n\n".join(context_parts)

    system = req.system_prompt
    if context:
        system += (
            "\n\n아래 참고 문서를 바탕으로 답변하세요. "
            "문서에 없는 내용은 '문서에서 찾을 수 없습니다'라고 하세요.\n\n"
            f"[참고 문서]\n{context}"
        )

    payload = {
        "model": LLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": req.message},
        ],
        "stream": False,
        "options": {
            "temperature": 0.5,
            "num_ctx": 4096,
        },
    }

    async with httpx.AsyncClient(timeout=180) as client:
        try:
            resp = await client.post(f"{OLLAMA_URL}/api/chat", json=payload)
        except httpx.ConnectError:
            raise HTTPException(
                status_code=503,
                detail="Ollama 서버에 연결할 수 없습니다. 'ollama serve' 명령으로 서버를 실행하세요."
            )

    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)

    reply = resp.json()["message"]["content"]
    return {
        "reply": reply,
        "rag_used": bool(context),
        "context_chunks": len(filtered) if context else 0,
    }


# ── 프론트엔드 서빙 ──────────────────────────────────────
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def root():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "ChatBot API is running. Place index.html in ./static/"}
    
    
@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": LLAMA_MODEL,
                "object": "model",
                "created": 0,
                "owned_by": "local"
            }
        ]
    }
    
    
    
class OpenAIMessage(BaseModel):
    role: str
    content: str

class OpenAIChatRequest(BaseModel):
    model: str
    messages: list[OpenAIMessage]
    temperature: float | None = 0.7
    stream: bool | None = False


@app.post("/v1/chat/completions")
async def openai_chat(req: OpenAIChatRequest):
    system_prompt = ""
    user_message = ""

    for msg in req.messages:
        if msg.role == "system":
            system_prompt = msg.content
        elif msg.role == "user":
            user_message = msg.content

    payload = {
        "model": req.model or LLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt or "You are a helpful assistant."},
            {"role": "user", "content": user_message},
        ],
        "stream": False,
        "options": {
            "temperature": req.temperature or 0.7,
            "num_ctx": 4096,
        },
    }

    async with httpx.AsyncClient(timeout=180) as client:
        resp = await client.post(f"{OLLAMA_URL}/api/chat", json=payload)

    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)

    content = resp.json()["message"]["content"]

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model or LLAMA_MODEL,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }
