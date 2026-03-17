"""
ChatBot API Server — main.py
Ubuntu 22.04 LTS / Ollama + Llama + ChromaDB RAG
"""

import os
import httpx
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# ── 환경 설정 ────────────────────────────────────────────
OLLAMA_URL  = os.getenv("OLLAMA_URL",  "http://localhost:11434")
LLAMA_MODEL = os.getenv("LLAMA_MODEL", "llama3.2:3b")   # ollama pull llama3.2:3b
CHROMA_DIR  = os.getenv("CHROMA_DIR",  "./chroma_db")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
STATIC_DIR  = os.getenv("STATIC_DIR",  "./static")

# ── FastAPI 초기화 ───────────────────────────────────────
app = FastAPI(title="ChatBot API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 임베딩 / 벡터스토어 ─────────────────────────────────
# paraphrase-multilingual: 한국어 포함 다국어 임베딩에 적합
embeddings   = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
vector_store = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings,
)

# ── 정적 파일 (프론트엔드) ───────────────────────────────
os.makedirs(STATIC_DIR, exist_ok=True)

# ── 요청 스키마 ──────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    use_rag: bool = False
    system_prompt: str = "You are a helpful assistant. Answer in the same language as the user's question."

class DeleteRequest(BaseModel):
    collection: str = "all"   # 현재는 "all" 만 지원

# ── 헬스 체크 ────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "model": LLAMA_MODEL, "embed_model": EMBED_MODEL}

# ── 업로드된 PDF 목록 ────────────────────────────────────
@app.get("/documents")
async def list_documents():
    """ChromaDB에 저장된 소스 파일 목록 반환"""
    try:
        results = vector_store.get(include=["metadatas"])
        sources = list({m.get("source", "unknown") for m in results["metadatas"]})
        sources = [os.path.basename(s) for s in sources]
        return {"documents": sorted(sources)}
    except Exception as e:
        return {"documents": [], "error": str(e)}

# ── PDF 업로드 → ChromaDB 저장 ───────────────────────────
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    PDF 파일을 업로드하여 ChromaDB에 임베딩 저장.

    개선 포인트:
    - chunk_size=800, overlap=100  → 문맥 유실 최소화
    - 같은 파일 재업로드 시 기존 청크 삭제 후 재저장 (중복 방지)
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드 가능합니다.")

    tmp_path = f"/tmp/{file.filename}"
    content  = await file.read()

    if len(content) == 0:
        raise HTTPException(status_code=400, detail="빈 파일입니다.")

    with open(tmp_path, "wb") as f:
        f.write(content)

    # ── 기존 동일 파일 청크 삭제 ─────────────────────────
    try:
        existing = vector_store.get(include=["metadatas"])
        ids_to_delete = [
            existing["ids"][i]
            for i, m in enumerate(existing["metadatas"])
            if os.path.basename(m.get("source", "")) == file.filename
        ]
        if ids_to_delete:
            vector_store.delete(ids=ids_to_delete)
    except Exception:
        pass  # 첫 업로드라면 무시

    # ── PDF 로드 및 청킹 ─────────────────────────────────
    try:
        loader = PyPDFLoader(tmp_path)
        docs   = loader.load()
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"PDF 파싱 실패: {e}")

    if not docs:
        raise HTTPException(status_code=422, detail="PDF에서 텍스트를 추출할 수 없습니다. (스캔 이미지 PDF 불가)")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", "。", ".", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    vector_store.add_documents(chunks)
    vector_store.persist()

    return {
        "message": f"{len(chunks)}개 청크가 저장되었습니다.",
        "file": file.filename,
        "pages": len(docs),
        "chunks": len(chunks),
    }

# ── 문서 전체 삭제 ───────────────────────────────────────
@app.delete("/documents")
async def delete_all_documents():
    """ChromaDB의 모든 문서 청크 삭제"""
    try:
        existing = vector_store.get()
        if existing["ids"]:
            vector_store.delete(ids=existing["ids"])
        return {"message": f"{len(existing['ids'])}개 청크 삭제 완료"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── 챗봇 응답 ────────────────────────────────────────────
@app.post("/chat")
async def chat(req: ChatRequest):
    """
    Ollama Llama 모델 호출 (RAG 선택 가능)

    RAG 개선 포인트:
    - k=5 로 검색 수 확대
    - MMR(최대 한계 관련성) 대신 similarity_search_with_score 사용해 임계값 필터링
    """
    context = ""

    if req.use_rag:
        results = vector_store.similarity_search_with_score(req.message, k=5)
        # score가 낮을수록 유사도가 높음 (코사인 거리 기준)
        filtered = [(doc, score) for doc, score in results if score < 1.2]
        if filtered:
            context_parts = []
            for doc, score in filtered:
                source = os.path.basename(doc.metadata.get("source", "unknown"))
                page   = doc.metadata.get("page", "?")
                context_parts.append(
                    f"[출처: {source}, {page+1}페이지]\n{doc.page_content}"
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
            {"role": "user",   "content": req.message},
        ],
        "stream": False,
        "options": {
            "temperature": 0.7,
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
