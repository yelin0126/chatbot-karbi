import os
import glob
import uuid
import time
import shutil
from pathlib import Path
from typing import List, Literal, Optional, Dict, Any

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

load_dotenv()

# =========================
# 기본 설정
# =========================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", BASE_DIR / "chroma_db"))

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434/api")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:latest")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
ENABLE_OCR = os.getenv("ENABLE_OCR", "true").lower() == "true"
AUTO_INGEST_ON_STARTUP = os.getenv("AUTO_INGEST_ON_STARTUP", "true").lower() == "true"

VECTOR_TOP_K = int(os.getenv("VECTOR_TOP_K", "4"))

# =========================
# LangChain / 문서 처리
# =========================
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# PDF / OCR
import fitz  # pymupdf
from PIL import Image
import pytesseract
from pdf2image import convert_from_path

# =========================
# FastAPI
# =========================
app = FastAPI(title="LLama OCR RAG OpenAI-Compatible API", version="4.0.0")


# =========================
# 요청/응답 모델
# =========================
class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    message: str
    system_prompt: Optional[str] = (
        "너는 한국어로 답하는 AI 챗봇이다. "
        "문서와 이미지에서 추출된 텍스트를 기반으로 답한다. "
        "짧은 입력에는 짧고 자연스럽게 답하고, "
        "문서에 없는 내용은 추측하지 않는다."
    )
    history: List[Message] = Field(default_factory=list)


class IngestRequest(BaseModel):
    folder_path: Optional[str] = None


class CountKeywordRequest(BaseModel):
    filename: str
    keyword: str


# OpenAI 호환 요청 모델
class OpenAIMessage(BaseModel):
    role: str
    content: str


class OpenAIChatRequest(BaseModel):
    model: str
    messages: List[OpenAIMessage]
    temperature: Optional[float] = 0.2
    stream: Optional[bool] = False


# =========================
# 전역 객체
# =========================
embedding_model = None
vectorstore = None


# =========================
# 유틸
# =========================
def get_embeddings():
    global embedding_model
    if embedding_model is None:
        embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return embedding_model


def get_vectorstore():
    global vectorstore
    if vectorstore is None:
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        vectorstore = Chroma(
            collection_name="rag_docs",
            embedding_function=get_embeddings(),
            persist_directory=str(CHROMA_DIR),
        )
    return vectorstore


def call_ollama(prompt: str, temperature: float = 0.2) -> Dict[str, Any]:
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": 1024,
                },
            },
            timeout=180,
        )

        if response.status_code != 200:
            raise HTTPException(
                status_code=500,
                detail=f"Ollama 오류: status={response.status_code}, body={response.text}",
            )

        return response.json()

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Ollama API request failed: {e}")


def extract_text_from_image(image_path: str) -> str:
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image, lang="kor+eng")
        return (text or "").strip()
    except Exception:
        return ""


def extract_text_from_pdf_page_with_ocr(pdf_path: str, page_number_1based: int) -> str:
    if not ENABLE_OCR:
        return ""

    try:
        images = convert_from_path(
            pdf_path,
            first_page=page_number_1based,
            last_page=page_number_1based,
            dpi=200,
        )
        if not images:
            return ""

        text = pytesseract.image_to_string(images[0], lang="kor+eng")
        return (text or "").strip()
    except Exception:
        return ""


def load_pdf_documents(pdf_path: str) -> List[Document]:
    docs: List[Document] = []
    pdf_file = Path(pdf_path)

    try:
        pdf_doc = fitz.open(pdf_path)
    except Exception:
        return docs

    for i, page in enumerate(pdf_doc):
        text = page.get_text("text").strip()
        extraction_method = "text"

        if len(text) < 30 and ENABLE_OCR:
            ocr_text = extract_text_from_pdf_page_with_ocr(pdf_path, i + 1)
            if ocr_text:
                text = ocr_text
                extraction_method = "ocr"

        if not text:
            continue

        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": pdf_file.name,
                    "source_path": str(pdf_file),
                    "page": i + 1,
                    "extraction_method": extraction_method,
                },
            )
        )

    pdf_doc.close()
    return docs


def load_image_document(image_path: str) -> List[Document]:
    text = extract_text_from_image(image_path)
    if not text:
        return []

    image_file = Path(image_path)

    return [
        Document(
            page_content=text,
            metadata={
                "source": image_file.name,
                "source_path": str(image_file),
                "page": 1,
                "extraction_method": "ocr_image",
            },
        )
    ]


def load_full_text_from_file(file_path: str) -> str:
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext == ".pdf":
        docs = load_pdf_documents(file_path)
    elif ext in [".png", ".jpg", ".jpeg", ".webp"]:
        docs = load_image_document(file_path)
    else:
        return ""

    return "\n".join(doc.page_content for doc in docs if doc.page_content)


def chunk_documents(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=[
            "\n\n",
            "\n",
            ". ",
            "。 ",
            "? ",
            "! ",
            "다. ",
            "요. ",
            " ",
            "",
        ],
    )

    chunked: List[Document] = []
    for d in docs:
        pieces = splitter.split_text(d.page_content)
        for idx, piece in enumerate(pieces):
            chunked.append(
                Document(
                    page_content=piece,
                    metadata={
                        **d.metadata,
                        "chunk_index": idx,
                        "doc_id": str(uuid.uuid4()),
                    },
                )
            )
    return chunked


def retrieve_context(query: str) -> List[Document]:
    vs = get_vectorstore()
    return vs.similarity_search(query, k=VECTOR_TOP_K)


def format_context(docs: List[Document]) -> str:
    if not docs:
        return ""

    formatted = []
    for idx, d in enumerate(docs, start=1):
        source = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", "?")
        method = d.metadata.get("extraction_method", "unknown")

        formatted.append(
            f"[문서 {idx}] 출처: {source} / 페이지: {page} / 추출방식: {method}\n{d.page_content}"
        )

    return "\n\n".join(formatted)


def format_history(history: List[Message]) -> str:
    if not history:
        return ""

    lines = []
    for msg in history[-8:]:
        lines.append(f"[{msg.role}]\n{msg.content}")
    return "\n\n".join(lines)


def build_prompt(system_prompt: str, history: List[Message], user_message: str, context: str) -> str:
    return f"""
[시스템 지침]
{system_prompt}

너의 답변 규칙:
1. 반드시 한국어로 답한다.
2. 모든 답변은 자연스럽고 간결한 한국어로 작성한다.
3. 사용자의 입력이 짧은 인사말이나 단순 표현이면 과도하게 해석하지 말고 짧게 답한다.
4. 문서/이미지에서 추출된 텍스트가 있으면 그것을 우선 근거로 사용한다.
5. 문서에 없는 내용을 추측하지 않는다.
6. 텍스트 추출 요청이면 원문 위주로 보여준다.
7. 요약 요청이면 핵심만 짧게 정리한다.
8. 근거가 부족하면 "문서 기준으로는 확인되지 않습니다."라고 말한다.
9. 이전 대화와 무관한 새 입력이면 이전 맥락을 억지로 이어붙이지 않는다.

[대화 이력]
{format_history(history)}

[검색된 문서]
{context if context else "검색된 관련 문서 없음"}

[사용자 질문]
{user_message}

[답변 형식]
- 핵심 답변:
- 근거 요약:
- 참고 문서:
""".strip()


def ingest_folder(folder_path: Path) -> Dict[str, Any]:
    folder_path.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(glob.glob(str(folder_path / "*.pdf")))
    image_files = []
    for ext in ["*.png", "*.jpg", "*.jpeg", "*.webp"]:
        image_files.extend(glob.glob(str(folder_path / ext)))
    image_files = sorted(image_files)

    all_chunks: List[Document] = []
    processed_files = []

    for pdf in pdf_files:
        page_docs = load_pdf_documents(pdf)
        if not page_docs:
            continue

        chunks = chunk_documents(page_docs)
        all_chunks.extend(chunks)
        processed_files.append(Path(pdf).name)

    for image_path in image_files:
        docs = load_image_document(image_path)
        if not docs:
            continue

        chunks = chunk_documents(docs)
        all_chunks.extend(chunks)
        processed_files.append(Path(image_path).name)

    if not all_chunks:
        return {"message": "추출 가능한 문서/이미지가 없습니다.", "count": 0}

    vs = get_vectorstore()
    vs.add_documents(all_chunks)

    return {
        "message": "문서 적재 완료",
        "count": len(all_chunks),
        "files": processed_files,
    }


def convert_openai_messages(messages: List[OpenAIMessage]):
    system_prompt = (
        "너는 한국어로 답하는 AI 챗봇이다. "
        "짧은 질문에는 짧게 답하고, 이전 문맥을 불필요하게 끌고 오지 않는다."
    )
    history: List[Message] = []
    user_message = ""

    for msg in messages:
        if msg.role == "system":
            system_prompt = msg.content
        elif msg.role == "user":
            user_message = msg.content
            history.append(Message(role="user", content=msg.content))
        elif msg.role == "assistant":
            history.append(Message(role="assistant", content=msg.content))

    if history and history[-1].role == "user":
        user_message = history[-1].content
        history = history[:-1]

    return system_prompt, history, user_message


# =========================
# API
# =========================
@app.on_event("startup")
def startup_event():
    get_vectorstore()

    if AUTO_INGEST_ON_STARTUP:
        try:
            ingest_folder(DATA_DIR)
        except Exception as e:
            print(f"[startup ingest skipped] {e}")


@app.get("/")
def root():
    return {
        "message": "LLama OCR RAG OpenAI-Compatible API is running",
        "model": OLLAMA_MODEL,
        "data_dir": str(DATA_DIR),
        "chroma_dir": str(CHROMA_DIR),
        "ocr_enabled": ENABLE_OCR,
    }


@app.get("/health")
def health():
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/tags", timeout=10)
        response.raise_for_status()

        vs = get_vectorstore()
        collection_count = vs._collection.count()

        return {
            "status": "ok",
            "ollama": "connected",
            "model": OLLAMA_MODEL,
            "documents_in_vectorstore": collection_count,
            "ocr_enabled": ENABLE_OCR,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {e}")


@app.post("/ingest")
def ingest(req: IngestRequest):
    folder = Path(req.folder_path) if req.folder_path else DATA_DIR

    try:
        result = ingest_folder(folder)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingest failed: {e}")


@app.post("/chat")
def chat(req: ChatRequest):
    try:
        retrieved_docs = retrieve_context(req.message)
        context = format_context(retrieved_docs)

        prompt = build_prompt(
            system_prompt=req.system_prompt or "",
            history=req.history,
            user_message=req.message,
            context=context,
        )

        result = call_ollama(prompt)
        answer = result.get("response", "").strip()

        sources = []
        for d in retrieved_docs:
            sources.append(
                {
                    "source": d.metadata.get("source"),
                    "page": d.metadata.get("page"),
                    "chunk_index": d.metadata.get("chunk_index"),
                    "extraction_method": d.metadata.get("extraction_method"),
                }
            )

        return {
            "model": result.get("model", OLLAMA_MODEL),
            "answer": answer,
            "done": result.get("done", True),
            "sources": sources,
            "retrieved_count": len(retrieved_docs),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {e}")


@app.get("/docs-list")
def docs_list():
    try:
        vs = get_vectorstore()
        store = vs.get(include=["metadatas"])

        unique_docs = {}
        for meta in store.get("metadatas", []):
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
            "documents": list(unique_docs.values())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"docs-list failed: {e}")


@app.delete("/reset-db")
def reset_db():
    global vectorstore

    try:
        vectorstore = None

        if CHROMA_DIR.exists():
            shutil.rmtree(CHROMA_DIR)

        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        get_vectorstore()

        return {"message": "벡터 DB 초기화 완료"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"reset-db failed: {e}")


@app.post("/count-keyword")
def count_keyword(req: CountKeywordRequest):
    try:
        target_path = DATA_DIR / req.filename

        if not target_path.exists():
            raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다.")

        text = load_full_text_from_file(str(target_path))
        if not text:
            return {
                "filename": req.filename,
                "keyword": req.keyword,
                "count": 0,
                "message": "추출된 텍스트가 없습니다."
            }

        count = text.lower().count(req.keyword.lower())

        return {
            "filename": req.filename,
            "keyword": req.keyword,
            "count": count
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"count-keyword failed: {e}")


# =========================
# OpenAI 호환 API
# =========================
@app.get("/v1/models")
def list_openai_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "llama3.1:latest",
                "object": "model",
                "created": 0,
                "owned_by": "local"
            },
            {
                "id": "llama3.2-vision:11b",
                "object": "model",
                "created": 0,
                "owned_by": "local"
            }
        ]
    }

@app.post("/v1/chat/completions")
def openai_chat_completions(req: OpenAIChatRequest):
    try:
        system_prompt, history, user_message = convert_openai_messages(req.messages)

        retrieved_docs = retrieve_context(user_message)
        context = format_context(retrieved_docs)

        prompt = build_prompt(
            system_prompt=system_prompt,
            history=history,
            user_message=user_message,
            context=context,
        )

        result = call_ollama(prompt, temperature=req.temperature or 0.2)
        answer = result.get("response", "").strip()

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model or OLLAMA_MODEL,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": answer
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI-compatible chat failed: {e}")
