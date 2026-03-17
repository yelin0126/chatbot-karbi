#!/bin/bash
# =============================================================================
# ChatBot 설치 스크립트
# - Ubuntu 22.04 LTS
# - Ollama + Qwen2.5 + Qwen2.5VL
# - FastAPI
# - RAG (Chroma + HuggingFace Embeddings)
# - Unstructured PDF parsing + OCR fallback + 구조 기반 chunking
# - DuckDuckGo 웹검색 옵션 (region/time/source/backend/safesearch)
# - Whisper STT
# - Open WebUI
# 실행 계정: root
# =============================================================================

set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

ok()   { echo -e "${GREEN}[OK]${NC}    $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC}  $1"; }
err()  { echo -e "${RED}[ERR]${NC}   $1"; exit 1; }
step() {
    echo -e "\n${GREEN}==============================${NC}"
    echo -e "${GREEN} $1${NC}"
    echo -e "${GREEN}==============================${NC}"
}

if [ "$(id -u)" -ne 0 ]; then
    err "root 계정으로 실행하세요. (sudo -i 후 재실행)"
fi

APP_DIR="/opt/chatbot"
VENV_DIR="$APP_DIR/chatbot-env"
APP_FILE="$APP_DIR/main.py"
ENV_FILE="/etc/default/chatbot"
SERVICE_FILE="/etc/systemd/system/chatbot.service"

WEB_PORT=8000
TEXT_MODEL="qwen2.5:7b"
VISION_MODEL="qwen2.5vl:7b"
EMBED_MODEL="jhgan/ko-sroberta-multitask"
WHISPER_MODEL="base"

TARGET_USER="${SUDO_USER:-${USER:-root}}"

step "STEP 1 | apt update"
apt update -y
ok "패키지 목록 업데이트 완료"

step "STEP 2 | 기본 패키지 설치"
apt install -y \
    ca-certificates \
    curl \
    gnupg \
    lsb-release \
    software-properties-common \
    ffmpeg \
    python3 \
    python3-pip \
    python3-venv \
    build-essential \
    libmagic-dev \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-kor
ok "기본 패키지 설치 완료"

step "STEP 3 | 기존 Docker 충돌 패키지 제거"
apt remove -y docker.io docker-compose docker-compose-v2 docker-doc podman-docker containerd runc 2>/dev/null || true
ok "충돌 가능 패키지 정리 완료"

step "STEP 4 | Docker 공식 저장소 등록"
install -m 0755 -d /etc/apt/keyrings

if [ ! -f /etc/apt/keyrings/docker.asc ]; then
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
    chmod a+r /etc/apt/keyrings/docker.asc
fi

cat > /etc/apt/sources.list.d/docker.sources <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
Components: stable
Signed-By: /etc/apt/keyrings/docker.asc
EOF

apt update -y
ok "Docker 공식 저장소 등록 완료"

step "STEP 5 | Docker Engine + Compose Plugin 설치"
apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
ok "Docker 설치 완료"
ok "docker version: $(docker --version)"
ok "docker compose version: $(docker compose version)"

step "STEP 6 | Docker 서비스 활성화"
systemctl enable --now docker
systemctl is-active --quiet docker || err "Docker 서비스 시작 실패"
ok "Docker 서비스 활성화 완료"

step "STEP 7 | docker 그룹 사용자 설정"
if id "$TARGET_USER" >/dev/null 2>&1; then
    usermod -aG docker "$TARGET_USER" || true
    ok "docker 그룹에 '$TARGET_USER' 추가 완료"
    warn "현재 세션에 docker 그룹 권한이 즉시 반영되지 않을 수 있습니다. 재로그인 후 적용됩니다."
else
    warn "대상 사용자 '$TARGET_USER'를 찾지 못했습니다. docker 그룹 추가는 건너뜁니다."
fi

step "STEP 8 | Ollama 설치"
if command -v ollama >/dev/null 2>&1; then
    warn "Ollama가 이미 설치되어 있습니다. 건너뜁니다."
else
    curl -fsSL https://ollama.com/install.sh | sh
    ok "Ollama 설치 완료"
fi

if systemctl list-unit-files | grep -q '^ollama\.service'; then
    systemctl enable --now ollama
    sleep 3
    systemctl is-active --quiet ollama || err "Ollama 서비스 시작 실패"
else
    warn "ollama.service를 찾지 못했습니다. 수동 실행 시도"
    nohup ollama serve >/var/log/ollama.log 2>&1 &
    sleep 5
fi
ok "Ollama 서비스 실행 확인"

step "STEP 9 | Ollama 모델 다운로드"
ollama pull "$TEXT_MODEL"
ok "텍스트 모델 다운로드 완료: $TEXT_MODEL"

ollama pull "$VISION_MODEL"
ok "비전 모델 다운로드 완료: $VISION_MODEL"

step "STEP 10 | 앱 디렉토리 / 가상환경 생성"
mkdir -p "$APP_DIR"
cd "$APP_DIR"

python3 -m venv "$VENV_DIR"
ok "가상환경 생성 완료: $VENV_DIR"

PIP="$VENV_DIR/bin/pip"

"$PIP" install --upgrade pip setuptools wheel
ok "pip 업그레이드 완료"

step "STEP 11 | Python 패키지 설치"
"$PIP" install \
    fastapi \
    "uvicorn[standard]" \
    langchain \
    langchain-community \
    langchain-text-splitters \
    langchain-huggingface \
    langchain-chroma \
    chromadb \
    pypdf \
    pdfplumber \
    sentence-transformers \
    python-multipart \
    pillow \
    httpx \
    openai-whisper \
    duckduckgo-search \
    "unstructured[pdf]"
ok "Python 패키지 설치 완료"

step "STEP 12 | 환경변수 파일 작성"
cat > "$ENV_FILE" <<EOF
OLLAMA_URL=http://127.0.0.1:11434
TEXT_MODEL=$TEXT_MODEL
VISION_MODEL=$VISION_MODEL
CHROMA_DIR=$APP_DIR/chroma_db
EMBED_MODEL=jhgan/ko-sroberta-multitask
WHISPER_MODEL=$WHISPER_MODEL
EOF

chmod 644 "$ENV_FILE"
ok "환경변수 파일 작성 완료: $ENV_FILE"

step "STEP 13 | main.py 작성"
cat > "$APP_FILE" <<'PYEOF'
import os
import base64
import tempfile
from typing import Optional, List, Dict, Any, Literal

import httpx
import whisper

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from pypdf import PdfReader
import pdfplumber

from unstructured.partition.pdf import partition_pdf

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
TEXT_MODEL = os.getenv("TEXT_MODEL", "qwen2.5:7b")
VISION_MODEL = os.getenv("VISION_MODEL", "qwen2.5vl:7b")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")

app = FastAPI(title="ChatBot API", version="5.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
vector_store = Chroma(
    collection_name="chatbot_docs",
    embedding_function=embeddings,
    persist_directory=CHROMA_DIR,
)

whisper_model = None


class ChatRequest(BaseModel):
    message: str

    use_rag: bool = False
    use_web_search: bool = False

    web_search_count: int = Field(default=5, ge=1, le=10)
    web_search_region: str = "kr-kr"
    web_search_time: Optional[Literal["d", "w", "m", "y"]] = None
    web_search_source: Literal["text", "news", "images"] = "text"
    web_search_backend: Literal["auto", "html", "lite"] = "auto"
    web_search_safesearch: Literal["strict", "moderate", "off"] = "moderate"

    system_prompt: str = "당신은 사내 AI 어시스턴트입니다. 한국어로 친절하고 명확하게 답변하세요."


class WebSearchRequest(BaseModel):
    query: str
    max_results: int = Field(default=5, ge=1, le=10)
    region: str = "kr-kr"
    time: Optional[Literal["d", "w", "m", "y"]] = None
    source: Literal["text", "news", "images"] = "text"
    backend: Literal["auto", "html", "lite"] = "auto"
    safesearch: Literal["strict", "moderate", "off"] = "moderate"


@app.on_event("startup")
def load_whisper_model():
    global whisper_model
    if whisper_model is None:
        print(f"Loading Whisper model: {WHISPER_MODEL}")
        whisper_model = whisper.load_model(WHISPER_MODEL)
        print("Whisper model loaded.")


def _ensure_whisper():
    global whisper_model
    if whisper_model is None:
        whisper_model = whisper.load_model(WHISPER_MODEL)
    return whisper_model


def _rough_text_signal(pdf_path: str, sample_pages: int = 5) -> int:
    """
    OCR 분기를 위한 간단한 텍스트 신호 체크.
    텍스트 레이어가 거의 없으면 스캔본으로 보고 OCR 경로 우선.
    """
    try:
        reader = PdfReader(pdf_path)
        total = 0
        for page in reader.pages[:sample_pages]:
            total += len(page.extract_text() or "")
        return total
    except Exception:
        return 0


def _extract_with_pdfplumber(pdf_path: str, source: str) -> List[Document]:
    """
    pdfplumber를 이용한 직접 텍스트/표 추출 (unstructured 실패 시 fallback 또는 보완용).
    페이지 단위 청크, 표는 별도 Document로 분리.
    """
    docs: List[Document] = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                # 표 추출
                tables = page.extract_tables()
                for tbl in (tables or []):
                    rows = ["\t".join([cell or "" for cell in row]) for row in tbl if row]
                    tbl_text = "\n".join(rows).strip()
                    if tbl_text:
                        docs.append(Document(
                            page_content=f"[표 - {page_num}페이지]\n{tbl_text}",
                            metadata={
                                "source": source,
                                "parser": "pdfplumber",
                                "strategy": "pdfplumber",
                                "section_title": f"{page_num}페이지 표",
                                "page_numbers": [page_num],
                                "chunk_role": "table",
                            }
                        ))

                # 본문 텍스트 (표 영역 제외)
                text = page.extract_text(x_tolerance=2, y_tolerance=3) or ""
                text = text.strip()
                if len(text) > 50:
                    # 2000자 이상이면 분할
                    for i in range(0, len(text), 2000):
                        chunk = text[i:i+2000].strip()
                        if chunk:
                            docs.append(Document(
                                page_content=f"[{page_num}페이지]\n{chunk}",
                                metadata={
                                    "source": source,
                                    "parser": "pdfplumber",
                                    "strategy": "pdfplumber",
                                    "section_title": f"{page_num}페이지",
                                    "page_numbers": [page_num],
                                    "chunk_role": "section",
                                }
                            ))
    except Exception as e:
        print(f"[pdfplumber] 오류: {e}")
    return docs


def _partition_pdf_adaptive(pdf_path: str):
    """
    1) 텍스트 신호가 약하면 ocr_only 우선
    2) 텍스트 신호가 있으면 hi_res 우선
    3) 실패 시 fast 등으로 fallback
    """
    text_signal = _rough_text_signal(pdf_path)
    primary = "ocr_only" if text_signal < 120 else "hi_res"

    candidates = [primary]
    for extra in ["hi_res", "ocr_only", "fast"]:
        if extra not in candidates:
            candidates.append(extra)

    last_error = None

    for strategy in candidates:
        attempts = []

        if strategy == "hi_res":
            attempts = [
                {"filename": pdf_path, "strategy": strategy, "languages": ["kor", "eng"], "infer_table_structure": True},
                {"filename": pdf_path, "strategy": strategy, "languages": ["kor", "eng"]},
                {"filename": pdf_path, "strategy": strategy, "infer_table_structure": True},
                {"filename": pdf_path, "strategy": strategy},
            ]
        elif strategy == "ocr_only":
            attempts = [
                {"filename": pdf_path, "strategy": strategy, "languages": ["kor", "eng"]},
                {"filename": pdf_path, "strategy": strategy},
            ]
        else:
            attempts = [
                {"filename": pdf_path, "strategy": strategy},
            ]

        for kwargs in attempts:
            try:
                elements = partition_pdf(**kwargs)
                if elements:
                    return elements, strategy, text_signal
            except TypeError as e:
                last_error = e
                continue
            except Exception as e:
                last_error = e
                continue

    raise RuntimeError(f"PDF 구조 파싱 실패: {last_error}")


def _elements_to_documents(
    elements: List[Any],
    source: str,
    strategy: str,
    max_chars: int = 2500,
    target_chars: int = 1800,
) -> List[Document]:
    """
    단순 문자수 분할이 아니라 구조 기반으로 청크 생성.
    - Title 나오면 섹션 전환
    - Table은 별도 청크
    - Narrative/List는 같은 섹션 안에서 묶음
    """
    docs: List[Document] = []

    current_title = "제목 미확인"
    buffer: List[str] = []
    buffer_pages = set()
    buffer_types: List[str] = []
    buffer_len = 0
    chunk_index = 0

    def flush_buffer():
        nonlocal buffer, buffer_pages, buffer_types, buffer_len, chunk_index

        body = "\n\n".join([x for x in buffer if x.strip()]).strip()
        if not body:
            buffer = []
            buffer_pages = set()
            buffer_types = []
            buffer_len = 0
            return

        docs.append(
            Document(
                page_content=f"[섹션]\n{current_title}\n\n{body}",
                metadata={
                    "source": source,
                    "parser": "unstructured",
                    "strategy": strategy,
                    "section_title": current_title,
                    "page_numbers": sorted(buffer_pages),
                    "element_types": list(dict.fromkeys(buffer_types)),
                    "chunk_role": "section",
                    "chunk_index": chunk_index,
                },
            )
        )

        chunk_index += 1
        buffer = []
        buffer_pages = set()
        buffer_types = []
        buffer_len = 0

    for elem in elements:
        text = str(elem).strip()
        if not text:
            continue

        elem_type = type(elem).__name__
        meta = getattr(elem, "metadata", None)
        page_number = getattr(meta, "page_number", None)

        if elem_type == "Title":
            flush_buffer()
            current_title = text[:300]
            continue

        if elem_type in ("Table", "TableChunk"):
            flush_buffer()
            docs.append(
                Document(
                    page_content=f"[섹션]\n{current_title}\n\n[표]\n{text}",
                    metadata={
                        "source": source,
                        "parser": "unstructured",
                        "strategy": strategy,
                        "section_title": current_title,
                        "page_numbers": [page_number] if page_number else [],
                        "element_types": [elem_type],
                        "chunk_role": "table",
                        "chunk_index": chunk_index,
                    },
                )
            )
            chunk_index += 1
            continue

        piece = text
        if elem_type == "ListItem":
            piece = f"• {text}"

        if buffer and (buffer_len + len(piece) > max_chars):
            flush_buffer()

        buffer.append(piece)
        if page_number:
            buffer_pages.add(page_number)
        buffer_types.append(elem_type)
        buffer_len += len(piece) + 2

        if buffer_len >= target_chars and elem_type in ("NarrativeText", "ListItem", "FigureCaption", "Caption"):
            flush_buffer()

    flush_buffer()
    return docs


def _make_search_wrapper(
    region: str = "kr-kr",
    max_results: int = 5,
    safesearch: str = "moderate",
    time: Optional[str] = None,
    source: str = "text",
    backend: str = "auto",
) -> DuckDuckGoSearchAPIWrapper:
    return DuckDuckGoSearchAPIWrapper(
        region=region,
        max_results=max_results,
        safesearch=safesearch,
        time=time,
        source=source,
        backend=backend,
    )


def _search_with_wrapper(
    query: str,
    max_results: int = 5,
    region: str = "kr-kr",
    time: Optional[str] = None,
    source: str = "text",
    backend: str = "auto",
    safesearch: str = "moderate",
) -> List[Dict[str, Any]]:
    try:
        wrapper = _make_search_wrapper(
            region=region,
            max_results=max_results,
            safesearch=safesearch,
            time=time,
            source=source,
            backend=backend,
        )
        results = wrapper.results(query, max_results=max_results, source=source)
        return results or []
    except Exception as e:
        return [{
            "title": "웹 검색 실패",
            "snippet": str(e),
            "link": "",
        }]


def _format_search_results(results: List[Dict[str, Any]]) -> str:
    if not results:
        return "검색 결과 없음"

    blocks = []
    for r in results:
        title = r.get("title", "")
        snippet = r.get("snippet", "")
        link = r.get("link", "")
        date = r.get("date", "")
        source = r.get("source", "")

        extra = []
        if source:
            extra.append(f"매체: {source}")
        if date:
            extra.append(f"일시: {date}")

        meta_line = "\n".join(extra).strip()
        if meta_line:
            blocks.append(f"[{title}]\n{snippet}\n{meta_line}\n출처: {link}")
        else:
            blocks.append(f"[{title}]\n{snippet}\n출처: {link}")

    return "\n\n".join(blocks)


async def _call_ollama(messages: List[Dict[str, Any]], model: str) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
    }

    timeout = httpx.Timeout(180.0, connect=30.0)

    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(f"{OLLAMA_URL}/api/chat", json=payload)

    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)

    data = resp.json()
    return data.get("message", {}).get("content", "").strip()


async def _transcribe_upload(file: UploadFile) -> str:
    allowed_exts = [".wav", ".mp3", ".m4a", ".ogg", ".flac"]
    ext = os.path.splitext(file.filename.lower())[1]

    if ext not in allowed_exts:
        raise HTTPException(
            status_code=400,
            detail=f"지원하지 않는 오디오 형식입니다. {allowed_exts}"
        )

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        model = _ensure_whisper()
        result = model.transcribe(tmp_path, language="ko")
        text = (result.get("text") or "").strip()

        if not text:
            raise HTTPException(status_code=400, detail="음성에서 텍스트를 인식하지 못했습니다.")

        return text

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"STT 오류: {str(e)}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "ollama_url": OLLAMA_URL,
        "text_model": TEXT_MODEL,
        "vision_model": VISION_MODEL,
        "embed_model": EMBED_MODEL,
        "stt_model": WHISPER_MODEL,
        "chroma_dir": CHROMA_DIR,
        "pdf_ingest": {
            "parser": "unstructured",
            "strategy_candidates": ["hi_res", "ocr_only", "fast"],
            "chunking": "structure-aware",
        },
        "web_search": {
            "provider": "DuckDuckGoSearchAPIWrapper",
            "default_region": "kr-kr",
        },
    }


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드 가능합니다.")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        elements, strategy, text_signal = _partition_pdf_adaptive(tmp_path)
        docs = _elements_to_documents(elements, file.filename, strategy)

        # unstructured 결과가 너무 빈약하면 pdfplumber로 보완
        if len(docs) < 3:
            print(f"[upload] unstructured 청크 부족({len(docs)}개), pdfplumber 보완 실행")
            plumber_docs = _extract_with_pdfplumber(tmp_path, file.filename)
            if plumber_docs:
                docs = plumber_docs
                strategy = "pdfplumber_fallback"

        if not docs:
            raise HTTPException(status_code=400, detail="구조 기반 청크를 만들지 못했습니다.")

        vector_store.add_documents(docs)

        return {
            "message": f"{len(docs)}개 구조 청크가 저장되었습니다.",
            "file": file.filename,
            "parser": "unstructured",
            "strategy": strategy,
            "text_signal": text_signal,
            "sample_metadata": docs[0].metadata if docs else {},
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF 처리 오류: {str(e)}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post("/upload-image")
async def upload_image(
    file: UploadFile = File(...),
    prompt: str = Form("이 이미지를 한국어로 자세히 설명해주세요.")
):
    allowed = [".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"]
    ext = os.path.splitext(file.filename.lower())[1]

    if ext not in allowed:
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능합니다.")

    try:
        image_data = await file.read()
        image_b64 = base64.b64encode(image_data).decode("utf-8")

        reply = await _call_ollama(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_b64],
                }
            ],
            model=VISION_MODEL,
        )

        return {
            "file": file.filename,
            "reply": reply,
            "model": VISION_MODEL,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"이미지 분석 오류: {str(e)}")


@app.post("/web-search")
async def web_search(req: WebSearchRequest):
    try:
        results = _search_with_wrapper(
            query=req.query,
            max_results=req.max_results,
            region=req.region,
            time=req.time,
            source=req.source,
            backend=req.backend,
            safesearch=req.safesearch,
        )

        return {
            "query": req.query,
            "count": len(results),
            "options": {
                "region": req.region,
                "time": req.time,
                "source": req.source,
                "backend": req.backend,
                "safesearch": req.safesearch,
                "max_results": req.max_results,
            },
            "results": results,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"웹 검색 오류: {str(e)}")


@app.post("/chat")
async def chat(req: ChatRequest):
    context_parts = []
    rag_preview = ""
    web_preview = ""
    rag_meta_preview = []

    if req.use_rag:
        try:
            docs = vector_store.similarity_search(req.message, k=6)

            formatted_docs = []
            for i, doc in enumerate(docs, start=1):
                meta = doc.metadata or {}
                section_title = meta.get("section_title", "제목 미확인")
                pages = meta.get("page_numbers", [])
                chunk_role = meta.get("chunk_role", "section")
                parser = meta.get("parser", "unknown")

                page_info = f"페이지: {pages}" if pages else "페이지 정보 없음"
                formatted_docs.append(
                    f"[문서 조각 {i}]\n"
                    f"섹션: {section_title}\n"
                    f"유형: {chunk_role} | 파서: {parser}\n"
                    f"{page_info}\n\n"
                    f"{doc.page_content}"
                )

                rag_meta_preview.append({
                    "section_title": section_title,
                    "page_numbers": pages,
                    "chunk_role": chunk_role,
                })

            rag_context = "\n\n---\n\n".join(formatted_docs)
            if rag_context:
                rag_preview = rag_context[:2000]
                context_parts.append(f"[업로드 문서 참고]\n{rag_context}")

        except Exception as e:
            context_parts.append(f"[업로드 문서 참고]\n문서 검색 중 오류가 발생했습니다: {str(e)}")

    if req.use_web_search:
        web_results = _search_with_wrapper(
            query=req.message,
            max_results=req.web_search_count,
            region=req.web_search_region,
            time=req.web_search_time,
            source=req.web_search_source,
            backend=req.web_search_backend,
            safesearch=req.web_search_safesearch,
        )
        web_context = _format_search_results(web_results)
        web_preview = web_context[:1500]
        context_parts.append(f"[웹 검색 결과]\n{web_context}")

    system = req.system_prompt
    if context_parts:
        system += (
            "\n\n"
            + "\n\n---\n\n".join(context_parts)
            + "\n\n"
            "【답변 지침】\n"
            "1. 위 문서 조각들을 꼼꼼히 읽고, 질문과 관련된 내용을 빠짐없이 반영하세요.\n"
            "2. 문서 내용을 인용할 때는 반드시 섹션명과 페이지 번호를 함께 밝히세요. 예: (3페이지 '계약 조건' 섹션)\n"
            "3. 여러 조각에 걸쳐 내용이 나뉘어 있으면 통합하여 완성된 답변을 제공하세요.\n"
            "4. 문서에 없는 내용을 추측하거나 지어내지 마세요. 없으면 '문서에서 찾을 수 없습니다'라고 명시하세요.\n"
            "5. 웹 검색 결과가 함께 있을 경우, 문서 내용과 웹 정보를 구분하여 출처를 표시하세요.\n"
        )

    reply = await _call_ollama(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": req.message},
        ],
        model=TEXT_MODEL,
    )

    return {
        "reply": reply,
        "used_rag": req.use_rag,
        "used_web_search": req.use_web_search,
        "model": TEXT_MODEL,
        "search_options": {
            "region": req.web_search_region,
            "time": req.web_search_time,
            "source": req.web_search_source,
            "backend": req.web_search_backend,
            "safesearch": req.web_search_safesearch,
            "max_results": req.web_search_count,
        } if req.use_web_search else None,
        "debug_preview": {
            "rag_context_preview": rag_preview if req.use_rag else None,
            "rag_metadata_preview": rag_meta_preview if req.use_rag else None,
            "web_context_preview": web_preview if req.use_web_search else None,
        },
    }


@app.post("/stt")
async def speech_to_text(file: UploadFile = File(...)):
    text = await _transcribe_upload(file)
    return {
        "file": file.filename,
        "text": text,
    }


@app.post("/chat-audio")
async def chat_audio(
    file: UploadFile = File(...),
    use_rag: bool = Form(False),
    use_web_search: bool = Form(False),
    web_search_count: int = Form(5),
    web_search_region: str = Form("kr-kr"),
    web_search_time: Optional[str] = Form(None),
    web_search_source: str = Form("text"),
    web_search_backend: str = Form("auto"),
    web_search_safesearch: str = Form("moderate"),
    system_prompt: str = Form("당신은 사내 AI 어시스턴트입니다. 한국어로 친절하고 명확하게 답변하세요."),
):
    transcribed_text = await _transcribe_upload(file)

    req = ChatRequest(
        message=transcribed_text,
        use_rag=use_rag,
        use_web_search=use_web_search,
        web_search_count=web_search_count,
        web_search_region=web_search_region,
        web_search_time=web_search_time,
        web_search_source=web_search_source,
        web_search_backend=web_search_backend,
        web_search_safesearch=web_search_safesearch,
        system_prompt=system_prompt,
    )
    result = await chat(req)

    return {
        "recognized_text": transcribed_text,
        "reply": result["reply"],
        "used_rag": result["used_rag"],
        "used_web_search": result["used_web_search"],
        "model": result["model"],
        "search_options": result["search_options"],
    }
PYEOF
ok "main.py 작성 완료: $APP_FILE"

step "STEP 14 | systemd 서비스 작성"
cat > "$SERVICE_FILE" <<EOF
[Unit]
Description=ChatBot FastAPI Service
After=network-online.target docker.service ollama.service
Wants=network-online.target

[Service]
Type=simple
User=root
Group=root
WorkingDirectory=$APP_DIR
EnvironmentFile=$ENV_FILE
ExecStart=$VENV_DIR/bin/uvicorn main:app --host 0.0.0.0 --port $WEB_PORT
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

chmod 644 "$SERVICE_FILE"
systemctl daemon-reload
systemctl enable --now chatbot
sleep 5

systemctl is-active --quiet chatbot || {
    journalctl -u chatbot --no-pager -n 100
    err "chatbot 서비스 시작 실패"
}
ok "chatbot 서비스 실행 완료"

step "STEP 15 | Open WebUI 컨테이너 실행"
docker rm -f open-webui 2>/dev/null || true

docker run -d \
  --name open-webui \
  --network=host \
  -e OLLAMA_BASE_URL=http://127.0.0.1:11434 \
  -v open-webui:/app/backend/data \
  --restart unless-stopped \
  ghcr.io/open-webui/open-webui:main

sleep 5
docker ps --format '{{.Names}}' | grep -q '^open-webui$' || err "Open WebUI 컨테이너 실행 실패"
ok "Open WebUI 컨테이너 실행 완료"

step "STEP 16 | 상태 확인"
echo ""
systemctl --no-pager --full status chatbot | sed -n '1,20p' || true
echo ""
docker ps --filter "name=open-webui" || true
echo ""

echo -e "${GREEN}=============================================${NC}"
ok "모든 시스템(API + Open WebUI + 웹검색 + STT + 구조형 PDF 처리) 설정 완료"
echo -e "  ▶ 앱 경로       : $APP_DIR"
echo -e "  ▶ 환경변수 파일 : $ENV_FILE"
echo -e "  ▶ FastAPI Docs  : http://<서버IP>:$WEB_PORT/docs"
echo -e "  ▶ Open WebUI    : http://<서버IP>:8080"
echo -e "  ▶ API 상태확인  : systemctl status chatbot"
echo -e "  ▶ API 로그확인  : journalctl -u chatbot -f"
echo -e "  ▶ Ollama 확인   : ollama list"
echo -e "  ▶ Docker 확인   : docker ps"
echo ""
echo -e "  [주요 엔드포인트]"
echo -e "  - GET  /health"
echo -e "  - POST /upload"
echo -e "  - POST /upload-image"
echo -e "  - POST /web-search"
echo -e "  - POST /chat"
echo -e "  - POST /stt"
echo -e "  - POST /chat-audio"
echo -e "${GREEN}=============================================${NC}"