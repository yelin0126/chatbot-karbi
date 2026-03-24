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
