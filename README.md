# Tilon AI Chatbot

Document-first RAG chatbot for English/Korean PDFs and images. Users can upload a text PDF, scanned PDF, or image, the backend extracts and ingests it, and the chatbot can summarize, answer questions, find specific information, and extract visible text.

## Quick Start

```bash
# 1. Setup
cd /home/tilon/chatbot-karbi
cp .env.example .env
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 2. Start Ollama in another terminal
ollama serve
ollama pull qwen2.5:7b
ollama pull qwen2.5vl:7b

# 3. Run the API
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Open:
- `http://127.0.0.1:8000/ui` for the built-in chat/upload UI
- `http://127.0.0.1:8000/docs` for Swagger

Architecture summary:
- [ARCHITECTURE.md](/home/tilon/chatbot-karbi/ARCHITECTURE.md)

## Storage Model

The project now separates persistent library documents from chat uploads:

- `data/library/`
  Permanent knowledge-base documents for the team. Startup ingest and the file watcher use this folder.

- `data/uploads/`
  User-uploaded files from chat. These are ingested for the current chat flow but are not auto-watched as permanent library docs.

- `data/temp/`
  Optional intermediate processing folder.

Why this split helps:
- cleaner restart behavior
- no accidental re-ingestion of chat uploads
- easier debugging
- better separation of persistent corpus vs temporary chat documents

## Current Ingestion Flow

### Library documents
1. Put PDFs/images into `data/library/`
2. Startup ingest can load them if `AUTO_INGEST_ON_STARTUP=true`
3. The watcher monitors `data/library/` for new files
4. Chunks are stored in ChromaDB and indexed for retrieval

### Chat uploads
1. User uploads a file through `/ui` or `/chat-with-file`
2. File is saved to `data/uploads/`
3. Parser extracts content from PDF/image
4. Content is chunked, enriched, embedded, and stored
5. Chat stays scoped to that uploaded file

## Parsing / Extraction Stack

The parser uses a multi-step extraction pipeline:

1. `marker_single`
   Good for structured digital PDFs

2. `PyMuPDF`
   Fast text-layer extraction for born-digital pages

3. `qwen2.5vl:7b`
   Vision fallback for scanned or image-heavy pages

4. `tesseract`
   OCR fallback

Recent parser improvements:
- per-page routing instead of whole-file routing
- page classification: `digital`, `hybrid`, `scanned`
- quality gates for low text yield / garbled text
- richer layout metadata such as heading hints and block counts

## Project Structure

```text
chatbot-karbi/
├── main.py
├── .env.example
├── requirements.txt
├── chroma_db/
├── data/
│   ├── library/                   # Persistent team documents
│   ├── uploads/                   # Chat-uploaded files
│   └── temp/                      # Optional temp files
├── app/
│   ├── api/
│   │   ├── routes.py              # Main API endpoints
│   │   ├── upload_ui.py           # Built-in chat/upload UI
│   │   └── openai_compat.py       # OpenAI-compatible endpoints
│   ├── chat/
│   │   ├── handlers.py            # Unified chat handling
│   │   ├── prompts.py
│   │   └── router.py
│   ├── core/
│   │   ├── embeddings.py
│   │   ├── llm.py
│   │   ├── vectorstore.py
│   │   └── watcher.py
│   ├── models/
│   │   └── schemas.py
│   ├── pipeline/
│   │   ├── parser.py
│   │   ├── chunker.py
│   │   ├── enricher.py
│   │   └── ingest.py
│   └── retrieval/
│       ├── retriever.py
│       ├── keyword_index.py
│       └── reranker.py
└── finetuning/
```

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Server status and path info |
| `GET` | `/health` | Health check (Ollama + vectorstore) |
| `GET` | `/models` | Model list for built-in UI |
| `POST` | `/chat` | Main chat endpoint |
| `POST` | `/chat-with-file` | Upload a file and ask about it in one request |
| `POST` | `/upload` | Upload and ingest one file |
| `POST` | `/upload-multiple` | Upload and ingest multiple files |
| `POST` | `/ingest` | Ingest a folder, default `data/library/` |
| `DELETE` | `/reset-db` | Clear vector DB |
| `GET` | `/docs-list` | List stored chunks/documents |
| `POST` | `/count-keyword` | Count a keyword in a stored source file |
| `GET` | `/v1/models` | OpenAI-compatible model list |
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat |

## Testing Checklist

### Clean reset
```bash
curl -X DELETE http://127.0.0.1:8000/reset-db
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/docs-list
```

### Ingest permanent library docs
Put files in `data/library/`, then:

```bash
curl -X POST http://127.0.0.1:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{}'
```

### Upload and ask in one step
```bash
curl -X POST http://127.0.0.1:8000/chat-with-file \
  -F "file=@your.pdf" \
  -F "message=이 문서를 요약해줘"
```

### Direct image text extraction
In `/ui`, upload an image or screenshot and ask:
- `give me the text in the image`
- `what does this image say`
- `텍스트 추출해줘`

These now use direct extraction intent handling instead of normal low-confidence RAG fallback.

## Notes

- `AUTO_INGEST_ON_STARTUP=false` is recommended while testing chat uploads.
- If you want a document to behave like part of the permanent knowledge base, place it in `data/library/`.
- If you only upload it in chat, it goes into `data/uploads/` and stays separate from the watched library corpus.
- A short system design overview lives in [ARCHITECTURE.md](/home/tilon/chatbot-karbi/ARCHITECTURE.md).
