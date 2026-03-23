# Tilon AI Chatbot

Document-first RAG chatbot for English/Korean PDFs and images. Users can upload a text PDF, scanned PDF, or image, the backend extracts and ingests it, and the chatbot can summarize, answer questions, find specific information, and extract visible text.

## Current Project State

The system is now in the middle stage of the product architecture:

- ingestion/parsing: strong and actively improved
- retrieval: hybrid, document-aware, and benchmarked
- UI: usable for testing, not production-polished yet
- QLoRA: starter dataset/scaffold created, full training pipeline still pending

What is already working well:
- text PDFs, scanned PDFs, and images can be uploaded
- the built-in UI can upload one or many files at a time
- extraction uses multiple methods with fallback routing
- uploaded files stay scoped to chat
- uploaded documents are remembered in the UI sidebar and can be re-selected later
- remembered uploads can now be multi-selected for comparison-style questions
- retrieval uses vector + keyword search
- reranking and confidence gating are integrated
- document identity now uses stable `doc_id`
- repeated uploads of the same scoped file are deduplicated cleanly
- bundled upload PDFs with multiple sub-guidelines can now ask for clarification or narrow to the named sub-guideline
- benchmark baselines now exist for both library docs and upload-scoped chat

What is still not finished:
- retrieval threshold tuning on real Tilon documents
- richer block-level structure storage
- full QLoRA training pipeline and model comparison

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

Why this matters:
- the current architecture is already good enough for structured evaluation
- the project has moved from one-off debugging into benchmark-driven tuning and QLoRA data preparation

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
5. The UI sidebar remembers uploaded files from the document registry
6. Chat can be re-scoped to any remembered upload from the sidebar

## Bigger Picture

This project is really two systems that must work together:

1. RAG inference system
- extract documents
- chunk and enrich them
- retrieve the right evidence
- answer with grounding

2. QLoRA training system
- improve how the answer model uses retrieved evidence
- improve multilingual consistency
- improve citation/refusal behavior

Important principle:
- RAG is responsible for getting the right evidence
- QLoRA is responsible for using that evidence better

QLoRA should not be used to compensate for:
- weak OCR
- poor chunking
- wrong retrieval
- missing evidence

Current evaluation assets:
- `finetuning/data/benchmark_tilon_v1.jsonl`
  Library-document benchmark
- `finetuning/data/benchmark_upload_scoped_v1.jsonl`
  Upload-scoped / bundled-PDF benchmark
- `finetuning/data/qlora_train_v1.jsonl`
  First supervised QLoRA starter set

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

## Retrieval Status

The current retrieval stack is:

1. Chroma vector retrieval
2. BM25-like keyword retrieval
3. reciprocal rank fusion
4. optional reranking
5. confidence gating

This is already a strong document-chatbot foundation.

What still needs tuning:
- confidence thresholds on real Tilon PDFs
- when screenshot/image uploads should influence normal chat
- reranker resource tradeoffs
- exact-match vs summary behavior across different document types
- broader upload-scoped evaluation coverage beyond the first bundled-PDF benchmark

Current runtime policy:
- live chat can use hybrid retrieval without reranking by default for lower latency
- reranking remains available for benchmark/ablation work and can be re-enabled for live chat with `LIVE_RERANKER_ENABLED=true`

## Benchmark Status

Current benchmark baselines:
- library benchmark: stable
  - strong source recall
  - strong answer-point recall
  - bilingual and not-found coverage
- upload-scoped benchmark: stable
  - bundled upload disambiguation
  - named sub-guideline narrowing
  - upload-specific not-found behavior

This means the repo now has a measurable RAG baseline for both:
- persistent library documents
- temporary scoped uploads

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
| `GET` | `/uploaded-docs` | List remembered chat uploads from the registry |
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

## Roadmap

### Phase 1: Stabilize RAG Core
- finalize document registry behavior
- tune retrieval thresholds
- make screenshot/image uploads safer for general chat
- freeze live prompt/context format

### Phase 2: Build Evaluation Benchmark
- collect real Tilon documents
- create representative test questions
- include:
  - exact lookup
  - summary
  - OCR/image text
  - section understanding
  - negative “not found” cases
  - Korean/English mixed queries

### Phase 3: Baseline Evaluation
- measure retrieval quality
- measure citation correctness
- measure hallucination rate
- measure multilingual answer consistency

### Phase 4: QLoRA Dataset Preparation
- build training examples from the same prompt/context format used in production
- ensure examples teach:
  - grounded answering
  - correct refusal when evidence is weak
  - Korean/English consistency
  - citation behavior

### Phase 5: QLoRA Training
- fine-tune the answer model, not the retriever
- compare:
  - base `qwen2.5:7b`
  - RAG + base
  - RAG + QLoRA

## QLoRA Start Conditions

QLoRA should begin only after these are true:

- retrieval is stable enough on real Tilon documents
- prompt/context format is frozen for training
- evaluation questions exist
- baseline results are recorded
- the target base answer model is chosen and frozen for the training run

Without those, fine-tuning will be hard to evaluate and easy to misattribute.

## Notes

- `AUTO_INGEST_ON_STARTUP=false` is recommended while testing chat uploads.
- If you want a document to behave like part of the permanent knowledge base, place it in `data/library/`.
- If you only upload it in chat, it goes into `data/uploads/` and stays separate from the watched library corpus.
- A short system design overview lives in [ARCHITECTURE.md](/home/tilon/chatbot-karbi/ARCHITECTURE.md).
