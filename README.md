# Tilon AI Chatbot — Refactored

RAG-based AI chatbot with PDF parsing, vector retrieval, reranking, and QLoRA fine-tuning support.

## Quick Start

```bash
# 1. Clone and setup
cd tilon-chatbot
cp .env.example .env          # Edit with your settings
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Start Ollama (separate terminal)
ollama serve
ollama pull qwen2.5:7b

# 3. Run the API
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## Project Structure

```
tilon-chatbot/
├── main.py                        ← Entry point (uvicorn main:app)
├── .env.example                   ← All configurable settings
├── requirements.txt
│
├── app/
│   ├── config.py                  ← Centralized configuration
│   │
│   ├── models/
│   │   └── schemas.py             ← Pydantic request/response models
│   │
│   ├── core/                      ← Shared infrastructure
│   │   ├── embeddings.py          ← Embedding model (BAAI/bge-m3)
│   │   ├── vectorstore.py         ← ChromaDB management
│   │   └── llm.py                 ← Ollama client with retry
│   │
│   ├── pipeline/                  ← Document processing (Team's work)
│   │   ├── parser.py              ← PDF/Image parsing + OCR
│   │   ├── chunker.py             ← Text splitting
│   │   └── ingest.py              ← Orchestrates parse → chunk → store
│   │
│   ├── retrieval/                 ← Search & ranking
│   │   ├── retriever.py           ← Vector search + context formatting
│   │   └── reranker.py            ← BGE reranker (NEW)
│   │
│   ├── chat/                      ← Conversation handling
│   │   ├── router.py              ← Mode detection (general/doc/web/ocr)
│   │   ├── prompts.py             ← Prompt templates (CRITICAL for fine-tuning)
│   │   └── handlers.py            ← Mode-specific logic
│   │
│   └── api/                       ← HTTP endpoints
│       ├── routes.py              ← Core API (/chat, /ingest, /health)
│       └── openai_compat.py       ← OpenAI-compatible (/v1/chat/completions)
│
├── finetuning/                    ← Usama's QLoRA workstream
│   ├── README.md
│   └── data/
│
├── data/                          ← PDF/image files for ingestion
└── tests/
```

## API Endpoints

| Method   | Path                   | Description                   |
|----------|------------------------|-------------------------------|
| GET      | `/`                    | Server status                 |
| GET      | `/health`              | Health check (Ollama + DB)    |
| POST     | `/chat`                | Main chat endpoint            |
| POST     | `/ingest`              | Ingest PDFs/images from folder|
| DELETE   | `/reset-db`            | Wipe vector database          |
| GET      | `/docs-list`           | List ingested documents       |
| POST     | `/count-keyword`       | Count keyword in a file       |
| GET      | `/v1/models`           | OpenAI-compatible model list  |
| POST     | `/v1/chat/completions` | OpenAI-compatible chat        |
