# Improvements & Bug Fixes — Original vs Refactored

This document lists every issue found in the original `app.py` and what was done about it.

---

## 🔴 BUGS (Things that were broken)

### 1. Web Search Mode Never Actually Searches
**File:** `app/chat/handlers.py`
**Original:** `handle_web_search()` detected the user wanted real-time info (weather, news, stock prices) but then just asked Ollama — which has NO internet access. The response was always hallucinated.
**Fix:** Added actual Tavily web search (`_tavily_search()`) that fetches real results before sending them to the LLM. The Tavily API key was already in the Notion docs but never used in code.

### 2. Embedding Model Hardcoded to CPU
**File:** `app/core/embeddings.py`
**Original:** `model_kwargs={"device": "cpu"}` — BAAI/bge-m3 is a 560M parameter model. Running on CPU means every embedding takes seconds instead of milliseconds. With 4 team members all having GPUs, this was a significant bottleneck.
**Fix:** Auto-detects CUDA availability, configurable via `EMBEDDING_DEVICE` env var.

### 3. Duplicate Document Ingestion
**File:** `app/pipeline/ingest.py`
**Original:** Every call to `/ingest` re-processed and re-stored ALL files in the folder, even ones already in ChromaDB. This caused duplicate chunks, inflated the DB, and degraded retrieval quality (same chunk returned multiple times).
**Fix:** `get_ingested_sources()` checks which files are already stored and skips them. Returns `skipped` list in the response.

### 4. Deprecated FastAPI Startup Event
**File:** `main.py`
**Original:** Used `@app.on_event("startup")` which is deprecated since FastAPI 0.103.
**Fix:** Uses modern `lifespan` context manager.

---

## 🟡 MISSING FEATURES (Should have been there)

### 5. No Reranker
**File:** `app/retrieval/reranker.py` (NEW)
**Original:** Retrieved top-4 chunks by raw vector similarity only. Vector similarity is a rough first pass — it often returns chunks that mention similar words but don't actually answer the question.
**Fix:** Added optional `BAAI/bge-reranker-v2-m3` that re-scores candidates using cross-attention. When enabled, fetches 2x candidates then narrows to the best matches. Dramatically improves answer quality, especially for Korean+English mixed content.
**Config:** `RERANKER_ENABLED=true` in `.env`

### 6. No Language Detection
**File:** `app/pipeline/parser.py`
**Original:** Every document's `language` metadata was set to `"unknown"`. This is useless information — the LLM and the retriever both benefit from knowing if a chunk is Korean, English, or mixed.
**Fix:** Added `langdetect` to detect language per chunk. Stored in metadata as ISO code (`ko`, `en`, etc.).

### 7. No Logging
**File:** `app/config.py`
**Original:** Used `print()` statements scattered throughout. No way to control verbosity, no timestamps, no module identification.
**Fix:** Proper Python `logging` with configurable `LOG_LEVEL`, module-tagged loggers (`tilon.parser`, `tilon.retriever`, etc.), and structured format.

### 8. No Retry on LLM Timeout
**File:** `app/core/llm.py`
**Original:** If Ollama timed out (common with 7B+ models on first load), the request failed immediately with a 500 error.
**Fix:** Retries once on timeout. Distinguishes between timeout (retryable) and connection errors (not retryable).

### 9. No Marker Timeout or Cleanup
**File:** `app/pipeline/parser.py`
**Original:** `marker_single` subprocess had no timeout — if it hung on a corrupt PDF, the entire server froze. Also left temp files in `marker_output/` forever.
**Fix:** Added 120s timeout. Cleans up temp directory after successful extraction.

---

## 🟢 CODE QUALITY IMPROVEMENTS

### 10. Single 550-Line File → Modular Structure
**Original:** Everything in one `app.py` — config, models, parsing, chunking, embedding, retrieval, prompts, handlers, routes.
**Fix:** 15 focused files across 6 modules. Each file has one responsibility.

### 11. Hardcoded Values → Environment Configuration
**Original:** Chunk size (1200), overlap (150), temperature (0.2), timeout (180), top_k (4) were all hardcoded.
**Fix:** All configurable via `.env` without code changes.

### 12. No Response Models
**Original:** API endpoints returned raw dicts. No validation on output, no documentation.
**Fix:** Added Pydantic response models (`ChatResponse`, `SourceInfo`, `IngestResponse`).

### 13. Prompt Templates Mixed with Logic
**Original:** Prompt strings were inside handler functions, making them hard to find and modify.
**Fix:** Extracted to `app/chat/prompts.py` with clear documentation about the format that fine-tuning must match.

---

## 📋 Summary Table

| #  | Issue                          | Severity | Status |
|----|--------------------------------|----------|--------|
| 1  | Web search never searches      | 🔴 Bug   | Fixed  |
| 2  | Embedding on CPU only          | 🔴 Bug   | Fixed  |
| 3  | Duplicate ingestion            | 🔴 Bug   | Fixed  |
| 4  | Deprecated startup event       | 🔴 Bug   | Fixed  |
| 5  | No reranker                    | 🟡 Missing | Added  |
| 6  | No language detection          | 🟡 Missing | Added  |
| 7  | No logging                     | 🟡 Missing | Added  |
| 8  | No LLM retry                   | 🟡 Missing | Added  |
| 9  | No marker timeout/cleanup      | 🟡 Missing | Added  |
| 10 | Monolithic file                | 🟢 Quality | Refactored |
| 11 | Hardcoded config               | 🟢 Quality | Configurable |
| 12 | No response models             | 🟢 Quality | Added  |
| 13 | Prompts mixed with logic       | 🟢 Quality | Separated |

---

## ⚠️ Not Changed (Left as-is for compatibility)

- **Mode detection:** Still keyword-based. Works for now. Future improvement: use LLM classification.
- **ChromaDB:** Still the vector DB. Sufficient for current scale.
- **Prompt content:** Korean prompt text kept identical — only extracted, not rewritten.
- **API routes/paths:** All endpoints identical to original — no breaking changes.
- **Ollama as LLM backend:** Kept as-is. Usama's fine-tuned model will plug in here.
