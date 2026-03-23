# Improvements & Current State

This file is the change log for the project as it evolved from a single-file prototype into the current document-first RAG system.

It now covers two phases:

1. the original modular refactor and baseline bug fixes
2. the newer document-RAG architecture improvements

Related docs:
- [README.md](/home/tilon/chatbot-karbi/README.md)
- [ARCHITECTURE.md](/home/tilon/chatbot-karbi/ARCHITECTURE.md)

---

## Phase 1 — Original Refactor & Baseline Fixes

### 1. Web Search Now Actually Searches
**Area:** `app/chat/handlers.py`

**Original problem**
- “web search mode” detected current-information queries but still only asked Ollama
- answers about news, weather, stocks, and current events were hallucinated

**What changed**
- added real Tavily search integration before LLM generation

### 2. Embeddings No Longer Hardcode CPU
**Area:** `app/core/embeddings.py`

**Original problem**
- embeddings always ran on CPU
- this made ingestion and retrieval slower than necessary on GPU machines

**What changed**
- embedding device is configurable and CUDA-aware

### 3. Duplicate Ingestion Was Reduced
**Area:** `app/pipeline/ingest.py`

**Original problem**
- repeated `/ingest` calls reprocessed the same files and polluted the vector DB

**What changed**
- already-ingested files are skipped instead of blindly re-added

### 4. Startup Logic Was Modernized
**Area:** `main.py`

**Original problem**
- used deprecated FastAPI startup hooks

**What changed**
- switched to `lifespan`

### 5. Reranker Was Added
**Area:** `app/retrieval/reranker.py`

**Original problem**
- retrieval relied only on raw vector similarity

**What changed**
- added optional `BAAI/bge-reranker-v2-m3`

### 6. Language Metadata Was Added
**Area:** `app/pipeline/parser.py`

**Original problem**
- document language metadata was always `unknown`

**What changed**
- language detection added and stored in metadata

### 7. Logging Was Standardized
**Area:** `app/config.py`, whole app

**Original problem**
- scattered `print()` debugging

**What changed**
- structured Python logging with module-tagged loggers

### 8. LLM Timeout Handling Was Improved
**Area:** `app/core/llm.py`

**Original problem**
- timeout meant immediate failure

**What changed**
- retry logic added for timeout scenarios

### 9. Marker Timeout/Cleanup Was Added
**Area:** `app/pipeline/parser.py`

**Original problem**
- parser subprocesses could hang and leave temp files behind

**What changed**
- timeout and cleanup added

### 10. Monolith → Modular Structure
**Area:** whole repo

**Original problem**
- one large `app.py` mixed config, parsing, retrieval, prompts, and routes

**What changed**
- modularized into focused packages:
  - `api`
  - `chat`
  - `core`
  - `pipeline`
  - `retrieval`
  - `models`

### 11. Hardcoded Settings → `.env`
**Area:** whole repo

**Original problem**
- chunking, timeout, top-k, and model parameters were scattered and hardcoded

**What changed**
- moved core settings into environment configuration

### 12. Response Models Were Added
**Area:** `app/models/schemas.py`

**Original problem**
- endpoints returned untyped raw dicts

**What changed**
- Pydantic schemas added for request/response modeling

### 13. Prompt Templates Were Separated
**Area:** `app/chat/prompts.py`

**Original problem**
- prompt content was mixed into logic

**What changed**
- prompts were extracted into dedicated modules

---

## Phase 2 — Document-First RAG Architecture

### 14. Storage Was Split by Purpose
**Area:** `app/config.py`, `main.py`, `app/core/watcher.py`, `app/api/routes.py`

**Original problem**
- all files lived under one `data/` flow
- chat uploads and permanent docs were mixed together
- restart/watch behavior was confusing

**What changed**
- introduced:
  - `data/library/`
  - `data/uploads/`
  - `data/temp/`
- startup ingest and watcher now target only the library corpus
- chat uploads are handled separately

### 15. Upload Flow Became First-Class
**Area:** `app/api/routes.py`, `app/api/upload_ui.py`

**Original problem**
- upload existed as an idea, but not as a clean end-to-end product flow

**What changed**
- `/chat-with-file`, `/upload`, and `/upload-multiple` now support real chat document workflows
- uploaded docs are ingested immediately
- chat keeps the uploaded document scoped for follow-up questions

### 15A. UI Now Remembers Uploaded Files
**Area:** `app/api/routes.py`, `app/api/upload_ui.py`, document registry

**Original problem**
- uploaded files behaved like one-off actions in the UI
- users could not easily see what had already been uploaded
- there was no simple sidebar flow to re-scope a chat back to an earlier upload

**What changed**
- the built-in UI can now upload multiple files in one action
- remembered uploads are loaded from the document registry
- a sidebar section lists uploaded files with basic metadata
- clicking a remembered upload re-scopes the current chat to that document

### 15B. Live Reranking Became a Runtime Policy
**Area:** `app/config.py`, `app/retrieval/retriever.py`

**Original problem**
- reranking improved some benchmark scores
- but live latency cost was too high for normal chat use

**What changed**
- live retrieval now respects a separate `LIVE_RERANKER_ENABLED` flag
- default chat can stay on lower-latency hybrid retrieval
- reranking remains available for benchmark and ablation comparisons

### 15C. Multi-Document Comparison Scope Was Added
**Area:** `app/models/schemas.py`, `app/chat/handlers.py`, `app/api/routes.py`, `app/api/upload_ui.py`

**Original problem**
- chat could only keep one active uploaded document at a time
- comparison across remembered uploads was not a real backend feature

**What changed**
- chat requests/responses now support multi-document scope fields
- the upload sidebar can toggle multiple remembered uploads
- when more than one document is selected, chat builds a comparison-oriented document context
- comparison responses stay grounded in the selected documents instead of relying on freeform general chat

### 16. Direct OCR/Text Extraction Flow Was Added
**Area:** `app/chat/handlers.py`

**Original problem**
- image text requests still went through normal retrieval and often failed awkwardly

**What changed**
- prompts like:
  - `give me the text in the image`
  - `what does this image say`
  - `텍스트 추출해줘`
  now return extracted text directly

### 17. Parser Routing Became Page-Aware
**Area:** `app/pipeline/parser.py`

**Original problem**
- parsing decisions were too coarse and not robust enough for mixed PDFs

**What changed**
- per-page routing added
- page classification added:
  - `digital`
  - `hybrid`
  - `scanned`
- richer metadata added:
  - `doc_id`
  - checksum
  - quality flags
  - text yield
  - gibberish ratio
  - extraction confidence
  - layout hints

### 18. Semantic Chunking Replaced Blind Splitting
**Area:** `app/pipeline/chunker.py`

**Original problem**
- chunking at fixed character boundaries broke document meaning

**What changed**
- heading-aware, table-aware, breadcrumb-carrying chunking

### 19. Contextual Enrichment Was Added
**Area:** `app/pipeline/enricher.py`

**Original problem**
- embeddings only saw raw text, not enough context about where chunks came from

**What changed**
- chunks are enriched with document/section/page context before embedding

### 20. Retrieval Became Hybrid
**Area:** `app/retrieval/retriever.py`, `app/retrieval/keyword_index.py`

**Original problem**
- vector-only retrieval was weak for exact tokens, codes, and mixed technical text

**What changed**
- hybrid retrieval now combines:
  - vector search
  - keyword/BM25-style search
  - reciprocal rank fusion

### 21. Full-Document Scoped Retrieval Was Added
**Area:** `app/chat/handlers.py`, `app/retrieval/retriever.py`, `app/core/vectorstore.py`

**Original problem**
- “summarize/analyze this document” still used only top-k chunks

**What changed**
- whole-document tasks can now load all chunks for the scoped document

### 22. Confidence Gating Was Added
**Area:** `app/chat/handlers.py`, `app/retrieval/retriever.py`

**Original problem**
- low-quality retrieval could still lead to confident wrong answers

**What changed**
- low-confidence scoped document retrieval now returns grounded fallback responses

### 23. Document Registry V1 Was Added
**Area:** `app/core/document_registry.py`

**Original problem**
- the system was mostly filename-aware, not truly document-aware

**What changed**
- added a persistent registry with stable `doc_id`
- tracks:
  - source
  - source type
  - checksum
  - page count
  - chunk count
  - status

### 24. Scoped Chat Now Carries `doc_id`
**Area:** `app/models/schemas.py`, `app/chat/handlers.py`, `app/api/routes.py`, `app/api/upload_ui.py`

**Original problem**
- scoped chat relied too much on filename identity

**What changed**
- `active_doc_id` was added end-to-end
- this reduces ambiguity when filenames collide

### 25. Duplicate Upload/Watcher Conflicts Were Reduced
**Area:** `app/core/watcher.py`, `app/api/routes.py`

**Original problem**
- upload and watcher paths could race and re-ingest the same file

**What changed**
- watcher suppression added for files already handled by upload endpoints

### 26. Vectorstore Reset Was Made Safer
**Area:** `app/core/vectorstore.py`

**Original problem**
- deleting the Chroma backing directory while the app still had a live handle could produce readonly DB errors later

**What changed**
- reset now clears the collection safely in place

### 27. Reranker Runtime Strategy Was Improved
**Area:** `app/retrieval/reranker.py`, `.env`

**Original problem**
- reranker could compete with Ollama for GPU memory and crash with CUDA OOM

**What changed**
- reranker is now configurable by device
- CPU-first reranking is supported for stability
- reranker load failure is cached instead of retried endlessly

### 28. Repeated Uploads Now Replace Scoped Chunks
**Area:** `app/core/vectorstore.py`, `app/pipeline/ingest.py`

**Original problem**
- re-uploading the same upload-scoped file could accumulate duplicate chunks for the same `doc_id`
- this polluted whole-document scoped prompts and produced unstable answers

**What changed**
- chunk IDs are now deterministic for stored chunks
- upload re-ingestion deletes old chunks for the same `doc_id` before adding new ones
- whole-document fetches dedupe repeated chunk signatures

### 29. Bundled Upload PDFs Now Handle Ambiguity Better
**Area:** `app/chat/handlers.py`

**Original problem**
- some uploaded PDFs bundle multiple internal guidelines in one file
- generic questions like `지원대상이 어떻게 돼?` or `제3조 알려줘` were ambiguous
- even named questions like `프로젝트Lab 제4조` could accidentally answer from another sub-guideline with the same article number

**What changed**
- ambiguous bundled-PDF questions now ask for clarification
- if the user names a specific sub-guideline such as `프로젝트Lab`, `대학원 인턴십`, or `캡스톤디자인`, full-document context is narrowed to that scope only
- continuation pages inherit the active sub-guideline label

### 30. Upload-Scoped Benchmark Was Added
**Area:** `finetuning/data/benchmark_upload_scoped_v1.jsonl`, `scripts/run_benchmark.py`

**Original problem**
- the main benchmark covered library docs well, but upload-scoped QA had no repeatable regression test
- bundled upload behavior was being tested manually only

**What changed**
- added `benchmark_upload_scoped_v1.jsonl`
- benchmark runner now supports optional `scope_source_type`
- upload-scoped evaluation now covers:
  - ambiguity clarification
  - named sub-guideline narrowing
  - upload-specific not-found behavior

### 31. QLoRA Starter Dataset Was Added
**Area:** `finetuning/data/qlora_train_v1.jsonl`, `finetuning/README.md`, `finetuning/data/README.md`

**Original problem**
- the repo had benchmark data but no first supervised dataset scaffold for answer tuning

**What changed**
- added a small benchmark-linked supervised dataset scaffold
- includes:
  - grounded lookup answers
  - bilingual grounded answers
  - upload ambiguity clarification answers
  - clean refusal answers
- documentation now explains how benchmark rows feed QLoRA data creation

---

## Current Strengths

The project is now strongest in:
- ingestion architecture
- parser fallback design
- upload/document flow
- hybrid retrieval foundation
- document-aware scoped chat
- benchmarked library-doc QA
- benchmarked upload-scoped QA

## Current Gaps

The project is still weakest in:
- full QLoRA training/evaluation scripts
- multi-document comparison
- richer block-level structured artifacts
- tuned production thresholds for real Tilon docs

---

## Current Summary Table

| Area | Status | Notes |
|---|---|---|
| Modular backend | Done | Core refactor complete |
| PDF/image ingestion | Strong | Multi-step parser working |
| Upload workflow | Strong | Chat uploads are first-class |
| Semantic chunking | Done | Heading-aware and table-aware |
| Context enrichment | Done | Supports better retrieval |
| Hybrid retrieval | Done | Vector + keyword + fusion |
| Reranking | Added | Still being tuned for runtime behavior |
| Confidence gating | Added | Needs benchmark tuning |
| Document registry | Started | `doc_id` exists, lifecycle not complete |
| Library benchmark | Done | Stable baseline exists |
| Upload benchmark | Done | Scoped upload regressions are now measurable |
| QLoRA starter dataset | Started | First supervised scaffold exists |
| UI polish | Partial | Functional, not product-polished |
| Evaluation expansion | In progress | Broader coverage still needed |
| QLoRA training script | Missing | Next major implementation step |

---

## Bigger-Picture Conclusion

The project is no longer blocked by basic architecture.

The main risk now is not lack of features, but lack of:
- broader benchmark coverage
- automated QLoRA training/evaluation scripts
- disciplined model-to-model comparison after fine-tuning

That is why the repo can feel “stuck” even though it has improved significantly.

The next milestone is:

1. keep expanding benchmark coverage on real Tilon documents and uploads
2. grow the supervised QLoRA dataset from the benchmark rows
3. add the actual QLoRA training/evaluation scripts
4. then compare base RAG vs RAG + QLoRA
