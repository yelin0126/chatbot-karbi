# Tilon AI Chatbot Architecture

## Overview

Tilon AI Chatbot is a document-first RAG backend for Korean/English PDFs and images with an attached QLoRA fine-tuning workflow for the answer model.

The repository should be understood as two connected systems:

1. document RAG inference
2. answer-model fine-tuning and evaluation

Both are now active. The repo is still stronger on the RAG side, but the QLoRA path is no longer just scaffolding.

## Product Scopes

The system works with two document scopes:

- `data/library/`
  Persistent team documents that form the long-term knowledge base

- `data/uploads/`
  Chat-uploaded files that are ingested for the current chat flow and kept separate from the permanent library corpus

## High-Level Flow

### Library documents
1. Files are added to `data/library/`
2. Startup ingest or watcher detects them
3. Parser extracts text from PDF/image pages
4. Semantic chunking runs
5. Contextual enrichment adds document/page/section cues
6. Vectorstore and keyword index are updated
7. Chat can retrieve them globally

### Chat uploads
1. User uploads in `/ui` or `/chat-with-file`
2. File is saved to `data/uploads/`
3. It is parsed, chunked, enriched, embedded, and registered
4. The UI sidebar remembers uploaded files
5. Chat can re-scope to one or many remembered uploads
6. Whole-document tasks can load full upload scope rather than top-k chunks only

## Bigger-Picture Architecture

The full product path is:

1. document input
2. extraction / parsing
3. semantic chunking
4. contextual enrichment
5. embeddings
6. vector + keyword storage
7. hybrid retrieval
8. optional reranking
9. confidence gating
10. prompt construction
11. answer generation
12. optional adapter-enhanced answer generation

Important distinction:
- RAG retrieves the evidence
- QLoRA improves how the model uses that evidence

QLoRA is not a substitute for:
- bad extraction
- weak chunking
- poor retrieval
- missing evidence

## Storage Layout

```text
data/
├── library/
├── uploads/
└── temp/
```

This separation keeps permanent corpus behavior and chat-upload behavior independent.

## Core Components

### API layer
- [main.py](/home/tilon/chatbot-karbi/main.py)
- [app/api/routes.py](/home/tilon/chatbot-karbi/app/api/routes.py)
- [app/api/upload_ui.py](/home/tilon/chatbot-karbi/app/api/upload_ui.py)
- [app/api/openai_compat.py](/home/tilon/chatbot-karbi/app/api/openai_compat.py)

Responsibilities:
- endpoints
- built-in UI
- upload handling
- OpenAI-compatible interface

### Parsing / ingestion layer
- [app/pipeline/parser.py](/home/tilon/chatbot-karbi/app/pipeline/parser.py)
- [app/pipeline/chunker.py](/home/tilon/chatbot-karbi/app/pipeline/chunker.py)
- [app/pipeline/enricher.py](/home/tilon/chatbot-karbi/app/pipeline/enricher.py)
- [app/pipeline/ingest.py](/home/tilon/chatbot-karbi/app/pipeline/ingest.py)

Responsibilities:
- PDF/image extraction
- semantic chunking
- contextual enrichment
- ingestion orchestration

### Retrieval layer
- [app/core/vectorstore.py](/home/tilon/chatbot-karbi/app/core/vectorstore.py)
- [app/retrieval/retriever.py](/home/tilon/chatbot-karbi/app/retrieval/retriever.py)
- [app/retrieval/keyword_index.py](/home/tilon/chatbot-karbi/app/retrieval/keyword_index.py)
- [app/retrieval/reranker.py](/home/tilon/chatbot-karbi/app/retrieval/reranker.py)

Responsibilities:
- vector search
- keyword search
- reciprocal rank fusion
- reranking
- scoped retrieval
- whole-document loading

### Chat layer
- [app/chat/handlers.py](/home/tilon/chatbot-karbi/app/chat/handlers.py)
- [app/chat/prompts.py](/home/tilon/chatbot-karbi/app/chat/prompts.py)

Responsibilities:
- prompt building
- answer-type handling
- low-confidence fallback behavior
- upload-scope behavior
- deterministic file-by-file multi-document summary

### Fine-tuning layer
- [finetuning/train.py](/home/tilon/chatbot-karbi/finetuning/train.py)
- [finetuning/evaluate_adapter.py](/home/tilon/chatbot-karbi/finetuning/evaluate_adapter.py)
- [finetuning/infer_compare.py](/home/tilon/chatbot-karbi/finetuning/infer_compare.py)

Responsibilities:
- Qwen chat-template-consistent training
- local adapter evaluation
- base vs adapter comparison
- strict Korean-heavy regression testing

## Extraction Strategy

Current layered extraction strategy:

1. `marker_single`
2. `PyMuPDF`
3. `qwen2.5vl`
4. `tesseract`

Recent parser improvements:
- per-page routing
- page type classification: `digital`, `hybrid`, `scanned`
- quality gates for low yield or gibberish
- OCR-first / VLM-later runtime improvements on difficult files
- richer metadata for page quality and origin

## Retrieval Strategy

Current retrieval path:

1. vector retrieval
2. keyword retrieval
3. reciprocal rank fusion
4. optional reranking
5. confidence gating

Scoped document behavior:
- specific question -> top-k scoped retrieval
- summary/analysis -> full scoped document loading
- OCR request -> direct extraction result
- bundled guideline ambiguity -> clarification or named-scope narrowing
- multi-file summary -> deterministic file-level grouping

## Evaluation Assets

Benchmark and evaluation assets now include:

- library benchmark:
  - [benchmark_tilon_v1.jsonl](/home/tilon/chatbot-karbi/finetuning/data/benchmark_tilon_v1.jsonl)
- upload-scoped benchmark:
  - [benchmark_upload_scoped_v1.jsonl](/home/tilon/chatbot-karbi/finetuning/data/benchmark_upload_scoped_v1.jsonl)
- comparison benchmark:
  - [benchmark_compare_v1.jsonl](/home/tilon/chatbot-karbi/finetuning/data/benchmark_compare_v1.jsonl)
- strict QLoRA eval sets:
  - [qlora_eval_ko_strict_v1.jsonl](/home/tilon/chatbot-karbi/finetuning/data/qlora_eval_ko_strict_v1.jsonl)
  - [qlora_eval_ko_strict_v2.jsonl](/home/tilon/chatbot-karbi/finetuning/data/qlora_eval_ko_strict_v2.jsonl)

## Current Maturity By Layer

### Strong
- upload vs library scope separation
- parser routing and fallback extraction
- semantic chunking
- contextual enrichment
- hybrid retrieval foundation
- document-scoped chat
- deterministic multi-file summary
- QLoRA training/eval workflow

### Working but still being tuned
- clarification behavior for bundled documents
- nuanced comparison answers
- OCR-heavy summary cleanliness
- confidence thresholds
- exact phrase fidelity on strict lookup questions

### Future structural upgrades
- richer block-level artifacts
- stronger document-type-aware routing
- broader automated regression coverage

## Advanced Approaches For Full-System Improvement

These were deferred until the foundation was stable. They should now be considered part of the architecture roadmap.

### Retrieval and indexing
- clause/article indexing for policy documents
- table-aware artifacts
- hierarchical retrieval over long bundled documents
- late-interaction reranking such as ColBERT-style retrieval

### Routing and orchestration
- query-type classifier for lookup, summary, comparison, OCR, and clarification
- document-type-aware context formatting
- distinct handling for screenshot/image-heavy uploads

### Generation reliability
- answer-type-specific prompt controllers
- citation/evidence consistency checks
- post-generation validation for risky answer classes

### Fine-tuning upgrades
- failure-driven dataset expansion
- preference tuning after supervised fine-tuning
- judge-assisted comparison and refusal refinement

## Current Limits

Still not fully solved:
- bundled-document clarification quality
- fine-grained comparison quality
- block-level reasoning over tables/forms
- fully polished production UI
- final production rollout decision for adapter defaulting

## Current Conclusion

The architecture is no longer missing its foundation. The next phase is to refine accuracy and reliability on top of a stable system, using benchmark-driven iteration and selected advanced methods where they address real failure modes.
