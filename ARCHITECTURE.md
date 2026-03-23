# Tilon AI Chatbot Architecture

## Overview

Tilon AI Chatbot is a document-first RAG backend for English/Korean PDFs and images.

This architecture should be understood as two connected systems:

1. a document RAG inference pipeline
2. a QLoRA fine-tuning pipeline for the final answer model

The current repo is much stronger on the RAG side than the QLoRA side. RAG is now stable enough to support benchmark-driven tuning, and the next milestone is to expand the supervised QLoRA dataset and training workflow rather than redesigning the parser again.

The product supports two document sources:

- `data/library/`
  Persistent team documents that form the long-term knowledge base

- `data/uploads/`
  Chat-uploaded files that are ingested for the current user flow and scoped to chat

The system is optimized for:
- summarization
- question answering
- exact information lookup
- text extraction from images/scanned pages
- section/structure-aware retrieval

## High-Level Flow

### 1. Library documents
1. Team adds files to `data/library/`
2. Startup ingest or file watcher detects them
3. Parser extracts text from PDF/image
4. Chunker creates semantic chunks
5. Enricher prepends contextual headers
6. Embeddings are stored in ChromaDB
7. Keyword index is rebuilt/updated for hybrid retrieval

### 2. Chat uploads
1. User uploads a file in `/ui` or `/chat-with-file`
2. File is saved to `data/uploads/`
3. It is parsed, chunked, enriched, and stored
4. The upload is written into the document registry as a remembered chat document
5. The `/ui` sidebar lists remembered uploads so users can re-select them later
6. Remembered uploads can be multi-selected for comparison-style chat
7. Chat keeps an `active_source` or multi-document selection so follow-up questions stay scoped
8. Temporary chat uploads are not treated as permanent library docs by the watcher
9. Re-uploading the same scoped file replaces old chunks for that `doc_id` instead of accumulating duplicates
10. Bundled upload PDFs with multiple internal guidelines can trigger clarification or be narrowed to the named sub-guideline

## Bigger-Picture Architecture

The full product flow is:

1. document input
2. extraction / parsing
3. semantic chunking
4. contextual enrichment
5. embeddings
6. vector + keyword storage
7. hybrid retrieval
8. reranking
9. confidence gating
10. prompt construction
11. answer generation
12. future QLoRA-enhanced answer generation

Important distinction:
- RAG is responsible for retrieving correct evidence
- QLoRA is responsible for improving how the model uses that evidence

QLoRA is not a substitute for:
- extraction quality
- chunking quality
- retrieval precision
- confidence gating

## Storage Layout

```text
data/
├── library/   # persistent corpus
├── uploads/   # chat-uploaded files
└── temp/      # optional intermediate files
```

This split avoids mixing permanent documents with one-off chat uploads and prevents unexpected re-ingestion after restart.

## Core Components

### API Layer
- [main.py](/home/tilon/chatbot-karbi/main.py)
- [app/api/routes.py](/home/tilon/chatbot-karbi/app/api/routes.py)
- [app/api/upload_ui.py](/home/tilon/chatbot-karbi/app/api/upload_ui.py)
- [app/api/openai_compat.py](/home/tilon/chatbot-karbi/app/api/openai_compat.py)

Responsibilities:
- HTTP endpoints
- built-in chat/upload UI
- OpenAI-compatible endpoints
- upload handling and folder routing

### Parsing / Ingestion Layer
- [app/pipeline/parser.py](/home/tilon/chatbot-karbi/app/pipeline/parser.py)
- [app/pipeline/chunker.py](/home/tilon/chatbot-karbi/app/pipeline/chunker.py)
- [app/pipeline/enricher.py](/home/tilon/chatbot-karbi/app/pipeline/enricher.py)
- [app/pipeline/ingest.py](/home/tilon/chatbot-karbi/app/pipeline/ingest.py)

Responsibilities:
- PDF/image text extraction
- semantic chunking
- contextual enrichment
- ingestion orchestration

### Retrieval Layer
- [app/core/vectorstore.py](/home/tilon/chatbot-karbi/app/core/vectorstore.py)
- [app/retrieval/retriever.py](/home/tilon/chatbot-karbi/app/retrieval/retriever.py)
- [app/retrieval/keyword_index.py](/home/tilon/chatbot-karbi/app/retrieval/keyword_index.py)
- [app/retrieval/reranker.py](/home/tilon/chatbot-karbi/app/retrieval/reranker.py)

Responsibilities:
- vector search
- BM25-style keyword retrieval
- reciprocal rank fusion
- optional reranking
- confidence-aware scoped retrieval
- deterministic chunk IDs for safer scoped re-ingestion
- doc-scoped deletion/replacement for repeated uploads

### Chat Layer
- [app/chat/handlers.py](/home/tilon/chatbot-karbi/app/chat/handlers.py)
- [app/chat/prompts.py](/home/tilon/chatbot-karbi/app/chat/prompts.py)

Responsibilities:
- build final prompt
- unify chat behavior
- direct extraction responses for image/text requests
- scoped document fallback when retrieval confidence is low

## Extraction Strategy

The parser uses a layered approach:

1. `marker_single`
   Best for structured digital PDFs

2. `PyMuPDF`
   Fast text-layer extraction for born-digital pages

3. `qwen2.5vl`
   Vision fallback for scanned or image-heavy pages

4. `tesseract`
   OCR fallback

Recent improvements:
- per-page routing
- page type classification: `digital`, `hybrid`, `scanned`
- quality gates for low text yield / garbled text
- richer page metadata such as heading hints and layout counts

## Retrieval Strategy

The current retrieval path is hybrid:

1. vector retrieval from ChromaDB
2. keyword retrieval from in-memory index
3. reciprocal rank fusion
4. optional reranking
5. confidence gating for scoped document chat

For scoped uploaded-file chat:
- specific question -> top-k scoped retrieval
- full-document tasks like summary/analysis -> full-document scoped context
- direct OCR/transcription requests -> return extracted text directly
- bundled multi-guideline uploads -> ask for clarification on ambiguous clause questions
- named sub-guideline questions -> narrow context to the relevant sub-guideline pages only

## Evaluation Assets

The repo now has two benchmark tracks:

1. library benchmark
   - `finetuning/data/benchmark_tilon_v1.jsonl`
   - measures persistent library-doc QA, summary, bilingual behavior, and not-found handling

2. upload-scoped benchmark
   - `finetuning/data/benchmark_upload_scoped_v1.jsonl`
   - measures upload-specific ambiguity handling, named sub-guideline narrowing, and refusal behavior

The first supervised QLoRA scaffold also exists:

- `finetuning/data/qlora_train_v1.jsonl`
  - benchmark-linked starter examples
  - includes grounded answers, clarification answers, and refusal answers

## Maturity By Layer

### Strong / mostly ready
- upload and storage split (`library` vs `uploads`)
- parser routing and fallback extraction
- semantic chunking
- contextual enrichment
- document-scoped chat
- hybrid retrieval foundation

### Working but still being tuned
- reranker runtime strategy
- confidence thresholds
- screenshot/image upload behavior in broader chat
- document registry behavior
- upload benchmark breadth beyond the first bundled-PDF case

### Not started or not complete
- full QLoRA training script / adapter workflow
- multi-document comparison
- richer block-level artifacts for tables/citations/versioning

## UI Behavior

The built-in `/ui` is the primary frontend.

Behavior:
- uploaded file is stored in `data/uploads/`
- one or many files can be uploaded from the built-in UI
- remembered uploaded documents appear in the left sidebar
- clicking a remembered upload re-scopes the current chat to that document
- selecting multiple remembered uploads enables comparison-style document chat
- the chat remembers `active_source`
- follow-up messages stay scoped to the uploaded file
- user can clear scope and return to global retrieval

## Current Limits

Still not fully implemented:
- full document registry lifecycle management
- document versioning
- multi-document comparison workflow
- richer block-level artifact storage beyond page metadata
- polished production UI design
- QLoRA training/evaluation pipeline

## Why The Project Can Feel “Stuck”

The repo has moved beyond the early prototype stage, but progress can still feel slow because most work is happening one example at a time:
- one screenshot
- one OCR case
- one threshold
- one reranker issue

That is useful engineering work, but it does not yet create a measurable product-quality loop.

What is still missing is:
- a larger benchmark set covering more uploaded-file patterns
- a broader representative query set
- automated QLoRA training/evaluation scripts
- model-to-model comparison after fine-tuning

Without these, it is hard to tell whether the project is improving globally or only locally.

## Recommended Next Steps

### Phase 1: Stabilize RAG Core
1. Finish tuning retrieval and confidence behavior on real Tilon docs
2. Finalize document registry behavior and upload lifecycle
3. Keep prompt/context format stable enough for training reuse

### Phase 2: Expand Evaluation Coverage
1. Add more representative Tilon manuals, guides, and internal docs
2. Expand both library and upload-scoped question sets for:
   - exact lookup
   - summary
   - OCR/image text
   - section understanding
   - negative “not found” cases
   - Korean/English mixed queries
3. Keep recording baseline system behavior as changes land

### Phase 3: Start QLoRA Properly
1. Expand the supervised dataset using the same live prompt/context format
2. Fine-tune the answer model, not the retriever
3. Compare:
   - base model
   - RAG + base
   - RAG + QLoRA

## Evaluation Adoption Plan

To translate the broader evaluation strategy into this specific repo, use a phased adoption model.

### Adopt Now

These fit the current codebase immediately and should become the default workflow:

1. Keep the repo's local benchmark runner as the primary evaluation backbone
   - `scripts/run_benchmark.py`
   - `finetuning/data/benchmark_tilon_v1.jsonl`
   - `finetuning/data/benchmark_upload_scoped_v1.jsonl`
2. Continue expanding benchmark coverage with real Tilon documents and real upload-scoped cases
3. Add deterministic retrieval metrics
   - `Recall@k`
   - `MRR`
   - `NDCG@k`
   - ablations for vector-only / keyword-only / hybrid / hybrid + reranker
4. Expand the supervised QLoRA dataset from benchmark-linked examples
   - `finetuning/data/qlora_train_v1.jsonl`

### Adopt Next

These are strong next steps after the current benchmark/data scaffolds are larger and stable:

1. Add deterministic faithfulness and citation checks
   - NLI-based entailment scoring
   - character-level overlap / Korean-friendly answer checks
2. Add a calibrated local LLM-as-judge layer
   - DeepEval with local Ollama-backed `qwen2.5:7b`
   - only after comparing its judgments against a small human-checked set
3. Expand evaluation categories for:
   - OCR/image-heavy docs
   - tables/numerical reasoning
   - more bilingual/code-switched cases
   - multi-hop section reasoning

### Adopt Later

These are valuable, but should wait until the benchmark + training loop is already routine:

1. Arize Phoenix for richer embedding/retrieval observability
2. Langfuse for production tracing and human review queues
3. MLflow for experiment tracking across RAG and QLoRA variants
4. automated production sampling / drift alerts / feedback flywheel infrastructure

## Model Choice And Fine-Tuning

Changing the inference LLM can affect fine-tuning significantly.

### What transfers

- benchmark datasets
- retrieval metrics
- most benchmark question sets
- most supervised training examples

### What does not transfer cleanly

- LoRA/QLoRA adapters themselves
- model-specific prompt behavior
- optimal training hyperparameters
- exact evaluation baselines for answer quality

In practice:

- if you change from `qwen2.5:7b` to a different base family such as Llama or Mistral, you should assume the fine-tuning artifacts are not reusable
- if you stay within the same target base model family, the datasets still transfer well, but the trained adapter is still tied to that specific base model

Recommended rule:

- freeze the target base answer model before starting serious QLoRA training
- keep benchmark datasets model-agnostic
- treat model swaps as new baseline comparisons, not as a small runtime tweak

## QLoRA Start Conditions

QLoRA should begin after:
- retrieval quality is stable enough on real Tilon docs
- the prompt/context format is frozen
- benchmark questions exist
- baseline metrics are recorded

That is the point where fine-tuning becomes measurable instead of speculative.
