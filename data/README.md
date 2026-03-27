# Data Layout

This project uses a document-first storage layout under `data/`.

## Directories

- `data/library/`
  Permanent shared documents for the knowledge base. Startup ingest and the watcher target this folder.

- `data/uploads/`
  Runtime chat uploads from `/ui`, `/upload`, `/upload-multiple`, and `/chat-with-file`.

- `data/temp/`
  Temporary processing files.

- `data/pdf/`
  Optional legacy or experimental PDF location kept for older development flows.

## Runtime Artifacts

- `data/document_registry.json`
  Persistent registry of all ingested files. Tracks stable `doc_id` (SHA-256 checksum), source path, scope (`library` or `upload`), and chunk count.

- `data/parent_store.json`
  JSON store of parent chunk texts. Created by the hierarchical chunker. Each child chunk embeds a `parent_id`; at retrieval time, matched children are expanded back to their parent text for richer LLM context. Created automatically on first ingest.

## Current Behavior

- files in `data/library/` are treated as persistent corpus documents
- files in `data/uploads/` are treated as chat-scoped documents
- chat uploads are ingested immediately and remembered in the document registry
- remembered uploads can later be re-scoped from the UI sidebar
- repeated uploads of the same file are tracked by stable `doc_id` and replace prior scoped chunks
- if exactly one uploaded file is active, deictic prompts like `이 문서`, `이 파일`, and `this document` auto-scope to that file
- library questions that explicitly name a document can auto-scope to that library PDF
- unscoped library retrieval can promote itself into an implicit single-document scope when the retrieved evidence is overwhelmingly from one document family
- strict-fact table/value questions can further promote from a document family to one exact source/doc when the scorer is confident enough
- hierarchical parent-child chunking: child chunks (≤300 chars) are indexed; parent sections saved to `parent_store.json` for context expansion
- structure-aware retrieval now expands explicit and section-heading-style article/section queries into fuller local context
- upload-mode whole-document summary questions can use a deterministic single-document summary path for one uploaded file
- supported two-document comparisons can answer directly from the selected pair rather than always using the generic generation path
- `data/test pdfs/` can be used as a staging area, but files only affect the live corpus after being copied into `data/library/`
- the watcher will auto-ingest new files placed in `data/library/`, so large batches can materially change the active benchmark corpus
- the expanded real-PDF corpus in `data/library/` is now part of the grounding regression workflow, not just ad hoc testing
- the active library corpus now includes near-duplicate guideline variants; exact-source disambiguation across those duplicates is the main remaining grounding bottleneck
- until that duplicate-source issue is fully resolved, those rows should not be treated as pure model failures when building new QLoRA data
- scanned/hybrid PDFs in the active corpus use the OCR stack documented in the main README:
  - preferred: PaddleOCR v5 Korean + PyKoSpacing
  - fallback: Tesseract kor+eng
  - escalation only when OCR is still weak: Qwen2.5-VL

## Why This Split Matters

- cleaner restart behavior
- no accidental mixing of one-off chat uploads with long-term corpus files
- easier debugging and evaluation
- clearer separation between library benchmarking and upload-scoped benchmarking
- easier to freeze a reproducible corpus snapshot before a new training round; the current pre-`v10` baseline is anchored to the March 27, 2026 library state

## Git Policy

- the folder structure is tracked so teammates can see the intended layout after cloning
- runtime upload files, temp files, and local-only registry artifacts are gitignored
- if a document should belong to the shared benchmark/library corpus, place it in `data/library/`

## Future Improvements To Consider

- richer block-level artifact storage beyond the current clause/table handling
- versioned document lifecycle handling
- cached OCR/layout artifacts for expensive scanned documents
- offline evaluation snapshots for reproducible benchmark runs
- a dedicated staging-to-library promotion flow for controlled real-PDF regression testing
- better long-document evidence selection for unseen narrative uploads
- canonical duplicate-source handling for near-identical library PDFs
