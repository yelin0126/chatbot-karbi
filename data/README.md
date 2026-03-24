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

## Current Behavior

- files in `data/library/` are treated as persistent corpus documents
- files in `data/uploads/` are treated as chat-scoped documents
- chat uploads are ingested immediately and remembered in the document registry
- remembered uploads can later be re-scoped from the UI sidebar
- repeated uploads of the same file are tracked by stable `doc_id` and replace prior scoped chunks

## Why This Split Matters

- cleaner restart behavior
- no accidental mixing of one-off chat uploads with long-term corpus files
- easier debugging and evaluation
- clearer separation between library benchmarking and upload-scoped benchmarking

## Git Policy

- the folder structure is tracked so teammates can see the intended layout after cloning
- runtime upload files, temp files, and local-only registry artifacts are ignored
- if a document should belong to the shared benchmark/library corpus, place it in `data/library/`

## Future Improvements To Consider

Advanced approaches that may affect data layout later:
- richer block-level artifact storage for tables and clause-level indexing
- versioned document lifecycle handling
- cached OCR/layout artifacts for expensive documents
- offline evaluation snapshots for reproducible benchmark runs
