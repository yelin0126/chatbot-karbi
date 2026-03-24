# Tilon AI Chatbot

Document-first RAG chatbot for Korean/English PDFs and images. The system supports permanent library documents, chat uploads, upload-scoped QA, comparison-style retrieval, deterministic multi-file summary, and a local QLoRA workflow for answer-model improvement.

## Current Project State

The project is now in late stabilization and evaluation, not early prototyping.

What is complete and working:
- text PDFs, scanned PDFs, and images can be uploaded and ingested
- library documents and chat uploads are stored in separate scopes
- the built-in UI supports multi-file upload and remembered upload selection
- retrieval is hybrid: vector + keyword + fusion, with optional reranking
- single-document grounded QA is strong
- upload-scoped QA is working
- multi-file summary works in deterministic file-by-file mode
- repeated upload ingestion is deduplicated by stable `doc_id`
- the QLoRA training and evaluation pipeline works end to end

What is strongest today:
- parser / ingestion architecture
- hybrid retrieval and scoped document chat
- Korean grounded-answer improvement from QLoRA

What is still weaker:
- clarification for ambiguous bundled uploads
- nuanced comparison answers
- exact phrase fidelity in some lookup cases
- OCR-heavy screenshot/image summary quality

## Best Current Fine-Tuned Adapter

Current best checkpoint:
- [qwen25-qlora-v6](/home/tilon/chatbot-karbi/finetuning/output/qwen25-qlora-v6)

Hard Korean-heavy eval result on [qlora_eval_ko_strict_v2.jsonl](/home/tilon/chatbot-karbi/finetuning/data/qlora_eval_ko_strict_v2.jsonl):
- base avg: `0.244`
- adapter avg: `0.4173`
- adapter wins: `12/15`
- base wins: `0/15`
- ties: `3/15`

Interpretation:
- QLoRA is helping
- `v6` is meaningfully better than the base model
- `v6` is the best current candidate, but broader app-level validation is still needed before making it the default everywhere

## Quick Start

```bash
cd /home/tilon/chatbot-karbi
cp .env.example .env
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Start Ollama in another terminal:

```bash
ollama serve
ollama pull qwen2.5:7b
ollama pull qwen2.5vl:7b
```

Run the API:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Open:
- `http://127.0.0.1:8000/ui`
- `http://127.0.0.1:8000/docs`

See also:
- [ARCHITECTURE.md](/home/tilon/chatbot-karbi/ARCHITECTURE.md)
- [IMPROVEMENTS.md](/home/tilon/chatbot-karbi/IMPROVEMENTS.md)
- [finetuning/README.md](/home/tilon/chatbot-karbi/finetuning/README.md)

## Storage Model

- `data/library/`
  Persistent knowledge-base documents. Startup ingest and the watcher use this folder.

- `data/uploads/`
  Chat-uploaded documents. These are ingested immediately but kept separate from the watched library corpus.

- `data/temp/`
  Optional intermediate processing files.

Why this split matters:
- clearer lifecycle management
- no accidental treatment of chat uploads as permanent library docs
- safer restart behavior
- easier benchmarking of library vs upload-scoped behavior

## Current Ingestion Flow

### Library documents
1. Add files to `data/library/`
2. Startup ingest or the watcher detects them
3. Parser extracts text from PDF/image
4. Semantic chunking and contextual enrichment run
5. Embeddings and keyword index are updated

### Chat uploads
1. Upload through `/ui`, `/upload`, `/upload-multiple`, or `/chat-with-file`
2. File is saved to `data/uploads/`
3. Content is parsed, chunked, enriched, embedded, and registered
4. The sidebar remembers uploaded files from the document registry
5. Later chat can re-scope to one or many remembered uploads

## Parsing / Extraction Stack

The parser uses a layered pipeline:

1. `marker_single`
2. `PyMuPDF`
3. `qwen2.5vl`
4. `tesseract`

Important recent improvements:
- per-page routing instead of file-level routing
- page classification: `digital`, `hybrid`, `scanned`
- quality gates for low-yield or garbled text
- checksum-backed `doc_id`
- faster OCR-first / VLM-later fallback policy on hard files

## Retrieval Status

Current retrieval stack:

1. Chroma vector retrieval
2. keyword retrieval
3. reciprocal rank fusion
4. optional reranking
5. confidence gating

Scoped behavior:
- specific question -> top-k scoped retrieval
- whole-document task -> full scoped document loading
- image text request -> direct extraction response
- bundled upload ambiguity -> clarification or sub-guideline narrowing

## Multi-File Summary Status

Multi-file summary is implemented and usable.

Current behavior:
- whole-upload-corpus requests can load all uploaded documents
- deterministic file-by-file summary mode is available
- each file is summarized once with its own heading

Current limitation:
- OCR-heavy or noisy image-derived uploads can still produce rough summaries
- this is a presentation and evidence-quality issue, not a core feature gap

## QLoRA Status

The QLoRA workflow is no longer “pending.” It is implemented and benchmarked.

Available assets:
- strict eval sets:
  - [qlora_eval_ko_strict_v1.jsonl](/home/tilon/chatbot-karbi/finetuning/data/qlora_eval_ko_strict_v1.jsonl)
  - [qlora_eval_ko_strict_v2.jsonl](/home/tilon/chatbot-karbi/finetuning/data/qlora_eval_ko_strict_v2.jsonl)
- training sets:
  - [qlora_train_v1.jsonl](/home/tilon/chatbot-karbi/finetuning/data/qlora_train_v1.jsonl)
  - [qlora_train_v2.jsonl](/home/tilon/chatbot-karbi/finetuning/data/qlora_train_v2.jsonl)
  - [qlora_train_v3.jsonl](/home/tilon/chatbot-karbi/finetuning/data/qlora_train_v3.jsonl)
  - [qlora_train_v4.jsonl](/home/tilon/chatbot-karbi/finetuning/data/qlora_train_v4.jsonl)
- scripts:
  - [train.py](/home/tilon/chatbot-karbi/finetuning/train.py)
  - [evaluate_adapter.py](/home/tilon/chatbot-karbi/finetuning/evaluate_adapter.py)
  - [infer_compare.py](/home/tilon/chatbot-karbi/finetuning/infer_compare.py)

The training/eval path now:
- uses Qwen chat templates correctly
- supports offline local loading
- cleanly compares base vs adapter
- tracks Korean-heavy failure modes such as mixed-language drift, not-found behavior, and ambiguous document questions

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Server status and path info |
| `GET` | `/health` | Health check |
| `GET` | `/models` | Model list for built-in UI |
| `POST` | `/chat` | Main chat endpoint |
| `POST` | `/chat-with-file` | Upload a file and ask in one request |
| `POST` | `/upload` | Upload and ingest one file |
| `POST` | `/upload-multiple` | Upload and ingest multiple files |
| `POST` | `/ingest` | Ingest `data/library/` |
| `DELETE` | `/reset-db` | Clear vector DB |
| `GET` | `/docs-list` | List stored chunks/documents |
| `GET` | `/uploaded-docs` | List remembered uploads |
| `POST` | `/count-keyword` | Count a keyword in a stored source file |
| `GET` | `/v1/models` | OpenAI-compatible model list |
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat |

## Project Structure

```text
chatbot-karbi/
├── main.py
├── .env.example
├── requirements.txt
├── chroma_db/
├── data/
│   ├── library/
│   ├── uploads/
│   └── temp/
├── app/
│   ├── api/
│   ├── chat/
│   ├── core/
│   ├── models/
│   ├── pipeline/
│   └── retrieval/
└── finetuning/
```

## Current Roadmap

### Phase A: Real-World Validation
- test `v6` against real chatbot prompts and uploaded documents
- compare base runtime behavior vs adapter behavior in the actual product flow
- identify which failures are retrieval-side vs answer-model-side

### Phase B: Accuracy Refinement
- improve clarification behavior for ambiguous bundled uploads
- improve finer-grained comparison answers
- improve exact phrase fidelity for strict lookups
- improve noisy OCR/image summary quality

### Phase C: Advanced Approaches

These were intentionally deferred earlier, but they are now part of the roadmap for improving the full project:

- structure-aware retrieval:
  - table-aware artifacts
  - clause/article indexing
  - richer block-level storage
- smarter routing:
  - query-type classifier for lookup vs summary vs comparison vs extraction
  - document-type-aware prompt routing
- stronger retrieval research options:
  - late-interaction retrieval such as ColBERT-style reranking
  - hierarchical retrieval for long bundled documents
  - learned query expansion / hard-negative mining
- answer verification:
  - citation / evidence consistency checks
  - answer-type guards for `clarify`, `not_found`, `mention_only`, `comparison`
- stronger training methods after SFT:
  - judge-assisted dataset expansion
  - preference tuning such as DPO/ORPO for refusal and comparison quality
  - targeted synthetic augmentation only after real-document coverage is strong

These are not current blockers for a working product, but they are valid next-stage approaches for pushing accuracy and robustness higher.

## One-Line Summary

The project is now a working document-RAG chatbot with a validated fine-tuning pipeline, and `qwen25-qlora-v6` is the best current adapter, but the next stage should focus on real-world validation and targeted accuracy refinement rather than broad architectural changes.
