# Tilon AI Chatbot

Document-first RAG chatbot for Korean/English PDFs and images. The system supports permanent library documents, chat uploads, upload-scoped QA, comparison-style retrieval, deterministic multi-file summary, and a local QLoRA workflow for answer-model improvement.

## Current Project State

The project is a general-purpose Korean/English document QA chatbot. Users upload any documents and ask questions; the system answers based solely on retrieved document content.

What is complete and working:
- text PDFs, scanned PDFs, and images can be uploaded and ingested
- library documents and chat uploads are stored in separate scopes
- the built-in UI supports multi-file upload and remembered upload selection
- chat handling is stage-based and modularized rather than monolithic
- query routing now uses a centralized policy layer plus an optional embedding classifier
- retrieval is hybrid: vector + keyword + RRF fusion, with BGE reranker on CUDA FP16
- structure-aware retrieval expands article/section-level hits into fuller section context
- clause-level chunking for Korean legal articles (`제N조`) — each article is preserved as an atomic chunk
- pdfplumber column-aware extraction for 2-column PDFs (correct reading order)
- single-document grounded QA is strong
- upload-scoped QA is working
- single-upload chats now auto-scope `이 문서`, `이 파일`, and `this document` style questions to the only uploaded file
- multi-file summary works in deterministic file-by-file mode
- context-relevance pre-check and tiered NLI faithfulness gate prevent ungrounded answers
- document-backed grounding now runs for both scoped and unscoped document answers, not only explicit single-document scope
- citation-aware verifier logic and inline-citation fallback are active
- citation footer maps inline `[1]`, `[2]` markers to source file + page and is compacted to the actually useful evidence set
- repeated upload ingestion is deduplicated by stable `doc_id`
- the QLoRA training and evaluation pipeline works end to end
- the answer model runs fully locally (Qwen2.5-7B-Instruct + QLoRA, 4-bit NF4 via BitsAndBytes)
- CJK/Cyrillic/Thai token suppression active (32,257 tokens banned) to prevent script leakage
- greedy decoding (temperature=0.0) for deterministic answers
- narrative single-upload PDFs now have a deterministic single-document summary path instead of always relying on long trimmed prompts
- `제N조` lookups on non-clause narrative PDFs now fail cleanly instead of hallucinating fake article text
- broad year/number questions on a scoped document can use deterministic numeric extraction before free-form generation
- library queries that explicitly name a document can now auto-scope to that library PDF before generation
- unscoped library retrieval can now promote itself to an implicit single-document scope when the top retrieved chunks are dominated by one document family
- two-document comparison queries now have a deterministic comparison path for purpose / procedure / support / role / presence questions
- OCR-heavy policy exact lookups such as award caps and student-instructor hourly rates now have stronger deterministic table/value extraction
- unscoped library grounding improved substantially through scope, retrieval, exact-answer, and comparison-path fixes without changing the deployed QLoRA adapter

What is strongest today:
- PDF-grounded chat over the persistent library and uploaded files
- parser / ingestion architecture (5-method pipeline + pdfplumber + clause splitting)
- hybrid retrieval and scoped document chat
- query routing for lookup, summary, comparison, OCR, and default explanation
- structure-aware retrieval for article/section/chapter questions
- answer reliability pipeline (context-relevance gate + tiered NLI + inline citations + citation footer)
- OCR/table-aware deterministic grounding for hard policy/amendment PDFs
- single-upload document UX after auto-scope and compact evidence formatting
- named-document and dominant-document handling inside the persistent library corpus
- deterministic two-document comparison on the main supported comparison shapes
- general-purpose Korean QA behavior — no document-specific hardcoding
- local LLM stack (no external API dependency)

What is still weaker:
- exact-source locking when the corpus contains near-duplicate PDF variants of the same guideline
- answer faithfulness on the hardest exact-source exact-lookup cases
- benchmark coverage for broader real-world retrieval/generation failures
- OCR-heavy screenshot/image summary quality
- long narrative PDF summarization and selective fact extraction on unseen uploads still need broader real-world validation
- table-level artifacts (extracted but rarely large enough for standalone chunks)
- long-context GPU pressure during full local generation evals
- the broader unscoped library grounding suite is still only partially green, especially on explanation-style and comparison-style rows

Current roadmap position:
- `Phase 10A` chat-flow refactor: complete
- `Phase 10B` routing/classifier layer: complete
- `Phase 10C` structure-aware retrieval: first validated slice complete
- `Phase 10D` verifier / grounding quality: validated on current `v1` / `v2` grounding suites

Latest Phase 10D checkpoint:
- `verifier_grounding_eval_v1`: improved to `6/8`
- `verifier_grounding_eval_v2`: `10/10` on the current hard real-PDF set
- `verifier_grounding_eval_unscoped_v1` (latest checkpoint on March 27, 2026): `22/24`, `avg_source_recall = 0.917`, `avg_answer_point_recall = 0.875`, `correct_not_found_rate = 1.0`
- pre-`v10` baseline snapshot on March 27, 2026:
  - routing `v3`: `40/40`
  - structure retrieval `v1`: `6/11` (`source_recall = 1.0`, `point_recall = 0.444`)
  - structure retrieval `v2`: `6/15` (`source_recall = 1.0`, `point_recall = 0.185`)
  - scoped grounding `v1`: `5/8`
  - external grounding: `2/10`
  - table grounding: `5/8`
- pre-`v10` coverage expansion is now staged with:
  - `verifier_grounding_eval_external_v1.jsonl` for non-primary-library PDF types
  - `verifier_grounding_eval_tables_v1.jsonl` for table/list/form-heavy exact lookups
- current conclusion: the biggest gains came from RAG/scoping/extraction/deterministic-answer fixes, not a new adapter round; the last remaining unscoped misses are now much more model-facing, while the external-PDF slice shows strong Chinese-drift and answer-extraction weakness on unseen PDFs

## Next Roadmap

Immediate pre-`v10` priorities:
- freeze the March 27, 2026 pre-`v10` baseline snapshot as the comparison point
- mine `qlora_train_v10` from the remaining `011` / `014` unscoped failures plus the external-PDF and table-grounding failures
- focus especially on Chinese-drift suppression and exact amount extraction from noisy policy/table context
- keep any further pipeline tweaks smaller than the training-data changes so post-`v10` comparison stays interpretable

Next QLoRA workstream:
- assemble `qlora_train_v10` from benchmark-linked and real PDF-grounded failures
- focus on exact lookup, refusal/not-found, section understanding, comparison, and OCR/table-heavy answer shaping
- keep the existing `question/context/answer` schema
- compare post-`v10` behavior against the single pre-`v10` baseline snapshot rather than scattered historical runs

Broader roadmap after `v10`:
- add more unseen external PDFs to the grounding suite
- add more multi-file and long-narrative PDF rows
- collect live failure logs and separate pipeline failures from true model-side failures
- only then consider a larger model-family migration or preference-tuning round

## Current Active Adapter

Current deployed adapter:
- `finetuning/output/qwen25-qlora-v9`

Training summary:
- 135 samples, 100% general-purpose (contracts, policies, tech docs, reports, greetings)
- 0 document-specific hardcoded content
- loss: 1.097 → 0.377 over 3 epochs
- v5 (90 diverse samples) + v9 new (45 general samples)

Previous adapter `v8` was retired because ~78% of its training data was domain-specific (church PDFs), causing biased answer behavior on general documents.

Current answer-model interpretation:
- `qwen25-qlora-v9` remains the deployed baseline
- recent benchmark improvement came mainly from PDF-grounded pipeline work, not from changing the adapter
- the next adapter round should therefore stay PDF-first and failure-mined, teaching the model to use retrieved PDF context better rather than trying to compensate for weak retrieval or duplicate-source ambiguity
- recommendation: stay on the Qwen2.5 family for the next adapter round (`v10`) so the new training data is evaluated against a stable, already-integrated baseline; test Qwen3 only as a separate controlled branch after the pre-`v10` baseline and first PDF-grounded failure-mined round are complete

LLM config (`.env`):
- `LLM_BACKEND=local_hf`
- `LOCAL_LLM_MODEL_NAME=qwen25-qlora-v9`
- `LOCAL_LLM_ADAPTER_PATH=./finetuning/output/qwen25-qlora-v9`
- `LLM_TEMPERATURE=0.0`
- `VECTOR_TOP_K=10`
- `RERANKER_ENABLED=true` / `RERANKER_DEVICE=cuda` / `RERANKER_USE_FP16=true`

## Quick Start

```bash
cd /home/tilon/chatbot-karbi
cp .env.example .env
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the API:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
```

Note: `main.py` has no `uvicorn.run()` block — launch via uvicorn directly.

Optional — Ollama (only needed for VLM extraction fallback):

```bash
ollama serve
ollama pull qwen2.5vl:7b
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

See [ARCHITECTURE.md](/home/tilon/chatbot-karbi/ARCHITECTURE.md) for the full Mermaid flow diagrams.

### Library documents
1. Add files to `data/library/`
2. Startup ingest or the watcher detects them
3. Parser runs the 5-method per-page extraction pipeline
4. pdfplumber enhancement: replaces PyMuPDF output on 2-column pages (correct reading order), extracts tables as separate documents
5. Clause detection: documents with Korean legal articles (`제N조`) are split into one atomic chunk per article
6. Remaining content goes through hierarchical parent-child chunking (parent sections + child chunks ≤300 chars)
7. Contextual enrichment adds document/page/section headers
8. Child and clause chunks are embedded (BGE-M3) and stored in ChromaDB + BM25 index
9. Parent texts saved to `data/parent_store.json` for post-retrieval context expansion

### Chat uploads
1. Upload through `/ui`, `/upload`, `/upload-multiple`, or `/chat-with-file`
2. File is saved to `data/uploads/`
3. Parsed → pdfplumber enhanced → clause/hierarchical chunked → enriched → embedded → registered
4. The sidebar remembers uploaded files from the document registry
5. If exactly one upload is active, deictic prompts like `이 문서`, `이 파일`, and `this document` auto-scope to that file
6. Later chat can re-scope to one or many remembered uploads

## Parsing / Extraction Stack

The parser uses a 5-method layered pipeline with per-page quality-gated routing.

**File-level (tried first):**

| Method | Confidence | Notes |
|--------|-----------|-------|
| `marker_single` subprocess | 0.95 | Best for digital/text-heavy PDFs |

**Per-page fallback (when marker fails or quality is low):**

| Method | Confidence | Activates when |
|--------|-----------|----------------|
| PyMuPDF text extraction | 0.92 | Page is `digital` |
| PaddleOCR v5 Korean + PyKoSpacing | 0.82 | Page is `scanned` or `hybrid` |
| Tesseract kor+eng | 0.72 | Baseline OCR fallback |
| Qwen2.5-VL via Ollama | 0.78 | When OCR candidates are still insufficient |

`_select_best_page_candidate()` picks the highest-scoring candidate per page.

**Post-extraction enhancement (pdfplumber):**

After primary extraction, `pdfplumber` runs a column-aware enhancement pass:
- detects 2-column layouts and extracts left-then-right for correct reading order
- extracts tables as separate `chunk_type="table"` documents
- replaces PyMuPDF output when column-aware extraction is cleaner or yields more content

Key improvements:
- per-page routing instead of file-level routing
- page classification: `digital`, `hybrid`, `scanned`
- quality gates for low-yield and garbled text
- PaddleOCR v5 with PyKoSpacing post-processing for Korean OCR accuracy
- pdfplumber column-aware extraction for 2-column PDFs
- pdfplumber table extraction as separate document artifacts
- PaddleOCR lazy-loaded on first scanned page, does not add startup latency

Current OCR stack:
- preferred for scanned/hybrid Korean PDFs: `PaddleOCR v5 Korean + PyKoSpacing`
- baseline fallback: `Tesseract kor+eng`
- only when OCR is still insufficient: `Qwen2.5-VL` via Ollama


## Retrieval Status

Current retrieval pipeline:

| Step | Component | Notes |
|------|-----------|-------|
| 1 | ChromaDB vector search | BGE-M3 embeddings, CUDA |
| 2 | BM25 keyword search | Korean tokenization |
| 3 | RRF fusion | k=60 |
| 4 | BGE reranker-v2-m3 | CUDA FP16 |
| 5 | Parent + structural expansion | Matched children → full parent or structurally matching section |
| 6 | Confidence gating | Weak results filtered, with lexical-overlap escape hatch for strong scoped matches |

Chat-level improvements on top of retrieval:
- General knowledge queries without doc context trigger Tavily web search
- Comparison/causal queries get chain-of-thought prompt injection
- Context-relevance pre-check (reranker scores question vs top chunks; < 0.25 → not_found)
- Tiered post-generation NLI faithfulness check (< 0.15 → replace; 0.15–0.35 → disclaimer; ≥ 0.35 → pass)
- Document-backed answers, including unscoped corpus answers, now pass through the grounding gate
- Citation footer maps inline markers to source file + page and is capped to a compact set of evidence rows
- Explicit library document names can auto-scope an otherwise unscoped library query
- Dominant-document retrieval can promote unscoped library QA into an implicit single-document exact-answer path
- Supported two-document comparisons can bypass the generic LLM path and answer directly from the selected document pair

Recent retrieval evaluation:
- query routing eval `v3`: live routing reached `40/40`
- structure retrieval eval `v1`: `6/11`, with `source_recall = 1.0` but weak answer extraction
- structure retrieval eval `v2`: `6/15`, with `source_recall = 1.0` but weak answer extraction
- verifier grounding eval `v1`: improved to `6/8`
- verifier grounding eval `v2`: `10/10` on newly ingested real PDFs
- verifier grounding eval `unscoped_v1` (March 27, 2026 checkpoint): `22/24`, `avg_source_recall = 0.917`, `avg_answer_point_recall = 0.875`, `correct_not_found_rate = 1.0`
- verifier grounding eval `external_v1`: `2/10`, with Chinese drift as the dominant failure mode
- verifier grounding eval `tables_v1`: `5/8`
- the earlier grounding bottleneck on OCR/table/amendment-style PDFs has been substantially reduced by deterministic exact-answer, amendment-summary, unscoped auto-scope, dominant-document promotion, and deterministic comparison handling

Current grounding interpretation:
- source recall is strong on the harder real-PDF sets
- refusal / not-found behavior remains safe
- inline citations and citation footer are stable on the current answerable grounding suites
- upload/live testing already improved single-upload auto-scope, non-clause article fallback, and compact citation formatting
- structure-retrieval answer quality is now clearly weaker than its source retrieval
- the external-PDF slice is the loudest remaining gap and the best `v10` mining target
- the next progress should come from failure-mined PDF-grounded training data, especially Chinese-drift and table/value extraction cases, with the March 27 baseline used as the reference point

## Real-PDF Regression Coverage

The project is no longer being tested only on the original small library set.

Current state:
- `data/library/` is being used as the live regression corpus for many real policy/guideline PDFs
- `data/test pdfs/` is useful as a staging area, but files only affect the live corpus after being copied into `data/library/`
- this larger corpus is now part of Phase 10D and future QLoRA failure mining

Practical note:
- for controlled experiments, move only selected PDFs into `data/library/`
- for broader hardening, ingest a larger batch and mine failures from the resulting evals

Scoped behavior:
- specific question → top-k scoped retrieval
- whole-document task → full scoped document loading
- image text request → direct extraction response
- bundled upload ambiguity → clarification or sub-guideline narrowing

## Multi-File Summary Status

Multi-file summary is implemented and usable.

Current behavior:
- whole-upload-corpus requests can load all uploaded documents
- deterministic file-by-file summary mode is available
- each file is summarized once with its own heading
- when exactly one uploaded file is active, whole-document summary questions use a deterministic single-document summary path instead of a generic whole-upload-corpus formatter

Current limitation:
- OCR-heavy or noisy image-derived uploads can still produce rough summaries
- very long narrative PDFs can still pressure the local 4096-token generation window on some summary/extraction tasks
- this is now more of a long-document evidence selection issue than a basic upload-flow issue

## QLoRA Status

The QLoRA workflow is implemented, benchmarked, and deployed.

Available eval assets:
- strict eval sets:
  - [qlora_eval_ko_strict_v1.jsonl](/home/tilon/chatbot-karbi/finetuning/data/qlora_eval_ko_strict_v1.jsonl)
  - [qlora_eval_ko_strict_v2.jsonl](/home/tilon/chatbot-karbi/finetuning/data/qlora_eval_ko_strict_v2.jsonl)
- training sets: `qlora_train_v1` – `v9_combined` (see [finetuning/README.md](/home/tilon/chatbot-karbi/finetuning/README.md))
- scripts:
  - [train.py](/home/tilon/chatbot-karbi/finetuning/train.py)
  - [evaluate_adapter.py](/home/tilon/chatbot-karbi/finetuning/evaluate_adapter.py)
  - [infer_compare.py](/home/tilon/chatbot-karbi/finetuning/infer_compare.py)

The training/eval path uses Qwen chat templates correctly, supports offline local loading, and cleanly compares base vs adapter.

Current recommendation:
- keep improving RAG first: routing, retrieval, and grounding are in a healthier place now
- use routing and structure-retrieval eval misses as future high-signal QLoRA data
- use unseen-PDF and live-upload failures to decide what is still pipeline-side vs what truly belongs in the next QLoRA round
- avoid broad new training rounds until those real-world failure patterns are measured


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

### Phase A: Real-World Failure Mining (active)
- test the current fine-tuned RAG on new unseen PDFs and live upload usage
- keep routing / structure retrieval / verifier grounding regression suites green
- log failures by parser/OCR, retrieval, grounding, and answer-shaping category
- fix pipeline-side issues before adding new training data

### Phase B: Training Data Expansion (if needed)
- convert true model-side failures into QLoRA training data
- focus on comparison, refusal, and exact lookup answer patterns
- train v10 adapter candidate on expanded dataset
- run strict base-vs-adapter evaluation

### Phase C: Accuracy Refinement
- improve clarification behavior for ambiguous bundled uploads
- improve finer-grained comparison answers
- improve exact phrase fidelity for strict lookups
- improve noisy OCR/image summary quality
- investigate table artifact survival (most pdfplumber tables too small for standalone chunks)

### Phase D: Advanced Approaches

These were intentionally deferred and remain valid next-stage improvements:

- structure-aware retrieval:
  - richer block artifacts for forms and numbered rules
  - hierarchical retrieval for long bundled documents
- smarter routing:
  - query-type classifier for lookup vs summary vs comparison vs extraction
  - document-type-aware prompt routing
- stronger retrieval:
  - late-interaction retrieval such as ColBERT-style reranking
  - learned query expansion / hard-negative mining
- answer verification:
  - post-generation validation passes for risky answer categories
- stronger training methods after SFT:
  - judge-assisted dataset expansion
  - preference tuning such as DPO/ORPO for refusal and comparison quality
  - targeted synthetic augmentation only after real-document coverage is strong

These are not current blockers for a working product, but they are valid approaches for pushing accuracy and robustness higher.

## One-Line Summary

The project is a working document-RAG chatbot with clause-level chunking, column-aware extraction, compact cited grounding, smarter single-upload behavior, and a validated QLoRA fine-tuning pipeline; the next step is failure mining on unseen PDFs and live usage before the next training round.
