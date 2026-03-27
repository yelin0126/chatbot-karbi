# Tilon AI Chatbot Architecture

## Overview

Tilon AI Chatbot is a document-first RAG backend for Korean/English PDFs and images with an attached QLoRA fine-tuning workflow for the answer model.

The repository should be understood as two connected systems:

1. document RAG inference
2. answer-model fine-tuning and evaluation

Both are active and deployed. The system is general-purpose — it works with any uploaded document, with no hardcoded behavior for specific document types.

Current roadmap status:
- `Phase 10A` chat-flow refactor: complete
- `Phase 10B` routing/classifier layer: complete
- `Phase 10C` structure-aware retrieval: validated on targeted retrieval evals
- `Phase 10D` verifier / grounding upgrades: validated on the current grounding regression suites
- next major QLoRA round should build on failure-mined cases from the expanded real-PDF corpus and live upload usage

Current Phase 10D reading:
- retrieval/source selection is working well on the targeted real-PDF evals
- grounding/citation behavior is now also strong on the current hard `v1` / `v2` scoped suites
- the broader unscoped library suite has improved to a high-but-not-green checkpoint
- the remaining misses are now concentrated in exact-source locking over near-duplicate JNU guideline PDFs rather than broad comparison or refusal failures
- upload/live testing also exposed a separate long-document narrative-PDF quality layer
- the current opportunity is targeted duplicate-source disambiguation plus broader real-world failure mining, not another immediate grounding-architecture rewrite
- the March 27, 2026 pre-`v10` baseline now also shows that:
  - routing is stable
  - structure retrieval finds the right sources but answer extraction is weak
  - external-PDF grounding is the loudest remaining gap
  - Chinese drift is the dominant visible model-side failure pattern on unseen PDFs

Current architectural next step:
- keep the current multi-model PDF-first RAG architecture stable
- finish duplicate-source disambiguation on the remaining unscoped exact-lookup misses
- record one clean pre-`v10` baseline snapshot
- then mine failure-linked PDF-grounded training data for the next QLoRA round

Current model recommendation:
- keep the production fine-tuning path on the Qwen2.5 family for `v10`
- evaluate Qwen3 only as a later branch experiment after pipeline-side ambiguity is reduced and a clean post-`v10` comparison can be made

## Product Scopes

The system works with two document scopes:

- `data/library/`
  Persistent team documents that form the long-term knowledge base

- `data/uploads/`
  Chat-uploaded files that are ingested for the current chat flow and kept separate from the permanent library corpus

---

## Ingestion Flow

```mermaid
flowchart TD
    A([File Input\nPDF / Image]) --> B{File type?}
    B -->|PDF| C[parse_pdf\nparser.py]
    B -->|Image| D[parse_image\nparser.py]

    C --> E[marker_single\nsubprocess\nconf 0.95]
    E -->|quality OK| PA[Marker pages]
    E -->|fail / low quality| PG[Per-page routing]

    D --> PG

    PG --> PC{Page kind\nclassification}
    PC -->|digital| MU[PyMuPDF\ntext extraction\nconf 0.92]
    PC -->|scanned / hybrid| FB[Fallback chain]
    MU -->|quality gate fail| FB

    FB --> PO[PaddleOCR v5\nKorean + PyKoSpacing\nconf 0.82]
    FB --> TS[Tesseract\nkor+eng\nconf 0.72]
    FB --> VM[Qwen2.5-VL\nvia Ollama\nconf 0.78]

    PO & TS & VM & MU --> SEL[_select_best_page_candidate\nhighest composite score]
    PA --> SEL

    SEL --> DOCS([Document pages\nwith quality metadata])

    DOCS --> PB{pdfplumber\nenhancement?}
    PB -->|column-aware or tables| PLB[pdfplumber\ncolumn detection + tables]
    PB -->|no improvement| DOCS2([Page documents])
    PLB --> DOCS2
    DOCS --> DOCS2

    DOCS2 --> CLC{Clause detection\n제N조 headers?}
    CLC -->|yes| CLK[Clause-level chunks\none per 제N조 article]
    CLC -->|no| CH[Heading-aware\nhierarchical chunking]
    CH --> PAR[Parent chunks\nup to CHUNK_SIZE tokens]
    PAR --> KID[Child chunks\nup to 300 chars]
    PAR --> PS[(parent_store.json\nJSON parent store)]
    CLK --> KID2([Clause + child chunks])

    KID --> EN[enrich_chunks\nenricher.py\ncontextual headers]
    KID2 --> EN
    EN --> EMB[BGE-M3 embeddings\nCUDA]
    EMB --> VDB[(ChromaDB\nvector store)]
    EN --> KWI[(BM25 keyword index\nkeyword_index.py)]
    KWI & VDB --> REG[(document_registry.json\nstable doc_id)]
```

---

## Query / Chat Flow

```mermaid
flowchart TD
    Q([User query]) --> INT{Intent detection\nhandlers.py}
    INT -->|greeting / chitchat| GR[Direct reply]
    INT -->|doc query| RET[Hybrid retrieval]
    INT -->|no doc context\n+ general knowledge| WS[Web search\nTavily]

    RET --> VS[ChromaDB\nvector search top-K]
    RET --> KS[BM25 keyword search]
    VS & KS --> RRF[RRF fusion\nk=60]
    RRF --> RNK[BGE reranker-v2-m3\nCUDA FP16]
    RNK --> EXP[Parent expansion\nchild → parent text]
    EXP --> CTX([Retrieved context])

    CTX --> COT{CoT needed?\ncomparison / causal}
    COT -->|yes| CI[Chain-of-thought\ninjection to prompt]
    COT -->|no| SP[Standard prompt]
    WS --> SP
    CI & SP --> LLM[Qwen2.5-7B-Instruct\n+ QLoRA v9 / 4-bit NF4]
    GR --> LLM

    LLM --> ANS([Generated answer])
    ANS --> NLI{NLI faithfulness\nnli_verifier.py}
    NLI -->|score ≥ 0.15| CITE[Append citation\nfooter]
    NLI -->|score < 0.15| DISC[Append uncertainty\ndisclaimer]
    DISC --> CITE
    CITE --> OUT([Return answer])
```

## High-Level Flow

### Library documents
1. Files are added to `data/library/`
2. Startup ingest or watcher detects them
3. Parser extracts text from PDF/image pages (layered 5-method pipeline with per-page routing)
4. pdfplumber enhancement: column-aware extraction replaces PyMuPDF when 2-column layout is detected; tables extracted as separate documents
5. Clause detection: documents with Korean legal articles (`제N조`) are split into one chunk per article
6. Remaining content goes through hierarchical parent-child chunking
7. Contextual enrichment adds document/page/section headers
8. Child chunks are embedded (BGE-M3) and stored in ChromaDB + BM25 index
9. Parent texts saved to `data/parent_store.json`
10. Chat can retrieve them globally

### Chat uploads
1. User uploads via `/ui`, `/upload`, `/upload-multiple`, or `/chat-with-file`
2. File is saved to `data/uploads/`
3. Parsed → pdfplumber enhanced → clause/hierarchical chunked → enriched → embedded → registered
4. UI sidebar remembers uploaded files via `document_registry`
5. If exactly one uploaded file is active, deictic prompts like `이 문서` / `this document` auto-scope to that file
6. Chat can re-scope to one or many remembered uploads
7. Whole-document tasks load full upload scope rather than top-k chunks only

## Bigger-Picture Architecture

The full product path is:

1. document input
2. extraction / parsing (5-method layered pipeline, per-page)
3. pdfplumber column-aware enhancement + table extraction
4. clause-level splitting for Korean legal articles (`제N조`), or hierarchical parent-child chunking for other content
5. contextual enrichment
6. BGE-M3 embeddings (child/clause chunks)
7. ChromaDB vector store + BM25 keyword index
8. hybrid retrieval (vector + keyword + RRF fusion)
9. BGE reranker-v2-m3
10. parent expansion (child match → parent context)
11. context-relevance gate (reranker scores question vs top chunks; < 0.25 → not_found)
12. prompt construction (with optional CoT injection)
13. answer generation (Qwen2.5-7B + QLoRA v9)
14. tiered NLI faithfulness gate (< 0.15 → replace; 0.15–0.35 → disclaimer; ≥ 0.35 → pass)
15. compact citation footer (maps [1], [2] markers → source file + page, capped to the evidence actually used)

Additional current reliability behavior:
- document-backed grounding now runs for unscoped corpus answers as well as explicitly scoped answers
- single-upload narrative summary questions can bypass giant full-context prompts via a deterministic single-document summary builder
- explicit document names inside library questions can auto-scope to that library PDF
- retrieval dominated by one library document family can promote an unscoped query into an implicit single-document exact-answer path
- supported two-document comparisons can use a deterministic comparison builder instead of the generic LLM path
- `제N조` lookups on non-clause documents return a clean format-mismatch explanation instead of hallucinated article text
- broad year/number questions can use deterministic numeric extraction before free-form generation
- strict-fact table/value queries can promote to one exact source/doc and then answer from the full exact document rather than mixed-family chunks when the scorer is confident enough

Key distinction:
- RAG retrieves the evidence
- QLoRA improves how the model uses that evidence
- NLI verifies the answer matches the evidence

QLoRA is not a substitute for bad extraction, weak chunking, or poor retrieval.

Current architectural interpretation:
- most recent benchmark gains came from retrieval, scoping, deterministic exact-answer, and not-found/comparison fixes
- the current deployed adapter did not change during that improvement
- therefore the system is still more pipeline-limited than model-limited on the active PDF-grounding benchmark

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
- [app/core/paddle_ocr.py](/home/tilon/chatbot-karbi/app/core/paddle_ocr.py)
- [app/core/parent_store.py](/home/tilon/chatbot-karbi/app/core/parent_store.py)

Responsibilities:
- PDF/image extraction (5-method per-page pipeline)
- PaddleOCR v5 Korean OCR with PyKoSpacing post-processing as the preferred scanned/hybrid OCR path
- pdfplumber column-aware extraction and table detection
- clause-level splitting for Korean legal articles (`제N조`)
- hierarchical parent-child chunking
- contextual enrichment
- parent text storage
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
- [app/chat/policy.py](/home/tilon/chatbot-karbi/app/chat/policy.py)
- [app/chat/scope.py](/home/tilon/chatbot-karbi/app/chat/scope.py)
- [app/chat/retrieval_flow.py](/home/tilon/chatbot-karbi/app/chat/retrieval_flow.py)
- [app/chat/query_classifier.py](/home/tilon/chatbot-karbi/app/chat/query_classifier.py)
- [app/chat/query_classifier_data.py](/home/tilon/chatbot-karbi/app/chat/query_classifier_data.py)
- [app/core/nli_verifier.py](/home/tilon/chatbot-karbi/app/core/nli_verifier.py)

Responsibilities:
- stage-based chat orchestration
- centralized query routing policy
- optional embedding-backed routing classifier
- prompt building with optional chain-of-thought injection
- answer-type handling (lookup, summary, comparison, clarify)
- deterministic exact-answer shortcuts for clause/table/value-heavy document questions
- deterministic two-document comparison answers for supported comparison dimensions
- single-upload auto-scope resolution
- library document-name auto-scope and dominant-document implicit scope promotion
- deterministic single-document summary for narrative uploads
- non-clause article-query fallback
- deterministic numeric fact extraction for year/number-style prompts
- low-confidence and not-found fallback behavior
- broadened web search fallback for general knowledge queries
- upload-scope behavior
- deterministic file-by-file multi-document summary
- structure-aware retrieval expansion
- NLI faithfulness verification post-generation

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

The parser uses a 5-method layered pipeline with per-page quality-gated routing, followed by an optional pdfplumber enhancement pass.

### File-level (tried first)

| Method | Confidence | Used when |
|--------|-----------|----------|
| `marker_single` subprocess | 0.95 | All PDFs — best for digital/text-heavy |

If marker fails or produces low-quality output, the parser falls back to per-page routing.

### Per-page routing

Each page is classified as `digital`, `hybrid`, or `scanned` based on text yield and pixel analysis.

| Method | Confidence | Activates when |
|--------|-----------|----------------|
| PyMuPDF text extraction | 0.92 | Page is `digital` |
| PaddleOCR v5 Korean + PyKoSpacing | 0.82 | Page is `scanned` or `hybrid` (PADDLE_OCR_ENABLED=true) |
| Qwen2.5-VL via Ollama | 0.78 | OCR still insufficient (VLM_EXTRACTION_ENABLED=true) |
| Tesseract kor+eng | 0.72 | Always tried as baseline OCR fallback |

All candidates are scored by `_select_best_page_candidate()` and the highest-scoring one wins.

### pdfplumber enhancement pass

After the primary extraction, `_apply_pdfplumber_enhancements()` runs on all PDF pages:

1. **Column-aware extraction:** detects 2-column layouts by analyzing word x-coordinate distribution around the page midpoint. When detected, extracts left column then right column for correct reading order.
2. **Table extraction:** pdfplumber tables are extracted as separate `chunk_type="table"` documents with pipe-delimited row format.
3. **Page replacement:** a PyMuPDF page is replaced with the pdfplumber version when: column-aware extraction produced a cleaner result, the original is garbled, or pdfplumber yields ≥25% more characters.

This solves the common problem of interleaved text from 2-column Korean PDFs (e.g., `제1조` and `제9조` mixed together in a single text stream).

PaddleOCR notes:
- Uses PPOCRv5 Korean mobile model (CPU mode during ingest)
- PyKoSpacing corrects Korean word-spacing errors post-OCR
- Set `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True` in env to avoid connectivity check
- Lazy-loaded on first scanned/hybrid page — does not affect startup if no scanned PDFs

Parser improvements delivered:
- per-page routing instead of file-level routing
- page type classification: `digital`, `hybrid`, `scanned`
- quality gates for low-yield and garbled text
- PaddleOCR v5 with PyKoSpacing (replaces Tesseract as preferred Korean OCR)
- pdfplumber column-aware extraction for 2-column PDFs
- pdfplumber table extraction as separate document artifacts
- richer metadata: `extraction_method`, `page_kind`, `routing_reason`, `extractors_used`, `confidence`

## Retrieval Strategy

### Hybrid retrieval pipeline

| Step | Component | Notes |
|------|-----------|-------|
| 1 | ChromaDB vector search | BGE-M3 embeddings, top-K by cosine |
| 2 | BM25 keyword search | `keyword_index.py`, Korean tokenization |
| 3 | RRF fusion | k=60, merges both ranked lists |
| 4 | BGE reranker-v2-m3 | CUDA FP16, cross-encoder rescoring |
| 5 | Parent + structural expansion | Matched children → parent chunk text or query-matched structural section |
| 6 | Confidence gating | Threshold filters weak results, except strong lexical-overlap scoped matches |

### Chunking strategy

Documents are chunked using a priority system:

1. **Clause-level splitting** (highest priority): documents containing Korean legal article headers (`제N조(title)`) are split into one atomic chunk per article. Each clause becomes a `chunk_type="clause"` chunk. This keeps regulatory content intact and prevents cross-article fragmentation.
2. **Hierarchical parent-child chunking** (default): remaining content is split into parent sections (≤ CHUNK_SIZE tokens, heading-aware), then each parent is further split into child chunks (≤300 chars) for high-precision vector search.
3. **Table chunks**: pdfplumber-extracted tables become separate `chunk_type="table"` documents.

### Parent expansion
Documents are indexed as small child chunks (≤300 chars) or atomic clause chunks for high-precision retrieval.
After reranking, each matched child is replaced with its parent text (the full section it came from).
This gives the LLM broader context without polluting the vector index with long chunks.

### Structure-aware expansion
When the query explicitly or implicitly names a structural target such as `제N조`, `제N장`, `article N`, `section N`, or a distinctive section-heading phrase, retrieval expands around the matched section instead of leaving the model with an isolated fragment. This is especially important for:

- article lookup
- article-to-article comparison
- chapter/section lookup
- scoped structure-summary questions

### Scoped document behavior
- specific question → top-k scoped retrieval
- summary/analysis → full scoped document loading
- OCR request → direct extraction result
- bundled upload ambiguity → clarification or named-scope narrowing
- multi-file summary → deterministic file-level grouping
- exactly one uploaded document → auto-scope `이 문서` / `this document` style questions to that file

### Chat-level reliability
- Context-relevance pre-check (reranker scores question vs top chunks; < 0.25 → not_found before LLM)
- General knowledge queries with no doc context → Tavily web search fallback
- Comparison/causal/procedural queries → chain-of-thought prompt injection
- Post-generation → tiered NLI faithfulness gate (< 0.15 → replace; 0.15–0.35 → disclaimer; ≥ 0.35 → pass)
- Document-backed grounding now covers both scoped and unscoped document answers
- Citation footer maps inline `[1]`, `[2]` markers to source file + page in a compact capped footer
- inline citation fallback and citation-aware faithfulness scoring are active

## Evaluation Assets

Benchmark and evaluation assets now include:

- library benchmark:
  - [benchmark_tilon_v1.jsonl](/home/tilon/chatbot-karbi/finetuning/data/benchmark_tilon_v1.jsonl)
- upload-scoped benchmark:
  - [benchmark_upload_scoped_v1.jsonl](/home/tilon/chatbot-karbi/finetuning/data/benchmark_upload_scoped_v1.jsonl)
- comparison benchmark:
  - [benchmark_compare_v1.jsonl](/home/tilon/chatbot-karbi/finetuning/data/benchmark_compare_v1.jsonl)
- routing eval sets:
  - [query_policy_eval_v1.jsonl](/home/tilon/chatbot-karbi/finetuning/data/query_policy_eval_v1.jsonl)
  - [query_policy_eval_v2.jsonl](/home/tilon/chatbot-karbi/finetuning/data/query_policy_eval_v2.jsonl)
  - [query_policy_eval_v3.jsonl](/home/tilon/chatbot-karbi/finetuning/data/query_policy_eval_v3.jsonl)
- structure retrieval eval sets:
  - [structure_retrieval_eval_v1.jsonl](/home/tilon/chatbot-karbi/finetuning/data/structure_retrieval_eval_v1.jsonl)
  - [structure_retrieval_eval_v2.jsonl](/home/tilon/chatbot-karbi/finetuning/data/structure_retrieval_eval_v2.jsonl)
- verifier grounding eval sets:
  - [verifier_grounding_eval_v1.jsonl](/home/tilon/chatbot-karbi/finetuning/data/verifier_grounding_eval_v1.jsonl)
  - [verifier_grounding_eval_v2.jsonl](/home/tilon/chatbot-karbi/finetuning/data/verifier_grounding_eval_v2.jsonl)
- strict QLoRA eval sets:
  - [qlora_eval_ko_strict_v1.jsonl](/home/tilon/chatbot-karbi/finetuning/data/qlora_eval_ko_strict_v1.jsonl)
  - [qlora_eval_ko_strict_v2.jsonl](/home/tilon/chatbot-karbi/finetuning/data/qlora_eval_ko_strict_v2.jsonl)
- benchmark runner:
  - [run_benchmark.py](/home/tilon/chatbot-karbi/scripts/run_benchmark.py) — supports `--http` mode (calls live server) and in-process mode
  - [eval_query_routing.py](/home/tilon/chatbot-karbi/scripts/eval_query_routing.py)
  - [eval_structure_retrieval.py](/home/tilon/chatbot-karbi/scripts/eval_structure_retrieval.py)
  - [eval_verifier_grounding.py](/home/tilon/chatbot-karbi/scripts/eval_verifier_grounding.py)

## Current Maturity By Layer

### Strong
- upload vs library scope separation
- stage-based chat flow
- parser routing and fallback extraction
- clause-level and hierarchical chunking
- column-aware PDF extraction (pdfplumber)
- semantic chunking
- contextual enrichment
- hybrid retrieval (vector + keyword + RRF + BGE reranker)
- query routing policy and conservative classifier override
- structure-aware retrieval for scoped article/section queries
- document-scoped chat
- deterministic multi-file summary
- QLoRA training/eval workflow
- general-purpose answer model (v9, no domain hardcoding)

### Working but still being tuned
- verifier / grounding behavior for the hardest exact-lookup and comparison generations
- OCR/table-heavy exact amount extraction and category extraction on noisy policy PDFs
- OCR-heavy summary cleanliness
- long narrative PDF summary/extraction on unseen uploads
- full-answer evaluation under local GPU limits
- generation quality on unseen external PDFs, especially Chinese drift and raw table dumps despite correct retrieval
- broader retrieval regression coverage beyond current v1/v2 structure evals
- broader answer-grounding regression coverage over the newly expanded real-PDF corpus

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
- broader regression coverage for lookup, summary, comparison, OCR, and clarification
- document-type-aware context formatting
- distinct handling for screenshot/image-heavy uploads

### Generation reliability
- answer-type-specific prompt controllers
- stronger post-generation validation for risky answer classes
- verifier-aware evaluation sets

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

## LLM Stack

The answer model runs fully locally:

- base model: `Qwen/Qwen2.5-7B-Instruct`
- adapter: `finetuning/output/qwen25-qlora-v9` (QLoRA, LoRA r=16, alpha=32)
- quantization: 4-bit NF4 via BitsAndBytes
- device: CUDA (RTX 4070 12GB)
- decoding: greedy (temperature=0.0), rep_penalty=1.05, ngram=3
- token suppression: CJK + Cyrillic + Thai (32,257 token IDs banned)
- max input: 4096 tokens, max output: 1024 tokens

No external LLM API is required. Ollama is optional (only used for VLM extraction fallback via `qwen2.5vl:7b`).

## Current Conclusion

The architecture is no longer missing its foundation. The system is general-purpose and deployed. Routing and structure-aware retrieval are benchmarked and green, Phase 10D grounding is green on the current hard suites, and recent upload-path fixes improved single-document live usage. The next major accuracy bottleneck is broader unseen-PDF failure mining, especially on long narrative uploads, followed by targeted failure-driven QLoRA iteration.
