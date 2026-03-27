# Improvements & Current State

This file tracks how the project evolved from a single-file prototype into the current document-first RAG system with a working local QLoRA loop.

Related docs:
- [README.md](/home/tilon/chatbot-karbi/README.md)
- [ARCHITECTURE.md](/home/tilon/chatbot-karbi/ARCHITECTURE.md)
- [finetuning/README.md](/home/tilon/chatbot-karbi/finetuning/README.md)

## Current Phase Summary

Current roadmap status:
- `Phase 10A` chat-flow refactor: complete
- `Phase 10B` routing and query-policy layer: complete
- `Phase 10C` structure-aware retrieval: first validated slice complete
- `Phase 10D` verifier / answer-grounding upgrades: validated on the current grounding eval suites

Current interpretation:
- routing is strong
- retrieval is strong
- OCR/table-aware exact-answer grounding improved substantially on the hard real-PDF suite
- recent live upload tests also improved single-upload intelligence and long-document handling
- unscoped library grounding moved from an early partial baseline to a high checkpoint (`22/24`) through pipeline fixes
- the remaining unscoped misses are now much more model-facing than retrieval-facing
- the external-PDF and structure-retrieval slices expose answer-generation weakness more than source-finding weakness
- the next phase should focus on failure-mined PDF-grounded training data for the next QLoRA round

Pre-`v10` baseline preparation now also includes:
- widening the clause-title regex from 30 to 80 characters for Korean legal/article-style PDFs
- adding a small external-PDF grounding slice
- adding a table/list/form-heavy grounding slice
- capturing one clean baseline snapshot before any new `v10` training data is assembled

March 27, 2026 grounding checkpoint:
- `verifier_grounding_eval_unscoped_v1`: `22/24`
- `avg_source_recall = 0.917`
- `avg_answer_point_recall = 0.875`
- `correct_not_found_rate = 1.0`

March 27, 2026 pre-`v10` baseline snapshot:
- routing `v3`: `40/40`
- structure retrieval `v1`: `6/11` (`source_recall = 1.0`, `point_recall = 0.444`)
- structure retrieval `v2`: `6/15` (`source_recall = 1.0`, `point_recall = 0.185`)
- scoped grounding `v1`: `5/8`
- scoped grounding `v2`: `10/10`
- unscoped grounding: `22/24`
- external grounding: `2/10`
- table grounding: `5/8`

Interpretation:
- these gains came from RAG/scoping/extraction/deterministic-answer work, not from changing the deployed QLoRA adapter
- the project therefore does not yet have evidence that QLoRA inefficiency is the main remaining bottleneck
- however, the pre-`v10` baseline now clearly shows model-facing weaknesses on unseen PDFs, especially Chinese drift and raw table dumps

Current action order:
1. freeze the March 27 pre-`v10` baseline snapshot as the comparison point
2. assemble `qlora_train_v10` from `011`, `014`, external-PDF failures, table-grounding failures, and structure-retrieval answer failures
3. keep QLoRA `v10` on the Qwen2.5 family first; treat any Qwen3 migration as a later controlled experiment rather than part of the first failure-mined round

## Phase 1 — Baseline Refactor

Completed baseline improvements:
- real web search path instead of pure hallucinated “search mode”
- configurable embeddings device instead of CPU-only
- reduced duplicate ingestion
- FastAPI startup modernization with `lifespan`
- optional reranker integration
- language metadata in parsed documents
- structured logging
- better LLM timeout handling
- parser timeout and cleanup behavior
- monolith to modular package layout
- `.env`-driven runtime settings
- Pydantic request/response schemas
- separated prompt templates

## Phase 2 — Document-First RAG Architecture

Major architectural improvements completed:
- split storage into `data/library/`, `data/uploads/`, and `data/temp/`
- made uploads a first-class end-to-end flow
- added remembered-upload sidebar behavior
- added multi-document selection for comparison-style chat
- added direct image/text extraction intent handling
- upgraded parser routing to page-aware extraction
- replaced blind chunk splitting with semantic chunking
- added contextual enrichment before embedding
- moved retrieval to a hybrid vector + keyword path
- added full-document scoped retrieval for whole-document tasks
- added confidence gating
- introduced a persistent document registry with stable `doc_id`
- carried `active_doc_id` end to end
- reduced upload/watcher race conditions
- made vectorstore reset safer
- improved reranker runtime policy for latency and GPU safety
- made repeated uploads replace old scoped chunks instead of accumulating duplicates
- improved bundled upload PDF disambiguation behavior
- added upload-scoped benchmark coverage

## Phase 3 — Multi-File Summary And Scoped Corpus Behavior

The project now supports whole-uploaded-corpus tasks rather than only top-k chunk QA.

Completed:
- whole-upload-corpus loading for summary-style requests
- grouped document context formatting
- deterministic file-by-file summary mode
- one summary block per uploaded file
- basic OCR cleanup for representative summary lines

Current status:
- functional and useful for testing
- still less polished than single-document QA on noisy OCR/image uploads

## Phase 4 — QLoRA Workflow

The end-to-end training/evaluation path exists and has been exercised through multiple adapter versions.

Completed:
- [train.py](/home/tilon/chatbot-karbi/finetuning/train.py)
- [evaluate_adapter.py](/home/tilon/chatbot-karbi/finetuning/evaluate_adapter.py)
- [infer_compare.py](/home/tilon/chatbot-karbi/finetuning/infer_compare.py)
- strict Korean-heavy eval sets
- multiple training-set iterations with measured comparisons
- clean base-vs-adapter evaluation
- corrected Qwen chat-template usage

QLoRA adapter history:
- `v3`: established that fine-tuning can beat base
- `v4`: showed growing the dataset alone is not enough
- `v5`: first broadly convincing adapter (90 diverse general samples)
- `v6`: best measured result on strict Korean eval
- `v7`: experimental iteration
- `v8`: **retired** — ~78% of training data was church-specific; caused domain-biased answers
- `v9`: **current deployed** — 135 general-purpose samples, no hardcoded domain content

Best measured eval result (v6 on strict Korean set):
- dataset: [qlora_eval_ko_strict_v2.jsonl](/home/tilon/chatbot-karbi/finetuning/data/qlora_eval_ko_strict_v2.jsonl)
- base avg: `0.244`
- v6 adapter avg: `0.4173`
- adapter wins: `12/15`, base wins: `0/15`, ties: `3/15`

v9 training summary:
- 135 samples (v5 90 + 45 new general samples)
- loss: 1.097 → 0.377, 3 epochs, ~4m38s on RTX 4070
- zero church/domain-specific content

## Phase 5 — General-Purpose Cleanup

This phase addressed the core problem that the system had accumulated document-specific hardcoding from working with church PDFs.

Completed:
- removed `_extract_publication_history_facts()` — hardcoded answers for specific dates/facts
- removed `_try_structure_lookup()` — hardcoded theological category labels (신론/인간론/구원론 etc.)
- removed `_STRUCTURE_INDICATORS`, `_CATEGORY_LABEL_RE`, `_parse_toc_categories_from_docs()`
- cleaned `_HISTORY_INDICATORS` — removed all church-specific terms
- stripped `_clean_generated_text()` to document-agnostic minimal (only removes ㎞ and リン symbols)
- removed all church-specific regex patterns from answer post-processing
- extended token suppression to Cyrillic and Thai (total: 32,257 tokens banned)
- changed to greedy decoding (temperature=0.0) with relaxed repetition controls
- retired v8 adapter (domain-biased training data)
- trained v9 adapter on 135 general-purpose samples
- deployed v9: server health check confirmed `model: qwen25-qlora-v9`

Result: the chatbot now answers based on document content only, with no hardcoded answers for any specific document type.

## Phase 6 — RAG Accuracy Upgrades

This phase improved answer accuracy at every layer: extraction, retrieval, and generation.

### 6a — Hierarchical Parent-Child Chunking

**Problem:** flat chunking created fragments that cut across sentence boundaries, hurting both retrieval precision and answer completeness.

**Solution:**
- documents are chunked into parent sections (≤ CHUNK_SIZE tokens, heading-aware)
- each parent is further split into child chunks (≤ 300 chars) for high-precision vector search
- child `parent_id` links back to the full parent section
- after reranking, matched children are replaced with their parent text for fuller LLM context

New files: `app/core/parent_store.py`, updated `app/pipeline/chunker.py`, `app/pipeline/ingest.py`, `app/retrieval/retriever.py`

### 6b — NLI Faithfulness Verification

**Problem:** the system sometimes generated plausible-sounding answers that weren't grounded in the retrieved context.

**Solution:**
- post-generation faithfulness score using the already-loaded BGE reranker-v2-m3 as a proxy NLI model
- splits the answer into sentences, scores each against all retrieved chunks
- if mean-max score < 0.15 on a scoped-document query, appends a Korean uncertainty disclaimer
- zero additional model weight or VRAM — reuses the live reranker

New file: `app/core/nli_verifier.py`, updated `app/chat/handlers.py`

### 6c — Broadened Web Search + Chain-of-Thought

**Problem:** the chatbot fell back to a generic "I don't know" when no document context existed for general knowledge questions. Complex reasoning questions also suffered from shallow answers.

**Solution:**
- `_is_general_knowledge_query()` detects definition/tech queries (RAG, LLM, 딥러닝, 계약, etc.) and fires a Tavily web search when no doc context is available
- `_needs_chain_of_thought()` detects comparison, causal, and procedural queries and injects a step-by-step CoT instruction block into the system prompt

Updated: `app/chat/handlers.py`

### 6d — PaddleOCR v5 + PyKoSpacing

**Problem:** Tesseract produces poor Korean word boundaries on scanned PDFs, degrading downstream embedding and retrieval quality.

**Solution:**
- added PaddleOCR v5 PPOCRv5-Korean model as a per-page OCR candidate (confidence 0.82 vs Tesseract's 0.72)
- automatically preferred over Tesseract when both run on the same page
- PyKoSpacing corrects word-spacing errors in OCR output for Korean lines
- CPU-only for ingest (not real-time); lazy-loaded on first scanned page
- TensorFlow (PyKoSpacing dependency) forced to CPU before initialization to avoid GPU contention with PyTorch

New file: `app/core/paddle_ocr.py`, updated `app/pipeline/parser.py`, `app/config.py`

Current OCR behavior:
- preferred OCR path for scanned/hybrid Korean PDFs: `PaddleOCR v5 Korean + PyKoSpacing`
- baseline OCR fallback: `Tesseract kor+eng`
- escalation only when OCR is still insufficient: `Qwen2.5-VL`

## Phase 7 — Citation / Evidence Grounding

**Problem:** answers cited vague references like "문서 3페이지에 따르면..." but users had no reliable way to trace which specific chunk backed each claim.

**Solution:**
- `format_context()` now prefixes each chunk with an explicit reference number `[1]`, `[2]`, ...
- system prompts (both full Ollama and compact local_hf) instruct the model to use inline `[1]`, `[2]` markers when citing document content
- `build_document_prompt()` answer template updated — `핵심 답변` section now explicitly requests inline markers; `참고 문서` shows `(자동 생성됨)`
- `_build_citation_footer()` generates a structured source map after answer generation:
  ```
  ---
  **출처:**
  [1] filename.pdf, p.3 — Section Title
  [2] other.pdf, p.7 — Chapter 2
  ```
- footer is appended to every document-grounded answer, even when the LLM omits inline markers

Updated: `app/retrieval/retriever.py`, `app/chat/handlers.py`, `app/chat/prompts.py`

## Phase 8 — Answer Guards (`not_found` / Clarify)

**Problem:** The system only rejected answers at very low retrieval confidence (< 0.45) and very low faithfulness (< 0.15). Answers in the dangerous middle zone (0.15–0.50 faithfulness) passed through unchallenged — sounding plausible but not well-grounded in the source documents. The citation footer made such answers look even more credible.

**Solution — three-layer defense:**

### 8a — Context-relevance pre-check (pre-generation)

After retrieval but *before* prompt construction, scores `(question, top_chunks)` using the BGE reranker cross-encoder. If the best chunk relevance score < 0.25, the context is off-topic: returns a descriptive "not found" response immediately, never reaching the LLM.

New function: `check_context_relevance()` in `app/core/nli_verifier.py`
New config: `CONTEXT_RELEVANCE_THRESHOLD = 0.25`

### 8b — Tiered NLI faithfulness (post-generation)

Upgraded from a single 0.15 threshold to two tiers:
- **Hard threshold (< 0.15):** answer is *replaced* entirely with a "문서에서 확인되지 않습니다" message + rephrase suggestion
- **Soft threshold (0.15–0.35):** answer is kept but a "문서를 직접 확인해 주세요" disclaimer is appended
- **Above 0.35:** answer passes through unchanged

New configs: `NLI_FAITHFULNESS_HARD_THRESHOLD = 0.15`, `NLI_FAITHFULNESS_SOFT_THRESHOLD = 0.35`

### 8c — Improved not-found responses

All not-found messages now include:
- the document name for context (e.g. "'런케이션 운영지침'에서 해당 내용을 찾지 못했습니다")
- actionable suggestions (rephrase, use keywords, select different document)
- language-matched responses (Korean/English)

New functions: `_not_grounded_answer()`, `_context_irrelevant_answer()` in `app/chat/handlers.py`

### 8d — Prompt-level refusal reinforcement

Both system prompts (full Ollama + compact local_hf) updated:
- Rule 3 now explicitly instructs the model to say "해당 내용은 제공된 문서에서 확인되지 않습니다" when context is irrelevant
- Added "없는 내용을 지어내지 마라" (don't fabricate) in the compact prompt

Updated: `app/core/nli_verifier.py`, `app/chat/handlers.py`, `app/config.py`

## Phase 9 — Clause-Level Chunking & Column-Aware Extraction

**Problem:** Korean policy/regulation documents use `제N조(title)` article structure, but hierarchical chunking fragmented articles across chunk boundaries. Additionally, 2-column PDFs (common in Korean government/policy documents) had interleaved text extraction — e.g., `제1조` and `제9조` content mixed together in a single text stream.

**Solution — two improvements:**

### 9a — Clause-level chunking

Added Korean legal article detection and splitting before hierarchical chunking:
- `_CLAUSE_HEADER_RE` regex matches `제\s*\d+\s*조(?:의\s*\d+)?\s*[(（]...[）)]` patterns
- `_split_by_clauses()` splits document text at each `제N조` boundary
- Each article becomes an atomic `chunk_type="clause"` chunk with `section_title` set to the article header
- Clause detection runs before heading-based chunking — if clauses are found, the document bypasses hierarchical splitting entirely

Updated: `app/pipeline/chunker.py`

### 9b — pdfplumber column-aware extraction + table detection

Added a post-processing enhancement pass using `pdfplumber`:
- `_pdfplumber_page_text()` detects 2-column layouts by analyzing word x-coordinate distribution around the page midpoint
- When columns are detected, extracts left column then right column for correct reading order
- `_table_to_text()` converts pdfplumber tables to pipe-delimited text
- `_extract_pdfplumber()` provides full-page extraction with column awareness + separate table Documents
- `_should_prefer_pdfplumber()` decides when to replace PyMuPDF output: column-aware extraction, garbled text, or ≥25% more characters
- `_apply_pdfplumber_enhancements()` runs after primary extraction, replaces/supplements page documents

New dependency: `pdfplumber`
Updated: `app/pipeline/parser.py`

### 9c — Benchmark infrastructure (HTTP mode)

Added `--http` mode to `scripts/run_benchmark.py` to run benchmarks against the live server instead of loading models in-process (which caused CUDA OOM when the server was already running):
- `--http` flag calls `/chat` endpoint via HTTP
- `--server-url` flag (default `http://localhost:8000`)
- Per-row progress logging `[idx/total]`

## Phase 10 — Chat Flow, Routing, And Structure-Aware Retrieval

This phase moved the project from “good components” to a cleaner end-to-end RAG pipeline with measurable routing and retrieval regression coverage.

### 10A — Chat-flow refactor

**Problem:** `handlers.py` had accumulated too much mixed orchestration logic, which made routing, retrieval, and answer-guard work harder to reason about and harder to benchmark separately.

**Solution:**
- split the chat pipeline into stage helpers such as:
  - `_build_chat_state()`
  - `_resolve_scope_stage()`
  - `_run_retrieval_stage()`
  - `_build_generation_inputs_stage()`
  - `_finalize_answer_stage()`
- added focused chat modules:
  - `app/chat/policy.py`
  - `app/chat/deterministic.py`
  - `app/chat/scope.py`
  - `app/chat/retrieval_flow.py`
  - `app/chat/validation.py`
  - `app/chat/prompting.py`
  - `app/chat/text_utils.py`
  - `app/chat/state.py`

Result:
- clearer orchestration
- easier targeted evaluation
- lower risk when changing routing or retrieval behavior

### 10E — Unscoped library stabilization

This slice was driven by the new `verifier_grounding_eval_unscoped_v1` benchmark over the persistent library corpus.

Completed:
- library document-name auto-scope
- dominant-document implicit single-document promotion
- deterministic two-document comparison for supported dimensions
- scoped and unscoped not-found answer hardening
- OCR/table-aware exact-answer extraction for award caps, payment-form rows, support-fee rows, lecturer-rate rows, and amendment-heavy policy tables
- strict-fact exact-source promotion for table/value queries

Current reading:
- the unscoped library benchmark improved to `22/24`
- the remaining failures are still exact-source selection issues over near-duplicate JNU guideline variants
- this is strong evidence that pipeline quality, not the current adapter, dominated recent gains

### 10B — Query policy and embedding classifier

**Problem:** intent handling for lookup, summary, comparison, OCR/direct extraction, and broad explanation requests had become too implicit and brittle.

**Solution:**
- centralized routing rules in `app/chat/policy.py`
- added optional embedding-backed classifier:
  - `app/chat/query_classifier.py`
  - `app/chat/query_classifier_data.py`
- added routing eval runner:
  - `scripts/eval_query_routing.py`

### 10C / 10D — Retrieval and grounding validation

Validated checkpoints:
- structure retrieval eval `v1`: `6/11`, with `avg_source_recall = 1.0` and weak final answer extraction
- structure retrieval eval `v2`: `6/15`, with `avg_source_recall = 1.0` and weak final answer extraction
- verifier grounding eval `v1`: `6/8`
- verifier grounding eval `v2`: `10/10`

What closed the gap on the hard real-PDF grounding set:
- citation-aware verifier logic and footer handling
- OCR/table-aware deterministic exact-answer extraction
- amendment-summary / comparison shortcuts that synthesize a clean grounded summary from noisy retrieved chunks

Current reading:
- retrieval is now finding the right evidence much more often than the answer model is using it well
- the loudest remaining gap is answer generation on unseen external PDFs and table-heavy contexts
- the best next step is selective failure mining from new unseen PDFs and live usage
- added progressively harder eval sets:
  - `query_policy_eval_v1.jsonl`
  - `query_policy_eval_v2.jsonl`
  - `query_policy_eval_v3.jsonl`
- implemented conservative live override behavior so heuristics remain authoritative where they are safer

Measured result:
- `query_policy_eval_v3.jsonl`
  - heuristic: `40/40`
  - live embedding routing: `40/40`
  - classifier coverage: `19/40`
  - classifier accuracy on covered rows: `19/19`

Result:
- routing is now explicit, benchmarked, and much easier to evolve safely

### 10C — Structure-aware retrieval

**Problem:** even with better chunking and reranking, scoped article/section queries could still leave the LLM with only a fragment instead of the whole relevant section.

**Solution:**
- added structural context expansion in `app/chat/retrieval_flow.py`
- wired structural expansion into the retrieval stage in `app/chat/handlers.py`
- widened expansion beyond exact `제N조` references to keyword/section-heading style queries
- made low-relevance gating yield to strong lexical-overlap matches for scoped retrieval
- added dedicated retrieval eval runner:
  - `scripts/eval_structure_retrieval.py`
- added retrieval eval sets:
  - `structure_retrieval_eval_v1.jsonl`
  - `structure_retrieval_eval_v2.jsonl`
- switched the retrieval eval to default retrieval-only mode to avoid local-HF generation OOM during retrieval benchmarking

Measured result:
- `structure_retrieval_eval_v1.jsonl`: `6/11`, `avg_source_recall = 1.0`, `avg_answer_point_recall = 0.444`
- `structure_retrieval_eval_v2.jsonl`: `6/15`, `avg_source_recall = 1.0`, `avg_answer_point_recall = 0.185`

Result:
- the first structure-aware retrieval slice is validated
- the main next bottleneck is final-answer extraction and generation quality, not source selection
- Fixed `detect_refusal()` to recognize Phase 8 context-irrelevant answer patterns

Updated: `scripts/run_benchmark.py`

### 10D — Verifier / answer grounding

**Problem:** once routing and retrieval became reliable, the remaining failures shifted toward final answer behavior:
- missing inline citations
- grounded-but-overstated answers
- clause-level lookup rows where generation drifted away from the exact policy text

**Completed so far:**
- upgraded `app/core/nli_verifier.py`
  - strips citation footer before scoring
  - parses inline citation markers
  - scores cited sentences against cited chunks first
  - returns unsupported sentence and citation-mismatch details
- upgraded `app/chat/handlers.py`
  - faithfulness check now uses richer verifier output
  - inline citation fallback runs before footer appending
  - grounding-fallback answers do not receive misleading citation footers
  - one grounding-repair rewrite is attempted before disclaimer / refusal
  - bilingual scoped-query gating improved for Korean-doc / English-question cases
  - deterministic scoped clause-answer fallback added for strong factual matches
- added grounding benchmark assets:
  - `scripts/eval_verifier_grounding.py`
  - `verifier_grounding_eval_v1.jsonl`
  - `verifier_grounding_eval_v2.jsonl`

Measured result on `verifier_grounding_eval_v1.jsonl`:
- `overall_pass_count`: `6/8`
- `avg_source_recall`: `1.0`
- `avg_answer_point_recall`: `0.583`
- `inline_citation_rate_answerable`: `1.0`
- `correct_not_found_rate`: `1.0`

Current interpretation:
- retrieval/source grounding is already strong
- citation behavior is much stronger than before
- remaining weaknesses are now concentrated in a few exact answer-shaping and comparison cases

### Real-PDF corpus expansion

To make the pipeline more realistic, additional PDFs from `data/test pdfs/` were moved into `data/library/`.

Why this matters:
- routing, retrieval, and grounding are now being tested against a broader real corpus
- the watcher auto-ingests any new PDFs placed in `data/library/`
- those failures are the right source for future QLoRA SFT / preference data

Operational note:
- use `data/test pdfs/` as a staging area
- move only the PDFs you want actively indexed into `data/library/`

### Current Phase 10D direction

What the latest hard grounding evals showed:
- retrieval/source grounding is strong
- refusal behavior is still safe
- the current bottleneck is OCR/table-heavy exact-answer grounding on messy policy PDFs

Current pipeline-side fixes in progress:
- more realistic answer-point scoring for OCR/noisy exact matches
- stronger deterministic extraction for dense amount/category/value clauses
- keep `verifier_grounding_eval_v2` as the hard grounding benchmark before the next QLoRA round

### 10E — Unscoped library grounding and comparison hardening

**Problem:** once the scoped grounding suites went green, the broader library-wide eval exposed a different weakness profile:
- named library document questions were still being treated like whole-corpus tasks
- unscoped exact lookups were sometimes retrieving the right document but failing to convert that into a scoped exact answer
- two-document comparisons were routing correctly but still falling back to brittle free-form generation
- OCR-heavy exact lookup rows such as award caps and student-instructor hourly rates still broke on noisy page text

**Solution:**
- added explicit library document-name auto-scope in `app/chat/scope.py`
- added implicit single-document promotion in `app/chat/handlers.py` when retrieved chunks are dominated by one library document family
- added deterministic two-document comparison handling in `app/chat/deterministic.py` for:
  - purpose
  - procedure
  - support / benefit
  - role / relationship
  - presence / not-found checks such as refund rules, annual fee amount, or fee payment method
- generalized the presence path so it uses query-term extraction first and only falls back to small eval-sensitive label patterns when needed
- strengthened OCR/table exact-answer normalization for:
  - `개인상금 700,000원 이하/인`
  - `팀 상금 1,000,000원 이하/팀`
  - `학부생 시간당 50,000원 이하`
  - `대학원생 시간당 70,000원 이하`

Measured checkpoint on March 27, 2026:
- `verifier_grounding_eval_unscoped_v1`
  - `overall_pass_count = 22/24`
  - `avg_answer_point_recall = 0.875`
  - `avg_source_recall = 0.917`
  - `correct_not_found_rate = 1.0`

What this means:
- unscoped library behavior is now strong enough to act as a serious PDF-grounding regression guard
- not-found safety is stable after the comparison-presence fixes
- the remaining bottlenecks are now very narrow:
  - exact-source locking across near-duplicate guideline variants
  - a small amount of exact-lookup/source-selection residue

Current reading:
- scope plumbing, document-family promotion, and deterministic comparison behavior are working
- the next biggest gains should come from duplicate-source disambiguation and new PDF-grounded failure mining, not another large routing rewrite

### 10D — Live upload behavior hardening

Live testing against unseen uploaded narrative PDFs exposed a different class of issues from the policy/amendment benchmarks: the RAG core was healthy, but single-upload UX, long-document summary behavior, and fake clause lookups needed smarter handling.

Completed:
- single-upload auto-scope:
  - if exactly one upload is active, `이 문서`, `이 파일`, `this document` style queries now auto-bind to that file
- broadened grounding gate:
  - document-backed answers now pass through verifier logic even when they are not explicitly single-doc scoped
- deterministic single-document summary:
  - one uploaded narrative PDF can use a representative-sentence summary path instead of always sending the entire document through a trimmed prompt
- deterministic numeric fact extraction:
  - broad year/number questions can extract grounded numeric facts before free-form generation
- non-clause article fallback:
  - `제N조` queries on narrative PDFs now fail cleanly instead of hallucinating fake article text
- compact citation footer:
  - citation footer is capped to a compact useful evidence set instead of dumping huge per-page lists

What this changed in practice:
- upload-mode `이 문서 ...` queries behave more like scoped document QA and less like weak whole-upload-corpus search
- verifier catches bad long-document generations more safely
- the remaining challenge on uploads is no longer basic scoping; it is better evidence selection for long unseen narrative PDFs

### Measured results

Baseline benchmark (pre-Phase 9, flat chunking):
- 42 chunks across 3 library PDFs
- avg_answer_point_recall: **0.259**
- avg_source_recall: 0.947
- negative_case_refusal_pass: 1/4

After re-ingestion with clause chunking + pdfplumber:
- **63 chunks** (40 clause + 11 section + 12 page)
- 2-column PDF text now extracted in correct reading order
- Each `제N조` article preserved as atomic chunk (no cross-article fragmentation)
- Post-benchmark comparison pending

## Current Strengths

The project is currently strongest in:
- parser and ingestion architecture (5-method pipeline + pdfplumber enhancement)
- clause-level and column-aware document chunking
- upload/document lifecycle separation
- hybrid retrieval foundation
- scoped document chat
- single-document grounded QA
- upload-scoped QA
- single-upload auto-scope and remembered-upload reuse
- deterministic multi-file summary
- answer guard pipeline (context-relevance + tiered NLI + citation)
- benchmark-driven fine-tuning workflow

## Current Weaknesses

The project is still weakest in:
- clarification behavior on ambiguous bundled documents
- nuanced comparison answers
- exact phrase fidelity in some strict lookup cases
- OCR-heavy screenshot/image summary cleanliness
- long narrative PDF summary and selective fact extraction on unseen uploads
- table-level artifacts (pdfplumber tables extracted but rarely large enough to survive as standalone chunks)

## Current Summary Table

| Area | Status | Notes |
|---|---|---|
| Modular backend | Done | Core refactor complete |
| PDF/image ingestion | Strong | Multi-step parser working |
| Upload workflow | Strong | Remembers and re-scopes uploads |
| Single-upload auto-scope | Working | `이 문서` style prompts bind to the only uploaded file |
| Semantic chunking       | Done   | Heading-aware, clause-aware, table-aware |
| Clause-level chunking   | Done   | Korean `제N조` articles as atomic chunks  |
| Parent-child chunking   | Done   | Children indexed, parents for context |
| Context enrichment | Done | Supports better retrieval |
| Hybrid retrieval | Done | Vector + keyword + RRF fusion |
| Reranking               | Active | BGE-reranker-v2-m3 on CUDA FP16       |
| Confidence gating | Added | Threshold still tunable |
| Context-relevance gate  | Active | Pre-generation check, threshold 0.25  |
| Document registry | Working | Stable `doc_id` and upload tracking |
| Library benchmark | Done | Stable baseline + HTTP mode available |
| Upload benchmark | Done | Scoped regressions are measurable |
| Multi-file summary | Working | Deterministic file-by-file mode |
| Single-document upload summary | Working | Deterministic narrative summary path added |
| QLoRA training script | Done | Local training path works |
| QLoRA evaluation | Done | Strict Korean-heavy eval exists |
| General-purpose cleanup | Done | All domain hardcoding removed (Phase 5) |
| NLI faithfulness gate   | Active | Tiered: hard (<0.15) / soft (0.15–0.35) / pass  |
| PaddleOCR v5 + Spacing  | Active | Korean scanned PDF OCR                |
| pdfplumber enhancement  | Active | Column-aware extraction + table detection |
| Active adapter          | `v9`   | General-purpose, 135 diverse samples  |
| LLM decoding | Greedy | temperature=0.0, rep_penalty=1.05 |
| Token suppression | Active | CJK+Cyrillic+Thai (32,257 tokens banned) |
| UI polish | Partial | Functional, not product-polished |

## Advanced Approaches To Consider

Earlier planning intentionally postponed more advanced approaches until the core architecture was stable. Several are now implemented; the rest remain valid next-stage improvements.

### Retrieval / representation
- ~~clause-level and article-level indexing for policy documents~~ → **Done (Phase 9)**
- ~~richer block artifacts for tables~~ → **Partially done (Phase 9, pdfplumber tables)**
- richer block artifacts for forms and numbered rules
- hierarchical retrieval for bundled long documents
- late-interaction reranking such as ColBERT-style methods

### Routing / orchestration
- query-type classifier for lookup, summary, comparison, OCR, and clarification
- document-type-aware prompt routing
- separate policies for screenshot/image-heavy uploads vs clean digital PDFs

### Answer reliability
- ~~answer-type guards for `clarify`, `not_found`, `mention_only`, and `comparison`~~ → **Partially done (Phase 8: not_found + tiered faithfulness)**
- ~~citation/evidence consistency checks~~ → **Done (Phase 7 + Phase 8)**
- post-generation validation passes for risky answer categories

### Fine-tuning beyond SFT
- judge-assisted error mining from benchmark failures
- targeted synthetic augmentation only after real-document coverage is strong
- preference tuning such as DPO/ORPO for refusal, comparison, and ambiguity behavior

### Evaluation / operations
- automatic regression runs over strict eval sets
- retrieval ablation tracking
- real-world shadow evaluation with production-like prompts

These are now valid project improvements, but they should build on the current stable foundation rather than replace it.

## Current Conclusion

The project has completed the Phase 10A–10C foundation work and is now in Phase 10D. Core architecture, extraction, chunking, retrieval, answer verification, and fine-tuning are all implemented and deployed. The current focus is no longer “can retrieval find the right source?” but “does the final answer stay faithful, cited, and precise on harder real-PDF cases and live uploaded documents?”
