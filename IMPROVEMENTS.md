# Improvements & Current State

This file tracks how the project evolved from a single-file prototype into the current document-first RAG system with a working local QLoRA loop.

Related docs:
- [README.md](/home/tilon/chatbot-karbi/README.md)
- [ARCHITECTURE.md](/home/tilon/chatbot-karbi/ARCHITECTURE.md)
- [finetuning/README.md](/home/tilon/chatbot-karbi/finetuning/README.md)

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

This phase is no longer “future only.” The end-to-end training/evaluation path now exists and has been exercised repeatedly.

Completed:
- [train.py](/home/tilon/chatbot-karbi/finetuning/train.py)
- [evaluate_adapter.py](/home/tilon/chatbot-karbi/finetuning/evaluate_adapter.py)
- [infer_compare.py](/home/tilon/chatbot-karbi/finetuning/infer_compare.py)
- strict Korean-heavy eval sets
- multiple training-set iterations
- clean base-vs-adapter evaluation
- corrected Qwen chat-template usage

Important QLoRA milestones:
- `v3` established that fine-tuning could beat base
- `v4` showed that simply growing the dataset was not enough
- `v5` became the first broadly convincing adapter
- `v6` is the current best adapter

Current best adapter:
- [qwen25-qlora-v6](/home/tilon/chatbot-karbi/finetuning/output/qwen25-qlora-v6)

Current hardest eval result:
- dataset: [qlora_eval_ko_strict_v2.jsonl](/home/tilon/chatbot-karbi/finetuning/data/qlora_eval_ko_strict_v2.jsonl)
- base avg: `0.244`
- adapter avg: `0.4173`
- adapter wins: `12/15`
- base wins: `0/15`
- ties: `3/15`

What this means:
- the adapter meaningfully improves Korean grounded-answer behavior
- mixed-language / Chinese drift is reduced
- the training workflow is valid and useful
- app-level validation is still required before default deployment

## Current Strengths

The project is currently strongest in:
- parser and ingestion architecture
- upload/document lifecycle separation
- hybrid retrieval foundation
- scoped document chat
- single-document grounded QA
- upload-scoped QA
- deterministic multi-file summary
- benchmark-driven fine-tuning workflow

## Current Weaknesses

The project is still weakest in:
- clarification behavior on ambiguous bundled documents
- nuanced comparison answers
- exact phrase fidelity in some strict lookup cases
- OCR-heavy screenshot/image summary cleanliness
- block-level structured artifacts for tables and clause-level reasoning

## Current Summary Table

| Area | Status | Notes |
|---|---|---|
| Modular backend | Done | Core refactor complete |
| PDF/image ingestion | Strong | Multi-step parser working |
| Upload workflow | Strong | Remembers and re-scopes uploads |
| Semantic chunking | Done | Heading-aware and table-aware |
| Context enrichment | Done | Supports better retrieval |
| Hybrid retrieval | Done | Vector + keyword + fusion |
| Reranking | Added | Runtime policy still tunable |
| Confidence gating | Added | Still needs threshold tuning |
| Document registry | Working | Stable `doc_id` and upload tracking exist |
| Library benchmark | Done | Stable baseline exists |
| Upload benchmark | Done | Scoped regressions are measurable |
| Multi-file summary | Working | Deterministic file-by-file mode |
| QLoRA training script | Done | Local training path works |
| QLoRA evaluation | Done | Strict Korean-heavy eval exists |
| Best adapter | `v6` | Strongest current candidate |
| UI polish | Partial | Functional, not product-polished |

## Advanced Approaches To Consider

Earlier planning intentionally postponed more advanced approaches until the core architecture was stable. The project is now far enough along to consider them as real next-stage improvements.

### Retrieval / representation
- clause-level and article-level indexing for policy documents
- richer block artifacts for tables, forms, and numbered rules
- hierarchical retrieval for bundled long documents
- late-interaction reranking such as ColBERT-style methods

### Routing / orchestration
- query-type classifier for lookup, summary, comparison, OCR, and clarification
- document-type-aware prompt routing
- separate policies for screenshot/image-heavy uploads vs clean digital PDFs

### Answer reliability
- answer-type guards for `clarify`, `not_found`, `mention_only`, and `comparison`
- citation/evidence consistency checks
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

The project is no longer blocked by missing core architecture. The main task now is to turn a strong engineering foundation into consistently accurate product behavior through targeted evaluation, targeted data, and selective advanced methods where they clearly help.
