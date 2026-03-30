# Benchmark And Fine-Tuning Datasets

This folder contains the benchmark, strict evaluation, and supervised fine-tuning datasets used to improve the Tilon chatbot.

## What Lives Here

Benchmark and eval files:
- [benchmark_tilon_v1.jsonl](/home/tilon/chatbot-karbi/finetuning/data/benchmark_tilon_v1.jsonl)
- [benchmark_upload_scoped_v1.jsonl](/home/tilon/chatbot-karbi/finetuning/data/benchmark_upload_scoped_v1.jsonl)
- [benchmark_compare_v1.jsonl](/home/tilon/chatbot-karbi/finetuning/data/benchmark_compare_v1.jsonl)
- [query_policy_eval_v1.jsonl](/home/tilon/chatbot-karbi/finetuning/data/query_policy_eval_v1.jsonl)
- [query_policy_eval_v2.jsonl](/home/tilon/chatbot-karbi/finetuning/data/query_policy_eval_v2.jsonl)
- [query_policy_eval_v3.jsonl](/home/tilon/chatbot-karbi/finetuning/data/query_policy_eval_v3.jsonl)
- [structure_retrieval_eval_v1.jsonl](/home/tilon/chatbot-karbi/finetuning/data/structure_retrieval_eval_v1.jsonl)
- [structure_retrieval_eval_v2.jsonl](/home/tilon/chatbot-karbi/finetuning/data/structure_retrieval_eval_v2.jsonl)
- [verifier_grounding_eval_v1.jsonl](/home/tilon/chatbot-karbi/finetuning/data/verifier_grounding_eval_v1.jsonl)
- [verifier_grounding_eval_v2.jsonl](/home/tilon/chatbot-karbi/finetuning/data/verifier_grounding_eval_v2.jsonl)
- [verifier_grounding_eval_unscoped_v1.jsonl](/home/tilon/chatbot-karbi/finetuning/data/verifier_grounding_eval_unscoped_v1.jsonl)
- [verifier_grounding_eval_external_v1.jsonl](/home/tilon/chatbot-karbi/finetuning/data/verifier_grounding_eval_external_v1.jsonl)
- [verifier_grounding_eval_tables_v1.jsonl](/home/tilon/chatbot-karbi/finetuning/data/verifier_grounding_eval_tables_v1.jsonl)
- [qlora_eval_ko_strict_v1.jsonl](/home/tilon/chatbot-karbi/finetuning/data/qlora_eval_ko_strict_v1.jsonl)
- [qlora_eval_ko_strict_v2.jsonl](/home/tilon/chatbot-karbi/finetuning/data/qlora_eval_ko_strict_v2.jsonl)

Training sets (active):
- [qlora_train_v5.jsonl](/home/tilon/chatbot-karbi/finetuning/data/qlora_train_v5.jsonl) — 90 diverse general samples
- [qlora_train_v9.jsonl](/home/tilon/chatbot-karbi/finetuning/data/qlora_train_v9.jsonl) — 45 new general samples
- [qlora_train_v9_combined.jsonl](/home/tilon/chatbot-karbi/finetuning/data/qlora_train_v9_combined.jsonl) — 135 combined (used for v9 adapter)
- [qlora_train_v10.jsonl](/home/tilon/chatbot-karbi/finetuning/data/qlora_train_v10.jsonl) — 35 failure-mined rows (v10-only, pre-v10 baseline failures)
- [qlora_train_v10_combined.jsonl](/home/tilon/chatbot-karbi/finetuning/data/qlora_train_v10_combined.jsonl) — 170 combined (v9_combined + v10, for v10 adapter training)

Retired training sets (domain-specific, do not use):
- `qlora_train_v8.jsonl` — 25 church-specific samples
- `qlora_train_v8_combined.jsonl` — 115 samples, ~78% church-specific

Legacy training sets (kept for reference):
- [qlora_train_v1.jsonl](/home/tilon/chatbot-karbi/finetuning/data/qlora_train_v1.jsonl)
- [qlora_train_v2.jsonl](/home/tilon/chatbot-karbi/finetuning/data/qlora_train_v2.jsonl)
- [qlora_train_v3.jsonl](/home/tilon/chatbot-karbi/finetuning/data/qlora_train_v3.jsonl)
- [qlora_train_v4.jsonl](/home/tilon/chatbot-karbi/finetuning/data/qlora_train_v4.jsonl)

## Purpose Of Each Dataset Family

### Benchmarks
Used to evaluate live RAG behavior.

They guide:
- retrieval tuning
- routing tuning
- structure-aware retrieval tuning
- verifier / grounding tuning
- OCR/table-aware exact-answer tuning
- confidence threshold tuning
- scoped upload behavior
- comparison behavior
- unscoped library auto-scope and dominant-document behavior
- external-PDF generalization checks
- table/list/form-heavy grounding checks

### Strict eval sets
Used for base-vs-adapter evaluation.

They focus on:
- Korean-heavy grounded QA
- exact lookup
- clarification
- not-found refusal
- mention-only behavior
- comparison quality

### QLoRA training sets
Used to train LoRA adapters after the prompt/context contract is stable.

They should teach:
- concise grounded answering
- clean refusal
- ambiguity clarification
- Korean answer stability
- bilingual answer matching from PDF context
- exact lookup and table/value extraction from noisy policy PDFs
- section-aware explanation and comparison over retrieved document chunks

## Recommended Workflow

1. benchmark live RAG first
2. identify failure patterns
3. convert high-signal rows into supervised examples
4. train adapter
5. evaluate base vs adapter on strict eval sets
6. validate best adapter inside the actual chatbot flow

## Benchmark JSONL Schema

Each benchmark row looks like:

```json
{
  "id": "tilon-001",
  "category": "exact_lookup",
  "language": "ko",
  "document_source": "admin_manual.pdf",
  "question": "관리자 비밀번호는 어디에서 변경하나요?",
  "should_answer_from_docs": true,
  "expected_answer_points": [
    "관리자 메뉴 또는 사용자 관리 페이지에서 변경 가능",
    "문서 근거로 답해야 함"
  ],
  "expected_sources": [
    {"source": "admin_manual.pdf", "page": 12}
  ],
  "notes": "정확한 메뉴 경로가 나와야 하는 문제"
}
```

Required fields:
- `id`
- `category`
- `language`
- `document_source`
- `question`
- `should_answer_from_docs`
- `expected_answer_points`

Optional scope fields:
- `scope_source_type`
- `document_sources`

Use `document_sources` for comparison-style rows.

If both `document_source` and `document_sources` are omitted, the row can act as a true unscoped corpus test. This is how `verifier_grounding_eval_unscoped_v1.jsonl` exercises library auto-scope, dominant-document promotion, and comparison selection without giving the backend an explicit document upfront.

## QLoRA Training Schema

Each training row looks like:

```json
{
  "id": "qlora-001",
  "source_benchmark_id": "tilon-rise-001",
  "language": "ko",
  "question": "사용자 질문",
  "context": "[Doc: ... | Page: ... | Section: ... | Lang: ko]\n근거 문맥",
  "answer": "근거 기반 최종 답변"
}
```

Rules when expanding:
- keep `context` close to the live retrieval format
- keep `answer` grounded and concise
- include clarification answers for ambiguous bundled uploads
- include refusal answers for true `not_found` cases
- preserve benchmark linkage where possible
- prefer the repo's current `question/context/answer` schema over a generic `instruction/input/output` rewrite
- include citations in the natural answer text when the benchmark behavior expects them

## Current Dataset Evolution

### `qlora_train_v1`
- first scaffold
- benchmark-linked starter set

### `qlora_train_v2`
- broader expansion
- useful, but showed that more data alone was not enough

### `qlora_train_v3`
- smaller, more selective high-signal set
- used to produce `v5`

### `qlora_train_v4`
- booster set focused on:
  - comparison
  - clarification
  - mention-only
  - stricter not-found wording
  - bundled summary behavior
- used to produce `v6`

## Current Adapter Status

Current deployed adapter:
- [qwen25-qlora-v9](/home/tilon/chatbot-karbi/finetuning/output/qwen25-qlora-v9)

Strongest legacy strict result:
- eval set: [qlora_eval_ko_strict_v2.jsonl](/home/tilon/chatbot-karbi/finetuning/data/qlora_eval_ko_strict_v2.jsonl)
- base avg: `0.244`
- adapter avg: `0.4173` (`v6`)

Current PDF-grounding status:
- `verifier_grounding_eval_v2`: `10/10`
- `verifier_grounding_eval_unscoped_v1`: `22/24`
- `verifier_grounding_eval_external_v1`: `3/10` (up from 2/10 after Phase 12 pipeline fixes)
- `verifier_grounding_eval_tables_v1`: `2/8` (correctly scored after Phase 12 — 3 old "passes" were mislabeled)
- **pipeline track frozen** (2026-03-30) — all remaining 13 failures are model generation quality
- current interpretation: Chinese drift, hallucination, incomplete extraction, narrative table reading — all require training track fixes, not pipeline
- 14B produces byte-identical outputs to 7B; model capacity is not the bottleneck

## Pre-v10 Baseline Snapshot (2026-03-27)

Official reference checkpoint recorded before `qlora_train_v10` assembly. v10 data is now assembled; full experiment cycle completed 2026-03-30 (Tier A, Tier A+B, 14B, token expansion, VLM hybrid) — all flat.

**Important**: Structure retrieval baseline numbers were corrected on 2026-03-30. The original report (6/11 and 6/15) included unevaluated rows, which were auto-counted as passes. The evaluation script default was changed from `--mode retrieval` to `--mode full` to run full generation on all rows. **True corrected baseline:**

| Eval Suite | Pass/Total | Key Metrics |
|---|---|---|
| Routing (v3) | 40/40 (100%) | — |
| Structure retrieval (v1) | **1/11** | source_recall=1.0, point_recall=0.273 (was reported 6/11) |
| Structure retrieval (v2) | **0/15** | source_recall=1.0, point_recall=0.178 (was reported 6/15) |
| Grounding (v1, scoped) | 5/8 | not_found_correct=0.667 |
| Grounding (v2, scoped) | 10/10 (100%) | — |
| Unscoped grounding | 22/24 | point_recall=0.875, source_recall=0.917 |
| External grounding | 2/10 | point_recall=0.267, disclaimer=33% |
| Table grounding | 5/8 | point_recall=0.708 |

Result files: `finetuning/results/verifier_grounding_results_20260327_16*.jsonl` (old), `finetuning/results/structure_retrieval_results_20260330_110432.jsonl` (corrected v1), `finetuning/results/structure_retrieval_results_20260330_110628.jsonl` (corrected v2)

Key observations:
- routing is solid, no pipeline work needed to start v10
- scoped grounding v2 is perfect
- structure retrieval shows a model-use problem (source_recall=1.0 but low answer extraction)
- external grounding (2/10) is the loudest gap and the primary v10 mining target
- Chinese language drift is the dominant failure mode across external, table, and unscoped slices
- unscoped 011 and 014 are still mixed failures:
  - 011 still misses the page-3 amount evidence in the live path
  - 014 still suffers from wrong-variant / wrong-member retrieval before generation
  - both are useful v10 cases, but they should not be treated as clean model-only failures

## Post-v10 Investigation & Pipeline Fixes (2026-03-30)

### v10 Training Results
- **Tier A (168 rows, pure model failures)**: No improvement vs baseline. 1/11 structure, 2/10 external, 5/8 tables — identical to v9.
- **Tier A+B (170 rows, mixed failures)**: Slightly worse. 4/8 scoped v1 (was 5/8), 2/10 external, 5/8 tables.
- **Conclusion**: Failure-mined SFT on v9-based 7B model did not transfer improvement. The problem is not simply "need more training data."

### Token Limit Test (4096 → 8192)
Hypothesis: Prompt trimming at 4096 was causing context-cutoff failures.

| Metric | 4096 | 8192 | Result |
|--------|------|------|--------|
| Structure v1 pass | 1/11 | 1/11 | — no change |
| Structure v2 pass | 0/15 | 0/15 | — no change |
| Trimming events | 9 | 0 | ✓ Eliminated |
| Language drift | 4 | 3 | ✓ Slightly reduced |

**Conclusion**: Trimming was real but was not the limiter. Expanding context to 8192 eliminated all truncation but pass counts stayed the same. The real bottleneck is model generation quality (language drift, hallucination, poor extraction), not input length.

### Failure Root Cause Audit (33 rows across external/structure/tables)

| Cause | Count | Example Rows | Fix Type |
|-------|-------|--------------|----------|
| Language drift to Chinese | 4+ | ext-005/006/007/008 | Model behavior (token suppression leaking, retry still fails) |
| Table context raw-dump | 4+ | ext-003/009, table-006/008 | Model behavior (pipe-delimited text; fix = noisy-context SFT) |
| Hallucination / low faithfulness | 5+ | ext-004/008 | Model behavior (generates unsupported content) |
| False refusal | 2 | ext-010, table-007 | Pipeline (context_relevance=0.00 on valid form data) |
| Faithfulness repair degradation | 1+ | ext-004 (0.34→0.03) | Pipeline (grounding-repair makes answer worse) |

**Key fact**: All 33 failures had `source_recall=1.0` — retrieval routing worked perfectly. Failures are all generation-side.

### Completed Action Items (all flat — 2026-03-30)
1. **VLM hybrid on table-heavy PDFs**: Enabled `VLM_HYBRID_PDF_ENABLED=true`, re-ingested 6 target PDFs. VLM never fired — all PDFs are digital with text layers. No effect.
2. **14B base model test**: Qwen2.5-14B produced byte-identical answers to 7B on all failing rows. No improvement.
3. **Token expansion 4096→8192**: Eliminated trimming but pass counts unchanged. Kept as default.

### Completed Pipeline Fixes (Phase 12, 2026-03-30)
1. **Pipe-table extractor**: Deterministic row/field extraction for pipe-delimited `col1 | col2 | col3` content — 7 functions, 3-gate design
2. **Adjacent-token overlap bypass**: Prevents scattered tokens from bypassing relevance gate
3. **Compound presence matching**: All query terms must co-occur in one chunk for presence answers
4. **Full-doc chain pipe extractor**: Covers small docs ≤15 chunks that use full-document loading

### Remaining Action Items (Training Track — primary)
1. ~~**Align train/serve prompt templates** (Step 1, prerequisite)~~ — **DONE (2026-03-30)**: prompts now match exactly.
2. **Mine 13 model-side failures into v11 training rows**: Label by failure type (`chinese_drift`, `hallucination`, `incomplete_extraction`, `narrative_table_reading`); exclude ingestion-broken ext-003
3. **SSFO preference pairs**: Generate 400-500 chosen/rejected pairs from same model (with/without context)
4. **RAFT-format training data**: Oracle + distractor documents, 80/20 split, chain-of-thought with inline citations
5. **Two-stage training**: SFT on combined data, then SimPO preference tuning

**Completed / corrected note**: current venv verified at `paddlepaddle-gpu==3.3.0` on 2026-03-30; earlier 3.3.1 notes were stale, so scanned-PDF OCR should be treated as environment-sensitive until re-verified.

v9 is not a sufficiently broad "any PDF" adapter — it is RISE-PDF biased. Next improvement path: pipeline context fixes then noisy-context SFT.

## Run Benchmark

```bash
python scripts/validate_benchmark.py --path finetuning/data/benchmark_tilon_v1.jsonl
python scripts/run_benchmark.py --path finetuning/data/benchmark_tilon_v1.jsonl --mode both
```

Upload-scoped benchmark:

```bash
python scripts/validate_benchmark.py --path finetuning/data/benchmark_upload_scoped_v1.jsonl
python scripts/run_benchmark.py --path finetuning/data/benchmark_upload_scoped_v1.jsonl --mode both
```

Comparison benchmark:

```bash
python scripts/validate_benchmark.py --path finetuning/data/benchmark_compare_v1.jsonl
python scripts/run_benchmark.py --path finetuning/data/benchmark_compare_v1.jsonl --mode both
```

## Advanced Dataset Directions

These were deferred earlier, but they are now part of the roadmap for full-project improvement:

- failure-mined booster sets from real chatbot logs
- richer comparison-only and clarification-only eval slices
- clause-level examples for article-number questions
- table-aware and form-aware examples
- preference data for refusal and comparison behavior
- controlled synthetic augmentation after real-document coverage is strong

## Current Recommendation

The dataset work should stay selective and failure-driven. The project already proved that broad expansion is not automatically better than targeted high-signal additions. The pipeline track is now frozen — training track is primary.

Current priority order:
1. keep routing and retrieval evals green
2. keep verifier / grounding evals green on real PDFs (`v1`, `v2` are current regression guards)
3. use `verifier_grounding_eval_unscoped_v1.jsonl` as the active corpus-wide stress slice; 22/24 with two remaining mixed retrieval-plus-generation failures at 011 and 014
4. use `verifier_grounding_eval_external_v1.jsonl` (3/10) and `verifier_grounding_eval_tables_v1.jsonl` (2/8) as the primary failure-mining sources
5. **training track (primary):**
   - ~~Step 1: align train/serve prompt templates~~ — **DONE (2026-03-30)**. **Active step: Step 2 — mine 13 model-side failures into labeled v11 training rows.**
   - Step 2: mine 13 model-side failures into labeled v11 training rows
   - Step 3: SSFO preference pairs (400-500)
   - Step 4: RAFT-format training with oracle + distractor documents
   - Step 5: general instruction mix + two-stage SFT+SimPO
6. do NOT pursue: Qwen3, 14B, broad SFT expansion, VLM hybrid on digital PDFs — all confirmed flat (2026-03-30)

### `qlora_train_v10` Design

Row rubric:
- **Tier A (pure model-side)**: source_recall=1.0, model had correct context but generated badly — highest priority, 33 rows
- **Tier B (mixed pipeline/model)**: retrieval gap on key pages — include sparingly with ideal context, 2 rows (v10-usc-001, v10-usc-002)

Task distribution (35 rows total):
- lookup (grounded QA): 9
- article_lookup: 4
- appendix_lookup: 3 (teaches appendix vs main-body disambiguation)
- section_lookup: 3 (teaches section-heading navigation)
- table_lookup: 6
- attachment_lookup: 1
- refusal / not-found: 4 (3 ko, 1 en)
- crosslang (English Q + Korean context): 2
- summary: 1
- structured_extraction: 1
- clarification: 1

Language: 32 ko, 3 en

Current training status:
- v10 stays on the Qwen2.5 family
- **All v10 experiments completed 2026-03-30 — no improvement found:**
  - Tier A (168 rows): identical to v9 baseline across all suites
  - Tier A+B (170 rows): slight regression (scoped v1: 5→4/8)
  - 14B experiment: byte-identical outputs to 7B on failing rows
  - Token 4096→8192: trimming eliminated but pass counts unchanged
  - VLM hybrid: never fired on digital PDFs, eval unchanged
- **Conclusion:** More SFT, larger models, and VLM are not the bottleneck. Pipeline context quality and model behavior are.
- Adapter artifacts saved at `finetuning/output/qwen25-qlora-v10-tier-a/` and `finetuning/output/qwen25-14b-qlora-v10-tier-a/`

## Pre-v10 Baseline Runner

The pre-v10 baseline was captured on 2026-03-27. To re-run it for comparison after v10 adapter training:

```bash
bash scripts/run_pre_v10_baseline.sh
```

This aggregates:
- routing
- structure retrieval
- scoped grounding
- unscoped library grounding
- external-style PDF grounding
- table/list/form-heavy grounding
