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
- `verifier_grounding_eval_external_v1`: `2/10`
- `verifier_grounding_eval_tables_v1`: `5/8`
- current interpretation: remaining errors are now much more model-facing than pipeline-facing; Chinese drift is the dominant failure mode across external, table, and unscoped slices

## Pre-v10 Baseline Snapshot (2026-03-27)

Official reference checkpoint recorded before `qlora_train_v10` assembly.

| Eval Suite | Pass/Total | Key Metrics |
|---|---|---|
| Routing (v3) | 40/40 (100%) | — |
| Structure retrieval (v1) | 6/11 | source_recall=1.0, point_recall=0.444 |
| Structure retrieval (v2) | 6/15 | source_recall=1.0, point_recall=0.185 |
| Grounding (v1, scoped) | 5/8 | not_found_correct=0.667 |
| Grounding (v2, scoped) | 10/10 (100%) | — |
| Unscoped grounding | 22/24 | point_recall=0.875, source_recall=0.917 |
| External grounding | 2/10 | point_recall=0.267, disclaimer=33% |
| Table grounding | 5/8 | point_recall=0.708 |

Result files: `finetuning/results/verifier_grounding_results_20260327_16*.jsonl`

Key observations:
- routing is solid, no pipeline work needed to start v10
- scoped grounding v2 is perfect
- structure retrieval shows a model-use problem (source_recall=1.0 but low answer extraction)
- external grounding (2/10) is the loudest gap and the primary v10 mining target
- Chinese language drift is the dominant failure mode across external, table, and unscoped slices
- unscoped 011 and 014 are model-side failures with correct scoping but poor generation

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

The dataset work should stay selective and failure-driven. The project already proved that broad expansion is not automatically better than targeted high-signal additions.

Current priority order:
1. keep routing and retrieval evals green
2. keep verifier / grounding evals green on real PDFs (`v1`, `v2` are current regression guards)
3. use `verifier_grounding_eval_unscoped_v1.jsonl` as the active corpus-wide stress slice; 22/24 with known model-side failures at 011 and 014
4. use `verifier_grounding_eval_external_v1.jsonl` (2/10) and `verifier_grounding_eval_tables_v1.jsonl` (5/8) as the primary v10 failure-mining sources
5. mine true model-side failure patterns into targeted supervised data
6. assemble `qlora_train_v10` from: external Chinese-drift failures, unscoped 011+014, structure-retrieval extraction failures, table-grounding extraction failures
7. train v10 adapter on Qwen2.5, evaluate against the same baseline
8. only test Qwen3 after v10 exists as a controlled comparison

Current training recommendation:
- keep the next failure-mined QLoRA round on the Qwen2.5 family first
- only test Qwen3 after the first `v10` dataset exists and can be compared under the same eval suite

## Pre-v10 Baseline Runner

To capture one clean reference checkpoint before creating `qlora_train_v10`, run:

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
