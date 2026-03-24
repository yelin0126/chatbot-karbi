# Benchmark And Fine-Tuning Datasets

This folder contains the benchmark, strict evaluation, and supervised fine-tuning datasets used to improve the Tilon chatbot.

## What Lives Here

Benchmark and eval files:
- [benchmark_tilon_v1.jsonl](/home/tilon/chatbot-karbi/finetuning/data/benchmark_tilon_v1.jsonl)
- [benchmark_upload_scoped_v1.jsonl](/home/tilon/chatbot-karbi/finetuning/data/benchmark_upload_scoped_v1.jsonl)
- [benchmark_compare_v1.jsonl](/home/tilon/chatbot-karbi/finetuning/data/benchmark_compare_v1.jsonl)
- [qlora_eval_ko_strict_v1.jsonl](/home/tilon/chatbot-karbi/finetuning/data/qlora_eval_ko_strict_v1.jsonl)
- [qlora_eval_ko_strict_v2.jsonl](/home/tilon/chatbot-karbi/finetuning/data/qlora_eval_ko_strict_v2.jsonl)

Training sets:
- [qlora_train_v1.jsonl](/home/tilon/chatbot-karbi/finetuning/data/qlora_train_v1.jsonl)
- [qlora_train_v2.jsonl](/home/tilon/chatbot-karbi/finetuning/data/qlora_train_v2.jsonl)
- [qlora_train_v3.jsonl](/home/tilon/chatbot-karbi/finetuning/data/qlora_train_v3.jsonl)
- [qlora_train_v4.jsonl](/home/tilon/chatbot-karbi/finetuning/data/qlora_train_v4.jsonl)

## Purpose Of Each Dataset Family

### Benchmarks
Used to evaluate live RAG behavior.

They guide:
- retrieval tuning
- confidence threshold tuning
- scoped upload behavior
- comparison behavior

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

## Current Best Adapter Link

Current best adapter:
- [qwen25-qlora-v6](/home/tilon/chatbot-karbi/finetuning/output/qwen25-qlora-v6)

Current strongest strict result:
- eval set: [qlora_eval_ko_strict_v2.jsonl](/home/tilon/chatbot-karbi/finetuning/data/qlora_eval_ko_strict_v2.jsonl)
- base avg: `0.244`
- adapter avg: `0.4173`

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
