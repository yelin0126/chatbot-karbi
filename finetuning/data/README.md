# Benchmark Dataset

This folder is the starting point for the evaluation benchmark that will guide:

1. retrieval tuning
2. confidence threshold tuning
3. QLoRA dataset preparation

The benchmark should be built from real Tilon documents, not synthetic toy examples.

## Why This Exists

Right now the project has strong ingestion and retrieval foundations, but it still needs a measurable evaluation loop.

Without a benchmark:
- retrieval tuning becomes guesswork
- confidence thresholds are hard to calibrate
- QLoRA quality is hard to compare against the base model

## Recommended Workflow

### Step 1: Build benchmark questions
Create JSONL rows in `benchmark_template.jsonl` using representative Tilon documents.

### Step 2: Run baseline evaluation
Evaluate the current RAG system using these questions before changing thresholds or prompts.

### Step 3: Tune retrieval
Use benchmark failures to improve:
- retrieval precision
- confidence gating
- scoped vs global behavior
- reranking runtime strategy

### Step 4: Convert benchmark items into QLoRA-ready training data
Once the prompt/context format is stable, selected benchmark items can become supervised fine-tuning examples.

## JSONL Schema

Each line should be one JSON object like this:

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

## Required Fields

- `id`
- `category`
- `language`
- `document_source`
- `question`
- `should_answer_from_docs`
- `expected_answer_points`

## Optional Scope Fields

- `scope_source_type`
- `document_sources`

Use `scope_source_type` when the same filename may exist in more than one registry scope.

Example:

```json
{
  "document_source": "policy_bundle.pdf",
  "scope_source_type": "upload"
}
```

This is especially useful for upload-scoped benchmarks, where the runner should prefer
the uploaded document instead of a library copy with the same filename.

Use `document_sources` for multi-document comparison benchmarks.

Example:

```json
{
  "document_source": "doc_a.pdf",
  "document_sources": ["doc_a.pdf", "doc_b.pdf"]
}
```

When `document_sources` contains two or more entries, the benchmark runner scopes the
question to all selected documents and evaluates comparison-style answers.

## Suggested Categories

- `exact_lookup`
- `summary`
- `ocr_extract`
- `section_understanding`
- `not_found`
- `bilingual`

## Upload-Scoped Benchmarks

The file `benchmark_upload_scoped_v1.jsonl` is for temporary upload-chat behavior rather than
the persistent library corpus.

Before running it:

1. start the backend
2. upload the target file once through `/ui` or `/chat-with-file`
3. confirm the uploaded filename exists in the document registry
4. run the benchmark

Example:

```bash
python scripts/validate_benchmark.py --path finetuning/data/benchmark_upload_scoped_v1.jsonl
python scripts/run_benchmark.py --path finetuning/data/benchmark_upload_scoped_v1.jsonl --mode both
```

## Comparison Benchmarks

The file `benchmark_compare_v1.jsonl` is for multi-document PDF comparison.

It uses library documents by default, so it can usually be run immediately after normal ingest:

```bash
python scripts/validate_benchmark.py --path finetuning/data/benchmark_compare_v1.jsonl
python scripts/run_benchmark.py --path finetuning/data/benchmark_compare_v1.jsonl --mode both
```

## Category Guidance

### `exact_lookup`
Use for:
- commands
- settings
- paths
- error codes
- dates

### `summary`
Use for:
- section summaries
- whole-document summaries
- key-point summaries

### `ocr_extract`
Use for:
- screenshot text
- scanned pages
- image-only pages

### `section_understanding`
Use for:
- “what does this section mean?”
- structure/navigation questions

### `not_found`
Use for:
- questions the document should *not* answer
- confidence gating tests

### `bilingual`
Use for:
- English question / Korean document
- Korean question / English document
- mixed technical terminology

## Good Benchmark Design Rules

- Use real Tilon documents whenever possible
- Keep questions realistic, like what teammates would actually ask
- Include both easy and hard questions
- Include negative questions where the correct answer is “not found”
- Include screenshot and scanned-document cases
- Keep `expected_answer_points` short and checkable

## QLoRA Connection

This benchmark is not the same thing as the final QLoRA dataset, but it is the best foundation for it.

The benchmark tells you:
- whether retrieval is good enough
- whether the live prompt format is stable
- what kinds of grounded answers you want the fine-tuned model to produce

When the benchmark is mature enough, selected items can be extended with:
- retrieved context
- ideal final answer
- citation style

That becomes the supervised QLoRA dataset.

## QLoRA Starter File

The file `qlora_train_v1.jsonl` is the first supervised training scaffold.

It uses this shape:

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

Recommended rules when expanding it:

- keep `context` close to the live retrieval format
- keep `answer` grounded and concise
- include clarification answers for ambiguous upload cases
- include refusal answers for true `not_found` cases
- preserve benchmark linkage through `source_benchmark_id`

## From Dataset To Training

Once the dataset is large enough, use:

```bash
python finetuning/train.py \
  --data finetuning/data/qlora_train_v1.jsonl \
  --model Qwen/Qwen2.5-7B-Instruct \
  --output finetuning/output/qwen25-qlora-v1
```

Current expectation:

- `qlora_train_v1.jsonl` is a starter set
- it is useful for testing the training workflow
- it is not yet large enough for a strong final adapter
