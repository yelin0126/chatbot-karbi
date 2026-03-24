# QLoRA Fine-Tuning Workstream

This folder contains the local fine-tuning and evaluation workflow for improving the answer model used on top of retrieved document context.

Related docs:
- [README.md](/home/tilon/chatbot-karbi/README.md)
- [ARCHITECTURE.md](/home/tilon/chatbot-karbi/ARCHITECTURE.md)
- [finetuning/data/README.md](/home/tilon/chatbot-karbi/finetuning/data/README.md)

## Current Status

This workstream is no longer just a scaffold.

What is working now:
- local QLoRA training for Qwen2.5 7B
- strict base-vs-adapter evaluation
- offline-friendly loading for local model files
- chat-template-correct training and inference
- multiple dataset iterations with measured comparisons

Current best adapter:
- [qwen25-qlora-v6](/home/tilon/chatbot-karbi/finetuning/output/qwen25-qlora-v6)

Current strongest measured result:
- eval set: [qlora_eval_ko_strict_v2.jsonl](/home/tilon/chatbot-karbi/finetuning/data/qlora_eval_ko_strict_v2.jsonl)
- base avg: `0.244`
- adapter avg: `0.4173`
- adapter wins: `12/15`
- base wins: `0/15`
- ties: `3/15`

Interpretation:
- QLoRA is helping
- `v6` is the best current checkpoint
- more app-level validation is still needed before production defaulting

## Key Files

- [train.py](/home/tilon/chatbot-karbi/finetuning/train.py)
- [evaluate_adapter.py](/home/tilon/chatbot-karbi/finetuning/evaluate_adapter.py)
- [infer_compare.py](/home/tilon/chatbot-karbi/finetuning/infer_compare.py)
- [requirements-qlora.txt](/home/tilon/chatbot-karbi/finetuning/requirements-qlora.txt)

Main datasets:
- [qlora_train_v1.jsonl](/home/tilon/chatbot-karbi/finetuning/data/qlora_train_v1.jsonl)
- [qlora_train_v2.jsonl](/home/tilon/chatbot-karbi/finetuning/data/qlora_train_v2.jsonl)
- [qlora_train_v3.jsonl](/home/tilon/chatbot-karbi/finetuning/data/qlora_train_v3.jsonl)
- [qlora_train_v4.jsonl](/home/tilon/chatbot-karbi/finetuning/data/qlora_train_v4.jsonl)
- [qlora_eval_ko_strict_v1.jsonl](/home/tilon/chatbot-karbi/finetuning/data/qlora_eval_ko_strict_v1.jsonl)
- [qlora_eval_ko_strict_v2.jsonl](/home/tilon/chatbot-karbi/finetuning/data/qlora_eval_ko_strict_v2.jsonl)

## Recommended Workflow

The right sequence is still:

1. stabilize retrieval and prompt/context contract
2. benchmark the live RAG system
3. convert high-signal benchmark rows into training data
4. fine-tune
5. run strict base-vs-adapter evaluation
6. test the best adapter inside the real chatbot flow

Why:
- retrieval must provide the right evidence first
- QLoRA should improve evidence usage, not compensate for bad retrieval

## Training Data Format

Each row uses the live document-QA context style:

```json
{
  "id": "qlora-001",
  "source_benchmark_id": "tilon-rise-001",
  "language": "ko",
  "question": "사용자 질문",
  "context": "[Doc: filename.pdf | Page: 3 | Section: Requirements | Lang: ko]\n근거 문맥",
  "answer": "근거 기반 최종 답변"
}
```

Important:
- training and inference now use the Qwen chat template, not an ad hoc raw plain-text wrapper
- context should stay close to live retrieval formatting
- answers should remain concise and grounded

## Install Training Dependencies

```bash
cd /home/tilon/chatbot-karbi
source .venv/bin/activate
pip install -r finetuning/requirements-qlora.txt
```

## Train An Adapter

Example:

```bash
python finetuning/train.py \
  --data finetuning/data/qlora_train_v4.jsonl \
  --model Qwen/Qwen2.5-7B-Instruct \
  --output finetuning/output/qwen25-qlora-v6 \
  --max-seq-len 640 \
  --batch-size 1 \
  --grad-accum 8 \
  --epochs 2 \
  --lora-r 8 \
  --lora-alpha 16 \
  --fp16
```

Recommended for 12GB-class GPUs:
- stop `uvicorn`
- stop `ollama serve`
- keep `batch_size=1`
- use `grad_accum=8`
- use `fp16`
- use `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

## Dry Run

Validate dataset formatting and split without loading full model weights:

```bash
python finetuning/train.py \
  --data finetuning/data/qlora_train_v4.jsonl \
  --output /tmp/tilon-qlora-dryrun \
  --dry-run
```

## Compare Base vs Adapter On One Prompt

```bash
python finetuning/infer_compare.py \
  --question "런케이션 프로그램의 지원조건은 무엇인가요?" \
  --context-file /tmp/context.txt \
  --adapter finetuning/output/qwen25-qlora-v6 \
  --fp16
```

Notes:
- `context.txt` should contain retrieved context in the live `[Doc: ...]` style
- `run_config.json` is reused automatically if present in the adapter folder

## Strict Adapter Evaluation

Small strict eval:

```bash
python finetuning/evaluate_adapter.py \
  --data finetuning/data/qlora_eval_ko_strict_v1.jsonl \
  --adapter finetuning/output/qwen25-qlora-v6 \
  --fp16
```

Harder eval:

```bash
python finetuning/evaluate_adapter.py \
  --data finetuning/data/qlora_eval_ko_strict_v2.jsonl \
  --adapter finetuning/output/qwen25-qlora-v6 \
  --fp16
```

Quick smoke test:

```bash
python finetuning/evaluate_adapter.py \
  --data finetuning/data/qlora_eval_ko_strict_v2.jsonl \
  --adapter finetuning/output/qwen25-qlora-v6 \
  --fp16 \
  --limit 2 \
  --max-new-tokens 96 \
  --output /tmp/tilon_eval_limit2.json
```

The evaluator:
- runs both base and adapter
- checks answer-type behavior and keyword coverage
- flags mixed-language drift and repetition
- writes incremental results

## Current Best Result Summary

### `v3`
- first clearly useful adapter
- proved QLoRA could beat base after chat-template correction

### `v4`
- larger dataset, but did not beat `v3`
- showed that more data alone was not enough

### `v5`
- became the strongest focused adapter at that stage
- improved comparison and not-found behavior

### `v6`
- current best
- improved the previously weak comparison case in strict eval v2
- best candidate for broader product-side testing

## Remaining Weak Spots

Even with `v6`, the workstream is not finished.

Still weak:
- ambiguous bundled-document clarification
- nuanced support/benefit comparison answers
- exact phrase fidelity on strict lookup rows
- some mention-only wording

## Advanced Fine-Tuning Approaches To Consider

These were postponed until the supervised fine-tuning path was proven. They should now be treated as valid next-stage options.

- failure-driven dataset expansion from strict eval misses
- judge-assisted hard-case mining
- targeted synthetic augmentation after real-data coverage is strong
- preference tuning such as DPO/ORPO for:
  - clarification
  - refusal
  - comparison
  - mention-only behavior
- adapter selection or routing by answer type if one adapter does not dominate all cases

## Recommended Next Step

Do not start a new broad training round immediately.

Recommended next move:
1. freeze `v6` as the current best adapter
2. test it inside the real chatbot product flow
3. log real failures
4. decide whether another targeted dataset round is actually needed
