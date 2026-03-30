# QLoRA Fine-Tuning Workstream

This folder contains the local fine-tuning and evaluation workflow for improving the answer model used on top of retrieved document context.

Related docs:
- [README.md](/home/tilon/chatbot-karbi/README.md)
- [ARCHITECTURE.md](/home/tilon/chatbot-karbi/ARCHITECTURE.md)
- [finetuning/data/README.md](/home/tilon/chatbot-karbi/finetuning/data/README.md)

## Current Status

This workstream is active and deployed. The **training track is now the primary improvement path** after the pipeline track was frozen (2026-03-30).

What is working now:
- local QLoRA training for Qwen2.5 7B (4-bit NF4)
- strict base-vs-adapter evaluation
- offline-friendly loading for local model files
- chat-template-correct training and inference
- multiple dataset iterations with measured comparisons
- v9 adapter deployed and serving live traffic
- post-Phase 10 routing and retrieval eval assets now exist for better failure mining
- grounding/verifier eval assets now exist for answer-faithfulness failure mining
- newer real-PDF grounding evals now expose OCR/table-heavy exact-answer failure modes before training
- the hard real-PDF grounding suite now has a stable green checkpoint (`verifier_grounding_eval_v2 = 10/10`)
- an unscoped library grounding suite now exists for broader corpus-wide verification (`verifier_grounding_eval_unscoped_v1`)
- the latest unscoped library checkpoint on March 27, 2026 is `22/24`, with `avg_source_recall = 0.917`, `avg_answer_point_recall = 0.875`, and `correct_not_found_rate = 1.0`
- Phase 12 pipeline fixes deployed (2026-03-30): pipe-table extractor, overlap bypass, presence handler, full-doc chain
- post-pipeline eval: external 3/10, tables 2/8 — pipeline track now frozen

Current deployed adapter: `qwen25-qlora-v9`

Post-pipeline-freeze status:
- **13 remaining eval failures are all model generation quality** (source_recall=1.0 everywhere)
- Chinese drift, hallucination, incomplete extraction, narrative table reading
- ~~prompt misalignment between `train.py` and `prompting.py`~~: **FIXED (2026-03-30)** — prompts now match exactly

Adapter history:

| Adapter | Samples | Notes |
|---------|---------|-------|
| v1–v4 | 44–60 | Iterative experiments |
| v5 | 90 | First broadly convincing general adapter |
| v6 | — | Best score on strict Korean eval |
| v7 | — | Experimental |
| v8 | 115 | **Retired** — ~78% church/domain-specific data |
| v9 | 135 | **Current** — 100% general-purpose |

v9 training details:
- data: `qlora_train_v9_combined.jsonl` (v5 90 + 45 new general samples)
- topics: contracts, policies, tech docs, manuals, reports, greetings
- loss: 1.097 → 0.377, 3 epochs, ~4m38s (trained on RTX 4070 12GB; current GPU is A6000 48GB)
- zero domain-specific content

Best measured eval result (v6 on strict Korean set):
- eval set: [qlora_eval_ko_strict_v2.jsonl](/home/tilon/chatbot-karbi/finetuning/data/qlora_eval_ko_strict_v2.jsonl)
- base avg: `0.244`
- adapter avg: `0.4173`
- adapter wins: `12/15`, base wins: `0/15`, ties: `3/15`

## Key Files

- [train.py](/home/tilon/chatbot-karbi/finetuning/train.py)
- [evaluate_adapter.py](/home/tilon/chatbot-karbi/finetuning/evaluate_adapter.py)
- [infer_compare.py](/home/tilon/chatbot-karbi/finetuning/infer_compare.py)
- [requirements-qlora.txt](/home/tilon/chatbot-karbi/finetuning/requirements-qlora.txt)

Main datasets:
- [qlora_train_v5.jsonl](/home/tilon/chatbot-karbi/finetuning/data/qlora_train_v5.jsonl) — 90 diverse general samples
- [qlora_train_v9.jsonl](/home/tilon/chatbot-karbi/finetuning/data/qlora_train_v9.jsonl) — 45 new general samples
- [qlora_train_v9_combined.jsonl](/home/tilon/chatbot-karbi/finetuning/data/qlora_train_v9_combined.jsonl) — 135 combined (used to train v9)
- [qlora_eval_ko_strict_v1.jsonl](/home/tilon/chatbot-karbi/finetuning/data/qlora_eval_ko_strict_v1.jsonl)
- [qlora_eval_ko_strict_v2.jsonl](/home/tilon/chatbot-karbi/finetuning/data/qlora_eval_ko_strict_v2.jsonl)

Retired (do not use for training):
- `qlora_train_v8.jsonl` — 25 samples, all church-specific
- `qlora_train_v8_combined.jsonl` — 115 samples, ~78% church-specific

## Recommended Workflow

The right sequence remains:

1. stabilize routing, retrieval, and prompt/context contract
2. benchmark the live RAG system on real queries and real PDFs
3. benchmark structure-aware retrieval and answer grounding separately
4. convert high-signal benchmark failures into training data (chosen + rejected pairs)
5. fine-tune (SFT first, then DPO/ORPO for preference alignment)
6. run strict base-vs-adapter evaluation
7. test the best adapter inside the real chatbot flow

Why this order matters:
- retrieval must provide the right evidence first
- QLoRA should improve evidence usage, not compensate for bad retrieval or weak grounding gates
- preference tuning (DPO/ORPO) requires real failure data to be meaningful
- synthetic augmentation should only expand coverage that real docs already partially cover

Current recommendation:
- pipeline track is frozen — all extractable pipeline fixes are deployed
- **training track is now primary**
- ~~**Step 1 (prerequisite):** align train/serve prompt templates~~ — **DONE (2026-03-30)**. **Current active step: Step 2 — mine 13 model-side failures into labeled training rows.**
- **Step 2:** mine 13 confirmed model-side failures into labeled v11 training rows (exclude ingestion-broken ext-003, label by failure type)
- **Step 3:** SSFO (Self-Supervised Faithfulness Optimization) — generate chosen (with context) and rejected (without context) responses from same model, 400-500 preference pairs
- **Step 4:** RAFT-format training — oracle document + k distractor documents per example, 80/20 split, chain-of-thought with inline citations
- **Step 5:** general instruction mix + two-stage SFT+SimPO (DPO variant, saves ~4-5GB VRAM)
- use the now-stable routing / retrieval / grounding suites as regression guards
- do NOT pursue: Qwen3, 14B, VLM hybrid on digital PDFs — all confirmed non-productive (2026-03-30)
- `v10` adapters saved as artifacts but not deployed; v9 remains the production adapter

Next adapter guidance:
- v9 is the current deployed baseline — general-purpose SFT, no domain content
- v10 experiment cycle completed 2026-03-30 — all approaches flat:
  - Tier A (168 rows): no improvement over v9 baseline
  - Tier A+B (170 rows): slight regression
  - 14B: byte-identical outputs to 7B on all failing rows
  - Token 4096→8192: trimming eliminated, pass counts unchanged
  - VLM hybrid: never fired on digital PDFs
- Phase 12 pipeline fixes completed 2026-03-30 — pipeline track frozen:
  - Pipe-table extractor: deterministic row/field extraction for pipe-delimited content
  - Overlap bypass: adjacent-token pair check prevents scattered-token false matches
  - Presence handler: compound concept matching (all terms must co-occur in one chunk)
  - Full-doc chain: pipe extractor added for small docs ≤15 chunks
  - Result: external 3/10 (up from 2/10), tables 2/8 (correctly scored)
- **v11 plan: prompt alignment → SSFO preference pairs → RAFT-format training → SFT+SimPO**
- model: stay on `Qwen/Qwen2.5-7B-Instruct`
  - reason 1: the current production stack, prompts, and eval history are already anchored to Qwen2.5
  - reason 2: 14B produced identical outputs — model size is not the bottleneck
  - reason 3: changing to Qwen3 at the same time as training changes would make attribution harder
- when to test Qwen3:
  - only after v11 training track is evaluated
  - as a separate branch experiment with the same dataset and evaluation suite

Current model-side conclusion:
- the v10 experiment cycle (2026-03-30) exhausted model-size, SFT volume, token expansion, and VLM hybrid approaches — all flat
- 14B produces byte-identical outputs to 7B on failing rows, proving model capacity is not the bottleneck
- the move from low unscoped grounding to `22/24` came mainly from RAG/scoping/extraction/deterministic-answer fixes
- Phase 12 pipeline fixes added 1 new pass (ext-010 correct refusal) and correctly reclassified table scores
- the remaining 13 failures are model generation quality: Chinese drift, hallucination, incomplete extraction, narrative table reading
- **next: align prompts, then SSFO + RAFT-format training on confirmed model-side failures**

Practical recommendation:
- ~~Step 1: align train/serve prompt templates~~ — **DONE (2026-03-30)**
- Step 2: mine 13 failures into labeled training rows by failure type
- Step 3-5: SSFO → RAFT → SFT+SimPO two-stage training
- only consider model-family changes after v11 training track is evaluated
- do NOT pursue: Qwen3, 14B, broad SFT expansion, VLM hybrid on digital PDFs — all confirmed flat


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
- keep the schema as `question/context/answer` for this repo; do not switch the main training workflow to a generic `instruction/input/output` format unless the trainer is changed too
- the current priority is PDF-grounded chat, not generic chat or image generation

## Install Training Dependencies

```bash
cd /home/tilon/chatbot-karbi
source .venv/bin/activate
pip install -r finetuning/requirements-qlora.txt
```

## Train An Adapter

Example:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python finetuning/train.py \
  --data finetuning/data/qlora_train_v9_combined.jsonl \
  --model Qwen/Qwen2.5-7B-Instruct \
  --output finetuning/output/qwen25-qlora-v10 \
  --max-seq-len 768 \
  --batch-size 1 \
  --grad-accum 16 \
  --epochs 3 \
  --lora-r 16 \
  --lora-alpha 32 \
  --fp16
```

Recommended for A6000 48GB:
- `batch_size=4` (up to 8)
- `grad_accum=4`
- `bf16`
- `max_seq_len=2048` (up to 4096)

For 12GB-class GPUs (legacy):
- stop `uvicorn` and `ollama serve` first
- `batch_size=1`, `grad_accum=16`, `fp16`
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

### `v9`
- current deployed adapter
- chosen because it is general-purpose and free of the domain bias that affected `v8`
- the right baseline for future failure-mined data collection, even though `v6` remains the strongest legacy strict-eval snapshot

## Remaining Weak Spots

Even with the current adapters, the workstream is not finished.

Still weak:
- ambiguous bundled-document clarification
- nuanced support/benefit comparison answers
- exact phrase fidelity on strict lookup rows
- some mention-only wording
- some long narrative upload summary / extraction behavior before evidence selection is tightened further
- unscoped library comparison and explanation rows are still weaker than scoped-document QA

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

Pipeline track is frozen. Training track is now primary.

Recommended next steps:
1. ~~**align train/serve prompt templates**~~ — **DONE (2026-03-30)**: `train.py`'s `DEFAULT_SYSTEM_PROMPT` now matches `prompting.py`'s `COMPACT_SYSTEM_PROMPT` exactly.
2. mine 13 confirmed model-side failures (exclude ingestion-broken ext-003) into labeled v11 training rows by failure type: `chinese_drift`, `hallucination`, `incomplete_extraction`, `narrative_table_reading`
3. generate 400-500 SSFO preference pairs (chosen with context, rejected without)
4. assemble RAFT-format training data (oracle + distractor documents, 80/20 split)
5. two-stage training: SFT on combined data, then SimPO preference tuning
6. evaluate against all existing regression suites

PaddleOCR status: current venv verified at `paddlepaddle-gpu==3.3.0` on 2026-03-30; earlier 3.3.1 notes were stale, so scanned-PDF OCR should be treated as environment-sensitive until re-verified.

Pre-`v10` baseline helper:

```bash
bash scripts/run_pre_v10_baseline.sh
```
