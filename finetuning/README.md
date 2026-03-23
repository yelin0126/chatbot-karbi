# Usama's QLoRA Fine-Tuning Workstream

This folder contains all fine-tuning work. It is independent of the
RAG pipeline but uses the same prompt format.

System context:
- Main product overview: [README.md](/home/tilon/chatbot-karbi/README.md)
- Short backend architecture: [ARCHITECTURE.md](/home/tilon/chatbot-karbi/ARCHITECTURE.md)

## Key files (to be created)
- `train.py` — QLoRA training script
- `evaluate.py` — Model evaluation
- `data/benchmark_template.jsonl` — Retrieval + answer benchmark seed
- `data/README.md` — Benchmark schema and workflow
- `data/qlora_train_v1.jsonl` — First supervised training scaffold linked to benchmark rows
- `output/` — Trained LoRA adapters

## Recommended Order

Do not start QLoRA first.

Recommended sequence:
1. build benchmark questions from real Tilon documents
2. evaluate the current RAG system
3. tune retrieval / confidence behavior
4. freeze prompt + context format
5. then create the QLoRA dataset

Why:
- RAG must provide the right evidence first
- QLoRA should improve answer behavior on top of stable evidence retrieval

## Training Data Format

Each sample must match the prompt format from `app/chat/prompts.py`:

```json
{
  "id": "qlora-001",
  "source_benchmark_id": "tilon-rise-001",
  "language": "ko",
  "question": "User's question",
  "context": "[Doc: filename.pdf | Page: 3 | Section: Requirements | Lang: ko]\nActual chunk text here...",
  "answer": "Expected high-quality answer with source citation."
}
```

## Critical: The context format MUST match `app/retrieval/retriever.py:format_context()`

The starter file `data/qlora_train_v1.jsonl` is a manual scaffold, not a final large training set.
Its purpose is to:

1. anchor the answer style you want after RAG is stable
2. preserve difficult benchmark cases such as:
   - bilingual grounded answers
   - bundled upload clarification
   - clean not-found refusals
3. make later data expansion consistent across teammates

## Validation

Use this to validate the benchmark file:

```bash
cd /home/tilon/chatbot-karbi
source .venv/bin/activate
python scripts/validate_benchmark.py --path finetuning/data/benchmark_template.jsonl
```

## Run Benchmark

Run the benchmark against the current system:

```bash
cd /home/tilon/chatbot-karbi
source .venv/bin/activate
python scripts/run_benchmark.py --path finetuning/data/benchmark_template.jsonl --mode both
```

Useful variants:

```bash
python scripts/run_benchmark.py --mode retrieval
python scripts/run_benchmark.py --mode answer
python scripts/run_benchmark.py --id tilon-001 --id tilon-004
```

Outputs are written to:

- `finetuning/results/benchmark_results_*.jsonl`
- `finetuning/results/benchmark_summary_*.json`

## Suggested Next Step For QLoRA

After the benchmark baseline is frozen:

1. expand `data/qlora_train_v1.jsonl` from the strongest benchmark rows
2. keep contexts close to the live retrieved format
3. prefer high-signal grounded answers over long stylistic responses
4. include both:
   - positive grounded answers
   - clarification / refusal answers

## Training Scaffold

The repo now includes:

- `finetuning/train.py`
- `finetuning/requirements-qlora.txt`

This script:

1. reads `data/qlora_train_v1.jsonl`
2. formats each sample to match the current document-QA prompt style
3. trains a LoRA adapter on top of a frozen base model
4. saves the adapter and a small `run_config.json`

Install training dependencies first:

```bash
cd /home/tilon/chatbot-karbi
source .venv/bin/activate
pip install -r finetuning/requirements-qlora.txt
```

Example training command:

```bash
python finetuning/train.py \
  --data finetuning/data/qlora_train_v1.jsonl \
  --model Qwen/Qwen2.5-7B-Instruct \
  --output finetuning/output/qwen25-qlora-v1
```

## Compare Base vs Adapter

After training an adapter, compare the base model and the adapter on the
same question:

```bash
python finetuning/infer_compare.py \
  --question "런케이션 프로그램의 지원조건은 무엇인가요?" \
  --context-file /tmp/context.txt \
  --adapter finetuning/output/qwen25-qlora-v1
```

Notes:

- `--context-file` should contain retrieved context in the same `[Doc: ...]`
  format used by the live RAG system
- if `run_config.json` exists in the adapter directory, the script will reuse
  the recorded base model automatically
- add `--fp16` on modest GPUs if needed

## Safe First Run

For a conservative first launch on a modest local GPU, use:

```bash
bash finetuning/run_first_qlora.sh
```

This preset keeps the run safer by:

- using 4-bit loading by default
- reducing `--max-seq-len` to `1024`
- using `--batch-size 1`
- using `--grad-accum 8`
- using `--fp16` instead of assuming bf16 support
- using only `2` epochs for the first pass
- enabling `PYTORCH_ALLOC_CONF=expandable_segments:True`

Before running the first real training job on a 12GB-class GPU, stop:

- `uvicorn`
- `ollama serve`
- any other Python or GPU-heavy process

If you only want to validate formatting and dataset split without loading model
weights, run:

```bash
python finetuning/train.py \
  --data finetuning/data/qlora_train_v1.jsonl \
  --output /tmp/tilon-qlora-dryrun \
  --dry-run
```

Important:

- the current dataset is still very small, so this script is a workflow scaffold, not a final training recipe
- serious QLoRA training should start only after `qlora_train_v1.jsonl` is expanded substantially
