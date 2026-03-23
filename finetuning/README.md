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
