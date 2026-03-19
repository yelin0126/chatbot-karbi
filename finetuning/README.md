# Usama's QLoRA Fine-Tuning Workstream

This folder contains all fine-tuning work. It is independent of the
RAG pipeline but uses the same prompt format.

## Key files (to be created)
- `train.py` — QLoRA training script
- `evaluate.py` — Model evaluation
- `data/` — Training data (JSON format)
- `output/` — Trained LoRA adapters

## Training Data Format

Each sample must match the prompt format from `app/chat/prompts.py`:

```json
{
  "question": "User's question",
  "context": "[Doc: filename.pdf | Page: 3 | Section: Requirements | Lang: ko]\nActual chunk text here...",
  "answer": "Expected high-quality answer with source citation."
}
```

## Critical: The context format MUST match `app/retrieval/retriever.py:format_context()`