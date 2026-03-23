#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -d ".venv" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

echo "Starting first QLoRA run..."
echo "Important: stop uvicorn, Ollama, and any other GPU-heavy process before training."
echo "This 7B QLoRA run expects the GPU to be mostly free."

export PYTORCH_ALLOC_CONF=expandable_segments:True

python finetuning/train.py \
  --data finetuning/data/qlora_train_v1.jsonl \
  --model Qwen/Qwen2.5-7B-Instruct \
  --output finetuning/output/qwen25-qlora-v1 \
  --max-seq-len 1024 \
  --epochs 2 \
  --batch-size 1 \
  --grad-accum 8 \
  --eval-ratio 0.15 \
  --logging-steps 1 \
  --save-steps 25 \
  --fp16
