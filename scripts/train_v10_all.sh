#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec .venv/bin/python3 finetuning/train.py \
  --data finetuning/data/qlora_train_v10_combined.jsonl \
  --output finetuning/output/qwen25-qlora-v10-all \
  --bf16 \
  --epochs 3 \
  --batch-size 1 \
  --grad-accum 16 \
  --lr 2e-4 \
  --lora-r 16 \
  --lora-alpha 32 \
  --save-steps 50 \
  --logging-steps 5
