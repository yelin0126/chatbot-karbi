#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
SERVER_URL="${SERVER_URL:-http://localhost:8000}"

echo "== Pre-v10 baseline: routing =="
"$PYTHON_BIN" scripts/eval_query_routing.py --path finetuning/data/query_policy_eval_v3.jsonl

echo
echo "== Pre-v10 baseline: structure retrieval =="
"$PYTHON_BIN" scripts/eval_structure_retrieval.py --path finetuning/data/structure_retrieval_eval_v1.jsonl
"$PYTHON_BIN" scripts/eval_structure_retrieval.py --path finetuning/data/structure_retrieval_eval_v2.jsonl

echo
echo "== Pre-v10 baseline: grounding/verifier (HTTP mode) =="
"$PYTHON_BIN" scripts/eval_verifier_grounding.py --http --server-url "$SERVER_URL" --path finetuning/data/verifier_grounding_eval_v1.jsonl
"$PYTHON_BIN" scripts/eval_verifier_grounding.py --http --server-url "$SERVER_URL" --path finetuning/data/verifier_grounding_eval_v2.jsonl
"$PYTHON_BIN" scripts/eval_verifier_grounding.py --http --server-url "$SERVER_URL" --path finetuning/data/verifier_grounding_eval_unscoped_v1.jsonl
"$PYTHON_BIN" scripts/eval_verifier_grounding.py --http --server-url "$SERVER_URL" --path finetuning/data/verifier_grounding_eval_external_v1.jsonl
"$PYTHON_BIN" scripts/eval_verifier_grounding.py --http --server-url "$SERVER_URL" --path finetuning/data/verifier_grounding_eval_tables_v1.jsonl

echo
echo "Pre-v10 baseline run complete."
