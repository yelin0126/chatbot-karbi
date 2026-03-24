"""
Live pipeline regression test for large-PDF deterministic shortcuts.

Usage:
    python finetuning/test_pipeline_eval.py \\
        --source "제칠일안식일예수재림교_기본교리.pdf" \\
        [--doc-id <doc_id>] \\
        [--eval finetuning/data/pipeline_eval_large_pdf_v1.jsonl] \\
        [--url http://localhost:8000] \\
        [--bucket 1,2,3]          # comma-separated buckets to run; default all
        [--verbose]               # print full answer for each case
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import requests

# ── Defaults ────────────────────────────────────────────────────────────────
DEFAULT_EVAL = Path(__file__).parent / "data" / "pipeline_eval_large_pdf_v1.jsonl"
DEFAULT_URL = "http://localhost:8000"


def _check(answer: str, expected: list[str], forbidden: list[str]) -> tuple[bool, list[str]]:
    """Return (pass, failed_reasons)."""
    reasons = []
    answer_lower = answer.lower()
    for kw in expected:
        if kw.lower() not in answer_lower:
            reasons.append(f"MISSING expected '{kw}'")
    for kw in forbidden:
        if kw.lower() in answer_lower:
            reasons.append(f"HIT forbidden '{kw}'")
    return (len(reasons) == 0), reasons


def run(
    source: Optional[str],
    doc_id: Optional[str],
    eval_path: Path,
    base_url: str,
    buckets: Optional[set[int]],
    verbose: bool,
) -> int:
    cases = []
    with open(eval_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))

    if buckets:
        cases = [c for c in cases if c.get("bucket") in buckets]

    if not cases:
        print("No test cases matched the selected buckets.")
        return 1

    passed = 0
    failed = 0
    errors = 0

    print(f"\n{'='*72}")
    print(f"  Pipeline eval: {eval_path.name}")
    print(f"  Source: {source or '(none)'}{f'  doc_id={doc_id}' if doc_id else ''}")
    print(f"  Cases : {len(cases)}")
    print(f"{'='*72}\n")

    for case in cases:
        cid = case["id"]
        bucket = case.get("bucket", "?")
        question = case["question"]
        expected = case.get("expected_keywords", [])
        forbidden = case.get("forbidden_keywords", [])
        notes = case.get("notes", "")

        payload = {
            "message": question,
            "history": [],
        }
        if source:
            payload["active_source"] = source
        if doc_id:
            payload["active_doc_id"] = doc_id

        try:
            resp = requests.post(f"{base_url}/chat", json=payload, timeout=90)
            resp.raise_for_status()
            data = resp.json()
            answer = data.get("answer", "")
        except Exception as exc:
            print(f"  [{cid}] B{bucket} ERROR: {exc}")
            errors += 1
            continue

        ok, reasons = _check(answer, expected, forbidden)
        status = "PASS" if ok else "FAIL"
        marker = "✓" if ok else "✗"

        if ok:
            passed += 1
            print(f"  {marker} [{cid}] B{bucket}  {status}  — {question[:60]}")
        else:
            failed += 1
            print(f"  {marker} [{cid}] B{bucket}  {status}  — {question[:60]}")
            for r in reasons:
                print(f"         → {r}")
            if notes:
                print(f"         note: {notes}")

        if verbose:
            # Indent and truncate long answers for readability
            preview = answer.replace("\n", " ")[:200]
            print(f"         answer: {preview}")

    total = passed + failed + errors
    print(f"\n{'─'*72}")
    print(f"  Result: {passed}/{total} passed  ({failed} failed, {errors} errors)")
    print(f"{'─'*72}\n")

    return 0 if (failed == 0 and errors == 0) else 1


def main():
    parser = argparse.ArgumentParser(description="Live pipeline regression test.")
    parser.add_argument("--source", default=None, help="active_source filename (no path)")
    parser.add_argument("--doc-id", default=None, help="active_doc_id UUID")
    parser.add_argument("--eval", default=str(DEFAULT_EVAL), help="JSONL eval file path")
    parser.add_argument("--url", default=DEFAULT_URL, help="API base URL")
    parser.add_argument(
        "--bucket",
        default=None,
        help="Comma-separated bucket numbers to run, e.g. '1,2'. Default: all.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print answer previews")
    args = parser.parse_args()

    buckets = None
    if args.bucket:
        buckets = {int(b.strip()) for b in args.bucket.split(",")}

    if not args.source and not args.doc_id:
        print(
            "WARNING: no --source or --doc-id specified.\n"
            "  The API will try to answer without document scope, which will likely fail.\n"
            "  Pass --source 'filename.pdf' to scope queries to the right document.\n"
        )

    sys.exit(
        run(
            source=args.source,
            doc_id=args.doc_id,
            eval_path=Path(args.eval),
            base_url=args.url.rstrip("/"),
            buckets=buckets,
            verbose=args.verbose,
        )
    )


if __name__ == "__main__":
    main()
