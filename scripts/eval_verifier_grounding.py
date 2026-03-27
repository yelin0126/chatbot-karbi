#!/usr/bin/env python3
"""
Evaluate answer grounding / verifier behavior through the live chat pipeline.

Focus:
- final-answer grounding quality
- citation usage and footer presence
- not-found / disclaimer behavior

This complements retrieval-only evals by measuring the post-generation output
that the user actually sees.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from scripts.run_benchmark import (
    BenchmarkScope,
    load_benchmark,
    resolve_scope,
    resolve_scopes,
    score_answer_points,
    score_sources,
)

_HTTP_MODE = False
_SERVER_URL = "http://localhost:8000"


def _lazy_imports():
    global handle_chat
    from app.chat.handlers import handle_chat  # noqa: F811


DEFAULT_EVAL_PATH = "finetuning/data/verifier_grounding_eval_v1.jsonl"
DEFAULT_RESULTS_DIR = "finetuning/results"
INLINE_CITATION_RE = re.compile(r"\[(\d+)\]")
FOOTER_MARKERS = ("\n---\n**출처:**", "\n---\n**Sources:**", "\n---\nSources:")
NOT_FOUND_PATTERNS = [
    "찾을 수 없습니다",
    "찾지 못했습니다",
    "확인되지 않습니다",
    "충분히 확인되지 않습니다",
    "관련된 내용이 포함되어 있지 않은",
    "i couldn't find relevant content",
    "don't contain enough information",
    "not covered in the provided documents",
    "doesn't appear to contain information related",
]
DISCLAIMER_PATTERNS = [
    "정확히 일치하지 않을 수 있습니다",
    "문서를 직접 확인해 주세요",
    "may not exactly match",
    "please check the document",
]


def strip_citation_footer(answer: str) -> str:
    text = answer or ""
    for marker in FOOTER_MARKERS:
        if marker in text:
            return text.split(marker, 1)[0].strip()
    return text.strip()


def has_citation_footer(answer: str) -> bool:
    return any(marker in (answer or "") for marker in FOOTER_MARKERS)


def count_inline_citations(answer: str) -> int:
    body = strip_citation_footer(answer or "")
    return len(INLINE_CITATION_RE.findall(body))


def detect_not_found(answer: str) -> bool:
    lower = (answer or "").lower()
    return any(pattern.lower() in lower for pattern in NOT_FOUND_PATTERNS)


def detect_disclaimer(answer: str) -> bool:
    lower = (answer or "").lower()
    return any(pattern.lower() in lower for pattern in DISCLAIMER_PATTERNS)


def resolve_eval_scope(item: dict[str, Any]) -> BenchmarkScope:
    preferred_source_type = str(item.get("scope_source_type") or "").strip() or None
    document_sources = [
        str(source).strip()
        for source in item.get("document_sources", []) or []
        if str(source).strip()
    ]
    document_source = str(item.get("document_source") or "").strip() or None

    if document_sources:
        return resolve_scopes(
            document_source,
            document_sources=document_sources,
            preferred_source_type=preferred_source_type,
        )

    if document_source:
        return resolve_scope(document_source, preferred_source_type=preferred_source_type)

    if preferred_source_type:
        return BenchmarkScope(
            source=None,
            doc_id=None,
            source_type=preferred_source_type,
            sources=[],
            doc_ids=[],
            source_types=[preferred_source_type],
            resolved=True,
            resolution_note=f"unscoped source_type='{preferred_source_type}'",
        )

    return BenchmarkScope(
        source=None,
        doc_id=None,
        source_type=None,
        sources=[],
        doc_ids=[],
        source_types=[],
        resolved=True,
        resolution_note="global unscoped query",
    )


def run_case(item: dict[str, Any], model: str | None) -> dict[str, Any]:
    scope = resolve_eval_scope(item)
    if not scope.resolved:
        return {
            "scope": asdict(scope),
            "resolved": False,
            "answer": "",
            "mode": "unresolved",
            "sources": [],
        }

    if _HTTP_MODE:
        import requests

        payload: dict[str, Any] = {"message": item["question"]}
        if scope.sources:
            payload["active_sources"] = scope.sources
        if scope.doc_ids:
            payload["active_doc_ids"] = scope.doc_ids
        if scope.source:
            payload["active_source"] = scope.source
        if scope.doc_id:
            payload["active_doc_id"] = scope.doc_id
        if scope.source_type:
            payload["active_source_type"] = scope.source_type
        if model:
            payload["model"] = model

        try:
            resp = requests.post(
                f"{_SERVER_URL}/chat",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=180,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            return {
                "scope": asdict(scope),
                "resolved": True,
                "answer": f"[HTTP ERROR] {exc}",
                "mode": "error",
                "sources": [],
            }

        raw_sources = data.get("sources", [])
        sources = [s if isinstance(s, dict) else vars(s) for s in raw_sources]
        return {
            "scope": asdict(scope),
            "resolved": True,
            "answer": data.get("answer", ""),
            "mode": data.get("mode", ""),
            "sources": sources,
        }

    result = handle_chat(
        item["question"],
        history=[],
        model=model,
        active_source=scope.source,
        active_doc_id=scope.doc_id,
        active_source_type=scope.source_type,
        active_sources=scope.sources,
        active_doc_ids=scope.doc_ids,
    )
    return {
        "scope": asdict(scope),
        "resolved": True,
        "answer": result.get("answer", ""),
        "mode": result.get("mode", ""),
        "sources": result.get("sources", []),
    }


def score_grounding(item: dict[str, Any], answer_result: dict[str, Any]) -> dict[str, Any]:
    answer = answer_result.get("answer", "")
    should_answer = bool(item.get("should_answer_from_docs", True))
    inline_count = count_inline_citations(answer)
    footer = has_citation_footer(answer)
    not_found = detect_not_found(answer)
    disclaimer = detect_disclaimer(answer)
    min_inline = int(item.get("min_inline_citations", 0))

    source_scoring = score_sources(
        answer_result.get("sources", []),
        item.get("expected_sources", []),
    )
    answer_scoring = score_answer_points(
        answer,
        item.get("expected_answer_points", []),
    )

    if should_answer:
        passed = (
            source_scoring["expected_source_recall"] == 1.0
            and answer_scoring["expected_points_recall"] == 1.0
            and footer
            and inline_count >= min_inline
            and not not_found
        )
    else:
        passed = not_found and inline_count == 0

    return {
        "should_answer_from_docs": should_answer,
        "inline_citation_count": inline_count,
        "has_citation_footer": footer,
        "not_found_detected": not_found,
        "disclaimer_detected": disclaimer,
        "source_scoring": source_scoring,
        "answer_scoring": answer_scoring,
        "passed": passed,
    }


def summarize_results(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    items = list(rows)
    answered = [row for row in items if row["grounding"]["should_answer_from_docs"]]
    refusals = [row for row in items if not row["grounding"]["should_answer_from_docs"]]

    def _avg(values: list[float]) -> float | None:
        return round(sum(values) / len(values), 3) if values else None

    return {
        "total_rows": len(items),
        "answerable_rows": len(answered),
        "not_found_rows": len(refusals),
        "avg_source_recall": _avg([row["grounding"]["source_scoring"]["expected_source_recall"] for row in items]),
        "avg_answer_point_recall": _avg([row["grounding"]["answer_scoring"]["expected_points_recall"] for row in items]),
        "overall_pass_count": sum(1 for row in items if row["grounding"]["passed"]),
        "footer_rate_answerable": _avg([1.0 if row["grounding"]["has_citation_footer"] else 0.0 for row in answered]),
        "inline_citation_rate_answerable": _avg([1.0 if row["grounding"]["inline_citation_count"] > 0 else 0.0 for row in answered]),
        "disclaimer_rate_answerable": _avg([1.0 if row["grounding"]["disclaimer_detected"] else 0.0 for row in answered]),
        "correct_not_found_rate": _avg([1.0 if row["grounding"]["not_found_detected"] else 0.0 for row in refusals]),
        "disclaimer_rate": _avg([1.0 if row["grounding"]["disclaimer_detected"] else 0.0 for row in items]),
        "category_counts": dict(Counter(row.get("category", "") for row in items)),
    }


def main() -> int:
    global _HTTP_MODE, _SERVER_URL

    parser = argparse.ArgumentParser(description="Evaluate answer grounding / verifier behavior.")
    parser.add_argument("--path", default=DEFAULT_EVAL_PATH, help="Evaluation JSONL path")
    parser.add_argument("--model", default=None, help="Override model for answer generation")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of rows")
    parser.add_argument("--id", dest="ids", action="append", default=[], help="Run only these IDs")
    parser.add_argument("--output-dir", default=DEFAULT_RESULTS_DIR, help="Where to write results")
    parser.add_argument(
        "--http",
        action="store_true",
        default=False,
        help="Call the running server's /chat endpoint instead of loading models in-process",
    )
    parser.add_argument(
        "--server-url",
        default="http://localhost:8000",
        help="Server base URL for --http mode (default: http://localhost:8000)",
    )
    args = parser.parse_args()

    _HTTP_MODE = args.http
    _SERVER_URL = args.server_url.rstrip("/")

    if not _HTTP_MODE:
        _lazy_imports()
    else:
        import requests

        try:
            requests.get(f"{_SERVER_URL}/health", timeout=5).raise_for_status()
            print(f"Server reachable at {_SERVER_URL}")
        except Exception as exc:
            print(f"ERROR: cannot reach server at {_SERVER_URL}: {exc}")
            return 1

    eval_path = Path(args.path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_benchmark(eval_path)
    if args.ids:
        wanted = set(args.ids)
        rows = [row for row in rows if row.get("id") in wanted]
    if args.limit:
        rows = rows[: args.limit]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = output_dir / f"verifier_grounding_results_{timestamp}.jsonl"
    summary_path = output_dir / f"verifier_grounding_summary_{timestamp}.json"

    results: list[dict[str, Any]] = []
    with result_path.open("w", encoding="utf-8") as handle:
        for idx, item in enumerate(rows, start=1):
            print(f"[{idx}/{len(rows)}] {item['id']} — {item['question'][:60]}…", flush=True)
            answer_result = run_case(item, args.model)
            row_result = {
                "id": item["id"],
                "category": item.get("category", ""),
                "language": item.get("language", ""),
                "document_source": item.get("document_source"),
                "question": item["question"],
                "answer_result": answer_result,
                "grounding": score_grounding(item, answer_result),
                "notes": item.get("notes", ""),
            }
            handle.write(json.dumps(row_result, ensure_ascii=False) + "\n")
            results.append(row_result)

    summary = summarize_results(results)
    summary["eval_path"] = str(eval_path)
    summary["result_path"] = str(result_path)
    summary["http_mode"] = _HTTP_MODE
    summary["server_url"] = _SERVER_URL if _HTTP_MODE else None
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("VERIFIER GROUNDING EVAL COMPLETE")
    print(f"results: {result_path}")
    print(f"summary: {summary_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
