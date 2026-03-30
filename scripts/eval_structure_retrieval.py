#!/usr/bin/env python3
"""
Focused Phase 10C evaluation for structure-aware retrieval and grounding.

Runs the real `handle_chat(...)` path, then scores:
- source/page recall
- answer-point recall

This is intentionally narrower than the main benchmark runner: it is meant to
catch regressions in article/chapter/section retrieval after retrieval-flow
changes.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# This eval is meant for local, cached models. Avoid long network retry loops.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from app.core.document_registry import list_documents


DEFAULT_EVAL_PATH = "finetuning/data/structure_retrieval_eval_v1.jsonl"
DEFAULT_RESULTS_DIR = "finetuning/results"
_ANSWER_STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "into", "only",
    "문서", "내용", "관련", "대한", "해야", "합니다", "한다", "되어야", "있다", "있는",
    "또는", "포함", "언급", "기반", "답해야", "좋음", "점", "것", "수", "로", "을", "를",
}
handle_chat = None
_build_chat_state = None
_resolve_scope_stage = None
_run_retrieval_stage = None


def _lazy_imports() -> None:
    global handle_chat, _build_chat_state, _resolve_scope_stage, _run_retrieval_stage
    from app.chat.handlers import (  # noqa: WPS442
        handle_chat as _handle_chat,
        _build_chat_state as _state_builder,
        _resolve_scope_stage as _scope_stage,
        _run_retrieval_stage as _retrieval_stage,
    )
    handle_chat = _handle_chat
    _build_chat_state = _state_builder
    _resolve_scope_stage = _scope_stage
    _run_retrieval_stage = _retrieval_stage


@dataclass
class EvalScope:
    source: str | None
    doc_id: str | None
    source_type: str | None
    resolved: bool
    resolution_note: str
    sources: list[str]
    doc_ids: list[str]


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            raw = raw.strip()
            if not raw:
                continue
            obj = json.loads(raw)
            obj["_line_no"] = line_no
            rows.append(obj)
    return rows


def resolve_scope(document_source: str) -> EvalScope:
    entries = [doc for doc in list_documents() if doc.get("source") == document_source]
    if not entries:
        return EvalScope(
            source=document_source,
            doc_id=None,
            source_type=None,
            resolved=False,
            resolution_note="document source not found in registry",
            sources=[],
            doc_ids=[],
        )

    best = sorted(
        entries,
        key=lambda item: (0 if item.get("source_type") == "library" else 1, str(item.get("updated_at") or "")),
    )[0]
    return EvalScope(
        source=best.get("source"),
        doc_id=best.get("doc_id"),
        source_type=best.get("source_type"),
        resolved=True,
        resolution_note="matched exact source",
        sources=[best.get("source")] if best.get("source") else [],
        doc_ids=[best.get("doc_id")] if best.get("doc_id") else [],
    )


def normalize_text(text: str) -> str:
    return " ".join((text or "").lower().split())


def _extract_keywords(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+|[가-힣]{2,}", normalize_text(text))
    return [tok for tok in tokens if tok not in _ANSWER_STOPWORDS]


def _point_matches(answer: str, point: str) -> bool:
    normalized_answer = normalize_text(answer)
    normalized_point = normalize_text(point)
    if normalized_point and normalized_point in normalized_answer:
        return True

    keywords = _extract_keywords(point)
    if not keywords:
        return False

    hits = sum(1 for keyword in keywords if keyword in normalized_answer)
    if len(keywords) <= 2:
        return hits == len(keywords)
    return hits >= max(2, len(keywords) - 1)


def score_answer_points(answer: str, expected_points: list[str]) -> dict[str, Any]:
    hits = []
    misses = []
    for point in expected_points:
        if _point_matches(answer, point):
            hits.append(point)
        else:
            misses.append(point)

    total = len(expected_points)
    recall = len(hits) / total if total else 1.0
    return {
        "expected_points_total": total,
        "expected_points_hit": len(hits),
        "expected_points_recall": round(recall, 3),
        "point_hits": hits,
        "point_misses": misses,
    }


def score_sources(actual_sources: list[dict[str, Any]], expected_sources: list[dict[str, Any]]) -> dict[str, Any]:
    if not expected_sources:
        return {
            "expected_source_total": 0,
            "expected_source_hit": 0,
            "expected_source_recall": 1.0,
            "source_hits": [],
            "source_misses": [],
        }

    hits = []
    misses = []
    for expected in expected_sources:
        source = expected.get("source")
        page = expected.get("page")
        matched = any(
            actual.get("source") == source and (page is None or actual.get("page") == page)
            for actual in actual_sources
        )
        if matched:
            hits.append(expected)
        else:
            misses.append(expected)

    total = len(expected_sources)
    recall = len(hits) / total if total else 1.0
    return {
        "expected_source_total": total,
        "expected_source_hit": len(hits),
        "expected_source_recall": round(recall, 3),
        "source_hits": hits,
        "source_misses": misses,
    }


def run_case_full(item: dict[str, Any], scope: EvalScope, model: str | None) -> dict[str, Any]:
    if not scope.resolved:
        return {
            "resolved": False,
            "resolution_note": scope.resolution_note,
            "answer": "",
            "mode": "unresolved",
            "sources": [],
        }

    result = handle_chat(
        user_message=item["question"],
        model=model,
        active_source=scope.source,
        active_doc_id=scope.doc_id,
        active_sources=scope.sources,
        active_doc_ids=scope.doc_ids,
    )
    return {
        "resolved": True,
        "resolution_note": scope.resolution_note,
        "answer": result.get("answer", ""),
        "mode": result.get("mode", ""),
        "sources": result.get("sources", []),
    }


def run_case_retrieval_only(item: dict[str, Any], scope: EvalScope, model: str | None) -> dict[str, Any]:
    if not scope.resolved:
        return {
            "resolved": False,
            "resolution_note": scope.resolution_note,
            "answer": "",
            "mode": "unresolved",
            "sources": [],
        }

    state = _build_chat_state(
        item["question"],
        [],
        model,
        scope.source,
        scope.doc_id,
        scope.source_type,
        scope.sources,
        scope.doc_ids,
        None,
    )

    scope_response = _resolve_scope_stage(
        state,
        user_message=item["question"],
        active_source=scope.source,
        active_doc_id=scope.doc_id,
        active_source_type=scope.source_type,
    )
    if scope_response:
        return {
            "resolved": True,
            "resolution_note": scope.resolution_note,
            "answer": scope_response.get("answer", ""),
            "mode": scope_response.get("mode", "document_qa"),
            "sources": scope_response.get("sources", []),
        }

    retrieval_response = _run_retrieval_stage(
        state,
        user_message=item["question"],
        active_source=scope.source,
        active_doc_id=scope.doc_id,
        active_source_type=scope.source_type,
    )
    if retrieval_response:
        return {
            "resolved": True,
            "resolution_note": scope.resolution_note,
            "answer": retrieval_response.get("answer", ""),
            "mode": retrieval_response.get("mode", "document_qa"),
            "sources": retrieval_response.get("sources", []),
        }

    return {
        "resolved": True,
        "resolution_note": scope.resolution_note,
        "answer": "",
        "mode": "retrieval_only",
        "sources": state.sources,
    }


def summarize_results(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    items = list(rows)
    source_recalls = [row["source_scoring"]["expected_source_recall"] for row in items]
    answer_recalls = [
        row["answer_scoring"]["expected_points_recall"]
        for row in items
        if row.get("answer_scoring") is not None
    ]
    unevaluated = sum(1 for row in items if row.get("answer_scoring") is None)
    full_pass = sum(
        1
        for row in items
        if row["source_scoring"]["expected_source_recall"] == 1.0
        and row.get("answer_scoring") is not None
        and row["answer_scoring"]["expected_points_recall"] == 1.0
    )
    return {
        "total_rows": len(items),
        "avg_source_recall": round(sum(source_recalls) / len(source_recalls), 3) if source_recalls else None,
        "avg_answer_point_recall": round(sum(answer_recalls) / len(answer_recalls), 3) if answer_recalls else None,
        "full_pass_count": full_pass,
        "unevaluated_count": unevaluated,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate structure-aware retrieval through handle_chat().")
    parser.add_argument("--path", default=DEFAULT_EVAL_PATH, help="Evaluation JSONL path")
    parser.add_argument("--model", default=None, help="Override model for answer generation")
    parser.add_argument(
        "--mode",
        choices=["retrieval", "full"],
        default="full",
        help="`retrieval` stops after the chat retrieval stage; `full` runs generation too.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Limit number of rows")
    parser.add_argument("--id", dest="ids", action="append", default=[], help="Run only these IDs")
    parser.add_argument("--output-dir", default=DEFAULT_RESULTS_DIR, help="Where to write results")
    args = parser.parse_args()

    _lazy_imports()

    eval_path = Path(args.path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(eval_path)
    if args.ids:
        wanted = set(args.ids)
        rows = [row for row in rows if row.get("id") in wanted]
    if args.limit:
        rows = rows[: args.limit]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = output_dir / f"structure_retrieval_results_{timestamp}.jsonl"
    summary_path = output_dir / f"structure_retrieval_summary_{timestamp}.json"

    results: list[dict[str, Any]] = []
    with result_path.open("w", encoding="utf-8") as handle:
        for idx, item in enumerate(rows, start=1):
            print(f"[{idx}/{len(rows)}] {item['id']} — {item['question'][:60]}…", flush=True)
            scope = resolve_scope(item["document_source"])
            if args.mode == "full":
                answer_result = run_case_full(item, scope, args.model)
            else:
                answer_result = run_case_retrieval_only(item, scope, args.model)
            row_result = {
                "id": item["id"],
                "category": item.get("category", ""),
                "language": item.get("language", ""),
                "document_source": item.get("document_source"),
                "question": item["question"],
                "scope": asdict(scope),
                "answer_result": answer_result,
                "source_scoring": score_sources(
                    answer_result.get("sources", []),
                    item.get("expected_sources", []),
                ),
                "answer_scoring": (
                    score_answer_points(
                        answer_result.get("answer", ""),
                        item.get("expected_answer_points", []),
                    )
                    if args.mode == "full" or answer_result.get("answer")
                    else None
                ),
                "notes": item.get("notes", ""),
            }
            handle.write(json.dumps(row_result, ensure_ascii=False) + "\n")
            results.append(row_result)

    summary = summarize_results(results)
    summary["mode"] = args.mode
    summary["eval_path"] = str(eval_path)
    summary["result_path"] = str(result_path)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("STRUCTURE RETRIEVAL EVAL COMPLETE")
    print(f"results: {result_path}")
    print(f"summary: {summary_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
