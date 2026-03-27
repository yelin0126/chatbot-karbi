#!/usr/bin/env python3
"""
Run the benchmark dataset against the current RAG system.

Supports:
- retrieval-only evaluation
- answer generation evaluation
- combined runs

Outputs:
- JSONL file with per-item results
- JSON summary file with aggregate metrics
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from difflib import SequenceMatcher
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.core.document_registry import list_documents

_HTTP_MODE = False
_SERVER_URL = "http://localhost:8000"


def _lazy_imports():
    """Import GPU-requiring modules only when running in-process (non-HTTP) mode."""
    global handle_chat, get_documents_by_doc_ids, extract_sources, retrieve
    from app.chat.handlers import handle_chat  # noqa: F811
    from app.core.vectorstore import get_documents_by_doc_ids  # noqa: F811
    from app.retrieval.retriever import extract_sources, retrieve  # noqa: F811


DEFAULT_BENCHMARK = "finetuning/data/benchmark_template.jsonl"
DEFAULT_RESULTS_DIR = "finetuning/results"
SUMMARY_CATEGORIES_FULL_DOC = {"summary", "section_understanding"}
_ANSWER_STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "into", "only",
    "문서", "내용", "관련", "대한", "해야", "합니다", "한다", "되어야", "있다", "있는",
    "또는", "포함", "언급", "기반", "답해야", "좋음", "점", "것", "수", "로", "을", "를",
}


@dataclass
class BenchmarkScope:
    source: str | None
    doc_id: str | None
    source_type: str | None
    resolved: bool
    resolution_note: str
    sources: list[str] = field(default_factory=list)
    doc_ids: list[str] = field(default_factory=list)
    source_types: list[str] = field(default_factory=list)


def load_benchmark(path: Path) -> list[dict[str, Any]]:
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


def _pick_best_registry_entry(
    entries: list[dict[str, Any]],
    preferred_source_type: str | None = None,
) -> dict[str, Any]:
    def sort_key(item: dict[str, Any]) -> tuple[int, str]:
        source_type = item.get("source_type")
        if preferred_source_type:
            source_type_score = 0 if source_type == preferred_source_type else 1
        else:
            source_type_score = 0 if source_type == "library" else 1
        updated = str(item.get("updated_at") or item.get("created_at") or "")
        return (source_type_score, updated)

    return sorted(entries, key=sort_key)[0]


def resolve_scope(document_source: str, preferred_source_type: str | None = None) -> BenchmarkScope:
    entries = [doc for doc in list_documents() if doc.get("source") == document_source]
    if not entries:
        return BenchmarkScope(
            source=document_source,
            doc_id=None,
            source_type=None,
            sources=[],
            doc_ids=[],
            source_types=[],
            resolved=False,
            resolution_note="document source not found in registry",
        )

    best = _pick_best_registry_entry(entries, preferred_source_type=preferred_source_type)
    note = "matched exact source"
    if len(entries) > 1:
        if preferred_source_type:
            note = f"matched {len(entries)} documents; preferred source_type='{preferred_source_type}'"
        else:
            note = f"matched {len(entries)} documents; selected preferred scope"

    return BenchmarkScope(
        source=best.get("source"),
        doc_id=best.get("doc_id"),
        source_type=best.get("source_type"),
        sources=[best.get("source")] if best.get("source") else [],
        doc_ids=[best.get("doc_id")] if best.get("doc_id") else [],
        source_types=[best.get("source_type")] if best.get("source_type") else [],
        resolved=True,
        resolution_note=note,
    )


def resolve_scopes(
    document_source: str | None,
    document_sources: list[str] | None = None,
    preferred_source_type: str | None = None,
) -> BenchmarkScope:
    requested_sources = [source for source in (document_sources or []) if str(source).strip()]
    if not requested_sources and document_source:
        requested_sources = [document_source]

    if not requested_sources:
        return BenchmarkScope(
            source=None,
            doc_id=None,
            source_type=None,
            sources=[],
            doc_ids=[],
            source_types=[],
            resolved=False,
            resolution_note="no document source provided",
        )

    resolved_scopes = [
        resolve_scope(source, preferred_source_type=preferred_source_type)
        for source in requested_sources
    ]
    unresolved = [scope for scope in resolved_scopes if not scope.resolved]
    if unresolved:
        missing = ", ".join(scope.source or "unknown" for scope in unresolved)
        return BenchmarkScope(
            source=requested_sources[0],
            doc_id=None,
            source_type=None,
            sources=[scope.source for scope in resolved_scopes if scope.source],
            doc_ids=[scope.doc_id for scope in resolved_scopes if scope.doc_id],
            source_types=[scope.source_type for scope in resolved_scopes if scope.source_type],
            resolved=False,
            resolution_note=f"one or more document sources not found in registry: {missing}",
        )

    note = resolved_scopes[0].resolution_note
    if len(resolved_scopes) > 1:
        note = f"matched {len(resolved_scopes)} comparison documents"

    return BenchmarkScope(
        source=resolved_scopes[0].source,
        doc_id=resolved_scopes[0].doc_id,
        source_type=resolved_scopes[0].source_type,
        sources=[scope.source for scope in resolved_scopes if scope.source],
        doc_ids=[scope.doc_id for scope in resolved_scopes if scope.doc_id],
        source_types=[scope.source_type for scope in resolved_scopes if scope.source_type],
        resolved=True,
        resolution_note=note,
    )


def should_use_full_document(item: dict[str, Any]) -> bool:
    if item.get("category") in SUMMARY_CATEGORIES_FULL_DOC:
        return True
    return bool(item.get("full_document", False))


def normalize_text(text: str) -> str:
    return " ".join((text or "").lower().split())


def _compact_text(text: str) -> str:
    return re.sub(r"[^0-9a-z가-힣]+", "", normalize_text(text))


def _normalize_money_token(token: str) -> str:
    compact = re.sub(r"[\s,]+", "", (token or "").lower())
    if not compact:
        return ""
    if compact.endswith("만원"):
        number_part = compact[:-2]
        try:
            value = int(float(number_part) * 10_000)
        except ValueError:
            return compact
        return f"{value}원"
    if compact.endswith("천원"):
        number_part = compact[:-2]
        try:
            value = int(float(number_part) * 1_000)
        except ValueError:
            return compact
        return f"{value}원"
    if compact.endswith("원"):
        number_part = compact[:-1]
        if number_part.isdigit():
            return f"{int(number_part)}원"
    return compact


def _extract_normalized_money_values(text: str) -> set[str]:
    values: set[str] = set()
    normalized = normalize_text(text)
    for match in re.finditer(r"\d[\d,]*(?:\.\d+)?\s*(?:만원|천원|원)", normalized):
        value = _normalize_money_token(match.group(0))
        if value:
            values.add(value)
    return values


def _point_money_matches(answer: str, point: str) -> bool:
    point_values = _extract_normalized_money_values(point)
    if not point_values:
        return False
    answer_values = _extract_normalized_money_values(answer)
    if not point_values.issubset(answer_values):
        return False

    point_keywords = [
        keyword
        for keyword in _extract_keywords(point)
        if not re.fullmatch(r"\d+", keyword)
        and keyword not in {"원", "만원", "천원", "이하", "이내", "최대"}
    ]
    normalized_answer = normalize_text(answer)
    if not point_keywords:
        return True
    hits = sum(1 for keyword in point_keywords if keyword in normalized_answer)
    return hits >= max(1, len(point_keywords) - 1)


def _keyword_fuzzy_hit(answer: str, keyword: str) -> bool:
    answer_keywords = _extract_keywords(answer)
    compact_keyword = _compact_text(keyword)
    if not compact_keyword:
        return False
    for candidate in answer_keywords:
        compact_candidate = _compact_text(candidate)
        if not compact_candidate:
            continue
        if compact_keyword == compact_candidate:
            return True
        if not re.fullmatch(r"[가-힣]+", compact_keyword):
            continue
        if len(compact_keyword) < 3 or len(compact_candidate) < 3:
            continue
        if compact_keyword[0] != compact_candidate[0] or compact_keyword[-1] != compact_candidate[-1]:
            continue
        if abs(len(compact_keyword) - len(compact_candidate)) > 2:
            continue
        if SequenceMatcher(None, compact_keyword, compact_candidate).ratio() >= 0.60:
            return True
    return False


def _fuzzy_contains(answer: str, point: str) -> bool:
    compact_answer = _compact_text(answer)
    compact_point = _compact_text(point)
    if not compact_answer or not compact_point or len(compact_point) < 4:
        return False
    if compact_point in compact_answer:
        return True

    target_len = len(compact_point)
    window_min = max(3, target_len - 2)
    window_max = min(len(compact_answer), target_len + 2)
    is_korean = bool(re.fullmatch(r"[가-힣]+", compact_point))
    for window_size in range(window_min, window_max + 1):
        for start in range(0, max(1, len(compact_answer) - window_size + 1)):
            candidate = compact_answer[start:start + window_size]
            ratio = SequenceMatcher(None, compact_point, candidate).ratio()
            if ratio >= 0.74:
                return True
            if (
                is_korean
                and compact_point[0] == candidate[0]
                and compact_point[-1] == candidate[-1]
                and ratio >= 0.60
            ):
                return True
    return False


def _extract_keywords(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+|[가-힣]{2,}", normalize_text(text))
    return [tok for tok in tokens if tok not in _ANSWER_STOPWORDS]


def _point_matches(answer: str, point: str) -> bool:
    normalized_answer = normalize_text(answer)
    normalized_point = normalize_text(point)
    if normalized_point and normalized_point in normalized_answer:
        return True
    compact_point = _compact_text(point)
    if compact_point and compact_point in _compact_text(answer):
        return True
    if _point_money_matches(answer, point):
        return True

    keywords = _extract_keywords(point)
    if not keywords:
        return _fuzzy_contains(answer, point)

    hits = sum(1 for keyword in keywords if keyword in normalized_answer)
    fuzzy_hits = sum(
        1 for keyword in keywords
        if keyword not in normalized_answer and _keyword_fuzzy_hit(answer, keyword)
    )
    total_hits = hits + fuzzy_hits
    if len(keywords) <= 2:
        if total_hits == len(keywords):
            return True
    elif total_hits >= max(2, len(keywords) - 1):
        return True

    return _fuzzy_contains(answer, point)


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


def _normalize_source_name(name: str) -> str:
    """Collapse whitespace so near-duplicate filenames compare equal.

    Handles cases like ``(붙임) 제주대학교`` vs ``(붙임)제주대학교`` where the
    only difference is optional whitespace between a parenthetical prefix and
    the main title.
    """
    return re.sub(r"\s+", "", name).lower() if name else ""


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
        norm_source = _normalize_source_name(source)
        matched = any(
            _normalize_source_name(actual.get("source")) == norm_source
            and (page is None or actual.get("page") == page)
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


def detect_refusal(answer: str) -> bool:
    lower = (answer or "").lower()
    indicators = [
        "couldn't find relevant information",
        "not found in the document",
        "could not find relevant information",
        "i couldn't find",
        "i could not find",
        "not mentioned in the document",
        "not described in the document",
        "not specified in the document",
        "제공된 문서에서 확인되지 않습니다",
        "질문과 관련된 정보를 찾지 못했습니다",
        "문서에서 찾을 수 없습니다",
        "문서에서 확인할 수 없습니다",
        "문서에 나와 있지 않습니다",
        "문서에 언급되어 있지 않습니다",
        "문서에 명시되어 있지 않습니다",
        "명시되어 있지 않습니다",
        "확인되지 않습니다",
        "확인할 수 없습니다",
        "찾을 수 없습니다",
        "언급되어 있지 않습니다",
        # Phase 8 context-irrelevant / not-grounded answers
        "이 질문과 관련된 내용이 포함되어 있지 않은 것 같습니다",
        "관련된 내용이 포함되어 있지 않은 것 같습니다",
        "충분히 확인되지 않습니다",
        "doesn't appear to contain information",
        "does not appear to contain information",
        "해당 내용은",
    ]
    return any(token in lower for token in indicators)


def run_retrieval(item: dict[str, Any], scope: BenchmarkScope) -> dict[str, Any]:
    if _HTTP_MODE:
        # In HTTP mode retrieval is derived from the answer call; return a placeholder.
        return {
            "resolved": scope.resolved,
            "resolution_note": scope.resolution_note,
            "retrieved_sources": [],
            "confidence": None,
            "strong_keyword_hit": None,
            "used_full_document": None,
            "note": "retrieval skipped in --http mode; use answer sources instead",
        }

    if not scope.resolved:
        return {
            "resolved": False,
            "resolution_note": scope.resolution_note,
            "retrieved_sources": [],
            "confidence": 0.0,
            "strong_keyword_hit": False,
            "used_full_document": False,
        }

    if len(scope.doc_ids) > 1:
        docs = get_documents_by_doc_ids(scope.doc_ids)
        return {
            "resolved": True,
            "resolution_note": scope.resolution_note,
            "retrieved_sources": extract_sources(docs),
            "confidence": 1.0 if docs else 0.0,
            "strong_keyword_hit": bool(docs),
            "used_full_document": True,
        }

    result = retrieve(
        query=item["question"],
        source_filter=scope.source,
        doc_id_filter=scope.doc_id,
        full_document=should_use_full_document(item),
    )
    actual_sources = extract_sources(result.docs)
    return {
        "resolved": True,
        "resolution_note": scope.resolution_note,
        "retrieved_sources": actual_sources,
        "confidence": result.confidence,
        "strong_keyword_hit": result.strong_keyword_hit,
        "used_full_document": result.used_full_document,
    }


def _run_answer_http(item: dict[str, Any], scope: BenchmarkScope, model: str | None) -> dict[str, Any]:
    import requests  # stdlib requests always available

    payload: dict[str, Any] = {"message": item["question"]}
    if scope.resolved:
        if scope.source:
            payload["active_source"] = scope.source
        if scope.doc_id:
            payload["active_doc_id"] = scope.doc_id
        if scope.sources:
            payload["active_sources"] = scope.sources
        if scope.doc_ids:
            payload["active_doc_ids"] = scope.doc_ids
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
            "resolved": scope.resolved,
            "resolution_note": scope.resolution_note,
            "answer": f"[HTTP ERROR] {exc}",
            "mode": "error",
            "sources": [],
        }

    raw_sources = data.get("sources", [])
    # normalise: sources may be dicts or SourceInfo-like objects
    sources = [s if isinstance(s, dict) else vars(s) for s in raw_sources]
    return {
        "resolved": scope.resolved,
        "resolution_note": scope.resolution_note,
        "answer": data.get("answer", ""),
        "mode": data.get("mode", ""),
        "sources": sources,
    }


def run_answer(item: dict[str, Any], scope: BenchmarkScope, model: str | None) -> dict[str, Any]:
    if _HTTP_MODE:
        return _run_answer_http(item, scope, model)

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


def summarize_results(results: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(results)
    category_counts = Counter(row["category"] for row in rows)
    unresolved = sum(1 for row in rows if not row["scope"]["resolved"])
    answer_recalls = [row["answer_scoring"]["expected_points_recall"] for row in rows if "answer_scoring" in row]
    source_recalls = [row["source_scoring"]["expected_source_recall"] for row in rows if "source_scoring" in row]
    refusal_cases = [row for row in rows if not row.get("should_answer_from_docs", True)]
    refusal_pass = sum(1 for row in refusal_cases if row.get("answer_refusal_detected"))

    return {
        "total_rows": len(rows),
        "unresolved_document_scope_rows": unresolved,
        "categories": dict(category_counts),
        "avg_answer_point_recall": round(sum(answer_recalls) / len(answer_recalls), 3) if answer_recalls else None,
        "avg_source_recall": round(sum(source_recalls) / len(source_recalls), 3) if source_recalls else None,
        "negative_case_count": len(refusal_cases),
        "negative_case_refusal_pass": refusal_pass,
    }


def main() -> int:
    global _HTTP_MODE, _SERVER_URL

    parser = argparse.ArgumentParser(description="Run benchmark JSONL against the current RAG system.")
    parser.add_argument("--path", default=DEFAULT_BENCHMARK, help="Benchmark JSONL path")
    parser.add_argument(
        "--mode",
        choices=["retrieval", "answer", "both"],
        default="both",
        help="What to evaluate",
    )
    parser.add_argument("--model", default=None, help="Override model for answer generation")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of benchmark rows")
    parser.add_argument("--id", dest="ids", action="append", default=[], help="Run only these benchmark IDs")
    parser.add_argument("--output-dir", default=DEFAULT_RESULTS_DIR, help="Where to store benchmark results")
    parser.add_argument(
        "--http",
        action="store_true",
        default=False,
        help="Call running server's /chat HTTP endpoint instead of importing modules in-process",
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
        # Verify server is reachable before running
        import requests
        try:
            requests.get(f"{_SERVER_URL}/health", timeout=5).raise_for_status()
            print(f"Server reachable at {_SERVER_URL}")
        except Exception as exc:
            print(f"ERROR: cannot reach server at {_SERVER_URL}: {exc}")
            return 1

    benchmark_path = Path(args.path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_benchmark(benchmark_path)
    if args.ids:
        wanted = set(args.ids)
        rows = [row for row in rows if row.get("id") in wanted]
    if args.limit:
        rows = rows[: args.limit]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = output_dir / f"benchmark_results_{timestamp}.jsonl"
    summary_path = output_dir / f"benchmark_summary_{timestamp}.json"

    results: list[dict[str, Any]] = []
    total = len(rows)
    with result_path.open("w", encoding="utf-8") as handle:
        for idx, item in enumerate(rows, start=1):
            print(f"[{idx}/{total}] {item['id']} — {item['question'][:60]}…", flush=True)
            scope = resolve_scopes(
                item.get("document_source"),
                document_sources=item.get("document_sources"),
                preferred_source_type=item.get("scope_source_type"),
            )
            row_result: dict[str, Any] = {
                "id": item["id"],
                "category": item["category"],
                "language": item["language"],
                "document_source": item.get("document_source"),
                "document_sources": item.get("document_sources", []),
                "question": item["question"],
                "should_answer_from_docs": item["should_answer_from_docs"],
                "scope": asdict(scope),
            }

            retrieval_payload = None
            answer_payload = None

            if args.mode in {"retrieval", "both"}:
                retrieval_payload = run_retrieval(item, scope)
                row_result["retrieval"] = retrieval_payload
                # In HTTP mode retrieval sources come from the answer payload; populated below.
                if not _HTTP_MODE:
                    row_result["source_scoring"] = score_sources(
                        retrieval_payload.get("retrieved_sources", []),
                        item.get("expected_sources", []),
                    )

            if args.mode in {"answer", "both"}:
                answer_payload = run_answer(item, scope, args.model)
                row_result["answer_result"] = answer_payload
                row_result["answer_scoring"] = score_answer_points(
                    answer_payload.get("answer", ""),
                    item.get("expected_answer_points", []),
                )
                row_result["answer_refusal_detected"] = detect_refusal(answer_payload.get("answer", ""))
                # In HTTP mode, derive source recall from answer sources
                if _HTTP_MODE and args.mode in {"both"}:
                    row_result["source_scoring"] = score_sources(
                        answer_payload.get("sources", []),
                        item.get("expected_sources", []),
                    )

            handle.write(json.dumps(row_result, ensure_ascii=False) + "\n")
            results.append(row_result)

    summary = summarize_results(results)
    summary["mode"] = args.mode
    summary["http_mode"] = _HTTP_MODE
    summary["server_url"] = _SERVER_URL if _HTTP_MODE else None
    summary["benchmark_path"] = str(benchmark_path)
    summary["result_path"] = str(result_path)

    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("BENCHMARK COMPLETE")
    print(f"rows: {summary['total_rows']}")
    print(f"results: {result_path}")
    print(f"summary: {summary_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
