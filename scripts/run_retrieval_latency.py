#!/usr/bin/env python3
"""
Measure latency and compute-cost tradeoffs for retrieval strategies.

Primary use:
- compare `hybrid` vs `hybrid_rerank`
- estimate reranker overhead on the current benchmark set

Outputs:
- JSONL file with per-row latency samples
- JSON summary with aggregate timing/cost stats
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import VECTOR_TOP_K
from app.core.vectorstore import similarity_search_with_scores
from app.retrieval.keyword_index import search_keyword_index
from app.retrieval.reranker import rerank
from scripts.run_benchmark import load_benchmark, resolve_scope
from scripts.run_retrieval_ablation import _fuse_results

DEFAULT_BENCHMARK = "finetuning/data/benchmark_tilon_v1.jsonl"
DEFAULT_RESULTS_DIR = "finetuning/results"
DEFAULT_STRATEGIES = ("hybrid", "hybrid_rerank")


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    index = (len(ordered) - 1) * pct
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def timed_strategy(
    strategy: str,
    query: str,
    source: str | None,
    doc_id: str | None,
    k: int,
) -> dict[str, Any]:
    started = perf_counter()
    vector_results = similarity_search_with_scores(
        query,
        k=k,
        filter_source=source,
        filter_doc_id=doc_id,
        min_score=None,
    )
    after_vector = perf_counter()

    keyword_results = search_keyword_index(
        query,
        k=k,
        source_filter=source,
        doc_id_filter=doc_id,
    )
    after_keyword = perf_counter()

    fused = _fuse_results(vector_results, keyword_results, limit=max(k, VECTOR_TOP_K * 2))
    docs = [entry["doc"] for entry in fused]
    after_fuse = perf_counter()

    rerank_ms = 0.0
    rerank_candidates = 0
    if strategy == "hybrid_rerank" and len(docs) > 1:
        rerank_candidates = len(docs)
        rerank_started = perf_counter()
        docs = rerank(query, docs, top_n=k)
        rerank_ms = (perf_counter() - rerank_started) * 1000.0
    else:
        docs = docs[:k]

    finished = perf_counter()
    return {
        "returned_docs": len(docs),
        "vector_candidates": len(vector_results),
        "keyword_candidates": len(keyword_results),
        "fused_candidates": len(fused),
        "rerank_candidates": rerank_candidates,
        "vector_ms": (after_vector - started) * 1000.0,
        "keyword_ms": (after_keyword - after_vector) * 1000.0,
        "fuse_ms": (after_fuse - after_keyword) * 1000.0,
        "rerank_ms": rerank_ms,
        "total_ms": (finished - started) * 1000.0,
    }


def summarize(strategy_rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_ms = [row["total_ms"] for row in strategy_rows]
    rerank_ms = [row["rerank_ms"] for row in strategy_rows]
    vector_ms = [row["vector_ms"] for row in strategy_rows]
    keyword_ms = [row["keyword_ms"] for row in strategy_rows]
    fuse_ms = [row["fuse_ms"] for row in strategy_rows]
    rerank_candidates = [row["rerank_candidates"] for row in strategy_rows if row["rerank_candidates"] > 0]

    return {
        "rows": len(strategy_rows),
        "avg_total_ms": round(statistics.fmean(total_ms), 3) if total_ms else None,
        "p50_total_ms": round(percentile(total_ms, 0.50), 3) if total_ms else None,
        "p95_total_ms": round(percentile(total_ms, 0.95), 3) if total_ms else None,
        "avg_vector_ms": round(statistics.fmean(vector_ms), 3) if vector_ms else None,
        "avg_keyword_ms": round(statistics.fmean(keyword_ms), 3) if keyword_ms else None,
        "avg_fuse_ms": round(statistics.fmean(fuse_ms), 3) if fuse_ms else None,
        "avg_rerank_ms": round(statistics.fmean(rerank_ms), 3) if rerank_ms else None,
        "rerank_rows": sum(1 for row in strategy_rows if row["rerank_candidates"] > 0),
        "avg_rerank_candidates": round(statistics.fmean(rerank_candidates), 3) if rerank_candidates else 0.0,
    }


def print_summary(summary: dict[str, Any]) -> None:
    print("RETRIEVAL LATENCY SUMMARY")
    print(
        f"{'strategy':<16} {'avg_ms':>9} {'p50_ms':>9} {'p95_ms':>9} "
        f"{'rerank_ms':>11} {'pairs':>8} {'rows':>6}"
    )
    for strategy, data in summary.items():
        print(
            f"{strategy:<16} "
            f"{str(data.get('avg_total_ms')):>9} "
            f"{str(data.get('p50_total_ms')):>9} "
            f"{str(data.get('p95_total_ms')):>9} "
            f"{str(data.get('avg_rerank_ms')):>11} "
            f"{str(data.get('avg_rerank_candidates')):>8} "
            f"{str(data.get('rows')):>6}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure retrieval latency/cost tradeoffs.")
    parser.add_argument("--path", default=DEFAULT_BENCHMARK, help="Benchmark JSONL path")
    parser.add_argument("--k", type=int, default=10, help="Top-k cutoff")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of benchmark rows")
    parser.add_argument("--id", dest="ids", action="append", default=[], help="Run only these benchmark IDs")
    parser.add_argument("--repeats", type=int, default=3, help="Measured repeats per row/strategy")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs per row/strategy")
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=list(DEFAULT_STRATEGIES),
        choices=["hybrid", "hybrid_rerank"],
        help="Strategies to measure",
    )
    parser.add_argument("--output-dir", default=DEFAULT_RESULTS_DIR, help="Where to store timing results")
    args = parser.parse_args()

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
    result_path = output_dir / f"retrieval_latency_{timestamp}.jsonl"
    summary_path = output_dir / f"retrieval_latency_summary_{timestamp}.json"

    per_sample: list[dict[str, Any]] = []
    for item in rows:
        scope = resolve_scope(
            item["document_source"],
            preferred_source_type=item.get("scope_source_type"),
        )
        if not scope.resolved:
            continue

        for strategy in args.strategies:
            for _ in range(args.warmup):
                timed_strategy(strategy, item["question"], scope.source, scope.doc_id, args.k)

            for repeat_idx in range(args.repeats):
                timing = timed_strategy(strategy, item["question"], scope.source, scope.doc_id, args.k)
                per_sample.append(
                    {
                        "id": item["id"],
                        "strategy": strategy,
                        "repeat": repeat_idx + 1,
                        "scope": {
                            "source": scope.source,
                            "doc_id": scope.doc_id,
                            "source_type": scope.source_type,
                        },
                        **timing,
                    }
                )

    with result_path.open("w", encoding="utf-8") as handle:
        for row in per_sample:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        strategy: summarize([row for row in per_sample if row["strategy"] == strategy])
        for strategy in args.strategies
    }
    payload = {
        "benchmark_path": str(benchmark_path),
        "k": args.k,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "summary": summary,
        "result_path": str(result_path),
    }
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print_summary(summary)
    print(f"results: {result_path}")
    print(f"summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
