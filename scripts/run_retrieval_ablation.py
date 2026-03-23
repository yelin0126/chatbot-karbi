#!/usr/bin/env python3
"""
Compare retrieval strategies on the benchmark set.

Strategies:
- vector
- keyword
- hybrid
- hybrid_rerank

Outputs:
- JSON file with per-strategy metrics
- console summary table
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import RERANKER_ENABLED, VECTOR_TOP_K
from app.core.vectorstore import similarity_search_with_scores
from app.retrieval.keyword_index import search_keyword_index
from app.retrieval.reranker import rerank
from app.retrieval.retriever import extract_sources
from scripts.run_benchmark import load_benchmark, resolve_scope


DEFAULT_BENCHMARK = "finetuning/data/benchmark_tilon_v1.jsonl"
DEFAULT_RESULTS_DIR = "finetuning/results"
RRF_K = 60


def _doc_key(doc) -> str:
    meta = doc.metadata
    return str(
        meta.get("chunk_id")
        or f"{meta.get('doc_id') or meta.get('source')}::{meta.get('page')}::{meta.get('chunk_index')}"
    )


def _fuse_results(vector_results, keyword_results, limit: int):
    merged: dict[str, dict[str, Any]] = {}

    for rank, (doc, score) in enumerate(vector_results, start=1):
        key = _doc_key(doc)
        entry = merged.setdefault(
            key,
            {"doc": doc, "rrf_score": 0.0, "vector_score": 0.0, "keyword_score": 0.0},
        )
        entry["rrf_score"] += 1.0 / (RRF_K + rank)
        entry["vector_score"] = max(entry["vector_score"], score)

    for rank, (doc, score) in enumerate(keyword_results, start=1):
        key = _doc_key(doc)
        entry = merged.setdefault(
            key,
            {"doc": doc, "rrf_score": 0.0, "vector_score": 0.0, "keyword_score": 0.0},
        )
        entry["rrf_score"] += 1.0 / (RRF_K + rank)
        entry["keyword_score"] = max(entry["keyword_score"], score)

    return sorted(
        merged.values(),
        key=lambda item: (item["rrf_score"], item["vector_score"], item["keyword_score"]),
        reverse=True,
    )[:limit]


def run_strategy(strategy: str, query: str, source: str | None, doc_id: str | None, k: int):
    vector_results = similarity_search_with_scores(
        query,
        k=k,
        filter_source=source,
        filter_doc_id=doc_id,
        min_score=None,
    )
    keyword_results = search_keyword_index(
        query,
        k=k,
        source_filter=source,
        doc_id_filter=doc_id,
    )

    if strategy == "vector":
        docs = [doc for doc, _ in vector_results[:k]]
    elif strategy == "keyword":
        docs = [doc for doc, _ in keyword_results[:k]]
    elif strategy == "hybrid":
        docs = [entry["doc"] for entry in _fuse_results(vector_results, keyword_results, limit=k)]
    elif strategy == "hybrid_rerank":
        docs = [entry["doc"] for entry in _fuse_results(vector_results, keyword_results, limit=max(k, VECTOR_TOP_K * 2))]
        if len(docs) > 1:
            docs = rerank(query, docs, top_n=k)
        else:
            docs = docs[:k]
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return extract_sources(docs)


def _source_key(item: dict[str, Any]) -> tuple[str | None, int | None]:
    return item.get("source"), item.get("page")


def _dedupe_ranked_keys(ranked: list[tuple[str | None, int | None]]) -> list[tuple[str | None, int | None]]:
    """Keep first occurrence only so one relevant page cannot earn credit twice."""
    deduped = []
    seen = set()
    for key in ranked:
        if key in seen:
            continue
        seen.add(key)
        deduped.append(key)
    return deduped


def _dcg(ranked: list[tuple[str | None, int | None]], relevant: set[tuple[str | None, int | None]], k: int) -> float:
    score = 0.0
    for i, key in enumerate(ranked[:k], start=1):
        rel = 1.0 if key in relevant else 0.0
        if rel:
            score += rel / math.log2(i + 1)
    return score


def score_row(actual_sources: list[dict[str, Any]], expected_sources: list[dict[str, Any]], k: int) -> dict[str, float]:
    relevant = {_source_key(item) for item in expected_sources}
    ranked = _dedupe_ranked_keys([_source_key(item) for item in actual_sources])
    hits = [key for key in ranked[:k] if key in relevant]

    hit_rate = 1.0 if hits else 0.0
    recall = len(set(hits)) / len(relevant) if relevant else 1.0

    rr = 0.0
    for rank, key in enumerate(ranked[:k], start=1):
        if key in relevant:
            rr = 1.0 / rank
            break

    ideal_ranked = list(relevant)
    ideal_dcg = _dcg(ideal_ranked, relevant, k)
    ndcg = (_dcg(ranked, relevant, k) / ideal_dcg) if ideal_dcg else 1.0

    return {
        "hit_rate": hit_rate,
        "recall": recall,
        "mrr": rr,
        "ndcg": ndcg,
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_strategy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        by_strategy[row["strategy"]].append(row)

    summary: dict[str, Any] = {}
    for strategy, rows in by_strategy.items():
        eval_rows = [row for row in rows if row["expected_source_total"] > 0]
        unresolved = sum(1 for row in rows if not row["scope"]["resolved"])
        if not eval_rows:
            summary[strategy] = {
                "rows": len(rows),
                "evaluated_rows": 0,
                "unresolved_rows": unresolved,
                "hit_rate@k": None,
                "recall@k": None,
                "mrr@k": None,
                "ndcg@k": None,
            }
            continue

        summary[strategy] = {
            "rows": len(rows),
            "evaluated_rows": len(eval_rows),
            "unresolved_rows": unresolved,
            "hit_rate@k": round(sum(r["hit_rate"] for r in eval_rows) / len(eval_rows), 3),
            "recall@k": round(sum(r["recall"] for r in eval_rows) / len(eval_rows), 3),
            "mrr@k": round(sum(r["mrr"] for r in eval_rows) / len(eval_rows), 3),
            "ndcg@k": round(sum(r["ndcg"] for r in eval_rows) / len(eval_rows), 3),
        }

    return summary


def print_summary(summary: dict[str, Any], k: int) -> None:
    print(f"RETRIEVAL ABLATION COMPLETE (k={k})")
    print(f"{'strategy':<16} {'hit_rate':>9} {'recall':>9} {'mrr':>9} {'ndcg':>9} {'rows':>6}")
    for strategy in ["vector", "keyword", "hybrid", "hybrid_rerank"]:
        data = summary.get(strategy, {})
        print(
            f"{strategy:<16} "
            f"{str(data.get('hit_rate@k')):>9} "
            f"{str(data.get('recall@k')):>9} "
            f"{str(data.get('mrr@k')):>9} "
            f"{str(data.get('ndcg@k')):>9} "
            f"{str(data.get('evaluated_rows')):>6}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run retrieval ablation on the benchmark set.")
    parser.add_argument("--path", default=DEFAULT_BENCHMARK, help="Benchmark JSONL path")
    parser.add_argument("--k", type=int, default=10, help="Top-k cutoff for retrieval metrics")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of benchmark rows")
    parser.add_argument("--id", dest="ids", action="append", default=[], help="Run only these benchmark IDs")
    parser.add_argument("--output-dir", default=DEFAULT_RESULTS_DIR, help="Where to store ablation results")
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

    strategies = ["vector", "keyword", "hybrid", "hybrid_rerank"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = output_dir / f"retrieval_ablation_{timestamp}.jsonl"
    summary_path = output_dir / f"retrieval_ablation_summary_{timestamp}.json"

    all_results: list[dict[str, Any]] = []
    with result_path.open("w", encoding="utf-8") as handle:
        for item in rows:
            scope = resolve_scope(
                item["document_source"],
                preferred_source_type=item.get("scope_source_type"),
            )
            expected_sources = item.get("expected_sources", [])
            for strategy in strategies:
                actual_sources = []
                if scope.resolved:
                    actual_sources = run_strategy(
                        strategy,
                        item["question"],
                        scope.source,
                        scope.doc_id,
                        args.k,
                    )

                scores = score_row(actual_sources, expected_sources, args.k)
                row = {
                    "id": item["id"],
                    "category": item["category"],
                    "strategy": strategy,
                    "question": item["question"],
                    "document_source": item["document_source"],
                    "scope": asdict(scope),
                    "expected_source_total": len(expected_sources),
                    "actual_sources": actual_sources,
                    **scores,
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                all_results.append(row)

    summary = {
        "benchmark_path": str(benchmark_path),
        "result_path": str(result_path),
        "k": args.k,
        "reranker_enabled_in_config": RERANKER_ENABLED,
        "strategies": summarize(all_results),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print_summary(summary["strategies"], args.k)
    print(f"results: {result_path}")
    print(f"summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
