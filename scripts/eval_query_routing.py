#!/usr/bin/env python3
"""
Evaluate query routing decisions against a small labeled JSONL set.

This script is intentionally lightweight and local-only. It compares:
  - heuristic intent
  - classifier intent
  - final routed intent

Examples:
  python scripts/eval_query_routing.py
  python scripts/eval_query_routing.py --classifier embedding
  python scripts/eval_query_routing.py --classifier embedding --live
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


DEFAULT_DATASET = Path("finetuning/data/query_policy_eval_v1.jsonl")


def _load_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                cases.append(json.loads(raw))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no}: {exc}") from exc
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate query routing policy.")
    parser.add_argument(
        "--path",
        default=str(DEFAULT_DATASET),
        help="Path to labeled routing JSONL dataset",
    )
    parser.add_argument(
        "--classifier",
        default="heuristic",
        choices=["heuristic", "embedding"],
        help="Classifier provider to evaluate",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Let classifier influence final routed intent (disables shadow mode).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print every case instead of only disagreements.",
    )
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"ERROR: dataset not found: {path}")
        return 1

    os.environ["QUERY_CLASSIFIER_ENABLED"] = "true" if args.classifier != "heuristic" else "false"
    os.environ["QUERY_CLASSIFIER_PROVIDER"] = args.classifier
    os.environ["QUERY_CLASSIFIER_SHADOW_MODE"] = "false" if args.live else "true"

    from app.chat.policy import route_query  # imported after env setup

    cases = _load_cases(path)

    final_correct = 0
    heuristic_correct = 0
    classifier_correct = 0
    classifier_available = 0
    disagreements = 0
    by_intent: dict[str, Counter[str]] = defaultdict(Counter)

    print("=" * 88)
    print(f"Routing eval: {path}")
    print(f"classifier={args.classifier} shadow_mode={not args.live}")
    print("=" * 88)

    for case in cases:
        expected = case["intent"]
        question = case["question"]
        policy = route_query(question)

        heuristic_ok = policy.heuristic_intent == expected
        final_ok = policy.intent == expected
        classifier_ok = policy.classifier_intent == expected if policy.classifier_intent else False

        heuristic_correct += int(heuristic_ok)
        final_correct += int(final_ok)
        if policy.classifier_intent is not None:
            classifier_available += 1
            classifier_correct += int(classifier_ok)

        by_intent[expected]["total"] += 1
        by_intent[expected]["final_correct"] += int(final_ok)

        if policy.classifier_disagrees:
            disagreements += 1

        should_print = args.verbose or policy.classifier_disagrees or not final_ok
        if should_print:
            print(
                f"[{case['id']}] expected={expected} "
                f"heuristic={policy.heuristic_intent} "
                f"classifier={policy.classifier_intent or '-'} "
                f"final={policy.intent} "
                f"source={policy.intent_source} "
                f"confidence={policy.intent_confidence:.2f}"
            )
            print(f"  Q: {question}")
            if case.get("notes"):
                print(f"  note: {case['notes']}")

    total = len(cases)
    print("-" * 88)
    print(
        f"heuristic_accuracy={heuristic_correct}/{total} ({heuristic_correct/total:.1%}) | "
        f"final_accuracy={final_correct}/{total} ({final_correct/total:.1%})"
    )
    if classifier_available:
        print(
            f"classifier_accuracy={classifier_correct}/{classifier_available} "
            f"({classifier_correct/classifier_available:.1%}) | "
            f"classifier_coverage={classifier_available}/{total} ({classifier_available/total:.1%})"
        )
    else:
        print("classifier_accuracy=n/a | classifier_coverage=0")
    print(f"disagreements={disagreements}/{total} ({disagreements/total:.1%})")
    print("-" * 88)
    print("per-intent final accuracy:")
    for intent in sorted(by_intent):
        total_intent = by_intent[intent]["total"]
        correct_intent = by_intent[intent]["final_correct"]
        print(f"  {intent}: {correct_intent}/{total_intent} ({correct_intent/total_intent:.1%})")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
