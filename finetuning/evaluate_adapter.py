"""
Evaluate the base model and a QLoRA adapter on a small strict eval set.

Example:
    python finetuning/evaluate_adapter.py \
      --data finetuning/data/qlora_eval_ko_strict_v1.jsonl \
      --adapter finetuning/output/qwen25-qlora-v2b \
      --output /tmp/tilon_eval_v2b.json
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import re
from dataclasses import dataclass
from pathlib import Path
from contextlib import nullcontext
from typing import Any

from train import DEFAULT_SYSTEM_PROMPT, render_document_chat


NOT_FOUND_HINTS = (
    "찾을 수 없습니다",
    "확인할 수 없습니다",
    "확인되지 않습니다",
    "명시되어 있지 않습니다",
    "제공되지 않습니다",
    "문맥에서는",
)
CLARIFY_HINTS = ("모호", "어떤 지침", "어느 지침", "지정해", "지정해 주세요", "말씀하시는지")
MENTION_ONLY_HINTS = ("언급", "별도 설명", "설명은 제공되지", "설명은 없습니다")
COMPARISON_HINTS = ("반면", "차이", "다릅니다", "공통", "한편")


@dataclass
class EvalSample:
    sample_id: str
    category: str
    language: str
    answer_type: str
    question: str
    context: str
    expected_keywords: list[str]
    forbidden_keywords: list[str]
    notes: str


def _dependency_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _require_dependencies() -> None:
    missing = [
        name
        for name in ["torch", "transformers", "peft", "accelerate"]
        if not _dependency_available(name)
    ]
    if missing:
        raise SystemExit(
            "Missing inference dependencies: "
            + ", ".join(missing)
            + "\nInstall them first with:\n  pip install -r finetuning/requirements-qlora.txt"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate base model vs QLoRA adapter.")
    parser.add_argument(
        "--data",
        default="finetuning/data/qlora_eval_ko_strict_v1.jsonl",
        help="Path to strict eval set JSONL.",
    )
    parser.add_argument(
        "--adapter",
        default="finetuning/output/qwen25-qlora-v2b",
        help="Path to trained adapter directory.",
    )
    parser.add_argument(
        "--model",
        default="",
        help="Optional base model override. Defaults to adapter run_config.json if present.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional JSON output path. Defaults to finetuning/results/eval_*.json",
    )
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=4)
    parser.add_argument("--disable-4bit", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt used to build prompts.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional number of eval samples to run from the top of the file.",
    )
    return parser.parse_args()


def _resolve_base_model(adapter_path: Path, explicit_model: str) -> str:
    if explicit_model:
        return explicit_model
    run_config_path = adapter_path / "run_config.json"
    if run_config_path.exists():
        try:
            payload = json.loads(run_config_path.read_text(encoding="utf-8"))
            model_name = str(payload.get("model") or "").strip()
            if model_name:
                return model_name
        except Exception:
            pass
    return "Qwen/Qwen2.5-7B-Instruct"


def _load_samples(path: Path) -> list[EvalSample]:
    rows: list[EvalSample] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        rows.append(
            EvalSample(
                sample_id=str(payload["id"]),
                category=str(payload["category"]),
                language=str(payload["language"]),
                answer_type=str(payload["answer_type"]),
                question=str(payload["question"]),
                context=str(payload["context"]),
                expected_keywords=[str(x) for x in payload.get("expected_keywords", [])],
                forbidden_keywords=[str(x) for x in payload.get("forbidden_keywords", [])],
                notes=str(payload.get("notes") or ""),
            )
        )
    return rows


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _contains_any(answer: str, hints: tuple[str, ...]) -> bool:
    normalized = _normalize(answer)
    return any(_normalize(hint) in normalized for hint in hints)


def _count_matches(answer: str, keywords: list[str]) -> tuple[int, list[str]]:
    normalized = _normalize(answer)
    matched = [kw for kw in keywords if _normalize(kw) in normalized]
    return len(matched), matched


def _count_forbidden(answer: str, keywords: list[str]) -> list[str]:
    normalized = _normalize(answer)
    return [kw for kw in keywords if _normalize(kw) in normalized]


def _contains_cjk_hanja(answer: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", answer))


def _looks_repetitive(answer: str) -> bool:
    chunks = [chunk.strip() for chunk in re.split(r"[.!?\n]+", answer) if chunk.strip()]
    seen: set[str] = set()
    for chunk in chunks:
        key = _normalize(chunk)
        if len(key) < 8:
            continue
        if key in seen:
            return True
        seen.add(key)
    return False


def _type_ok(answer: str, answer_type: str) -> bool:
    if answer_type == "not_found":
        return _contains_any(answer, NOT_FOUND_HINTS)
    if answer_type == "clarify":
        return _contains_any(answer, CLARIFY_HINTS)
    if answer_type == "mention_only":
        return _contains_any(answer, MENTION_ONLY_HINTS)
    if answer_type == "comparison":
        return _contains_any(answer, COMPARISON_HINTS)
    return True


def _score_answer(answer: str, sample: EvalSample) -> dict[str, Any]:
    keyword_match_count, matched = _count_matches(answer, sample.expected_keywords)
    forbidden_hits = _count_forbidden(answer, sample.forbidden_keywords)
    type_ok = _type_ok(answer, sample.answer_type)
    hanja_flag = sample.language == "ko" and _contains_cjk_hanja(answer)
    repetitive_flag = _looks_repetitive(answer)

    score = 0.0
    if sample.expected_keywords:
        score += 0.6 * (keyword_match_count / len(sample.expected_keywords))
    if type_ok:
        score += 0.3
    if not forbidden_hits:
        score += 0.1
    if hanja_flag:
        score -= 0.15
    if repetitive_flag:
        score -= 0.1
    score = max(0.0, min(1.0, score))

    return {
        "score": round(score, 4),
        "matched_keywords": matched,
        "missing_keywords": [kw for kw in sample.expected_keywords if kw not in matched],
        "forbidden_hits": forbidden_hits,
        "type_ok": type_ok,
        "hanja_flag": hanja_flag,
        "repetitive_flag": repetitive_flag,
    }


def _build_generation_config(model, tokenizer, args):
    generation_config = copy.deepcopy(model.generation_config)
    generation_config.max_new_tokens = args.max_new_tokens
    generation_config.do_sample = args.temperature > 0
    generation_config.repetition_penalty = args.repetition_penalty
    generation_config.no_repeat_ngram_size = args.no_repeat_ngram_size
    generation_config.pad_token_id = tokenizer.pad_token_id
    generation_config.eos_token_id = tokenizer.eos_token_id
    if generation_config.do_sample:
        generation_config.temperature = args.temperature
        generation_config.top_p = args.top_p
    else:
        generation_config.temperature = None
        generation_config.top_p = None
        if hasattr(generation_config, "top_k"):
            generation_config.top_k = None
    return generation_config


def _generate_answer(model, tokenizer, prompt: str, generation_config) -> str:
    import torch

    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {key: value.to(model.device) for key, value in encoded.items()}
    with torch.no_grad():
        output = model.generate(**encoded, generation_config=generation_config)
    return tokenizer.decode(
        output[0][encoded["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()


def _adapter_disabled_context(adapter_model):
    disable_adapter = getattr(adapter_model, "disable_adapter", None)
    if callable(disable_adapter):
        return disable_adapter()
    return nullcontext()


def main() -> None:
    args = parse_args()
    _require_dependencies()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    data_path = Path(args.data)
    adapter_path = Path(args.adapter)
    if not data_path.exists():
        raise SystemExit(f"Eval set not found: {data_path}")
    if not adapter_path.exists():
        raise SystemExit(f"Adapter path not found: {adapter_path}")

    samples = _load_samples(data_path)
    if args.limit and args.limit > 0:
        samples = samples[: args.limit]
    base_model_name = _resolve_base_model(adapter_path, args.model)

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        use_fast=True,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = None
    if not args.disable_4bit and _dependency_available("bitsandbytes"):
        compute_dtype = torch.float16
        if args.bf16:
            compute_dtype = torch.bfloat16
        elif args.fp16:
            compute_dtype = torch.float16
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    model_kwargs: dict[str, Any] = {"trust_remote_code": True}
    if quant_config is not None:
        model_kwargs["quantization_config"] = quant_config
        model_kwargs["device_map"] = "auto"

    model_kwargs["local_files_only"] = True
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, **model_kwargs)
    base_model.eval()
    adapter_model = PeftModel.from_pretrained(base_model, str(adapter_path))
    adapter_model.eval()

    base_generation_config = _build_generation_config(base_model, tokenizer, args)
    adapter_generation_config = _build_generation_config(adapter_model, tokenizer, args)

    results: list[dict[str, Any]] = []
    output_path = Path(args.output) if args.output else Path("finetuning/results/eval_qwen25_v2b_ko_strict.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def write_partial() -> None:
        base_avg = round(sum(item["base_eval"]["score"] for item in results) / max(1, len(results)), 4)
        adapter_avg = round(sum(item["adapter_eval"]["score"] for item in results) / max(1, len(results)), 4)
        payload = {
            "eval_set": str(data_path),
            "base_model": base_model_name,
            "adapter_path": str(adapter_path),
            "sample_count": len(samples),
            "completed_count": len(results),
            "base_avg_score": base_avg,
            "adapter_avg_score": adapter_avg,
            "base_wins": sum(1 for item in results if item["winner"] == "base"),
            "adapter_wins": sum(1 for item in results if item["winner"] == "adapter"),
            "ties": sum(1 for item in results if item["winner"] == "tie"),
            "results": results,
        }
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    for index, sample in enumerate(samples, start=1):
        print(f"[{index}/{len(samples)}] Evaluating {sample.sample_id}: {sample.question}", flush=True)
        prompt = render_document_chat(
            tokenizer=tokenizer,
            system_prompt=args.system_prompt,
            user_message=sample.question,
            context=sample.context,
            add_generation_prompt=True,
        )
        with _adapter_disabled_context(adapter_model):
            base_answer = _generate_answer(adapter_model, tokenizer, prompt, base_generation_config)
        adapter_answer = _generate_answer(adapter_model, tokenizer, prompt, adapter_generation_config)

        base_eval = _score_answer(base_answer, sample)
        adapter_eval = _score_answer(adapter_answer, sample)
        winner = "tie"
        if base_eval["score"] > adapter_eval["score"]:
            winner = "base"
        elif adapter_eval["score"] > base_eval["score"]:
            winner = "adapter"

        results.append(
            {
                "id": sample.sample_id,
                "category": sample.category,
                "answer_type": sample.answer_type,
                "question": sample.question,
                "notes": sample.notes,
                "base_answer": base_answer,
                "adapter_answer": adapter_answer,
                "base_eval": base_eval,
                "adapter_eval": adapter_eval,
                "winner": winner,
            }
        )
        write_partial()
        print(
            json.dumps(
                {
                    "id": sample.sample_id,
                    "winner": winner,
                    "base_score": base_eval["score"],
                    "adapter_score": adapter_eval["score"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    summary = {
        "eval_set": str(data_path),
        "base_model": base_model_name,
        "adapter_path": str(adapter_path),
        "sample_count": len(results),
        "base_avg_score": round(sum(item["base_eval"]["score"] for item in results) / max(1, len(results)), 4),
        "adapter_avg_score": round(sum(item["adapter_eval"]["score"] for item in results) / max(1, len(results)), 4),
        "base_wins": sum(1 for item in results if item["winner"] == "base"),
        "adapter_wins": sum(1 for item in results if item["winner"] == "adapter"),
        "ties": sum(1 for item in results if item["winner"] == "tie"),
        "results": results,
    }
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
