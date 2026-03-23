"""
QLoRA training entrypoint for the current document-answering prompt format.

This script is intentionally lightweight and repo-local:
- loads JSONL samples from finetuning/data/
- formats them to match the live document QA prompt style
- trains a LoRA adapter on top of a frozen base model

Example:
    python finetuning/train.py \
      --data finetuning/data/qlora_train_v1.jsonl \
      --model Qwen/Qwen2.5-7B-Instruct \
      --output finetuning/output/qwen25-qlora-v1
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List


DEFAULT_SYSTEM_PROMPT = (
    "너는 한국어로 답하는 AI 챗봇이다. "
    "짧은 질문에는 짧고 자연스럽게 답한다. "
    "문서 질문은 문서 근거로만 답하고, 근거가 없으면 모른다고 말한다."
)

LANG_RULE = (
    "CRITICAL: Respond in the SAME language the user is using. "
    "If the user writes in Korean, respond in Korean. "
    "If the user writes in English, respond in English. "
    "NEVER respond in Chinese (中文) under any circumstances."
)


def _dependency_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _require_training_dependencies() -> None:
    missing = [
        name
        for name in ["torch", "transformers", "datasets", "peft", "accelerate"]
        if not _dependency_available(name)
    ]
    if missing:
        joined = ", ".join(missing)
        raise SystemExit(
            "Missing QLoRA dependencies: "
            f"{joined}\n"
            "Install them first, for example:\n"
            "  pip install -r finetuning/requirements-qlora.txt"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a QLoRA adapter for document QA.")
    parser.add_argument(
        "--data",
        default="finetuning/data/qlora_train_v1.jsonl",
        help="Training dataset path (JSONL).",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Base Hugging Face model to fine-tune.",
    )
    parser.add_argument(
        "--output",
        default="finetuning/output/qwen25-qlora-v1",
        help="Output directory for the adapter.",
    )
    parser.add_argument("--eval-ratio", type=float, default=0.15, help="Eval split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--max-seq-len", type=int, default=2048, help="Tokenizer max length.")
    parser.add_argument("--epochs", type=float, default=3.0, help="Training epochs.")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate.")
    parser.add_argument("--batch-size", type=int, default=2, help="Per-device train batch size.")
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=8,
        help="Gradient accumulation steps.",
    )
    parser.add_argument("--warmup-ratio", type=float, default=0.03, help="Warmup ratio.")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay.")
    parser.add_argument("--logging-steps", type=int, default=5, help="Logging step interval.")
    parser.add_argument("--save-steps", type=int, default=50, help="Checkpoint save interval.")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank.")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha.")
    parser.add_argument("--lora-dropout", type=float, default=0.1, help="LoRA dropout.")
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt used when formatting training samples.",
    )
    parser.add_argument(
        "--disable-4bit",
        action="store_true",
        help="Disable 4-bit loading even if bitsandbytes is available.",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Force bf16 training args. Useful when the GPU supports bf16.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Force fp16 training args. Use when bf16 is not supported.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate dataset formatting and split without loading training dependencies or model weights.",
    )
    return parser.parse_args()


def _read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                rows.append(json.loads(raw))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} is not valid JSON: {exc}") from exc
    return rows


def _validate_rows(rows: Iterable[Dict], data_path: Path) -> List[Dict]:
    required = {"id", "question", "context", "answer"}
    validated: List[Dict] = []
    seen = set()

    for idx, row in enumerate(rows, start=1):
        missing = [field for field in required if not row.get(field)]
        if missing:
            raise ValueError(f"{data_path}:{idx} missing required fields: {', '.join(missing)}")
        row_id = str(row["id"])
        if row_id in seen:
            raise ValueError(f"Duplicate sample id found: {row_id}")
        seen.add(row_id)
        validated.append(row)

    if not validated:
        raise ValueError(f"No training rows found in {data_path}")
    return validated


def build_document_prompt(system_prompt: str, user_message: str, context: str) -> str:
    return f"""[시스템 지침]
{system_prompt}

답변 규칙:
1. {LANG_RULE}
2. 제공된 문서 문맥만 근거로 답한다.
3. 문서에 없는 내용은 추측하지 않는다.
4. 문맥이 부족하면 "해당 내용은 제공된 문서에서 확인되지 않습니다."라고 답한다.
5. 핵심 답변을 먼저 말한다.
6. 가능하면 페이지와 문서를 근거로 설명한다.
7. 이미지에서 추출된 텍스트가 제공되면 해당 텍스트를 기반으로 답한다.

[대화 이력]

[검색된 문서]
{context if context else "검색된 관련 문서 없음"}

[사용자 질문]
{user_message}

[답변 형식]
- 핵심 답변:
- 근거 요약:
- 참고 문서:""".strip()


@dataclass
class FormattedSample:
    sample_id: str
    source_benchmark_id: str
    prompt: str
    answer: str
    language: str


def format_samples(rows: Iterable[Dict], system_prompt: str) -> List[FormattedSample]:
    formatted: List[FormattedSample] = []
    for row in rows:
        prompt = build_document_prompt(
            system_prompt=system_prompt,
            user_message=str(row["question"]).strip(),
            context=str(row["context"]).strip(),
        )
        formatted.append(
            FormattedSample(
                sample_id=str(row["id"]),
                source_benchmark_id=str(row.get("source_benchmark_id") or ""),
                prompt=prompt,
                answer=str(row["answer"]).strip(),
                language=str(row.get("language") or ""),
            )
        )
    return formatted


def split_samples(
    samples: List[FormattedSample],
    eval_ratio: float,
    seed: int,
) -> tuple[List[FormattedSample], List[FormattedSample]]:
    if len(samples) < 4 or eval_ratio <= 0:
        return samples, []

    eval_count = max(1, int(round(len(samples) * eval_ratio)))
    eval_count = min(eval_count, len(samples) - 1)
    rng = random.Random(seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)
    return shuffled[eval_count:], shuffled[:eval_count]


def _find_lora_target_modules(model) -> List[str]:
    import torch

    target_names = set()
    preferred_suffixes = {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    }

    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        suffix = name.split(".")[-1]
        if suffix in preferred_suffixes:
            target_names.add(suffix)

    if target_names:
        return sorted(target_names)

    fallback = []
    for candidate in ["q_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]:
        fallback.append(candidate)
    return fallback


def main() -> None:
    args = parse_args()

    data_path = Path(args.data)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _validate_rows(_read_jsonl(data_path), data_path)
    formatted = format_samples(rows, system_prompt=args.system_prompt)
    train_samples, eval_samples = split_samples(
        formatted,
        eval_ratio=args.eval_ratio,
        seed=args.seed,
    )

    summary = {
        "data_path": str(data_path),
        "model": args.model,
        "total_samples": len(formatted),
        "train_samples": len(train_samples),
        "eval_samples": len(eval_samples),
        "max_seq_len": args.max_seq_len,
        "dry_run": args.dry_run,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "system_prompt": args.system_prompt,
        "languages": sorted({sample.language for sample in formatted if sample.language}),
        "sample_ids_preview": [sample.sample_id for sample in formatted[:5]],
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print("\nDry run complete: dataset, formatting, and split are valid.")
        return

    _require_training_dependencies()

    from datasets import Dataset
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = None
    use_4bit = not args.disable_4bit and _dependency_available("bitsandbytes")
    if use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    model_kwargs = {"trust_remote_code": True}
    if quant_config is not None:
        model_kwargs["quantization_config"] = quant_config
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model.config.use_cache = False
    if quant_config is not None:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=_find_lora_target_modules(model),
    )
    model = get_peft_model(model, lora_config)

    def encode_sample(sample: FormattedSample) -> Dict:
        prompt_text = sample.prompt + "\n"
        answer_text = sample.answer + tokenizer.eos_token

        prompt_ids = tokenizer(prompt_text, add_special_tokens=True, truncation=False)["input_ids"]
        answer_ids = tokenizer(answer_text, add_special_tokens=False, truncation=False)["input_ids"]

        input_ids = (prompt_ids + answer_ids)[: args.max_seq_len]
        labels = ([-100] * len(prompt_ids) + answer_ids)[: args.max_seq_len]
        attention_mask = [1] * len(input_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    train_dataset = Dataset.from_list([encode_sample(sample) for sample in train_samples])
    eval_dataset = Dataset.from_list([encode_sample(sample) for sample in eval_samples]) if eval_samples else None

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        return_tensors="pt",
    )

    training_kwargs = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "num_train_epochs": args.epochs,
        "learning_rate": args.lr,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_total_limit": 2,
        "eval_steps": args.save_steps if eval_dataset is not None else None,
        "bf16": args.bf16,
        "fp16": args.fp16,
        "gradient_checkpointing": True,
        "lr_scheduler_type": "cosine",
        "report_to": [],
        "remove_unused_columns": False,
        "seed": args.seed,
    }

    strategy_value = "steps" if eval_dataset is not None else "no"
    training_signature = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in training_signature.parameters:
        training_kwargs["eval_strategy"] = strategy_value
    else:
        training_kwargs["evaluation_strategy"] = strategy_value

    training_args = TrainingArguments(**training_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    summary["use_4bit"] = use_4bit
    (output_dir / "run_config.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"Saved QLoRA adapter to {output_dir}")


if __name__ == "__main__":
    main()
