"""
Compare base-model and QLoRA-adapter answers on the same prompt.

Example:
    python finetuning/infer_compare.py \
      --question "런케이션 프로그램의 지원조건은 무엇인가요?" \
      --context-file /tmp/context.txt \
      --adapter finetuning/output/qwen25-qlora-v1
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from train import DEFAULT_SYSTEM_PROMPT, build_document_prompt


def _dependency_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _require_inference_dependencies() -> None:
    missing = [
        name
        for name in ["torch", "transformers", "peft", "accelerate"]
        if not _dependency_available(name)
    ]
    if missing:
        joined = ", ".join(missing)
        raise SystemExit(
            "Missing inference dependencies: "
            f"{joined}\n"
            "Install them first, for example:\n"
            "  pip install -r finetuning/requirements-qlora.txt"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare base and adapter answers.")
    parser.add_argument(
        "--adapter",
        default="finetuning/output/qwen25-qlora-v1",
        help="Path to trained adapter directory.",
    )
    parser.add_argument(
        "--model",
        default="",
        help="Base Hugging Face model. If omitted, try adapter run_config.json first.",
    )
    parser.add_argument("--question", required=True, help="User question to test.")
    parser.add_argument(
        "--context",
        default="",
        help="Retrieved context string in live [Doc: ...] format.",
    )
    parser.add_argument(
        "--context-file",
        default="",
        help="Optional text file containing retrieved context.",
    )
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt used to build the document-QA prompt.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Generation length.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p sampling.")
    parser.add_argument(
        "--disable-4bit",
        action="store_true",
        help="Disable 4-bit model loading even if bitsandbytes is available.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use fp16 compute dtype when loading in 4-bit mode.",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Use bf16 compute dtype when loading in 4-bit mode.",
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


def _load_context(args: argparse.Namespace) -> str:
    if args.context_file:
        return Path(args.context_file).read_text(encoding="utf-8").strip()
    return str(args.context).strip()


def main() -> None:
    args = parse_args()
    _require_inference_dependencies()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    adapter_path = Path(args.adapter)
    if not adapter_path.exists():
        raise SystemExit(f"Adapter path not found: {adapter_path}")

    base_model_name = _resolve_base_model(adapter_path, args.model)
    context = _load_context(args)
    prompt = build_document_prompt(
        system_prompt=args.system_prompt,
        user_message=args.question.strip(),
        context=context,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
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

    model_kwargs = {"trust_remote_code": True}
    if quant_config is not None:
        model_kwargs["quantization_config"] = quant_config
        model_kwargs["device_map"] = "auto"

    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, **model_kwargs)
    base_model.eval()

    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {key: value.to(base_model.device) for key, value in encoded.items()}

    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
        "temperature": args.temperature if args.temperature > 0 else None,
        "top_p": args.top_p if args.temperature > 0 else None,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    generation_kwargs = {k: v for k, v in generation_kwargs.items() if v is not None}

    with torch.no_grad():
        base_output = base_model.generate(**encoded, **generation_kwargs)
    base_text = tokenizer.decode(
        base_output[0][encoded["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()

    adapter_model = PeftModel.from_pretrained(base_model, str(adapter_path))
    adapter_model.eval()
    with torch.no_grad():
        adapter_output = adapter_model.generate(**encoded, **generation_kwargs)
    adapter_text = tokenizer.decode(
        adapter_output[0][encoded["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()

    result = {
        "base_model": base_model_name,
        "adapter_path": str(adapter_path),
        "question": args.question.strip(),
        "context_chars": len(context),
        "base_answer": base_text,
        "adapter_answer": adapter_text,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
