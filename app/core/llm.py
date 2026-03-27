"""
LLM runtime client.

Supports:
- Ollama-backed inference
- local Hugging Face + PEFT adapter inference
"""

from __future__ import annotations

import copy
import gc
import importlib.util
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from fastapi import HTTPException

from app.config import (
    AVAILABLE_MODELS,
    LLM_BACKEND,
    LLM_MAX_TOKENS,
    LLM_SUPPRESS_FOREIGN_SCRIPTS,
    LLM_TEMPERATURE,
    LLM_TIMEOUT,
    LOCAL_LLM_ADAPTER_PATH,
    LOCAL_LLM_BASE_MODEL,
    LOCAL_LLM_DEVICE,
    LOCAL_LLM_LOAD_IN_4BIT,
    LOCAL_LLM_LOCAL_FILES_ONLY,
    LOCAL_LLM_MAX_INPUT_TOKENS,
    LOCAL_LLM_MODEL_NAME,
    LOCAL_LLM_OOM_RETRY_INPUT_TOKENS,
    LOCAL_LLM_OOM_RETRY_MAX_TOKENS,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
)

logger = logging.getLogger("tilon.llm")

MAX_RETRIES = 2
LOCAL_BACKEND_NAME = "local_hf"
OLLAMA_BACKEND_NAME = "ollama"

_LOCAL_RUNTIME: Optional[dict[str, Any]] = None
_CJK_BAD_TOKEN_IDS: Optional[list] = None


def _dependency_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _from_pretrained_with_cache_fallback(loader, model_name: str, **kwargs):
    """
    Prefer local-only loading when configured, but recover gracefully if the
    local HF cache is incomplete and network access is actually available.
    """
    try:
        return loader(model_name, **kwargs)
    except Exception as exc:
        local_only = kwargs.get("local_files_only", False)
        if not local_only:
            raise
        if not isinstance(exc, (OSError, ValueError)):
            raise
        logger.warning(
            "Local HF cache miss for '%s' with local_files_only=True; retrying with network access enabled (%s)",
            model_name,
            exc,
        )
        retry_kwargs = dict(kwargs)
        retry_kwargs["local_files_only"] = False
        return loader(model_name, **retry_kwargs)


def get_default_model_name() -> str:
    if LLM_BACKEND == LOCAL_BACKEND_NAME:
        return LOCAL_LLM_MODEL_NAME
    return OLLAMA_MODEL


def get_available_models() -> list[str]:
    if LLM_BACKEND == LOCAL_BACKEND_NAME:
        return [LOCAL_LLM_MODEL_NAME]
    return [model.strip() for model in AVAILABLE_MODELS if model.strip()]


def _call_ollama(
    prompt: str,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    model = model or OLLAMA_MODEL
    temperature = temperature if temperature is not None else LLM_TEMPERATURE
    max_tokens = max_tokens or LLM_MAX_TOKENS

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.debug("Ollama request (attempt %d) -> model=%s", attempt, model)
            response = requests.post(
                f"{OLLAMA_BASE_URL}/generate",
                json=payload,
                timeout=LLM_TIMEOUT,
            )

            if response.status_code != 200:
                raise HTTPException(
                    status_code=500,
                    detail=f"Ollama error: status={response.status_code}, body={response.text[:500]}",
                )

            data = response.json()
            logger.debug("Ollama response received (%d chars)", len(data.get("response", "")))
            return data

        except requests.exceptions.Timeout:
            last_error = f"Ollama timeout after {LLM_TIMEOUT}s (attempt {attempt})"
            logger.warning(last_error)
        except requests.exceptions.ConnectionError as exc:
            last_error = f"Cannot connect to Ollama at {OLLAMA_BASE_URL}: {exc}"
            logger.error(last_error)
            break
        except HTTPException:
            raise
        except Exception as exc:
            last_error = str(exc)
            logger.error("Ollama unexpected error: %s", exc)
            break

    raise HTTPException(status_code=500, detail=f"Ollama request failed: {last_error}")


def _resolve_local_base_model(adapter_path: Path) -> str:
    if LOCAL_LLM_BASE_MODEL:
        return LOCAL_LLM_BASE_MODEL

    run_config_path = adapter_path / "run_config.json"
    if run_config_path.exists():
        try:
            payload = json.loads(run_config_path.read_text(encoding="utf-8"))
            model_name = str(payload.get("model") or "").strip()
            if model_name:
                return model_name
        except Exception:
            logger.warning("Failed to read base model from %s", run_config_path, exc_info=True)

    return "Qwen/Qwen2.5-7B-Instruct"


def _build_local_generation_config(model, tokenizer, temperature: Optional[float], max_tokens: Optional[int]):
    global _CJK_BAD_TOKEN_IDS

    generation_config = copy.deepcopy(model.generation_config)
    generation_config.max_new_tokens = max_tokens or LLM_MAX_TOKENS
    _eff_temp = temperature if temperature is not None else LLM_TEMPERATURE
    generation_config.do_sample = _eff_temp > 0
    # For greedy decoding (temp=0) use minimal repetition controls so the model
    # won't substitute correct Korean syllables with alternatives to avoid repeats.
    if _eff_temp == 0:
        generation_config.repetition_penalty = 1.05
        generation_config.no_repeat_ngram_size = 3
    else:
        generation_config.repetition_penalty = 1.15
        generation_config.no_repeat_ngram_size = 5
    generation_config.pad_token_id = tokenizer.pad_token_id
    generation_config.eos_token_id = tokenizer.eos_token_id
    if generation_config.do_sample:
        generation_config.temperature = _eff_temp
        generation_config.top_p = 1.0
    else:
        generation_config.temperature = None
        generation_config.top_p = None
        if hasattr(generation_config, "top_k"):
            generation_config.top_k = None

    # Suppress foreign-script tokens that must never appear in Korean output:
    # CJK (Chinese/Japanese), Cyrillic (Russian etc.), and Thai characters.
    # Builds the list once and caches it for all subsequent calls.
    # Can be disabled via LLM_SUPPRESS_FOREIGN_SCRIPTS=false in .env for testing.
    if not LLM_SUPPRESS_FOREIGN_SCRIPTS:
        return generation_config

    if _CJK_BAD_TOKEN_IDS is None:
        bad_ids = []
        for token_id in range(tokenizer.vocab_size):
            decoded = tokenizer.decode([token_id])
            if any(
                "\u4e00" <= ch <= "\u9fff"  # CJK Unified Ideographs
                or "\u0400" <= ch <= "\u04ff"  # Cyrillic (Russian/Ukrainian/etc.)
                or "\u0e00" <= ch <= "\u0e7f"  # Thai
                for ch in decoded
            ):
                bad_ids.append(token_id)
        _CJK_BAD_TOKEN_IDS = bad_ids
        logger.info("Built foreign-script suppression list (CJK+Cyrillic+Thai): %d token IDs banned", len(bad_ids))

    if _CJK_BAD_TOKEN_IDS:
        generation_config.suppress_tokens = _CJK_BAD_TOKEN_IDS

    return generation_config


def _load_local_runtime() -> dict[str, Any]:
    global _LOCAL_RUNTIME
    if _LOCAL_RUNTIME is not None:
        return _LOCAL_RUNTIME

    missing = [
        name
        for name in ("torch", "transformers", "peft", "accelerate")
        if not _dependency_available(name)
    ]
    if missing:
        raise HTTPException(
            status_code=500,
            detail=(
                "Local HF backend is not available because inference dependencies are missing: "
                + ", ".join(missing)
            ),
        )

    adapter_path = Path(LOCAL_LLM_ADAPTER_PATH)
    if not adapter_path.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Local adapter path not found: {adapter_path}",
        )

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    quant_config = None
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "local_files_only": LOCAL_LLM_LOCAL_FILES_ONLY,
    }

    if LOCAL_LLM_DEVICE == "cuda" and LOCAL_LLM_LOAD_IN_4BIT and _dependency_available("bitsandbytes"):
        from transformers import BitsAndBytesConfig

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = quant_config
        model_kwargs["device_map"] = "auto"

    base_model_name = _resolve_local_base_model(adapter_path)
    tokenizer = _from_pretrained_with_cache_fallback(
        AutoTokenizer.from_pretrained,
        base_model_name,
        use_fast=True,
        local_files_only=LOCAL_LLM_LOCAL_FILES_ONLY,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = _from_pretrained_with_cache_fallback(
        AutoModelForCausalLM.from_pretrained,
        base_model_name,
        **model_kwargs,
    )
    base_model.eval()
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    model.eval()

    _LOCAL_RUNTIME = {
        "adapter_path": str(adapter_path),
        "base_model_name": base_model_name,
        "tokenizer": tokenizer,
        "model": model,
    }
    logger.info(
        "Loaded local HF backend -> adapter=%s base_model=%s",
        adapter_path,
        base_model_name,
    )
    return _LOCAL_RUNTIME


def _free_local_cuda_cache() -> None:
    try:
        import torch
    except Exception:
        return

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _encode_local_prompt(tokenizer, rendered_prompt: str, max_input_tokens: int):
    import torch

    encoded = tokenizer(rendered_prompt, return_tensors="pt", truncation=False)
    input_ids = encoded["input_ids"]
    seq_len = input_ids.shape[1]
    if seq_len <= max_input_tokens:
        return encoded

    # Smart trimming: preserve system instructions (head) AND document context +
    # user question (tail).  The middle section is typically conversation history,
    # which is least critical for answer quality.
    head_tokens = min(768, max_input_tokens // 3)   # system prompt + instructions
    tail_tokens = max_input_tokens - head_tokens      # doc context + user question
    trimmed = {}
    for key, value in encoded.items():
        trimmed[key] = torch.cat(
            [value[:, :head_tokens], value[:, -tail_tokens:]],
            dim=1,
        )
    logger.warning(
        "Trimmed local HF prompt from %d to %d tokens (head=%d, tail=%d)",
        seq_len,
        max_input_tokens,
        head_tokens,
        tail_tokens,
    )
    return trimmed


def _clean_generated_text(text: str) -> str:
    """
    Remove stray non-Korean/non-ASCII Unicode symbols that the adapter occasionally
    emits due to tokenizer pressure from the CJK/Cyrillic/Thai suppression list.

    Intentionally kept minimal and document-agnostic: no domain-specific word
    corrections, only character-level cleanup of known garbage symbols.
    """
    # ㎞ (U+338E, "km" compatibility symbol) is never a valid Korean output character.
    # The adapter sometimes emits it when forced away from CJK by the suppression list.
    text = text.replace("㎞", "")
    # リン (Katakana) and other stray half/full-width Katakana — invalid in Korean output.
    text = text.replace("リン", "")
    return text


def _generate_local_text(
    prompt: str,
    messages: Optional[List[Dict[str, str]]] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> str:
    runtime = _load_local_runtime()
    tokenizer = runtime["tokenizer"]
    model = runtime["model"]

    # Use structured messages when provided — matches the system+user split used
    # during QLoRA training (render_document_chat in finetuning/train.py).
    # Falling back to a single user-role wrap only when no messages are given
    # (e.g. raw string prompts from the language-retry path).
    chat_messages = messages if messages else [{"role": "user", "content": prompt}]
    rendered_prompt = tokenizer.apply_chat_template(
        chat_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    generation_config = _build_local_generation_config(model, tokenizer, temperature, max_tokens)

    import torch

    encoded = _encode_local_prompt(
        tokenizer,
        rendered_prompt,
        LOCAL_LLM_MAX_INPUT_TOKENS,
    )
    encoded = {key: value.to(model.device) for key, value in encoded.items()}

    try:
        with torch.inference_mode():
            output = model.generate(**encoded, generation_config=generation_config)
    except torch.cuda.OutOfMemoryError:
        logger.warning(
            "Local HF generation OOM; retrying with shorter input/output budget",
            exc_info=True,
        )
        _free_local_cuda_cache()
        retry_max_tokens = min(max_tokens or LLM_MAX_TOKENS, LOCAL_LLM_OOM_RETRY_MAX_TOKENS)
        retry_config = _build_local_generation_config(model, tokenizer, temperature, retry_max_tokens)
        retry_encoded = _encode_local_prompt(
            tokenizer,
            rendered_prompt,
            LOCAL_LLM_OOM_RETRY_INPUT_TOKENS,
        )
        retry_encoded = {key: value.to(model.device) for key, value in retry_encoded.items()}
        try:
            with torch.inference_mode():
                output = model.generate(**retry_encoded, generation_config=retry_config)
            encoded = retry_encoded
        except torch.cuda.OutOfMemoryError as exc:
            _free_local_cuda_cache()
            raise HTTPException(
                status_code=500,
                detail=(
                    "Local HF inference ran out of GPU memory. "
                    "문서 범위를 줄이거나 더 짧은 질문으로 다시 시도해 주세요."
                ),
            ) from exc
    finally:
        _free_local_cuda_cache()

    answer = tokenizer.decode(
        output[0][encoded["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()
    del output
    return _clean_generated_text(answer)


def generate_text(
    prompt: str,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    messages: Optional[List[Dict[str, str]]] = None,
) -> str:
    if LLM_BACKEND == LOCAL_BACKEND_NAME:
        return _generate_local_text(
            prompt=prompt,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    result = _call_ollama(
        prompt=prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return (result.get("response") or "").strip()


def check_llm_health() -> Dict[str, Any]:
    if LLM_BACKEND == LOCAL_BACKEND_NAME:
        adapter_path = Path(LOCAL_LLM_ADAPTER_PATH)
        missing = [
            name
            for name in ("torch", "transformers", "peft", "accelerate")
            if not _dependency_available(name)
        ]
        if missing:
            return {
                "status": "disconnected",
                "backend": LOCAL_BACKEND_NAME,
                "model": LOCAL_LLM_MODEL_NAME,
                "error": f"Missing dependencies: {', '.join(missing)}",
            }
        if not adapter_path.exists():
            return {
                "status": "disconnected",
                "backend": LOCAL_BACKEND_NAME,
                "model": LOCAL_LLM_MODEL_NAME,
                "error": f"Adapter path not found: {adapter_path}",
            }
        return {
            "status": "connected",
            "backend": LOCAL_BACKEND_NAME,
            "model": LOCAL_LLM_MODEL_NAME,
            "adapter_path": str(adapter_path),
            "loaded": _LOCAL_RUNTIME is not None,
        }

    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/tags", timeout=10)
        response.raise_for_status()
        return {
            "status": "connected",
            "backend": OLLAMA_BACKEND_NAME,
            "model": OLLAMA_MODEL,
            "models": response.json(),
        }
    except Exception as exc:
        return {
            "status": "disconnected",
            "backend": OLLAMA_BACKEND_NAME,
            "model": OLLAMA_MODEL,
            "error": str(exc),
        }


def call_ollama(
    prompt: str,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    return _call_ollama(
        prompt=prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def check_ollama_health() -> Dict[str, Any]:
    return check_llm_health()


def get_response_text(result: Dict[str, Any]) -> str:
    if "response" in result:
        return (result.get("response") or "").strip()
    return (result.get("text") or "").strip()
