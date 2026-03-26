"""
Ollama LLM client.

IMPROVEMENTS over original:
- Retry logic (original failed immediately on timeout)
- Structured response parsing
- Connection health check
- Configurable timeout / temperature / max_tokens from config
"""

import logging
import time
from threading import BoundedSemaphore
from typing import Dict, Any, Optional

import requests
from fastapi import HTTPException

from app.config import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    LLM_TIMEOUT,
    LLM_REPEAT_PENALTY,
    LLM_REPEAT_LAST_N,
    LLM_MAX_CONCURRENT_REQUESTS,
    LLM_QUEUE_TIMEOUT,
)

logger = logging.getLogger("tilon.llm")

MAX_RETRIES = 2
_OLLAMA_GATE = BoundedSemaphore(value=LLM_MAX_CONCURRENT_REQUESTS)


def _acquire_ollama_slot(model: str) -> None:
    started = time.monotonic()
    acquired = _OLLAMA_GATE.acquire(timeout=LLM_QUEUE_TIMEOUT)
    waited = time.monotonic() - started

    if not acquired:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Ollama is busy (queue timeout after {LLM_QUEUE_TIMEOUT}s). "
                "Please retry shortly."
            ),
        )

    if waited >= 0.25:
        logger.info("Ollama queue wait %.2fs (model=%s)", waited, model)


def call_ollama(
    prompt: str,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Send a prompt to Ollama and return the response dict.

    Retries once on timeout. Raises HTTPException on failure.
    """
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
            "repeat_penalty": LLM_REPEAT_PENALTY,
            "repeat_last_n": LLM_REPEAT_LAST_N,
        },
    }

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            _acquire_ollama_slot(model)
            try:
                logger.debug("Ollama request (attempt %d) -> model=%s", attempt, model)
                response = requests.post(
                    f"{OLLAMA_BASE_URL}/generate",
                    json=payload,
                    timeout=LLM_TIMEOUT,
                )
            finally:
                _OLLAMA_GATE.release()

            if response.status_code != 200:
                detail = f"Ollama error: status={response.status_code}, body={response.text[:500]}"
                if response.status_code in {429, 503} and attempt < MAX_RETRIES:
                    last_error = detail
                    logger.warning("%s (attempt %d)", detail, attempt)
                    time.sleep(min(3.0, 1.5 * attempt))
                    continue

                raise HTTPException(status_code=500, detail=detail)

            data = response.json()
            logger.debug("Ollama response received (%d chars)", len(data.get("response", "")))
            return data

        except requests.exceptions.Timeout:
            last_error = f"Ollama timeout after {LLM_TIMEOUT}s (attempt {attempt})"
            logger.warning(last_error)
        except requests.exceptions.ConnectionError as e:
            last_error = f"Cannot connect to Ollama at {OLLAMA_BASE_URL}: {e}"
            logger.error(last_error)
            break  # no point retrying connection errors
        except HTTPException:
            raise
        except Exception as e:
            last_error = str(e)
            logger.error("Ollama unexpected error: %s", e)
            break

    raise HTTPException(status_code=500, detail=f"Ollama request failed: {last_error}")


def check_ollama_health() -> Dict[str, Any]:
    """Check if Ollama is reachable and return available models."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/tags", timeout=10)
        response.raise_for_status()
        return {"status": "connected", "models": response.json()}
    except Exception as e:
        return {"status": "disconnected", "error": str(e)}


def get_response_text(ollama_result: Dict[str, Any]) -> str:
    """Extract and clean the response text from Ollama output."""
    return (ollama_result.get("response") or "").strip()
