"""
OpenAI-compatible API endpoints (/v1/models, /v1/chat/completions).

Allows using this server as a drop-in replacement for OpenAI API
in tools like Open WebUI, Continue.dev, etc.

No changes from original logic — just cleanly separated.
"""

import uuid
import time
import logging
from typing import List

from fastapi import APIRouter, HTTPException

from app.config import OLLAMA_MODEL
from app.models.schemas import Message, OpenAIMessage, OpenAIChatRequest
from app.chat.router import detect_mode
from app.chat.handlers import (
    handle_general,
    handle_document_qa,
    handle_ocr_extract,
    handle_web_search,
)

logger = logging.getLogger("tilon.openai_compat")

router = APIRouter(prefix="/v1", tags=["OpenAI Compatible"])


def _convert_openai_messages(messages: List[OpenAIMessage]):
    """Convert OpenAI-format messages to internal format."""
    system_prompt = (
        "너는 한국어로 답하는 AI 챗봇이다. "
        "짧은 질문에는 짧게 답하고, 이전 문맥을 불필요하게 끌고 오지 않는다."
    )
    history: List[Message] = []
    user_message = ""

    for msg in messages:
        if msg.role == "system":
            system_prompt = msg.content
        elif msg.role == "user":
            user_message = msg.content
            history.append(Message(role="user", content=msg.content))
        elif msg.role == "assistant":
            history.append(Message(role="assistant", content=msg.content))

    # Remove last user message from history (it's the current query)
    if history and history[-1].role == "user":
        user_message = history[-1].content
        history = history[:-1]

    return system_prompt, history, user_message


@router.get("/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": OLLAMA_MODEL,
                "object": "model",
                "created": 0,
                "owned_by": "local",
            }
        ],
    }


@router.post("/chat/completions")
def chat_completions(req: OpenAIChatRequest):
    try:
        system_prompt, history, user_message = _convert_openai_messages(req.messages)
        selected_model = req.model or OLLAMA_MODEL
        mode = detect_mode(user_message)

        if mode == "general":
            result = handle_general(system_prompt, history, user_message, selected_model)
        elif mode == "document_qa":
            result = handle_document_qa(system_prompt, history, user_message, selected_model)
        elif mode == "ocr_extract":
            result = handle_ocr_extract(user_message)
        elif mode == "web_search":
            result = handle_web_search(system_prompt, history, user_message, selected_model)
        else:
            result = handle_general(system_prompt, history, user_message, selected_model)

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": selected_model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": result["answer"],
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("OpenAI-compatible chat failed")
        raise HTTPException(status_code=500, detail=f"OpenAI-compatible chat failed: {e}")
