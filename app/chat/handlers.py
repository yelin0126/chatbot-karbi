"""
Mode-specific chat handlers.

IMPROVEMENTS over original:
- Web search actually calls Tavily (original just asked Ollama)
- Handlers return structured dicts consistently
- OCR handler simplified
"""

import logging
from typing import List, Dict, Any

from app.models.schemas import Message
from app.core.llm import call_ollama, get_response_text
from app.retrieval.retriever import retrieve, format_context, extract_sources
from app.chat.prompts import (
    build_general_prompt,
    build_document_prompt,
    build_web_prompt,
)
from app.config import OLLAMA_MODEL, TAVILY_API_KEY

logger = logging.getLogger("tilon.handlers")


def handle_general(
    system_prompt: str,
    history: List[Message],
    user_message: str,
    model: str = None,
) -> Dict[str, Any]:
    """Handle general conversation — no retrieval needed."""
    prompt = build_general_prompt(system_prompt, history, user_message)
    result = call_ollama(prompt, model=model or OLLAMA_MODEL)

    return {
        "answer": get_response_text(result),
        "sources": [],
        "mode": "general",
    }


def handle_document_qa(
    system_prompt: str,
    history: List[Message],
    user_message: str,
    model: str = None,
) -> Dict[str, Any]:
    """Handle document-based Q&A — retrieve context then generate."""
    docs = retrieve(user_message)
    context = format_context(docs)

    prompt = build_document_prompt(system_prompt, history, user_message, context)
    result = call_ollama(prompt, model=model or OLLAMA_MODEL)

    return {
        "answer": get_response_text(result),
        "sources": extract_sources(docs),
        "mode": "document_qa",
    }


def handle_ocr_extract(user_message: str) -> Dict[str, Any]:
    """Handle OCR text extraction — return raw text from documents."""
    docs = retrieve(user_message)

    if not docs:
        return {
            "answer": "추출 가능한 문서 텍스트를 찾지 못했습니다.",
            "sources": [],
            "mode": "ocr_extract",
        }

    extracted_text = "\n\n".join(d.page_content for d in docs)

    return {
        "answer": extracted_text,
        "sources": extract_sources(docs),
        "mode": "ocr_extract",
    }


def _tavily_search(query: str) -> str:
    """
    NEW: Actually perform web search using Tavily API.
    Original code had a web_search mode but never searched — it just
    asked Ollama, which has no internet access and can't answer
    real-time questions (weather, stock prices, news, etc.).
    """
    if not TAVILY_API_KEY:
        logger.warning("Tavily API key not set — web search unavailable.")
        return ""

    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=TAVILY_API_KEY)
        response = client.search(query, max_results=3)

        results = []
        for r in response.get("results", []):
            title = r.get("title", "")
            content = r.get("content", "")
            url = r.get("url", "")
            results.append(f"- {title}\n  {content}\n  ({url})")

        return "\n\n".join(results)

    except ImportError:
        logger.warning("tavily-python not installed. Install with: pip install tavily-python")
        return ""
    except Exception as e:
        logger.error("Tavily search failed: %s", e)
        return ""


def handle_web_search(
    system_prompt: str,
    history: List[Message],
    user_message: str,
    model: str = None,
) -> Dict[str, Any]:
    """
    Handle web search queries — search first, then generate answer.

    IMPROVEMENT: Actually searches the web via Tavily before asking the LLM.
    """
    search_results = _tavily_search(user_message)

    prompt = build_web_prompt(system_prompt, history, user_message, search_results)
    result = call_ollama(prompt, model=model or OLLAMA_MODEL)

    return {
        "answer": get_response_text(result),
        "sources": [],
        "mode": "web_search",
    }
