import re
from typing import Any, Dict, List, Tuple

from app.chat.policy import (
    is_smalltalk_query,
    needs_chain_of_thought,
    needs_comparison_style,
    needs_multi_document_summary_style,
    needs_section_understanding_style,
)
from app.chat.validation import infer_target_language
from app.models.schemas import Message


SYSTEM_PROMPT = """You are Tilon AI, a helpful document-based chatbot.

CRITICAL RULES:
1. Respond in the SAME language the user is using. Korean → Korean. English → English. NEVER respond in Chinese (中文).
2. If document context is provided and relevant to the question, answer based on that context and cite the source (document name, page number).
3. If document context is provided but NOT relevant to the question, say so clearly: "해당 내용은 제공된 문서에서 확인되지 않습니다" (Korean) or "This is not covered in the provided documents" (English). Do NOT fabricate an answer from irrelevant context.
4. If no document context is available, answer from your general knowledge.
5. If web search results are provided, use them for current/real-time information.
6. If you don't know, say so honestly. Never make up information.
7. Be concise and direct. Answer the question first, then explain if needed.
8. When citing documents, use inline citation markers like [1], [2] that match the chunk numbers in the provided context. Example: "지원 대상은 중소기업입니다 [1]." or "The budget is $5M [3]."
9. If you answer from document evidence, every important factual sentence or bullet should include at least one inline citation marker before the sentence ends.
10. If a sentence cannot be supported with a citation, remove or rewrite that sentence. """


COMPACT_SYSTEM_PROMPT = (
    "너는 한국어로 답하는 AI 챗봇이다. "
    "짧은 질문에는 짧고 자연스럽게 답한다.\n\n"
    "답변 규칙:\n"
    "1. CRITICAL: Respond in the SAME language the user is using. "
    "Korean→Korean, English→English. NEVER respond in Chinese (中文).\n"
    "2. 문서 문맥이 제공된 경우 그것을 근거로 답하고, 인라인 출처 번호 [1], [2] 등을 표기한다.\n"
    "3. 문서 문맥이 질문과 무관하면 '해당 내용은 제공된 문서에서 확인되지 않습니다'라고 답한다. 없는 내용을 지어내지 마라.\n"
    "4. 문서 문맥이 없으면 일반 지식으로 자연스럽게 답한다.\n"
    "5. 핵심 답변을 먼저 말한다.\n"
    "6. 이미지에서 추출된 텍스트가 제공되면 해당 텍스트를 기반으로 답한다.\n"
    "7. 문서 근거로 답하는 문장이나 bullet에는 반드시 [1], [2] 같은 인라인 출처 번호를 넣어라."
)

LOCAL_HF_DOC_CONTEXT_CHAR_LIMIT = 8500
_INLINE_CITATION_RE = re.compile(r"\[\d+\]")
_CITATION_FOOTER_RE = re.compile(r"\n---\n\*\*(출처|Sources):\*\*.*$", re.DOTALL)
_NOT_FOUND_RE = re.compile(
    r"(찾을 수 없|찾지 못했|확인되지 않|충분히 확인되지 않|관련된 내용이 포함되어 있지 않|"
    r"couldn't find|could not find|don't contain enough information|not covered in the provided documents|"
    r"doesn't appear to contain information)",
    re.IGNORECASE,
)


def build_citation_footer(docs, max_entries: int = 8) -> str:
    """Build a compact citation footer mapping [1], [2], ... to source filenames and pages."""
    if not docs:
        return ""
    unique_docs = []
    seen = set()
    for d in docs:
        source = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", "?")
        section = (
            d.metadata.get("section_breadcrumb", "")
            or d.metadata.get("section_title", "")
        )
        key = (source, page, section)
        if key in seen:
            continue
        seen.add(key)
        unique_docs.append(d)

    lines: List[str] = []
    for idx, d in enumerate(unique_docs[:max_entries], start=1):
        source = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", "?")
        section = (
            d.metadata.get("section_breadcrumb", "")
            or d.metadata.get("section_title", "")
        )
        label = f"[{idx}] {source}"
        if page and page != "?":
            label += f", p.{page}"
        if section:
            label += f" — {section}"
        lines.append(label)
    if len(unique_docs) > len(lines):
        lines.append(f"... and {len(unique_docs) - len(lines)} more cited chunk(s)")
    return "\n---\n**출처:**\n" + "\n".join(lines)


def has_inline_citations(answer: str) -> bool:
    body = _CITATION_FOOTER_RE.sub("", answer or "")
    return bool(_INLINE_CITATION_RE.search(body))


def is_grounding_fallback_answer(answer: str) -> bool:
    return bool(_NOT_FOUND_RE.search(answer or ""))


def ensure_minimum_inline_citations(answer: str, docs) -> str:
    """
    Add a minimal inline citation fallback when the model omitted all inline markers.

    This is intentionally conservative:
    - do nothing if citations already exist
    - do nothing for not-found / grounding fallback answers
    - cite only the top retrieved chunk as a last-resort anchor
    """
    if not answer or not docs or has_inline_citations(answer) or is_grounding_fallback_answer(answer):
        return answer

    body = _CITATION_FOOTER_RE.sub("", answer).rstrip()
    if not body:
        return answer

    lines = body.splitlines()
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("---"):
            continue
        if _INLINE_CITATION_RE.search(stripped):
            return answer
        lines[idx] = line.rstrip() + " [1]"
        return "\n".join(lines).strip()

    return answer


def format_history(history: List[Message], max_turns: int = 8) -> str:
    if not history:
        return ""
    return "\n\n".join(
        f"[{msg.role}]\n{msg.content}" for msg in history[-max_turns:]
    )


def trim_doc_context_for_local_hf(doc_context: str) -> str:
    """Trim retrieved doc context by dropping trailing chunk blocks."""
    if not doc_context or len(doc_context) <= LOCAL_HF_DOC_CONTEXT_CHAR_LIMIT:
        return doc_context

    blocks = re.split(r"(?=\[Doc: )", doc_context)
    kept: List[str] = []
    total = 0
    for block in blocks:
        stripped = block.strip()
        if not stripped:
            continue
        if total + len(stripped) > LOCAL_HF_DOC_CONTEXT_CHAR_LIMIT and kept:
            break
        kept.append(stripped)
        total += len(stripped)

    trimmed = "\n\n".join(kept)
    n_dropped = sum(1 for b in blocks if b.strip()) - len(kept)
    if n_dropped > 0:
        trimmed += f"\n\n[Note: {n_dropped} lower-ranked chunk(s) omitted to fit context window]"
    return trimmed


def build_prompt(
    user_message: str,
    history: List[Message],
    doc_context: str = "",
    web_context: str = "",
    system_prompt: str = "",
    selected_doc_count: int = 0,
    document_scope_count: int = 0,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Build a unified prompt and matching structured messages."""
    sys_parts: List[str] = [f"[System]\n{system_prompt or SYSTEM_PROMPT}"]
    target_lang = infer_target_language(user_message)
    if target_lang == "ko":
        sys_parts.append("[Required response language]\nKorean only. Do not use Chinese characters.")
    elif target_lang == "en":
        sys_parts.append(
            "[Required response language]\nEnglish only. Do not use Chinese characters.\n"
            "[Cross-language instruction]\n"
            "If the retrieved document evidence is in Korean, translate and explain it in English. "
            "Do not refuse only because the source document is in another language."
        )
    if needs_section_understanding_style(user_message):
        sys_parts.append(
            "[Task style]\n"
            "Answer as a section-understanding question. "
            "Explain the role, purpose, differences, or kinds of support described in the document. "
            "Name concrete items from the evidence instead of giving only a generic summary."
        )
    if document_scope_count > 1 and needs_multi_document_summary_style(user_message) and not needs_comparison_style(user_message):
        sys_parts.append(
            "[Multi-document summary instruction]\n"
            "You are answering a multi-document summary question.\n"
            "Summarize the retrieved documents file by file before giving any overall theme.\n"
            "You MUST create exactly one summary block per distinct file in the retrieved context.\n"
            "You MUST cover every distinct file that appears in the retrieved context.\n"
            "Use the actual filename or title as the heading for each block.\n"
            "Do not split one file into multiple headings just because it has several sections or clauses.\n"
            "Do not repeat the same file under slightly different names.\n"
            "Keep each file summary short and concrete.\n"
            "For each file, mention only:\n"
            "1. what the document is about,\n"
            "2. the most important points,\n"
            "3. anything notably different or special.\n"
            "Do not merge unrelated documents into one topic.\n"
            "If some documents are unrelated, say that clearly.\n"
            "Do not invent broad common themes unless multiple documents clearly support them.\n"
            "Only include a final overall summary if it is genuinely useful.\n"
            "Prefer this structure.\n"
            "If the user is writing in Korean:\n"
            "- 업로드된 문서별 요약:\n"
            "- 파일명: ...\n"
            "  - 요약:\n"
            "  - 핵심 내용:\n"
            "  - 특이사항:\n"
            "- 전체 요약(선택 사항):\n"
            "If the user is writing in English:\n"
            "- Uploaded file summaries:\n"
            "- File: ...\n"
            "  - Summary:\n"
            "  - Key points:\n"
            "  - Notable notes:\n"
            "- Overall summary (optional):\n"
            "Every point must stay grounded in the retrieved evidence."
        )
    elif selected_doc_count > 1 or needs_comparison_style(user_message):
        sys_parts.append(
            "[Comparison instruction]\n"
            "You are answering a document-comparison question.\n"
            "Compare only the selected documents in the retrieved document context.\n"
            "Do not merge multiple documents into one policy or invent shared rules.\n"
            "For each comparison point, explicitly say which document it belongs to.\n"
            "If a point is supported by only one document, say that clearly.\n"
            "If the evidence is insufficient for a comparison point, say it is not confirmed in the provided documents.\n"
            "Focus only on the dimension asked by the user, such as purpose, eligibility, procedure, support, or termination.\n"
            "Prefer a compact side-by-side structure.\n"
            "If the user is writing in Korean, use this structure:\n"
            "- 핵심 비교:\n"
            "- 문서별 요약:\n"
            "- 공통점:\n"
            "- 차이점:\n"
            "If the user is writing in English, use this structure:\n"
            "- Key comparison:\n"
            "- By document:\n"
            "- Similarities:\n"
            "- Differences:\n"
            "Every section must stay grounded in the retrieved evidence."
        )
    if needs_chain_of_thought(user_message) and not is_smalltalk_query(user_message):
        sys_parts.append(
            "[Reasoning instruction]\n"
            "This question requires step-by-step reasoning.\n"
            "First, gather and organize the key facts and evidence from the provided context.\n"
            "Then, reason through the answer before giving your final response.\n"
            "Structure your answer as: (1) Key facts, (2) Analysis, (3) Final answer."
        )

    usr_parts: List[str] = []
    history_text = format_history(history)
    if history_text:
        usr_parts.append(f"[Conversation history]\n{history_text}")
    if doc_context:
        usr_parts.append(f"[Retrieved document context]\n{doc_context}")
    if web_context:
        usr_parts.append(f"[Web search results]\n{web_context}")
    usr_parts.append(f"[User message]\n{user_message}")

    system_content = "\n\n".join(sys_parts)
    user_content = "\n\n".join(usr_parts)
    prompt = system_content + "\n\n" + user_content

    compact_sys_parts: List[str] = [COMPACT_SYSTEM_PROMPT]
    if target_lang == "en":
        compact_sys_parts.append(
            "If document evidence is in Korean and the question is in English, "
            "translate and explain the answer in English."
        )
    if needs_section_understanding_style(user_message):
        compact_sys_parts.append(
            "Task: Explain the role, purpose, or differences of the described section "
            "with concrete items from the evidence."
        )
    if document_scope_count > 1 and needs_multi_document_summary_style(user_message) and not needs_comparison_style(user_message):
        compact_sys_parts.append(
            "Task: 파일별로 요약하되 각 파일을 독립적으로 다루고 파일명을 헤딩으로 써라."
        )
    elif selected_doc_count > 1 or needs_comparison_style(user_message):
        compact_sys_parts.append(
            "Task: 문서별로 비교하고 각 포인트가 어느 문서에서 나왔는지 명시해라."
        )
    if needs_chain_of_thought(user_message) and not is_smalltalk_query(user_message):
        compact_sys_parts.append(
            "Task: 먼저 관련 사실과 근거를 단계별로 정리한 뒤 최종 답변을 작성하라."
        )

    compact_usr_parts: List[str] = []
    if history_text:
        compact_usr_parts.append(f"[Conversation history]\n{history_text}")
    if doc_context:
        compact_usr_parts.append(
            f"[Retrieved document context]\n{trim_doc_context_for_local_hf(doc_context)}"
        )
    if web_context:
        compact_usr_parts.append(f"[Web search results]\n{web_context}")
    compact_usr_parts.append(f"[User message]\n{user_message}")

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": "\n".join(compact_sys_parts)},
        {"role": "user", "content": "\n\n".join(compact_usr_parts)},
    ]
    return prompt, messages
