import re
from dataclasses import dataclass

from app.chat.query_classifier import classify_query as _classify_query
from app.config import QUERY_CLASSIFIER_SHADOW_MODE


_ENGLISH_SMALLTALK_RE = re.compile(
    r"\b(?:hello|hi|thanks|thank you|okay|ok|got it)\b",
    re.IGNORECASE,
)
_KOREAN_ACK_RE = re.compile(
    r"^\s*(?:응|네|넵|예|그래|좋아)\s*[.!?~]*\s*$",
    re.IGNORECASE,
)
_ASSISTANT_REFERENCE_RE = re.compile(
    r"(?:\b(?:you|your|yourself|assistant|chatbot|bot|helper)\b|너|넌|네\b|네가|너는|당신|당신은|챗봇|봇|도우미|어시스턴트)",
    re.IGNORECASE,
)
_ASSISTANT_META_RE = re.compile(
    r"(?:\b(?:who|what|introduce|about|capabilities|capable|role|help\s+with|can\s+you\s+do)\b|누구|자기소개|소개|정체|역할|무엇을\s*할\s*수|뭘\s*할\s*수|뭐\s*할\s*수|무슨\s*일|도와줄\s*수)",
    re.IGNORECASE,
)
_TECH_COMPARISON_TERMS = (
    "embedding", "임베딩", "reranker", "리랭커", "rag",
    "transformer", "트랜스포머", "llm", "벡터", "vector search",
)
_DOC_COMPARISON_RE = re.compile(
    r"(?:이|저|두)\s*(?:문서|파일|규정|지침|정책)|\b(?:these|two)\s+(?:documents|files|policies|rules|guidelines)\b",
    re.IGNORECASE,
)
_ARTICLE_TO_ARTICLE_COMPARISON_RE = re.compile(
    r"제\s*\d+\s*조\s*(?:와|하고|및)\s*제\s*\d+\s*조",
    re.IGNORECASE,
)
_ARTICLE_RANGE_RE = re.compile(
    r"제\s*\d+\s*조(?:\s*(?:부터|-|~)\s*제?\s*\d+\s*조|(?:\s*,\s*제?\s*\d+\s*조)+)",
    re.IGNORECASE,
)


def might_need_web_search(text: str) -> bool:
    """Light heuristic: does this query likely need real-time info?"""
    indicators = [
        "오늘", "현재", "최신", "최근", "실시간", "지금", "올해",
        "날씨", "뉴스", "환율", "주가", "시세", "속보",
        "today", "current", "latest", "recent", "now", "weather",
        "news", "stock", "price", "score", "live",
        "검색", "search", "look up", "find out",
    ]
    lower = text.lower()
    return any(kw in lower for kw in indicators)


def is_general_knowledge_query(text: str) -> bool:
    """
    True when the query looks like a factual general-knowledge question that
    has no real-time component but would benefit from web search when no
    document context was found.
    """
    lower = text.lower()
    if _DOC_COMPARISON_RE.search(text) and re.search(r"(차이|다르|다른\s*점|비교|difference|differ|compare)", lower):
        return False
    define_patterns = [
        r"(이란|란)\s*(뭐|무엇|어떤|어떻게|왜)",
        r"(정의|개념|원리|이론|역사)\s*(해줘|해\s*주세요|해봐|알려줘|알려\s*주세요)?",
        r"\b(?:define|what\s+is|how\s+does|why\s+is)\b",
    ]
    if any(re.search(p, lower) for p in define_patterns):
        return True
    tech_terms = [
        "rag", "llm", "ai", "ml", "transformer", "bert", "gpt",
        "딥러닝", "머신러닝", "강화학습", "신경망", "embedding", "임베딩", "벡터",
        "벡터 검색", "reranker", "리랭커", "트랜스포머", "검색 증강 생성",
        "retrieval augmented generation",
        "python", "java", "javascript", "api", "docker", "kubernetes",
        "blockchain", "nft", "chatgpt", "claude", "gemini",
        "경제", "gdp", "인플레이션", "금리", "환율이란",
    ]
    tech_hit = any(t in lower for t in tech_terms if len(t) >= 3)
    if tech_hit:
        return True
    if re.search(r"(차이|다르|비교|difference|compare)", lower):
        if sum(1 for term in _TECH_COMPARISON_TERMS if term in lower) >= 1:
            return True
    return False


def needs_chain_of_thought(text: str) -> bool:
    """
    True when the query is complex enough to benefit from explicit step-by-step
    reasoning in the LLM's answer.
    """
    lower = text.lower()
    cot_patterns = [
        r"(왜|어떻게|어떤\s*이유|차이|비교|장단점|문제점|원인|결과|장점|단점|분석)",
        r"(why|how\s+does|compare|difference\s+between|pros\s+and\s+cons|explain\s+why|analyze)",
        r"(단계|절차|과정|프로세스|flow|step|procedure)",
        r"[?？].*[?？]",
    ]
    return any(re.search(p, lower) for p in cot_patterns)


def needs_full_document_context(text: str) -> bool:
    """
    Detect requests that need the whole uploaded document, not just top-k chunks.
    """
    lower = text.lower().strip()
    indicators = [
        "요약", "요약해", "정리", "정리해", "전체", "전반", "구조", "목차",
        "분석", "분석해", "핵심", "주요 내용", "전체 내용", "섹션", "section",
        "structure", "outline", "overview", "summarize", "summary",
        "analyze", "analysis", "key points", "main points", "extract key",
        "extract data", "important information", "important info",
        "부칙", "경과조치", "지원대상", "운영절차", "시행일", "준용", "위원장",
        "article", "clause", "section title",
    ]
    if any(keyword in lower for keyword in indicators):
        return True
    if re.search(r"총\s*몇|몇\s*가지|몇\s*항목|몇\s*개|몇\s*장|how\s+many", lower):
        return True
    return bool(re.search(r"제\s*\d+\s*조", text))


def needs_multi_document_summary_style(text: str) -> bool:
    """Detect summary-style requests over multiple documents."""
    lower = text.lower().strip()
    if needs_strict_fact_style(text):
        return False
    indicators = [
        "요약", "전체적으로", "전반적으로", "한번에", "묶어서", "파일별",
        "문서별", "각 문서", "각 파일", "차례", "목차", "구성 범주", "전체 구조",
        "전체 흐름", "흐름", "개괄", "한눈에", "큰 그림",
        "summarize", "summary", "overview", "overall", "together",
        "by file", "each file", "each document", "structure overview",
        "big picture", "walk through the whole document", "document flow",
        "outline the structure", "big-picture flow",
    ]
    if any(keyword in lower for keyword in indicators):
        return True
    regexes = [
        r"정리(?:해|해줘|해\s*줘|해주세요|해\s*주세요|해서|좀)",
        r"\boutline\s+the\s+structure\b",
        r"\bhow\s+the\s+whole\s+document\s+flows\b",
    ]
    return any(re.search(pattern, lower) for pattern in regexes)


def needs_whole_corpus_full_context(text: str) -> bool:
    """
    Be much stricter for "all uploaded documents" mode.
    """
    lower = text.lower().strip()
    if needs_multi_document_summary_style(text):
        return True

    indicators = [
        "업로드된 문서 전체", "업로드된 전체", "모든 업로드", "전체 업로드",
        "all uploaded", "entire upload set", "whole upload corpus",
    ]
    if any(keyword in lower for keyword in indicators):
        return True
    return bool(re.search(r"총\s*몇|몇\s*가지|몇\s*항목|몇\s*개|몇\s*장|how\s+many", lower))


def is_assistant_meta_query(text: str) -> bool:
    """
    Detect identity/capability questions about the assistant itself.

    This stays broader than phrase-specific matching by looking for:
    1. a reference to the assistant ("you", "assistant", "너", "챗봇")
    2. identity/capability cues ("who", "introduce", "role", "무엇을 할 수")
    """
    lower = text.lower().strip()
    if "자기소개" in lower:
        return True
    return bool(
        _ASSISTANT_REFERENCE_RE.search(lower)
        and _ASSISTANT_META_RE.search(lower)
    )


def is_smalltalk_query(text: str) -> bool:
    """Detect greetings/acknowledgements that should bypass document scoping."""
    lower = text.lower().strip()
    korean_indicators = [
        "안녕", "고마워", "감사", "오케이", "알겠",
    ]
    if any(keyword in lower for keyword in korean_indicators) or _KOREAN_ACK_RE.match(lower):
        return True
    if is_assistant_meta_query(lower):
        return True
    return bool(_ENGLISH_SMALLTALK_RE.search(lower))


def is_direct_extraction_query(text: str) -> bool:
    """Detect requests that want raw OCR/text output from the uploaded file."""
    lower = text.lower().strip()
    if needs_article_lookup(text):
        return False
    indicators = [
        "텍스트 추출", "문자 추출", "글자 추출", "읽어줘", "텍스트만", "원문", "ocr",
        "내용 추출", "내용 보여", "전문 보여", "본문 보여", "여기 안에 있는 내용 추출",
        "글자 그대로", "텍스트 그대로", "그대로 보여", "문구 그대로", "OCR 결과만",
        "문구를 그대로", "한 글자도 바꾸지 말고", "보이는 문구", "텍스트를 적어줘",
        "가사 추출", "가사 보여", "가사 읽어", "lyrics", "lyric",
        "what does this image say", "what does the image say",
        "give me the text", "extract the text", "read the text",
        "read this image", "text in the image", "transcribe",
        "what text appears", "exactly what text", "show me the text",
        "text in this screenshot", "screenshot text", "exact words visible",
        "without summarizing", "uploaded image without summarizing",
    ]
    if any(keyword in lower for keyword in indicators):
        return True
    regexes = [
        r"스크린샷.*(?:문구|텍스트).*(?:그대로|적어줘|보여줘)",
        r"이미지.*텍스트.*(?:그대로|바꾸지\s*말고|보여줘)",
        r"\bwrite\s+out\s+the\s+exact\s+words\b",
        r"\bexact\s+text\b.*\bwithout\s+summarizing\b",
    ]
    return any(re.search(pattern, lower) for pattern in regexes)


def needs_section_understanding_style(text: str) -> bool:
    """Detect prompts asking for the role/meaning of a section rather than raw lookup."""
    lower = text.lower().strip()
    indicators = [
        "어떤 역할", "무슨 역할", "어떤 의미", "무슨 의미", "설명해줘", "설명해 줘",
        "차이", "구분", "성격", "의미를", "역할을", "what role", "what does this section do",
        "what does it mean", "explain the section", "difference between", "how is it different",
    ]
    return any(keyword in lower for keyword in indicators)


def needs_comparison_style(text: str) -> bool:
    lower = text.lower().strip()
    if not (_DOC_COMPARISON_RE.search(text) or needs_article_lookup(text)):
        if any(term in lower for term in _TECH_COMPARISON_TERMS):
            return False
    indicators = [
        "비교", "차이", "다른 점", "공통점", "구분", "대조",
        "compare", "comparison", "difference", "differences", "similarity", "differ",
    ]
    if any(keyword in lower for keyword in indicators):
        return True
    if _ARTICLE_TO_ARTICLE_COMPARISON_RE.search(text):
        return True
    regexes = [
        r"(?:이|저|두)\s*(?:문서|파일|규정|지침|정책).*(?:다르|다른\s*점|차이)",
        r"두\s*(?:문서|파일|규정|지침|정책).*(?:어떻게\s*다르|어떻게\s*다른지|차이|다른\s*점)",
        r"\bhow\s+do\s+these\s+two\s+(?:documents|files|policies|rules|guidelines)\s+differ\b",
        r"\b(?:compare|difference\s+between)\s+(?:these|the)\s+two\s+(?:documents|files|policies|rules|guidelines)\b",
    ]
    return any(re.search(pattern, lower) for pattern in regexes)


def needs_strict_fact_style(text: str) -> bool:
    """Detect exact-lookup questions where numbers/titles/dates must stay literal."""
    lower = text.lower().strip()
    indicators = [
        "몇 개", "몇장", "몇 장", "총 몇", "몇 항", "몇 조", "언제", "연도", "년도", "제목",
        "명칭", "이름", "처음", "최초", "어디", "장소", "추가된", "추가", "몇 명", "몇인",
        "얼마", "금액", "한도", "최대", "지급한도", "지급액", "강사비", "상금", "형태", "종류",
        "what year", "when", "how many", "how much", "which title", "title", "name",
        "first", "where", "added", "count", "number of", "starts the", "begins the",
        "amount", "limit", "limits", "payment form", "payment type",
    ]
    if any(keyword in lower for keyword in indicators):
        return True
    regexes = [
        r"\bwhich\s+chapter\s+(?:starts|begins)\b",
        r"정리된.*기본교리.*몇\s*개",
        r"\bwhat\s+chapter\s+starts\b",
    ]
    return any(re.search(pattern, lower) for pattern in regexes)


def needs_article_lookup(text: str) -> bool:
    """Detect specific article/clause lookup queries (제N조, Article N, Section N)."""
    if re.search(r"제\s*\d+\s*조", text):
        return True
    if re.search(r"\b(?:article|section|clause)\s*\d+\b", text, re.IGNORECASE):
        return True
    return False


_SAFE_CLASSIFIER_OVERRIDE_INTENTS = {
    "comparison",
    "default",
    "direct_extraction",
    "summarization",
}


@dataclass(frozen=True)
class QueryPolicy:
    intent: str
    heuristic_intent: str
    classifier_intent: str | None
    intent_source: str
    intent_confidence: float
    shadow_mode: bool
    classifier_disagrees: bool
    smalltalk: bool
    direct_extraction: bool
    general_knowledge: bool
    web_search_candidate: bool
    strict_fact: bool
    article_lookup: bool
    full_document: bool
    whole_corpus_full_document: bool
    multi_document_summary: bool
    comparison: bool
    section_understanding: bool
    chain_of_thought: bool


def classify_intent(text: str) -> str:
    """Return a stable top-level query label for routing and observability."""
    if is_smalltalk_query(text):
        return "smalltalk"
    if needs_comparison_style(text):
        return "comparison"
    if needs_article_lookup(text):
        return "article_lookup"
    if is_direct_extraction_query(text):
        return "direct_extraction"
    if needs_multi_document_summary_style(text):
        return "summarization"
    if needs_strict_fact_style(text):
        return "factual_lookup"
    if is_general_knowledge_query(text):
        return "general_knowledge"
    return "default"


def route_query(text: str) -> QueryPolicy:
    """Bundle the current heuristic routing decisions into one policy object."""
    heuristic_intent = classify_intent(text)
    classification = _classify_query(text)
    classifier_intent = classification.intent if classification else None
    classifier_disagrees = bool(classifier_intent and classifier_intent != heuristic_intent)
    if classification and not QUERY_CLASSIFIER_SHADOW_MODE:
        can_override = (
            classifier_intent == heuristic_intent
            or (
                classifier_intent in _SAFE_CLASSIFIER_OVERRIDE_INTENTS
                and heuristic_intent in {"default", "general_knowledge"}
            )
        )
        if can_override:
            resolved_intent = classifier_intent
            intent_source = classification.provider
            intent_confidence = classification.confidence
        else:
            resolved_intent = heuristic_intent
            intent_source = "heuristic"
            intent_confidence = classification.confidence
    else:
        resolved_intent = heuristic_intent
        intent_source = "heuristic"
        intent_confidence = classification.confidence if classification else 1.0

    return QueryPolicy(
        intent=resolved_intent,
        heuristic_intent=heuristic_intent,
        classifier_intent=classifier_intent,
        intent_source=intent_source,
        intent_confidence=intent_confidence,
        shadow_mode=QUERY_CLASSIFIER_SHADOW_MODE,
        classifier_disagrees=classifier_disagrees,
        smalltalk=is_smalltalk_query(text),
        direct_extraction=is_direct_extraction_query(text),
        general_knowledge=is_general_knowledge_query(text),
        web_search_candidate=might_need_web_search(text),
        strict_fact=needs_strict_fact_style(text),
        article_lookup=needs_article_lookup(text),
        full_document=needs_full_document_context(text),
        whole_corpus_full_document=needs_whole_corpus_full_context(text),
        multi_document_summary=needs_multi_document_summary_style(text),
        comparison=needs_comparison_style(text),
        section_understanding=needs_section_understanding_style(text),
        chain_of_thought=needs_chain_of_thought(text),
    )
