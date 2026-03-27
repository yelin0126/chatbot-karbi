import re


def infer_target_language(text: str) -> str:
    """Infer the user's requested answer language from the current message."""
    if re.search(r"[가-힣]", text or ""):
        return "ko"
    if re.search(r"[A-Za-z]", text or ""):
        return "en"
    return "same"


def contains_chinese(text: str) -> bool:
    """Detect Chinese Han characters that should not appear in normal answers."""
    return bool(re.search(r"[\u4e00-\u9fff]", text or ""))


def contains_excessive_english(text: str) -> bool:
    """True when the response contains >= 10 consecutive ASCII-only words."""
    run = 0
    for token in re.split(r"[\s\-_/,.;:!?()\[\]\"\']+", text or ""):
        if token and re.fullmatch(r"[A-Za-z]+", token):
            run += 1
            if run >= 10:
                return True
        elif token:
            run = 0
    return False


def needs_language_retry(user_message: str, answer: str) -> bool:
    """Retry when the model drifted into Chinese or excessive English for a Korean query."""
    target_lang = infer_target_language(user_message)
    if target_lang not in {"ko", "en"}:
        return False
    if contains_chinese(answer):
        return True
    if target_lang == "ko" and contains_excessive_english(answer):
        return True
    return False


def language_correction_prompt(
    original_prompt: str,
    user_message: str,
    bad_answer: str,
) -> str:
    target_lang = infer_target_language(user_message)
    lang_name = "Korean" if target_lang == "ko" else "English"
    return (
        f"{original_prompt}\n\n"
        "[Critical correction]\n"
        f"The previous draft answer was invalid because it used Chinese characters.\n"
        f"Rewrite the final answer entirely in {lang_name}.\n"
        "Do not use any Chinese characters.\n"
        "Keep the same facts and stay grounded in the provided context.\n\n"
        f"[Invalid draft answer]\n{bad_answer}"
    )
