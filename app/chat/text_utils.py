import re


def strip_enrichment_header(text: str) -> str:
    """Remove enrichment header prepended before embedding/retrieval."""
    return re.sub(r"^\[Document:.*?\]\n", "", text or "", flags=re.DOTALL).strip()
