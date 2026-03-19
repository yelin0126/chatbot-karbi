"""
In-memory BM25 keyword index for document chunks.

This complements vector search for exact tokens like error codes, commands,
product names, and mixed Korean/English technical terms.
"""

import logging
import math
import re
from collections import Counter
from typing import List, Optional, Tuple

from langchain_core.documents import Document

logger = logging.getLogger("tilon.keyword_index")

_TOKEN_RE = re.compile(r"[A-Za-z0-9._-]+|[가-힣]+")


def tokenize_text(text: str) -> List[str]:
    """Tokenize Korean/English technical text while preserving codes like E-401."""
    return [token.lower() for token in _TOKEN_RE.findall(text or "")]


class InMemoryBM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.clear()

    def clear(self) -> None:
        self._documents: List[Document] = []
        self._tokenized_docs: List[List[str]] = []
        self._doc_freq: Counter = Counter()
        self._avg_doc_len = 0.0

    def rebuild(self, docs: List[Document]) -> None:
        self._documents = list(docs)
        self._tokenized_docs = [tokenize_text(doc.page_content) for doc in self._documents]
        self._doc_freq = Counter()

        total_len = 0
        for tokens in self._tokenized_docs:
            total_len += len(tokens)
            self._doc_freq.update(set(tokens))

        self._avg_doc_len = total_len / len(self._tokenized_docs) if self._tokenized_docs else 0.0
        logger.info("Keyword index rebuilt with %d chunks", len(self._documents))

    def add_documents(self, docs: List[Document]) -> None:
        if not docs:
            return
        self.rebuild(self._documents + list(docs))

    def search(
        self,
        query: str,
        k: int = 4,
        source_filter: Optional[str] = None,
    ) -> List[Tuple[Document, float]]:
        query_tokens = tokenize_text(query)
        if not query_tokens or not self._documents:
            return []

        results: List[Tuple[Document, float]] = []
        total_docs = len(self._documents)

        for doc, doc_tokens in zip(self._documents, self._tokenized_docs):
            if source_filter and doc.metadata.get("source") != source_filter:
                continue
            if not doc_tokens:
                continue

            score = self._score(query_tokens, doc_tokens, total_docs)
            if score > 0:
                results.append((doc, score))

        results.sort(key=lambda item: item[1], reverse=True)
        return results[:k]

    def _score(self, query_tokens: List[str], doc_tokens: List[str], total_docs: int) -> float:
        term_freq = Counter(doc_tokens)
        doc_len = len(doc_tokens) or 1
        avg_doc_len = self._avg_doc_len or 1.0
        score = 0.0

        for token in query_tokens:
            freq = term_freq.get(token, 0)
            if not freq:
                continue

            doc_freq = self._doc_freq.get(token, 0)
            idf = math.log(1 + ((total_docs - doc_freq + 0.5) / (doc_freq + 0.5)))
            denom = freq + self.k1 * (1 - self.b + self.b * (doc_len / avg_doc_len))
            score += idf * ((freq * (self.k1 + 1)) / denom)

        return score


_keyword_index = InMemoryBM25Index()


def rebuild_keyword_index(docs: List[Document]) -> None:
    _keyword_index.rebuild(docs)


def add_keyword_documents(docs: List[Document]) -> None:
    _keyword_index.add_documents(docs)


def clear_keyword_index() -> None:
    _keyword_index.clear()


def search_keyword_index(
    query: str,
    k: int = 4,
    source_filter: Optional[str] = None,
) -> List[Tuple[Document, float]]:
    return _keyword_index.search(query=query, k=k, source_filter=source_filter)
