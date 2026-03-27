"""
Semantic chunker — splits documents by meaning, not character count.

Strategy:
1. Detect markdown headings → split at section boundaries
2. Detect tables → keep as independent chunks
3. Large sections → split at paragraph boundaries
4. Oversized paragraphs → character-based fallback with Korean separators
5. Each chunk carries heading breadcrumb (e.g., "설치 가이드 > 요구사항")

Why: Old chunker split at 1200 chars blindly. A chunk could be half of
"Requirements" + half of "Installation". This chunker keeps sections whole.
"""

import re
import uuid
import time
import logging
from typing import List, Tuple
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import CHUNK_SIZE, CHUNK_OVERLAP, CHILD_CHUNK_SIZE, CHILD_CHUNK_OVERLAP

logger = logging.getLogger("tilon.chunker")

_HEADING_RE = re.compile(r'^(#{1,4})\s+(.+)$', re.MULTILINE)
_TABLE_RE = re.compile(r'(?:^[|].*[|]\s*\n?){2,}', re.MULTILINE)

# Korean legal article pattern: 제N조(title) or 제N조의N(title)
# Handles spaces and both parenthesis styles: （） and ()
_CLAUSE_HEADER_RE = re.compile(
    r'제\s*\d+\s*조(?:의\s*\d+)?\s*[（(]\s*[^）)]{1,80}\s*[）)]',
    re.MULTILINE,
)
# Lookahead-based split: split just before each 제N조( pattern
_CLAUSE_SPLIT_RE = re.compile(
    r'(?=제\s*\d+\s*조(?:의\s*\d+)?\s*[（(])',
    re.MULTILINE,
)

_KOREAN_SEPARATORS = [
    "\n\n", "\n", "다. ", "요. ", "다.\n", "요.\n",
    ". ", "? ", "! ", " ", "",
]


@dataclass
class Section:
    heading: str
    level: int
    content: str
    breadcrumb: str
    is_table: bool = False


def _split_by_headings(text: str) -> List[Section]:
    """Split text into sections based on markdown headings with breadcrumb tracking."""
    sections = []
    heading_stack: List[Tuple[int, str]] = []
    matches = list(_HEADING_RE.finditer(text))

    if not matches:
        return [Section(heading="", level=0, content=text.strip(), breadcrumb="")]

    pre = text[:matches[0].start()].strip()
    if pre:
        sections.append(Section(heading="", level=0, content=pre, breadcrumb=""))

    for i, match in enumerate(matches):
        level = len(match.group(1))
        heading_text = match.group(2).strip()
        content_start = match.end()
        content_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[content_start:content_end].strip()

        while heading_stack and heading_stack[-1][0] >= level:
            heading_stack.pop()
        heading_stack.append((level, heading_text))
        breadcrumb = " > ".join(h[1] for h in heading_stack)

        sections.append(Section(
            heading=heading_text, level=level,
            content=content, breadcrumb=breadcrumb,
        ))

    return sections


def _extract_tables(text: str) -> Tuple[str, List[str]]:
    """Extract markdown tables, return (remaining_text, [tables])."""
    tables = [m.group(0).strip() for m in _TABLE_RE.finditer(text) if len(m.group(0).strip()) > 50]
    remaining = _TABLE_RE.sub('\n\n', text).strip() if tables else text
    return remaining, tables


def _split_paragraphs(text: str) -> List[str]:
    """Split by paragraphs, group small ones together."""
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    grouped, current, current_len = [], [], 0

    for para in paragraphs:
        if current_len + len(para) > CHUNK_SIZE and current:
            grouped.append("\n\n".join(current))
            current, current_len = [para], len(para)
        else:
            current.append(para)
            current_len += len(para)
    if current:
        grouped.append("\n\n".join(current))
    return grouped


def _char_split(text: str) -> List[str]:
    """Fallback character-based split with Korean separators."""
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=_KOREAN_SEPARATORS,
    ).split_text(text)


def _make_chunk(text, metadata, idx, section_title="", breadcrumb="", chunk_type="section"):
    return Document(
        page_content=text,
        metadata={
            **metadata,
            "chunk_index": idx,
            "chunk_id": str(uuid.uuid4()),
            "chunk_chars": len(text),
            "chunk_type": chunk_type,
            "section_title": section_title,
            "section_breadcrumb": breadcrumb,
            "timestamp": int(time.time()),
        },
    )


def _extract_clause_header(text: str) -> str:
    """Return the leading 제N조(...) header from a clause block, or empty string."""
    m = _CLAUSE_HEADER_RE.match(text.lstrip())
    return m.group(0) if m else ""


def _split_by_clauses(text: str) -> List[Tuple[str, str]]:
    """
    Split text at Korean article boundaries (제N조(...)) and return a list of
    (clause_header, full_clause_text) tuples.  Non-clause preamble text before
    the first article is returned as ("", preamble).
    """
    parts = _CLAUSE_SPLIT_RE.split(text)
    result: List[Tuple[str, str]] = []
    for part in parts:
        stripped = part.strip()
        if not stripped:
            continue
        header = _extract_clause_header(stripped)
        result.append((header, stripped))
    return result


def _chunk_section(section: Section, base_meta: dict, start_idx: int) -> List[Document]:
    """Chunk one section: tables → independent, text → whole or split."""
    chunks = []
    if not section.content.strip():
        return chunks

    remaining, tables = _extract_tables(section.content)

    # Table chunks
    for table in tables:
        pieces = [table] if len(table) <= CHUNK_SIZE else _char_split(table)
        for piece in pieces:
            chunks.append(_make_chunk(
                piece, base_meta, start_idx + len(chunks),
                section.heading, section.breadcrumb, "table",
            ))

    # Text chunks
    if not remaining.strip():
        return chunks

    if len(remaining) <= CHUNK_SIZE:
        chunks.append(_make_chunk(
            remaining, base_meta, start_idx + len(chunks),
            section.heading, section.breadcrumb, "section",
        ))
    else:
        for para in _split_paragraphs(remaining):
            pieces = [para] if len(para) <= CHUNK_SIZE else _char_split(para)
            for piece in pieces:
                chunks.append(_make_chunk(
                    piece, base_meta, start_idx + len(chunks),
                    section.heading, section.breadcrumb, "paragraph",
                ))

    return chunks


def chunk_documents(docs: List[Document]) -> List[Document]:
    """
    Semantic chunking: split documents by meaning, not character count.

    For documents with Korean legal articles (제N조): split into atomic
    clause chunks first — this keeps each article intact for precise retrieval.
    For documents with markdown headings: split at section boundaries.
    For documents without: split at paragraph boundaries.
    Tables always become independent chunks.
    """
    all_chunks: List[Document] = []
    total_sections = 0

    for doc in docs:
        text = doc.page_content
        # Copy metadata without chunk-specific fields
        base_meta = {k: v for k, v in doc.metadata.items()
                     if k not in ("chunk_index", "chunk_id", "chunk_chars",
                                  "chunk_type", "section_breadcrumb", "timestamp")}

        has_clauses = bool(_CLAUSE_HEADER_RE.search(text))
        has_headings = bool(_HEADING_RE.search(text))

        if has_clauses:
            # Korean legal article mode — each 제N조(...) block is one chunk
            clause_pairs = _split_by_clauses(text)
            total_sections += len(clause_pairs)
            for header, clause_text in clause_pairs:
                clause_type = "clause" if header else "section"
                chunks_for_clause: List[str] = (
                    [clause_text] if len(clause_text) <= CHUNK_SIZE
                    else _char_split(clause_text)
                )
                for piece in chunks_for_clause:
                    all_chunks.append(_make_chunk(
                        piece, base_meta, len(all_chunks),
                        section_title=header, breadcrumb=header,
                        chunk_type=clause_type,
                    ))
        elif has_headings:
            sections = _split_by_headings(text)
            total_sections += len(sections)
            for section in sections:
                all_chunks.extend(_chunk_section(section, base_meta, len(all_chunks)))
        else:
            # No headings — paragraph-based
            remaining, tables = _extract_tables(text)
            section_title = base_meta.get("section_title", "")

            for table in tables:
                pieces = [table] if len(table) <= CHUNK_SIZE else _char_split(table)
                for piece in pieces:
                    all_chunks.append(_make_chunk(
                        piece, base_meta, len(all_chunks), section_title, "", "table",
                    ))

            if remaining.strip():
                if len(remaining) <= CHUNK_SIZE:
                    all_chunks.append(_make_chunk(
                        remaining, base_meta, len(all_chunks), section_title, "", "page",
                    ))
                else:
                    for para in _split_paragraphs(remaining):
                        pieces = [para] if len(para) <= CHUNK_SIZE else _char_split(para)
                        for piece in pieces:
                            all_chunks.append(_make_chunk(
                                piece, base_meta, len(all_chunks), section_title, "", "paragraph",
                            ))

    types = {}
    for c in all_chunks:
        t = c.metadata.get("chunk_type", "?")
        types[t] = types.get(t, 0) + 1

    logger.info(
        "Semantic chunking: %d docs → %d sections → %d chunks | %s",
        len(docs), total_sections, len(all_chunks), types,
    )
    return all_chunks


def chunk_documents_hierarchical(docs: List[Document]) -> Tuple[List[Document], List[Document]]:
    """
    Two-level hierarchical chunking.

    Step 1 — produce PARENT chunks via the existing semantic chunker.
              Each parent is a full section (≤ CHUNK_SIZE chars) with heading
              context.  Parents are NOT embedded; they are stored in
              parent_store.py for context expansion at answer time.

    Step 2 — split every parent into smaller CHILD chunks
              (≤ CHILD_CHUNK_SIZE chars).  Children ARE embedded and indexed
              for retrieval.  Each child carries ``parent_id`` pointing back to
              its parent so the retriever can fetch the full-context parent text.

    Returns:
        parents  — List[Document] (full semantic sections, for parent_store)
        children — List[Document] (small retrieval units, for vectorstore)
    """
    parents: List[Document] = chunk_documents(docs)

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE,
        chunk_overlap=CHILD_CHUNK_OVERLAP,
        separators=_KOREAN_SEPARATORS,
    )

    children: List[Document] = []

    for parent in parents:
        # The parent's chunk_id doubles as the parent lookup key.
        parent_id = parent.metadata["chunk_id"]
        parent.metadata["parent_id"] = parent_id  # self-reference for clear_parents_for_doc

        child_texts = child_splitter.split_text(parent.page_content)

        for i, child_text in enumerate(child_texts):
            child = Document(
                page_content=child_text,
                metadata={
                    **parent.metadata,
                    "chunk_id": str(uuid.uuid4()),
                    "chunk_index": len(children),       # global, unique per doc
                    "chunk_chars": len(child_text),
                    "parent_id": parent_id,
                    "child_index_within_parent": i,
                },
            )
            children.append(child)

    logger.info(
        "Hierarchical chunking: %d docs → %d parents → %d children",
        len(docs), len(parents), len(children),
    )
    return parents, children
