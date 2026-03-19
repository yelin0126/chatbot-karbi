"""
Document chunking using LangChain text splitters.

IMPROVEMENTS over original:
- Chunk size/overlap configurable via environment (was hardcoded)
- Chunk metadata enriched with char count for debugging
"""

import uuid
import time
import logging
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger("tilon.chunker")

# Korean-aware separators
SEPARATORS = [
    "\n\n",
    "\n",
    ". ",
    "。 ",
    "? ",
    "! ",
    "다. ",    # Korean sentence ending
    "요. ",    # Korean polite ending
    " ",
    "",
]


def chunk_documents(docs: List[Document]) -> List[Document]:
    """Split documents into smaller chunks with metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS,
    )

    chunked: List[Document] = []
    for doc in docs:
        pieces = splitter.split_text(doc.page_content)
        for idx, piece in enumerate(pieces):
            chunked.append(
                Document(
                    page_content=piece,
                    metadata={
                        **doc.metadata,
                        "chunk_index": idx,
                        "chunk_id": str(uuid.uuid4()),
                        "chunk_chars": len(piece),
                        "timestamp": int(time.time()),
                    },
                )
            )

    logger.info(
        "Chunked %d documents → %d chunks (size=%d, overlap=%d)",
        len(docs), len(chunked), CHUNK_SIZE, CHUNK_OVERLAP,
    )
    return chunked
