from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.models.schemas import Message


@dataclass
class ChatExecutionState:
    user_message: str
    history: List[Message]
    selected_model: str
    active_source: Optional[str]
    active_doc_id: Optional[str]
    active_source_type: Optional[str]
    system_prompt: Optional[str]
    query_policy: Any = None
    normalized_sources: List[str] = field(default_factory=list)
    normalized_doc_ids: List[str] = field(default_factory=list)
    selected_docs: List[Dict[str, str]] = field(default_factory=list)
    multi_scope: bool = False
    scoped_source: Optional[str] = None
    scoped_doc_id: Optional[str] = None
    scoped_sources: List[str] = field(default_factory=list)
    scoped_doc_ids: List[str] = field(default_factory=list)
    use_full_document: bool = False
    retrieval: Any = None
    docs: List[Any] = field(default_factory=list)
    doc_context: str = ""
    web_context: str = ""
    sources: List[Dict[str, Any]] = field(default_factory=list)
    prompt: str = ""
    lm_messages: List[Dict[str, Any]] = field(default_factory=list)
    answer: str = ""

    def sync_scope(self) -> None:
        self.multi_scope = len(self.selected_docs) > 1
        self.scoped_source = self.selected_docs[0]["source"] if len(self.selected_docs) == 1 else None
        self.scoped_doc_id = self.selected_docs[0]["doc_id"] if len(self.selected_docs) == 1 else None
        self.scoped_sources = [doc["source"] for doc in self.selected_docs if doc.get("source")]
        self.scoped_doc_ids = [doc["doc_id"] for doc in self.selected_docs if doc.get("doc_id")]

    def document_scope_count(self) -> int:
        return max(
            len(self.selected_docs),
            len({
                source.get("doc_id") or source.get("source")
                for source in self.sources
                if source.get("doc_id") or source.get("source")
            }),
        )

    def response(
        self,
        answer: str,
        *,
        mode: str,
        sources: Optional[List[Dict[str, Any]]] = None,
        active_source: Optional[str] = None,
        active_doc_id: Optional[str] = None,
        active_source_type: Optional[str] = None,
        active_sources: Optional[List[str]] = None,
        active_doc_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        return {
            "answer": answer,
            "sources": self.sources if sources is None else sources,
            "mode": mode,
            "active_source": self.scoped_source if active_source is None else active_source,
            "active_doc_id": self.scoped_doc_id if active_doc_id is None else active_doc_id,
            "active_source_type": self.active_source_type if active_source_type is None else active_source_type,
            "active_sources": self.scoped_sources if active_sources is None else active_sources,
            "active_doc_ids": self.scoped_doc_ids if active_doc_ids is None else active_doc_ids,
        }
