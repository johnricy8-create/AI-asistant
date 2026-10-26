from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

RISK_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}


@dataclass
class Document:
    doc_id: str
    title: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _max_risk(values: list[str]) -> str:
    if not values:
        return "medium"
    return max(values, key=lambda item: RISK_ORDER.get(item, 1))


def split_text_by_blank_line(text: str) -> list[str]:
    """Split a document on blank lines (\\n\\n). No length limit is applied."""
    return [part.strip() for part in (text or "").split("\n\n") if part.strip()]


def build_documents_from_seed(chunks: list[dict[str, Any]]) -> list[Document]:
    """Reconstruct full documents from the demo seed chunks.

    Seed chunks are already section-level. We join sections with blank lines so
    that the runtime chunker applies the same "split on \\n\\n" rule as real
    uploaded documents.
    """
    grouped: dict[str, dict[str, Any]] = {}
    for chunk in chunks:
        doc_id = chunk["doc_id"]
        doc = grouped.setdefault(
            doc_id,
            {
                "doc_id": doc_id,
                "title": chunk["title"],
                "parts": [],
                "keywords": [],
                "risks": [],
                "section_keywords": {},
                "section_risks": {},
                "category": chunk.get("category", "未分类"),
                "doc_type": chunk.get("doc_type", "policy"),
                "source_uri": chunk.get("source_uri", ""),
                "status": chunk.get("status", "published"),
                "effective_from": chunk.get("effective_from"),
                "effective_to": chunk.get("effective_to"),
            },
        )
        section = chunk.get("section", "")
        doc["parts"].append(f"{section}\n{chunk.get('content', '')}")
        doc["keywords"].extend(chunk.get("keywords", []))
        doc["risks"].append(chunk.get("risk_level", "medium"))
        doc["section_keywords"][section] = chunk.get("keywords", [])
        doc["section_risks"][section] = chunk.get("risk_level", "medium")

    documents: list[Document] = []
    for doc in grouped.values():
        metadata = {
            "category": doc["category"],
            "doc_type": doc["doc_type"],
            "source_uri": doc["source_uri"],
            "status": doc["status"],
            "effective_from": doc["effective_from"],
            "effective_to": doc["effective_to"],
            "risk_level": _max_risk(doc["risks"]),
            "keywords": list(dict.fromkeys(doc["keywords"])),
            "section_keywords": doc["section_keywords"],
            "section_risks": doc["section_risks"],
        }
        documents.append(
            Document(
                doc_id=doc["doc_id"],
                title=doc["title"],
                text="\n\n".join(doc["parts"]),
                metadata=metadata,
            )
        )
    return documents


def split_documents_by_blank_line(
    documents: list[Document],
) -> list[dict[str, Any]]:
    """Split documents into chunks on blank lines.

    Each chunk keeps the document metadata. If the first line looks like a
    section heading, it is stored in `section`; the remaining text becomes
    `content`. No character limit or overlap is applied.
    """
    chunks: list[dict[str, Any]] = []
    for document in documents:
        section_keywords = document.metadata.get("section_keywords", {})
        section_risks = document.metadata.get("section_risks", {})
        for index, part in enumerate(split_text_by_blank_line(document.text)):
            lines = part.split("\n", 1)
            if len(lines) > 1:
                section = lines[0].strip()
                content = lines[1].strip()
            else:
                section = document.title
                content = part.strip()
            if not content:
                continue
            chunks.append(
                {
                    "id": f"{document.doc_id}-{index:03d}",
                    "doc_id": document.doc_id,
                    "title": document.title,
                    "section": section,
                    "content": content,
                    "category": document.metadata.get("category", "未分类"),
                    "doc_type": document.metadata.get("doc_type", "policy"),
                    "risk_level": section_risks.get(
                        section, document.metadata.get("risk_level", "medium")
                    ),
                    "effective_from": document.metadata.get("effective_from"),
                    "effective_to": document.metadata.get("effective_to"),
                    "source_uri": document.metadata.get("source_uri", ""),
                    "keywords": section_keywords.get(
                        section, document.metadata.get("keywords", [])
                    ),
                    "status": document.metadata.get("status", "published"),
                }
            )
    return chunks


def build_chunks_from_seed(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return split_documents_by_blank_line(build_documents_from_seed(chunks))
