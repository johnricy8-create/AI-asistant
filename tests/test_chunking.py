from __future__ import annotations

from app.data.seed import KB_SEED
from app.rag.chunking import (
    build_chunks_from_seed,
    build_documents_from_seed,
    split_text_by_blank_line,
)


def test_split_text_by_blank_line() -> None:
    text = "第一段\n\n第二段\n\n\n第三段"
    assert split_text_by_blank_line(text) == ["第一段", "第二段", "第三段"]


def test_split_does_not_limit_length() -> None:
    long_paragraph = "这是一个很长的段落。" * 500
    chunks = split_text_by_blank_line(long_paragraph)
    assert len(chunks) == 1
    assert chunks[0] == long_paragraph


def test_seed_documents_and_chunks() -> None:
    documents = build_documents_from_seed(KB_SEED)
    chunks = build_chunks_from_seed(KB_SEED)
    assert len(documents) == 12
    assert len(chunks) == 36
    assert all(chunk["content"] for chunk in chunks)
    assert all(chunk["id"] for chunk in chunks)
    assert all(chunk["keywords"] for chunk in chunks)
    assert all(chunk["status"] == "published" for chunk in chunks)


def test_chunk_metadata_shape() -> None:
    chunk = build_chunks_from_seed(KB_SEED)[0]
    for key in [
        "id",
        "doc_id",
        "title",
        "section",
        "content",
        "category",
        "risk_level",
        "effective_from",
        "source_uri",
        "keywords",
        "status",
    ]:
        assert key in chunk
