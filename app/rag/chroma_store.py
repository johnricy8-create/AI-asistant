from __future__ import annotations

import hashlib
import json
import logging
from datetime import date
from pathlib import Path
from typing import Any

from app.config import Settings
from app.core.text import first_sentence

logger = logging.getLogger(__name__)


class ChromaRetriever:
    def __init__(
        self,
        settings: Settings,
        chunks: list[dict[str, Any]],
        embedder: Any,
    ):
        import chromadb  # lazy import: tests may run without chromadb installed

        if not chunks:
            raise ValueError("chunks is empty")
        if embedder is None:
            raise ValueError("embedder is None")

        self.settings = settings
        self.chunks = chunks
        self.embedder = embedder
        self.embedding_model = getattr(embedder, "name", "unknown")
        self.name = f"ChromaDB + {self.embedding_model}"
        self.path = Path(settings.chroma_path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.collection_name = settings.chroma_collection
        self.manifest_path = self.path / f"{self.collection_name}.manifest.json"

        self.client = chromadb.PersistentClient(path=str(self.path))
        self._ensure_collection()

    def _build_hash(self) -> str:
        payload = "\n".join(
            f"{self.embedding_model}|{chunk['id']}|{chunk['content']}"
            for chunk in self.chunks
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _metadata(chunk: dict[str, Any]) -> dict[str, Any]:
        def date_int(value: Any, default: int) -> int:
            if not value:
                return default
            try:
                return int(str(value).replace("-", "")[:8])
            except ValueError:
                return default

        return {
            "chunk_id": chunk["id"],
            "doc_id": chunk["doc_id"],
            "title": chunk["title"],
            "section": chunk["section"],
            "category": chunk.get("category", "未分类"),
            "doc_type": chunk.get("doc_type", "policy"),
            "risk_level": chunk.get("risk_level", "medium"),
            "source_uri": chunk.get("source_uri", ""),
            "status": chunk.get("status", "published"),
            "effective_from": date_int(chunk.get("effective_from"), 0),
            "effective_to": date_int(chunk.get("effective_to"), 99991231),
            "keywords": json.dumps(chunk.get("keywords", []), ensure_ascii=False),
        }

    def _read_manifest(self) -> dict[str, Any] | None:
        if not self.manifest_path.exists():
            return None
        try:
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def _write_manifest(self, build_hash: str, count: int) -> None:
        self.manifest_path.write_text(
            json.dumps(
                {
                    "build_hash": build_hash,
                    "count": count,
                    "embedding_model": self.embedding_model,
                    "built_at": date.today().isoformat(),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def _ensure_collection(self) -> None:
        build_hash = self._build_hash()
        manifest = self._read_manifest()
        try:
            collection = self.client.get_collection(self.collection_name)
            if (
                manifest
                and manifest.get("build_hash") == build_hash
                and collection.count() == len(self.chunks)
            ):
                self.collection = collection
                logger.info(
                    "Chroma collection reused: %s (%s chunks)",
                    self.collection_name,
                    collection.count(),
                )
                return
        except Exception:  # noqa: BLE001 - collection may not exist
            pass

        try:
            self.client.delete_collection(self.collection_name)
        except Exception:  # noqa: BLE001 - first build
            pass

        collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Embedding %s chunks with %s", len(self.chunks), self.embedder.name)
        embeddings = self.embedder.embed([chunk["content"] for chunk in self.chunks])
        if len(embeddings) != len(self.chunks):
            raise RuntimeError(
                f"embedding count mismatch: {len(embeddings)} != {len(self.chunks)}"
            )
        collection.add(
            ids=[chunk["id"] for chunk in self.chunks],
            documents=[chunk["content"] for chunk in self.chunks],
            embeddings=embeddings,
            metadatas=[self._metadata(chunk) for chunk in self.chunks],
        )
        self.collection = collection
        self._write_manifest(build_hash, len(self.chunks))
        logger.info(
            "Chroma collection built: %s (%s chunks)",
            self.collection_name,
            len(self.chunks),
        )

    def search(
        self,
        query: str,
        top_k: int = 15,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if not query.strip():
            return []
        query_embedding = self.embedder.embed([query])[0]
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where or {"status": "published"},
            include=["documents", "metadatas", "distances"],
        )
        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        hits: list[dict[str, Any]] = []
        for index, chunk_id in enumerate(ids):
            metadata = metadatas[index] or {}
            content = documents[index] or ""
            distance = distances[index] if index < len(distances) else 1.0
            score = max(0.0, min(1.0, 1.0 - float(distance)))
            hits.append(
                {
                    "chunk_id": chunk_id,
                    "doc_id": metadata.get("doc_id", ""),
                    "title": metadata.get("title", ""),
                    "section": metadata.get("section", ""),
                    "content": content,
                    "category": metadata.get("category", ""),
                    "risk_level": metadata.get("risk_level", "medium"),
                    "effective_from": metadata.get("effective_from"),
                    "source_uri": metadata.get("source_uri", ""),
                    "score": round(score, 4),
                    "retrieval_score": round(score, 4),
                    "snippet": first_sentence(content, 180),
                }
            )
        return hits


class FallbackRetriever:
    """Try the vector retriever first; fall back to lexical retrieval on error."""

    def __init__(self, primary: Any, fallback: Any):
        self.primary = primary
        self.fallback = fallback
        self.chunks = primary.chunks
        self.name = primary.name
        self.embedding_model = getattr(primary, "embedding_model", "")

    def search(self, query: str, top_k: int = 15) -> list[dict[str, Any]]:
        try:
            return self.primary.search(query, top_k=top_k)
        except Exception:  # noqa: BLE001 - retrieval must degrade gracefully
            logger.exception("vector retrieval failed; falling back to lexical search")
            return self.fallback.search(query, top_k=top_k)
