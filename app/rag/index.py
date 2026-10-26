from __future__ import annotations

import math
from collections import Counter
from datetime import date
from typing import Any

from app.core.text import (
    char_ngrams,
    clamp,
    cosine_counter,
    first_sentence,
    keyword_hit_rate,
    normalize_text,
    tokenize,
)


class BM25Index:
    def __init__(self, tokenized_docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.tokenized_docs = tokenized_docs
        self.k1 = k1
        self.b = b
        self.doc_count = len(tokenized_docs)
        self.doc_lengths = [len(doc) for doc in tokenized_docs]
        self.avg_length = sum(self.doc_lengths) / max(1, self.doc_count)
        self.term_freqs = [Counter(doc) for doc in tokenized_docs]
        self.doc_freqs: Counter[str] = Counter()
        for doc in tokenized_docs:
            self.doc_freqs.update(set(doc))
        self.idf = {
            term: math.log(1 + (self.doc_count - freq + 0.5) / (freq + 0.5))
            for term, freq in self.doc_freqs.items()
        }

    def raw_scores(self, query_tokens: list[str]) -> list[float]:
        scores: list[float] = []
        for index, freqs in enumerate(self.term_freqs):
            doc_len = self.doc_lengths[index] or 1
            score = 0.0
            for term in query_tokens:
                tf = freqs.get(term, 0)
                if not tf:
                    continue
                idf = self.idf.get(term, 0.0)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / max(1.0, self.avg_length))
                score += idf * (tf * (self.k1 + 1)) / max(1e-9, denominator)
            scores.append(score)
        return scores


def _active(chunk: dict[str, Any]) -> bool:
    if chunk.get("status", "published") != "published":
        return False
    today = date.today().isoformat()
    if chunk.get("effective_from") and chunk["effective_from"] > today:
        return False
    if chunk.get("effective_to") and chunk["effective_to"] < today:
        return False
    return True


class HybridRetriever:
    """Dense-free hybrid retrieval for the demo.

    It combines BM25, character-bigram cosine, curated keyword hits and title
    similarity, then applies a light heuristic reranker. In production this is
    replaced by Milvus dense+sparse retrieval + BGE reranker.
    """

    def __init__(self, chunks: list[dict[str, Any]]):
        self.chunks = [chunk for chunk in chunks if _active(chunk)]
        self.searchable_texts = [self._searchable_text(chunk) for chunk in self.chunks]
        self.tokenized = [tokenize(text) for text in self.searchable_texts]
        self.bm25 = BM25Index(self.tokenized)
        self.ngrams = [char_ngrams(text) for text in self.searchable_texts]
        self.title_ngrams = [
            char_ngrams(f"{chunk['title']} {chunk['section']}") for chunk in self.chunks
        ]

    @staticmethod
    def _searchable_text(chunk: dict[str, Any]) -> str:
        return " ".join(
            [
                chunk.get("title", ""),
                chunk.get("section", ""),
                chunk.get("content", ""),
                " ".join(chunk.get("keywords", [])),
            ]
        )

    def search(self, query: str, top_k: int = 6) -> list[dict[str, Any]]:
        query = normalize_text(query)
        if not query:
            return []

        query_tokens = tokenize(query)
        query_ngrams = char_ngrams(query)
        raw_bm25 = self.bm25.raw_scores(query_tokens)
        max_bm25 = max(raw_bm25) if raw_bm25 else 0.0

        candidates: list[tuple[float, int]] = []
        for index, chunk in enumerate(self.chunks):
            char_sim = cosine_counter(query_ngrams, self.ngrams[index])
            kw_hit = keyword_hit_rate(query, chunk.get("keywords", []))
            title_sim = cosine_counter(query_ngrams, self.title_ngrams[index])
            bm25_norm = raw_bm25[index] / max_bm25 if max_bm25 else 0.0
            exact_bonus = 0.12 if query in normalize_text(chunk.get("content", "")) else 0.0
            score = (
                0.30 * char_sim
                + 0.35 * bm25_norm
                + 0.25 * kw_hit
                + 0.10 * title_sim
                + exact_bonus
            )
            score = clamp(score)
            if score > 0:
                candidates.append((score, index))

        candidates.sort(key=lambda item: item[0], reverse=True)

        reranked: list[tuple[float, int]] = []
        for base_score, index in candidates[:12]:
            section = normalize_text(self.chunks[index].get("section", ""))
            section_bonus = 0.0
            if section and any(
                token in section for token in query_tokens if len(token) >= 2
            ):
                section_bonus = 0.08
            reranked.append((clamp(0.92 * base_score + section_bonus), index))
        reranked.sort(key=lambda item: item[0], reverse=True)

        results: list[dict[str, Any]] = []
        per_doc: Counter[str] = Counter()
        for score, index in reranked:
            chunk = self.chunks[index]
            if per_doc[chunk["doc_id"]] >= 2:
                continue
            per_doc[chunk["doc_id"]] += 1
            results.append(
                {
                    "chunk_id": chunk["id"],
                    "doc_id": chunk["doc_id"],
                    "title": chunk["title"],
                    "section": chunk["section"],
                    "content": chunk["content"],
                    "category": chunk["category"],
                    "risk_level": chunk.get("risk_level", "medium"),
                    "effective_from": chunk.get("effective_from"),
                    "source_uri": chunk.get("source_uri", ""),
                    "score": round(score, 4),
                    "snippet": first_sentence(chunk["content"], 180),
                }
            )
            if len(results) >= top_k:
                break

        return results

    @staticmethod
    def build_context(hits: list[dict[str, Any]], max_chars: int = 6000) -> str:
        parts: list[str] = []
        total = 0
        for index, hit in enumerate(hits, start=1):
            block = (
                f"[{index}] 《{hit['title']}》 {hit['section']}\n"
                f"{hit['content']}\n"
                f"来源：{hit.get('source_uri', '')}"
            )
            if total + len(block) > max_chars:
                break
            parts.append(block)
            total += len(block)
        return "\n\n".join(parts)
