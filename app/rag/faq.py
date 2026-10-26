from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.core.text import (
    char_ngrams,
    cosine_counter,
    keyword_hit_rate,
    normalize_text,
    overlap_coefficient,
    tokenize,
)


@dataclass
class FAQMatch:
    faq_id: str
    std_q: str
    answer: str
    category: str
    risk_level: str
    score: float
    margin: float
    threshold: float
    passed: bool
    matched_question: str
    owner: str = ""
    candidates: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "faq_id": self.faq_id,
            "std_q": self.std_q,
            "category": self.category,
            "risk_level": self.risk_level,
            "score": round(self.score, 4),
            "margin": round(self.margin, 4),
            "threshold": self.threshold,
            "passed": self.passed,
            "matched_question": self.matched_question,
            "candidates": self.candidates,
        }


class FAQIndex:
    """Semantic-ish FAQ matcher for the demo.

    Production replaces this with BGE-M3 + Milvus. The scoring below combines
    character bigram cosine, token overlap and curated keyword hits so that the
    demo behaves deterministically without external embedding calls.
    """

    def __init__(self, entries: list[dict[str, Any]], default_margin: float = 0.03):
        self.entries = entries
        self.default_margin = default_margin
        self._prepared: list[dict[str, Any]] = []
        for entry in entries:
            variants = [entry["std_q"], *entry.get("variants", [])]
            self._prepared.append(
                {
                    **entry,
                    "variants": variants,
                    "variant_ngrams": [char_ngrams(item) for item in variants],
                    "variant_tokens": [tokenize(item) for item in variants],
                }
            )

    def _score_variant(self, query: str, entry: dict[str, Any], index: int) -> float:
        variant = entry["variants"][index]
        q_norm = normalize_text(query)
        v_norm = normalize_text(variant)
        if q_norm == v_norm:
            return 1.0

        char_sim = cosine_counter(char_ngrams(q_norm), entry["variant_ngrams"][index])
        token_sim = overlap_coefficient(q_norm, " ".join(entry["variant_tokens"][index]))
        kw_hit = keyword_hit_rate(q_norm, entry.get("keywords", []))
        score = 0.55 * char_sim + 0.25 * token_sim + 0.20 * kw_hit

        # Strong signal when the user's question contains the core topic words.
        if kw_hit >= 0.5 and char_sim >= 0.35:
            score += 0.05
        return min(score, 1.0)

    def search(self, query: str, limit: int = 3) -> FAQMatch | None:
        q_norm = normalize_text(query)
        if not q_norm:
            return None

        candidates: list[dict[str, Any]] = []
        for entry in self._prepared:
            best_score = 0.0
            best_variant = entry["std_q"]
            for index, variant in enumerate(entry["variants"]):
                score = self._score_variant(q_norm, entry, index)
                if score > best_score:
                    best_score = score
                    best_variant = variant
            candidates.append(
                {
                    "faq_id": entry["id"],
                    "std_q": entry["std_q"],
                    "answer": entry["answer"],
                    "category": entry["category"],
                    "risk_level": entry["risk_level"],
                    "threshold": entry.get("threshold", 0.82),
                    "owner": entry.get("owner", ""),
                    "score": best_score,
                    "matched_question": best_variant,
                }
            )

        candidates.sort(key=lambda item: item["score"], reverse=True)
        top = candidates[0]
        second_score = candidates[1]["score"] if len(candidates) > 1 else 0.0
        margin = top["score"] - second_score
        passed = top["score"] >= top["threshold"] and margin >= self.default_margin

        return FAQMatch(
            faq_id=top["faq_id"],
            std_q=top["std_q"],
            answer=top["answer"],
            category=top["category"],
            risk_level=top["risk_level"],
            score=top["score"],
            margin=margin,
            threshold=top["threshold"],
            passed=passed,
            matched_question=top["matched_question"],
            owner=top["owner"],
            candidates=[
                {
                    "faq_id": item["faq_id"],
                    "std_q": item["std_q"],
                    "score": round(item["score"], 4),
                    "threshold": item["threshold"],
                }
                for item in candidates[:limit]
            ],
        )
