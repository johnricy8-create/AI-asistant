from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections import Counter

_CN_SEQ = re.compile(r"[\u4e00-\u9fff]+")
_TOKEN = re.compile(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]+")
_SPACE = re.compile(r"\s+")
_PHONE = re.compile(r"(?<!\d)(1[3-9]\d{9})(?!\d)")
_ID_CARD = re.compile(r"(?<!\d)(\d{17}[\dXx])(?!\d)")
_EMAIL = re.compile(r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)")

# A small set of common traditional characters seen in customer messages.
_TRAD_TO_SIMP = str.maketrans(
    {
        "嗎": "吗",
        "麼": "么",
        "發": "发",
        "貨": "货",
        "錢": "钱",
        "訂": "订",
        "單": "单",
        "運": "运",
        "費": "费",
        "退": "退",
        "換": "换",
        "價": "价",
        "專": "专",
        "業": "业",
        "務": "务",
        "開": "开",
        "張": "张",
        "還": "还",
        "這": "这",
        "個": "个",
        "預": "预",
        "售": "售",
        "簽": "签",
        "收": "收",
        "請": "请",
        "問": "问",
        "題": "题",
        "幫": "帮",
        "轉": "转",
        "人": "人",
        "工": "工",
    }
)


def normalize_text(text: str | None) -> str:
    """Normalize user text without destroying meaning."""
    if not text:
        return ""
    value = unicodedata.normalize("NFKC", str(text))
    value = value.translate(_TRAD_TO_SIMP)
    value = value.replace("\u3000", " ")
    value = value.replace("？", "?").replace("！", "!").replace("，", ",")
    value = _SPACE.sub(" ", value).strip()
    return value


def tokenize(text: str | None) -> list[str]:
    """Chinese-friendly tokenizer: unigrams + bigrams + latin words."""
    value = normalize_text(text).lower()
    tokens: list[str] = []
    for match in _TOKEN.finditer(value):
        piece = match.group(0)
        if piece.isascii():
            tokens.append(piece)
            continue
        chars = list(piece)
        tokens.extend(chars)
        if len(chars) >= 2:
            tokens.extend("".join(chars[i : i + 2]) for i in range(len(chars) - 1))
            if len(chars) <= 6:
                tokens.append(piece)
    return tokens


def char_ngrams(text: str | None, n: int = 2) -> Counter[str]:
    value = normalize_text(text).lower()
    value = re.sub(r"[^\u4e00-\u9fffa-zA-Z0-9]", "", value)
    if not value:
        return Counter()
    if len(value) <= n:
        return Counter({value: 1})
    return Counter(value[i : i + n] for i in range(len(value) - n + 1))


def cosine_counter(a: Counter[str], b: Counter[str]) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    if not common:
        return 0.0
    numerator = sum(a[key] * b[key] for key in common)
    denominator = math.sqrt(sum(v * v for v in a.values())) * math.sqrt(
        sum(v * v for v in b.values())
    )
    return numerator / denominator if denominator else 0.0


def overlap_coefficient(left: str | None, right: str | None) -> float:
    a = set(tokenize(left))
    b = set(tokenize(right))
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def keyword_hit_rate(query: str | None, keywords: list[str] | None) -> float:
    if not keywords:
        return 0.0
    value = normalize_text(query)
    hits = sum(1 for keyword in keywords if normalize_text(keyword) in value)
    return hits / len(keywords)


def contains_any(text: str | None, words: list[str] | tuple[str, ...]) -> bool:
    value = normalize_text(text)
    return any(normalize_text(word) in value for word in words)


def mask_pii(text: str) -> str:
    value = _PHONE.sub(lambda m: m.group(1)[:3] + "****" + m.group(1)[-4:], text or "")
    value = _ID_CARD.sub(lambda m: m.group(1)[:4] + "**********" + m.group(1)[-4:], value)
    value = _EMAIL.sub(lambda m: m.group(1)[:2] + "***@" + m.group(1).split("@", 1)[1], value)
    return value


def stable_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


def first_sentence(text: str, max_chars: int = 220) -> str:
    value = re.sub(r"\s+", " ", (text or "").strip())
    if not value:
        return ""
    parts = re.split(r"(?<=[。！？!?；;])", value)
    sentence = next((part.strip() for part in parts if part.strip()), value)
    if len(sentence) > max_chars:
        sentence = sentence[:max_chars].rstrip("，,。；; ") + "……"
    return sentence


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))
