from __future__ import annotations

import math
from typing import Protocol


class EncoderLike(Protocol):
    def encode(self, text: str) -> list[float]: ...


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return max(-1.0, min(1.0, dot / (na * nb)))


class _FallbackEncoder:
    def encode(self, text: str) -> list[float]:
        text = (text or "")[:128]
        vec = [0.0] * 16
        for i, ch in enumerate(text):
            vec[i % 16] += (ord(ch) % 97) / 97.0
        return vec


class SemanticFitnessScorer:
    def __init__(self, encoder: EncoderLike | None = None):
        self.encoder = encoder or _FallbackEncoder()

    def score(self, target_intent: str, response: str) -> float:
        sim = _cosine(self.encoder.encode(target_intent), self.encoder.encode(response))
        return (sim + 1.0) / 2.0

