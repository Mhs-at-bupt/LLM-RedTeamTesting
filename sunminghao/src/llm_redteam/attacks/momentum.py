from __future__ import annotations

import random
from collections import defaultdict


class MomentumVocab:
    def __init__(self, mu: float = 0.9):
        self.mu = mu
        self._scores: dict[str, float] = defaultdict(float)

    def update_momentum_vocab(self, tokens: list[str], scores: list[float]) -> None:
        for token, score in zip(tokens, scores):
            prev = self._scores.get(token, 0.0)
            self._scores[token] = self.mu * prev + (1.0 - self.mu) * float(score)

    def get_top_tokens(self, k: int = 20) -> list[str]:
        return [x[0] for x in sorted(self._scores.items(), key=lambda item: item[1], reverse=True)[:k]]

    def apply_momentum_mutation(self, prompt: str) -> str:
        words = prompt.split()
        if not words:
            return prompt
        top = self.get_top_tokens(k=min(20, len(self._scores)))
        if not top:
            return prompt
        idx = random.randint(0, len(words) - 1)
        words[idx] = random.choice(top)
        return " ".join(words)

