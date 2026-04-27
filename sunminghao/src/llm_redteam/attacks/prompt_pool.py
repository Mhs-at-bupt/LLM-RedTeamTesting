from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PromptCandidate:
    text: str
    fitness: float = 0.0
    metadata: dict = field(default_factory=dict)


class PromptPool:
    def __init__(self) -> None:
        self._items: list[PromptCandidate] = []

    def add(self, text: str, fitness: float = 0.0, metadata: dict | None = None) -> None:
        self._items.append(PromptCandidate(text=text, fitness=fitness, metadata=metadata or {}))

    def sample(self, n: int) -> list[PromptCandidate]:
        if not self._items:
            return []
        n = min(n, len(self._items))
        return random.sample(self._items, n)

    def rank_by_fitness(self, descending: bool = True) -> list[PromptCandidate]:
        self._items = sorted(self._items, key=lambda x: x.fitness, reverse=descending)
        return self._items

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump([item.__dict__ for item in self._items], f, ensure_ascii=False, indent=2)

    def load(self, path: str | Path) -> None:
        with Path(path).open("r", encoding="utf-8") as f:
            raw = json.load(f)
        self._items = [PromptCandidate(**x) for x in raw]

    @property
    def items(self) -> list[PromptCandidate]:
        return self._items

