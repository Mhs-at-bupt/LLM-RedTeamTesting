from __future__ import annotations

import re
from dataclasses import dataclass

from llm_redteam.evaluation.refusal_detector import RefusalDetector
from llm_redteam.schemas import DefenseDecision


@dataclass
class KeywordMatch:
    category_hits: list[str]
    pattern_hits: list[str]
    refusal_hint_hits: list[str]


class KeywordRuleFilter:
    """Rule-based pre-filter inspired by keyword/policy guardrails in safety pipelines.

    This module is intentionally lightweight and deterministic. It is used as the first
    gate before higher-cost stochastic defenses.
    """

    def __init__(self, config: dict):
        cfg = config.get("keyword_filter", config)
        self.enabled = bool(cfg.get("enabled", True))
        self.threshold = float(cfg.get("threshold", 0.5))
        self.risk_categories = cfg.get("risk_categories", ["weapon", "malware", "fraud"])
        self.category_weight = float(cfg.get("category_weight", 0.25))
        self.pattern_weight = float(cfg.get("pattern_weight", 0.25))
        self.refusal_hint_weight = float(cfg.get("refusal_hint_weight", 0.10))
        self.refusal_patterns = cfg.get("refusal_patterns", RefusalDetector.DEFAULT_PATTERNS)
        self.pattern_rules = [re.compile(x, re.I) for x in cfg.get("pattern_rules", [])]
        self._category_regex = [re.compile(rf"\b{re.escape(x)}\b", re.I) for x in self.risk_categories]
        self._refusal_regex = [re.compile(x, re.I) for x in self.refusal_patterns]

    def find_matches(self, prompt: str) -> KeywordMatch:
        text = prompt or ""
        category_hits = [
            category
            for category, pat in zip(self.risk_categories, self._category_regex, strict=False)
            if pat.search(text)
        ]
        pattern_hits = [pat.pattern for pat in self.pattern_rules if pat.search(text)]
        refusal_hint_hits = [pat.pattern for pat in self._refusal_regex if pat.search(text)]
        return KeywordMatch(
            category_hits=category_hits,
            pattern_hits=pattern_hits,
            refusal_hint_hits=refusal_hint_hits,
        )

    def score(self, prompt: str) -> float:
        hits = self.find_matches(prompt)
        score = 0.0
        score += len(hits.category_hits) * self.category_weight
        score += len(hits.pattern_hits) * self.pattern_weight
        score += len(hits.refusal_hint_hits) * self.refusal_hint_weight
        return min(1.0, score)

    def is_risky(self, prompt: str) -> bool:
        return self.score(prompt) >= self.threshold

    def filter(self, prompt: str) -> DefenseDecision:
        rs = self.score(prompt)
        hits = self.find_matches(prompt)
        metadata = {
            "category_hits": hits.category_hits,
            "pattern_hits": hits.pattern_hits,
            "refusal_hint_hits": hits.refusal_hint_hits,
            "threshold": self.threshold,
        }
        if self.enabled and rs >= self.threshold:
            return DefenseDecision(
                blocked=True,
                reason="keyword_rule_filter",
                risk_score=rs,
                action="block",
                metadata=metadata,
            )
        return DefenseDecision(
            blocked=False,
            reason="pass",
            risk_score=rs,
            action="allow",
            metadata=metadata,
        )
