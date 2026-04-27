from __future__ import annotations

import random
import string
from collections import Counter

from llm_redteam.evaluation.refusal_detector import RefusalDetector
from llm_redteam.schemas import DefenseDecision


class SmoothLLMDefense:
    """Character-level randomized smoothing style defense.

    Adaptation note:
    - Follows SmoothLLM's high-level idea (perturb prompt copies and check response consistency).
    - Uses a lightweight disagreement score for engineering simplicity.
    """

    def __init__(self, config: dict):
        cfg = config.get("smoothllm", config)
        self.enabled = bool(cfg.get("enabled", True))
        self.paper_profile = str(cfg.get("paper_profile", "smoothllm_v1"))
        self.num_copies = int(cfg.get("num_copies", 10))
        self.q = float(cfg.get("perturbation_ratio", 0.10))
        legacy_type = cfg.get("perturbation_type")
        default_types = [legacy_type] if legacy_type else ["swap", "patch", "insert"]
        self.perturbation_types = list(cfg.get("perturbation_types", default_types))
        self.jailbreak_vote_threshold = float(cfg.get("jailbreak_vote_threshold", cfg.get("block_threshold", 0.5)))
        self.seed = cfg.get("seed")
        self.refusal_detector = RefusalDetector(cfg.get("refusal_patterns"))

    @staticmethod
    def _normalize_response(text: str) -> str:
        return " ".join((text or "").strip().lower().split())

    def _perturb_swap(self, chars: list[str], idx: int) -> None:
        if idx < len(chars) - 1:
            chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]

    def _perturb_patch(self, chars: list[str], idx: int) -> None:
        chars[idx] = random.choice(string.ascii_letters + string.digits + " ")

    def _perturb_insert(self, chars: list[str], idx: int) -> None:
        chars.insert(idx, random.choice(string.ascii_letters + string.digits + " "))

    def perturb(self, prompt: str) -> str:
        chars = list(prompt)
        if not chars:
            return prompt
        n = max(1, int(len(chars) * self.q))
        for _ in range(n):
            i = random.randint(0, len(chars) - 1)
            p = random.choice(self.perturbation_types) if self.perturbation_types else "swap"
            if p == "swap":
                self._perturb_swap(chars, i)
            elif p == "insert":
                self._perturb_insert(chars, i)
            else:
                self._perturb_patch(chars, i)
        return "".join(chars)

    def generate_variants(self, prompt: str) -> list[str]:
        if self.seed is not None:
            random.seed(int(self.seed))
        return [self.perturb(prompt) for _ in range(self.num_copies)]

    def defend(self, prompt: str, model_client) -> DefenseDecision:
        if not self.enabled:
            return DefenseDecision(blocked=False, reason="smoothllm_disabled", risk_score=0.0, action="allow")
        variants = self.generate_variants(prompt)
        responses = [model_client.generate(v).text for v in variants]
        normalized = [self._normalize_response(r) for r in responses]
        counter = Counter(normalized)
        label = counter.most_common(1)[0][0] if normalized else ""
        safe_votes = [self.refusal_detector.detect(r) for r in responses]
        jailbreak_votes = [not x for x in safe_votes]
        jailbreak_ratio = sum(1 for x in jailbreak_votes if x) / max(1, len(jailbreak_votes))
        blocked = jailbreak_ratio >= self.jailbreak_vote_threshold
        return DefenseDecision(
            blocked=blocked,
            reason="smoothllm_majority_vote",
            risk_score=jailbreak_ratio,
            action="block" if blocked else "allow",
            response=label,
            metadata={
                "responses": responses,
                "normalized_counts": dict(counter),
                "num_copies": self.num_copies,
                "safe_votes": safe_votes,
                "jailbreak_votes": jailbreak_votes,
                "jailbreak_ratio": jailbreak_ratio,
                "jailbreak_vote_threshold": self.jailbreak_vote_threshold,
                "paper_profile": self.paper_profile,
                "perturbation_types": self.perturbation_types,
            },
        )
