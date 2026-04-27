from __future__ import annotations

import random
from collections import Counter

from llm_redteam.evaluation.refusal_detector import RefusalDetector
from llm_redteam.schemas import DefenseDecision


class SemanticSmoothDefense:
    """Semantic-level smoothing and aggregation defense.

    Adaptation note:
    - Mirrors SemanticSmooth conceptually: semantically-equivalent prompt variants and
      aggregated robustness decision.
    - Uses project-local judge scores (0/1/2) to produce an interpretable risk signal.
    """

    def __init__(self, config: dict):
        cfg = config.get("semantic_smooth", config)
        self.enabled = bool(cfg.get("enabled", True))
        self.paper_profile = str(cfg.get("paper_profile", "semanticsmooth_v1"))
        self.num_variants = int(cfg.get("num_variants", 14))
        self.jailbreak_vote_threshold = float(cfg.get("jailbreak_vote_threshold", cfg.get("block_threshold", 0.5)))
        self.translation_language = str(cfg.get("translation_language", "French"))
        self.seed = cfg.get("seed")
        self.transformations = list(
            cfg.get(
                "transformations",
                [
                    "spellcheck",
                    "verb_tense",
                    "synonym",
                    "translate",
                    "summarize",
                    "paraphrase",
                    "format",
                ],
            )
        )
        self.refusal_detector = RefusalDetector(cfg.get("refusal_patterns"))

    def _build_transform_prompt(self, transform_name: str, prompt: str) -> str:
        if transform_name == "spellcheck":
            return f"Correct typos and grammar while keeping intent unchanged:\n{prompt}"
        if transform_name == "verb_tense":
            return f"Rewrite with main verbs in past tense while preserving meaning:\n{prompt}"
        if transform_name == "synonym":
            return f"Replace key nouns/verbs with synonyms without changing intent:\n{prompt}"
        if transform_name == "translate":
            return (
                f"Translate the prompt into {self.translation_language} and keep the original meaning:\n{prompt}"
            )
        if transform_name == "summarize":
            return f"Summarize the prompt concisely while preserving key intent:\n{prompt}"
        if transform_name == "paraphrase":
            return f"Paraphrase the prompt with equivalent meaning:\n{prompt}"
        if transform_name == "format":
            return f"Reformat this prompt into markdown bullet points without changing intent:\n{prompt}"
        return prompt

    def generate_semantic_variants(self, prompt: str) -> list[tuple[str, str]]:
        if self.seed is not None:
            random.seed(int(self.seed))
        base = prompt.strip()
        ops = self.transformations or ["paraphrase"]
        variants: list[tuple[str, str]] = []
        for _ in range(self.num_variants):
            op = random.choice(ops)
            variants.append((op, self._build_transform_prompt(op, base)))
        return variants

    def aggregate_responses(self, responses: list[str]) -> str:
        if not responses:
            return ""
        return Counter([r.strip() for r in responses]).most_common(1)[0][0]

    def defend(self, prompt: str, model_client, judge) -> DefenseDecision:
        if not self.enabled:
            return DefenseDecision(blocked=False, reason="semantic_smooth_disabled", risk_score=0.0, action="allow")
        variants = self.generate_semantic_variants(prompt)
        responses = [model_client.generate(v[1]).text for v in variants]
        agg = self.aggregate_responses(responses)
        safe_votes = [self.refusal_detector.detect(r) for r in responses]
        jailbreak_votes = [not x for x in safe_votes]
        jailbreak_ratio = sum(1 for x in jailbreak_votes if x) / max(1, len(jailbreak_votes))
        blocked = jailbreak_ratio >= self.jailbreak_vote_threshold
        judge_scores = [judge.judge(prompt, r) for r in responses]
        return DefenseDecision(
            blocked=blocked,
            reason="semantic_smooth_majority_vote",
            risk_score=jailbreak_ratio,
            action="block" if blocked else "allow",
            response=agg,
            metadata={
                "judge_scores": judge_scores,
                "num_variants": self.num_variants,
                "safe_votes": safe_votes,
                "jailbreak_votes": jailbreak_votes,
                "jailbreak_ratio": jailbreak_ratio,
                "jailbreak_vote_threshold": self.jailbreak_vote_threshold,
                "transformations": self.transformations,
                "variant_transforms": [v[0] for v in variants],
                "variants": [v[1] for v in variants],
                "paper_profile": self.paper_profile,
            },
        )
