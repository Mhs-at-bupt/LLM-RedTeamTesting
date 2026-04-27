from __future__ import annotations

from llm_redteam.defense.jailguard_detector import JailGuardStyleDetector
from llm_redteam.defense.keyword_filter import KeywordRuleFilter
from llm_redteam.defense.semantic_smooth import SemanticSmoothDefense
from llm_redteam.defense.smoothllm import SmoothLLMDefense
from llm_redteam.schemas import DefenseDecision


class IntegratedDefensePipeline:
    """Unified defense orchestrator.

    Layer order:
    1) Keyword / rule pre-filter
    2) SmoothLLM response-consistency check
    3) SemanticSmooth aggregation check
    4) JailGuard-style discrepancy detector
    """

    def __init__(self, config: dict, judge):
        self.config = config
        self.judge = judge
        self.keyword = KeywordRuleFilter(config)
        self.smooth = SmoothLLMDefense(config)
        self.semantic = SemanticSmoothDefense(config)
        self.jailguard = JailGuardStyleDetector(config)
        flags = config.get("integrated_defense", {})
        self.use_keyword = bool(flags.get("use_keyword_filter", True))
        self.use_smooth = bool(flags.get("use_smoothllm", True))
        self.use_semantic = bool(flags.get("use_semantic_smooth", True))
        self.use_jailguard = bool(flags.get("use_jailguard", False))
        self.use_peft_if_available = bool(flags.get("use_peft_if_available", True))

    def defend(self, prompt: str, model_client) -> DefenseDecision:
        layer_trace: list[dict] = []
        if self.use_keyword:
            d = self.keyword.filter(prompt)
            layer_trace.append({"layer": "keyword_filter", "blocked": d.blocked, "risk_score": d.risk_score})
            if d.blocked:
                d.metadata["layer_trace"] = layer_trace
                return d
        if self.use_smooth:
            d = self.smooth.defend(prompt, model_client)
            layer_trace.append({"layer": "smoothllm", "blocked": d.blocked, "risk_score": d.risk_score})
            if d.blocked:
                d.metadata["layer_trace"] = layer_trace
                return d
        if self.use_semantic:
            d = self.semantic.defend(prompt, model_client, self.judge)
            layer_trace.append({"layer": "semantic_smooth", "blocked": d.blocked, "risk_score": d.risk_score})
            if d.blocked:
                d.metadata["layer_trace"] = layer_trace
                return d
        if self.use_jailguard:
            jailguard_score = self.jailguard.discrepancy_score(prompt, model_client)
            layer_trace.append(
                {
                    "layer": "jailguard",
                    "blocked": jailguard_score >= self.jailguard.threshold,
                    "risk_score": jailguard_score,
                }
            )
            if jailguard_score >= self.jailguard.threshold:
                return DefenseDecision(
                    blocked=True,
                    reason="jailguard_discrepancy",
                    risk_score=jailguard_score,
                    action="block",
                    metadata={"threshold": self.jailguard.threshold, "layer_trace": layer_trace},
                )
            return DefenseDecision(
                blocked=False,
                reason="passed_all_layers",
                risk_score=0.0,
                action="allow",
                metadata={"peft_offline_supported": self.use_peft_if_available, "layer_trace": layer_trace},
            )
        return DefenseDecision(
            blocked=False,
            reason="passed_all_layers",
            risk_score=0.0,
            action="allow",
            metadata={
                "peft_offline_supported": self.use_peft_if_available,
                "layer_trace": layer_trace,
                "paper_profiles": {
                    "smoothllm": getattr(self.smooth, "paper_profile", "unknown"),
                    "semantic_smooth": getattr(self.semantic, "paper_profile", "unknown"),
                    "jailguard": getattr(self.jailguard, "paper_profile", "unknown"),
                },
            },
        )

    def defend_batch(self, prompts: list[str], model_client) -> list[DefenseDecision]:
        return [self.defend(p, model_client) for p in prompts]
