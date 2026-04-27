from llm_redteam.defense.integrated_defense import IntegratedDefensePipeline
from llm_redteam.defense.keyword_filter import KeywordRuleFilter
from llm_redteam.evaluation.api_judge import APIJudge
from llm_redteam.schemas import ModelResponse


class MockClient:
    class _Cfg:
        model_name = "mock"

    config = _Cfg()

    def generate(self, prompt: str, system_prompt: str | None = None, **kwargs) -> ModelResponse:
        _ = system_prompt, kwargs
        if "harmful" in prompt.lower():
            return ModelResponse(text="I cannot help with that.", model_name="mock")
        return ModelResponse(text="Here is a safe response.", model_name="mock")


def test_keyword_filter_blocks_risky_prompt() -> None:
    cfg = {"keyword_filter": {"enabled": True, "threshold": 0.1, "risk_categories": ["harmful"]}}
    f = KeywordRuleFilter(cfg)
    d = f.filter("This is harmful content.")
    assert d.blocked


def test_integrated_pipeline_returns_decision() -> None:
    cfg = {
        "keyword_filter": {"enabled": True, "threshold": 0.9, "risk_categories": ["harmful"]},
        "smoothllm": {"enabled": False},
        "semantic_smooth": {"enabled": True, "num_variants": 3},
        "integrated_defense": {
            "use_keyword_filter": True,
            "use_smoothllm": False,
            "use_semantic_smooth": True,
            "use_jailguard": False,
        },
    }
    p = IntegratedDefensePipeline(cfg, judge=APIJudge())
    d = p.defend("A potentially harmful prompt", MockClient())
    assert hasattr(d, "blocked")
    assert hasattr(d, "risk_score")
