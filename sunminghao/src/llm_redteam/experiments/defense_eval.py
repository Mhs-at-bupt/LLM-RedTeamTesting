from __future__ import annotations

from llm_redteam.defense.integrated_defense import IntegratedDefensePipeline
from llm_redteam.evaluation.api_judge import APIJudge


def run_defense_eval(prompts: list[str], model_client, defense_config: dict) -> list[dict]:
    pipeline = IntegratedDefensePipeline(defense_config, judge=APIJudge())
    decisions = pipeline.defend_batch(prompts, model_client)
    rows = []
    for i, d in enumerate(decisions):
        rows.append(
            {
                "sample_id": i,
                "defense_method": "integrated_defense",
                "model": model_client.config.model_name,
                "blocked": d.blocked,
                "risk_score": d.risk_score,
                "decision": d.reason,
                "response": d.response,
                "success_after_defense": not d.blocked,
                "benign_utility_label": 1 if not d.blocked else 0,
            }
        )
    return rows

