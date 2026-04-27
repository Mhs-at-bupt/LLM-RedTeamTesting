from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from llm_redteam.attacks.enhanced_autodan import AttackEvaluator, EnhancedAutoDAN
from llm_redteam.evaluation.api_judge import APIJudge
from llm_redteam.evaluation.refusal_detector import RefusalDetector
from llm_redteam.evaluation.semantic_fitness import SemanticFitnessScorer
from llm_redteam.utils.io_utils import save_jsonl


def run_main_attack(config: dict, model_client, dataset_path: str, output_dir: str) -> list[dict]:
    ds = Path(dataset_path)
    if ds.exists():
        data = pd.read_csv(ds)
    else:
        data = pd.DataFrame({"goal": ["Explain safe evaluation methodology for language models."]})
    evaluator = AttackEvaluator(
        semantic=SemanticFitnessScorer(),
        refusal=RefusalDetector(config.get("refusal_patterns")),
        judge=APIJudge(),
    )
    attacker = EnhancedAutoDAN(config=config, model_client=model_client, evaluator=evaluator)
    results = []
    for idx, row in data.iterrows():
        ret = attacker.run(str(row["goal"]))
        results.append(
            {
                "goal_id": ret.goal_id,
                "goal": ret.goal,
                "method": "enhanced_autodan",
                "model": model_client.config.model_name,
                "user_prompt": ret.best_user_prompt,
                "adversarial_prompt": ret.best_adversarial_prompt,
                "combined_prompt": ret.best_combined_prompt,
                "response": ret.best_response,
                "fitness": ret.best_fitness,
                "refusal_label": bool(ret.metadata.get("refusal", True)),
                "target_hit_label": int(ret.metadata.get("judge", 0)),
                "success": ret.success_label,
                "num_queries": ret.num_queries,
                "runtime_seconds": ret.runtime_seconds,
                "sample_id": idx,
            }
        )
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_jsonl(results, out / "results.jsonl")
    return results


def make_run_dir(base: str = "results/runs") -> str:
    ts = time.strftime("%Y%m%d-%H%M%S")
    p = Path(base) / ts
    p.mkdir(parents=True, exist_ok=True)
    return str(p)
