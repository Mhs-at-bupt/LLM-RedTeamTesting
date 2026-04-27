from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from llm_redteam.experiments.ablation import get_ablation_variants
from llm_redteam.experiments.main_results import summarize_main_results
from llm_redteam.experiments.runner import make_run_dir, run_main_attack
from llm_redteam.models import build_open_source_client
from llm_redteam.utils.config import load_yaml
from llm_redteam.utils.io_utils import save_csv, save_jsonl
from llm_redteam.utils.logging_utils import get_logger


def main(config_path: str = "configs/experiment_config.yaml") -> None:
    exp_cfg = load_yaml(config_path)
    attack_cfg = load_yaml(exp_cfg.get("attack_config", "configs/attack_config.yaml"))
    model_cfg = load_yaml(exp_cfg.get("model_config", "configs/model_config.yaml"))
    dataset = exp_cfg.get("dataset", "data/advbench/harmful_behaviors.csv")
    logger = get_logger("run_ablation")
    client = build_open_source_client(model_cfg, logger=logger)
    variants = get_ablation_variants()
    rows = []
    for v in variants:
        cfg = deepcopy(attack_cfg)
        if v == "autodan_hga_baseline":
            cfg["user_prompt_mutation_probability"] = 0.0
            cfg["character_mutation_probability"] = 0.0
            cfg["word_mutation_probability"] = 0.0
            cfg["sentence_mutation_probability"] = 0.0
        elif v == "structured_mutation_only":
            cfg["momentum_coefficient"] = 0.0
        elif v == "structured_mutation_joint_search":
            cfg["momentum_coefficient"] = 0.0
            cfg["user_prompt_mutation_probability"] = 0.1
        elif v == "full_method":
            cfg["momentum_coefficient"] = float(attack_cfg.get("momentum_coefficient", 0.9))
            cfg["user_prompt_mutation_probability"] = float(attack_cfg.get("user_prompt_mutation_probability", 0.1))
            cfg["character_mutation_probability"] = float(attack_cfg.get("character_mutation_probability", 0.2))
            cfg["word_mutation_probability"] = float(attack_cfg.get("word_mutation_probability", 0.4))
            cfg["sentence_mutation_probability"] = float(attack_cfg.get("sentence_mutation_probability", 0.4))
        run_dir = make_run_dir("results/runs")
        results = run_main_attack(cfg, client, dataset, run_dir)
        s = summarize_main_results(results)
        rows.append({"variant": v, "asr": s["asr"], "avg_fitness": s["avg_fitness"], "avg_queries": s["avg_queries"]})
    save_csv(rows, "results/ablation_results.csv")
    save_jsonl(rows, "results/ablation_results.jsonl")
    Path("results/ablation_summary.md").write_text(
        "# Ablation Summary\n\n" + "\n".join([f"- {r['variant']}: ASR={r['asr']:.4f}, fitness={r['avg_fitness']:.4f}" for r in rows]),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
