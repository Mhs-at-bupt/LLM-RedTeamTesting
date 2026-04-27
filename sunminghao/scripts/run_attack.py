from __future__ import annotations

import json
import shutil
from pathlib import Path

from llm_redteam.experiments.main_results import summarize_main_results
from llm_redteam.experiments.runner import make_run_dir, run_main_attack
from llm_redteam.models import build_open_source_client
from llm_redteam.utils.config import dump_yaml, load_yaml
from llm_redteam.utils.io_utils import save_csv
from llm_redteam.utils.logging_utils import get_logger
from llm_redteam.utils.seed import set_seed


def main(attack_cfg_path: str = "configs/attack_config.yaml") -> None:
    attack_cfg = load_yaml(attack_cfg_path)
    model_cfg = load_yaml("configs/model_config.yaml")
    set_seed(int(attack_cfg.get("random_seed", 42)))
    run_dir = make_run_dir("results/runs")
    logger = get_logger("run_attack", f"{run_dir}/logs.txt")
    client = build_open_source_client(model_cfg, logger=logger)
    results = run_main_attack(
        config=attack_cfg,
        model_client=client,
        dataset_path=str(Path("data/advbench/harmful_behaviors.csv")),
        output_dir=run_dir,
    )
    save_csv(results, Path(run_dir) / "summary.csv")
    dump_yaml({"attack": attack_cfg, "model": model_cfg}, Path(run_dir) / "config_used.yaml")
    summary = summarize_main_results(results)
    (Path(run_dir) / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    latest = Path("results/runs/latest")
    if latest.exists():
        shutil.rmtree(latest)
    shutil.copytree(run_dir, latest)
    logger.info("Attack run completed: %s", run_dir)


if __name__ == "__main__":
    main()
