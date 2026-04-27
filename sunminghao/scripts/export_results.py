from __future__ import annotations

from pathlib import Path

from llm_redteam.experiments.export_tables import export_simple_latex


def main() -> None:
    main_jsonl = "results/runs/latest/results.jsonl"
    if Path(main_jsonl).exists():
        export_simple_latex(main_jsonl, "results/latex_table_main_results.tex", caption="Main Results")
    ablation_jsonl = "results/ablation_results.jsonl"
    if Path(ablation_jsonl).exists():
        export_simple_latex(ablation_jsonl, "results/latex_table_ablation.tex", caption="Ablation Results")
    defense_jsonl = "results/defense_results.jsonl"
    if Path(defense_jsonl).exists():
        export_simple_latex(defense_jsonl, "results/latex_table_defense.tex", caption="Defense Results")


if __name__ == "__main__":
    main()
