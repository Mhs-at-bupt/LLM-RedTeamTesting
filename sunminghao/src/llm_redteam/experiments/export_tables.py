from __future__ import annotations

from pathlib import Path
from typing import Any

from llm_redteam.utils.io_utils import load_jsonl


def export_simple_latex(results_jsonl: str, output_tex: str, caption: str = "Results") -> None:
    rows = load_jsonl(results_jsonl)
    p = Path(output_tex)
    p.parent.mkdir(parents=True, exist_ok=True)
    def _pick(r: dict[str, Any], *keys: str, default: float = 0.0) -> float:
        for k in keys:
            if k in r:
                try:
                    return float(r[k])
                except Exception:
                    return default
        return default

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\begin{tabular}{lccc}",
        "\\hline",
        "Sample & Fitness & Success & Queries \\\\",
        "\\hline",
    ]
    for i, r in enumerate(rows[:20]):
        fit = _pick(r, "fitness", "risk_score")
        succ = int(bool(r.get("success", r.get("success_after_defense", False))))
        qry = int(_pick(r, "num_queries", default=0.0))
        lines.append(f"{i} & {fit:.3f} & {succ} & {qry} \\\\")
    lines += ["\\hline", "\\end{tabular}", f"\\caption{{{caption}}}", "\\end{table}"]
    p.write_text("\n".join(lines), encoding="utf-8")
