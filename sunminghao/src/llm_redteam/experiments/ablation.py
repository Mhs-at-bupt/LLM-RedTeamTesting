from __future__ import annotations


def get_ablation_variants() -> list[str]:
    return [
        "autodan_hga_baseline",
        "structured_mutation_only",
        "structured_mutation_joint_search",
        "full_method",
    ]

