from __future__ import annotations

from llm_redteam.evaluation.metrics import (
    compute_asr,
    compute_avg_fitness,
    compute_avg_queries,
    compute_runtime_stats,
)


def summarize_main_results(results: list[dict]) -> dict:
    return {
        "asr": compute_asr(results),
        "avg_fitness": compute_avg_fitness(results),
        "avg_queries": compute_avg_queries(results),
        "runtime": compute_runtime_stats(results),
    }

