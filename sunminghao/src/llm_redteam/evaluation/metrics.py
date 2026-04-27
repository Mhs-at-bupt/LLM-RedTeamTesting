from __future__ import annotations

from statistics import mean


def compute_asr(results: list[dict]) -> float:
    if not results:
        return 0.0
    return sum(1 for r in results if bool(r.get("success", False))) / len(results)


def compute_avg_fitness(results: list[dict]) -> float:
    vals = [float(r.get("fitness", 0.0)) for r in results]
    return mean(vals) if vals else 0.0


def compute_avg_queries(results: list[dict]) -> float:
    vals = [float(r.get("num_queries", 0.0)) for r in results]
    return mean(vals) if vals else 0.0


def compute_runtime_stats(results: list[dict]) -> dict:
    vals = [float(r.get("runtime_seconds", 0.0)) for r in results]
    if not vals:
        return {"avg": 0.0, "min": 0.0, "max": 0.0}
    return {"avg": mean(vals), "min": min(vals), "max": max(vals)}

