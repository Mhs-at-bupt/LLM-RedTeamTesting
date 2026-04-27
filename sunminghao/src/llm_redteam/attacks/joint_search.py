from __future__ import annotations

import random

from llm_redteam.attacks.prompt_pool import PromptCandidate, PromptPool


def sample_user_prompts(pool: PromptPool, batch_size: int) -> list[PromptCandidate]:
    return pool.sample(batch_size) if pool.items else []


def combine_prompt(adversarial_prompt: str, user_prompt: str, goal: str) -> str:
    return f"{user_prompt}\n\n{adversarial_prompt}\n\nTask Context: {goal}".strip()


def update_prompt_pool(pool: PromptPool, candidates: list[PromptCandidate], top_k: int = 50) -> None:
    for c in candidates:
        pool.add(c.text, c.fitness, c.metadata)
    pool.rank_by_fitness(descending=True)
    if len(pool.items) > top_k:
        del pool.items[top_k:]


def select_best_candidates(candidates: list[PromptCandidate], num_elites: int) -> list[PromptCandidate]:
    ranked = sorted(candidates, key=lambda x: x.fitness, reverse=True)
    elites = ranked[: max(1, num_elites)]
    rest = ranked[max(1, num_elites) :]
    random.shuffle(rest)
    return elites + rest

