from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelResponse:
    text: str
    model_name: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    raw: dict[str, Any] | None = None


@dataclass
class AttackResult:
    goal_id: str
    goal: str
    best_user_prompt: str
    best_adversarial_prompt: str
    best_combined_prompt: str
    best_response: str
    best_fitness: float
    success_label: bool
    num_queries: int
    runtime_seconds: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DefenseDecision:
    blocked: bool
    reason: str
    risk_score: float
    action: str
    response: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

