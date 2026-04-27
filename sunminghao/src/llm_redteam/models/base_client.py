from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from llm_redteam.schemas import ModelResponse


@dataclass
class ClientConfig:
    model_name: str
    timeout: float = 60.0
    max_retries: int = 3


class ModelClient(ABC):
    def __init__(self, config: ClientConfig):
        self.config = config

    @abstractmethod
    def generate(self, prompt: str, system_prompt: str | None = None, **kwargs) -> ModelResponse:
        raise NotImplementedError

    def batch_generate(self, prompts: list[str], **kwargs) -> list[ModelResponse]:
        return [self.generate(prompt=p, **kwargs) for p in prompts]

