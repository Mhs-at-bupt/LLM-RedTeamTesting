from __future__ import annotations

import os
import time
from typing import Any

from llm_redteam.models.base_client import ClientConfig, ModelClient
from llm_redteam.schemas import ModelResponse


class OpenAIClient(ModelClient):
    def __init__(self, config: ClientConfig):
        super().__init__(config)
        key = os.getenv("OPENAI_API_KEY", "")
        if key:
            try:
                from openai import OpenAI

                self.client = OpenAI(api_key=key)
            except Exception:  # pragma: no cover
                self.client = None
        else:
            self.client = None

    def generate(self, prompt: str, system_prompt: str | None = None, **kwargs) -> ModelResponse:
        if self.client is None:
            return ModelResponse(text="", model_name=self.config.model_name, raw={"error": "missing_openai_key"})
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        last_err: Any = None
        for _ in range(self.config.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.config.model_name,
                    messages=messages,
                    timeout=self.config.timeout,
                    temperature=kwargs.get("temperature", 0.7),
                    top_p=kwargs.get("top_p", 0.9),
                    max_tokens=kwargs.get("max_tokens", 256),
                )
                msg = (resp.choices[0].message.content or "").strip()
                usage = getattr(resp, "usage", None)
                return ModelResponse(
                    text=msg,
                    model_name=self.config.model_name,
                    prompt_tokens=getattr(usage, "prompt_tokens", None),
                    completion_tokens=getattr(usage, "completion_tokens", None),
                    raw=resp.model_dump() if hasattr(resp, "model_dump") else None,
                )
            except Exception as e:  # pragma: no cover
                last_err = e
                time.sleep(1.0)
        return ModelResponse(text="", model_name=self.config.model_name, raw={"error": str(last_err)})
