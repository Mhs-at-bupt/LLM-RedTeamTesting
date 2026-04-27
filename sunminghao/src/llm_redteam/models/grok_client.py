from __future__ import annotations

import os
import time

import requests

from llm_redteam.models.base_client import ClientConfig, ModelClient
from llm_redteam.schemas import ModelResponse


class GrokClient(ModelClient):
    def __init__(self, config: ClientConfig):
        super().__init__(config)
        self.api_key = os.getenv("XAI_API_KEY", "")
        self.base_url = "https://api.x.ai/v1/chat/completions"

    def generate(self, prompt: str, system_prompt: str | None = None, **kwargs) -> ModelResponse:
        if not self.api_key:
            return ModelResponse(text="", model_name=self.config.model_name, raw={"error": "missing_xai_key"})
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.config.model_name,
            "messages": [{"role": "system", "content": system_prompt or "You are a helpful assistant."}, {"role": "user", "content": prompt}],
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9),
            "max_tokens": kwargs.get("max_tokens", 256),
        }
        last_err = None
        for _ in range(self.config.max_retries):
            try:
                r = requests.post(self.base_url, headers=headers, json=payload, timeout=self.config.timeout)
                r.raise_for_status()
                data = r.json()
                text = data["choices"][0]["message"]["content"]
                return ModelResponse(text=text, model_name=self.config.model_name, raw=data)
            except Exception as e:  # pragma: no cover
                last_err = e
                time.sleep(1)
        return ModelResponse(text="", model_name=self.config.model_name, raw={"error": str(last_err)})

