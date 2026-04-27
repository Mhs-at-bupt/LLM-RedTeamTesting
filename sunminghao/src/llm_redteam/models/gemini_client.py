from __future__ import annotations

import os
import time

import requests

from llm_redteam.models.base_client import ClientConfig, ModelClient
from llm_redteam.schemas import ModelResponse


class GeminiClient(ModelClient):
    def __init__(self, config: ClientConfig):
        super().__init__(config)
        self.api_key = os.getenv("GEMINI_API_KEY", "")

    def generate(self, prompt: str, system_prompt: str | None = None, **kwargs) -> ModelResponse:
        if not self.api_key:
            return ModelResponse(text="", model_name=self.config.model_name, raw={"error": "missing_gemini_key"})
        payload = {"contents": [{"parts": [{"text": f"{system_prompt or ''}\n{prompt}".strip()}]}]}
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.config.model_name}:generateContent?key={self.api_key}"
        )
        last_err = None
        for _ in range(self.config.max_retries):
            try:
                r = requests.post(url, json=payload, timeout=self.config.timeout)
                r.raise_for_status()
                data = r.json()
                text = data["candidates"][0]["content"]["parts"][0]["text"]
                return ModelResponse(text=text, model_name=self.config.model_name, raw=data)
            except Exception as e:  # pragma: no cover
                last_err = e
                time.sleep(1)
        return ModelResponse(text="", model_name=self.config.model_name, raw={"error": str(last_err)})

