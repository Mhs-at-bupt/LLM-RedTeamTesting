from __future__ import annotations

from typing import Any

from llm_redteam.models.base_client import ClientConfig, ModelClient
from llm_redteam.schemas import ModelResponse


class OpenSourceClient(ModelClient):
    """Wrapper around local HF causal LM generation."""

    def __init__(
        self,
        config: ClientConfig,
        model: Any = None,
        tokenizer: Any = None,
        use_mock_when_unavailable: bool = True,
    ):
        super().__init__(config)
        self.model = model
        self.tokenizer = tokenizer
        self.use_mock_when_unavailable = use_mock_when_unavailable

    @staticmethod
    def _mock_response(prompt: str) -> str:
        p = (prompt or "").lower()
        risky_markers = [
            "harmful",
            "malware",
            "weapon",
            "bypass",
            "exploit",
            "illegal",
            "attack",
            "hack",
        ]
        if any(x in p for x in risky_markers):
            return "I'm sorry, but I cannot assist with harmful, illegal, or unsafe requests."
        return "Here is a safe and policy-compliant response."

    def generate(self, prompt: str, system_prompt: str | None = None, **kwargs) -> ModelResponse:
        full_prompt = f"{system_prompt}\n{prompt}" if system_prompt else prompt
        if self.model is None or self.tokenizer is None:
            if self.use_mock_when_unavailable:
                return ModelResponse(
                    text=self._mock_response(full_prompt),
                    model_name=self.config.model_name,
                    raw={"warning": "model_not_loaded_mock_mode"},
                )
            return ModelResponse(text="", model_name=self.config.model_name, raw={"error": "model_not_loaded"})
        inputs = self.tokenizer(full_prompt, return_tensors="pt")
        # Keep tensor device aligned with model to avoid cpu/cuda mismatch.
        model_device = getattr(self.model, "device", None)
        if model_device is not None:
            try:
                inputs = {k: v.to(model_device) for k, v in inputs.items()}
            except Exception:
                pass
        output = self.model.generate(
            **inputs,
            max_new_tokens=kwargs.get("max_tokens", 256),
            temperature=kwargs.get("temperature", 0.7),
            top_p=kwargs.get("top_p", 0.9),
            do_sample=kwargs.get("do_sample", True),
        )
        text = self.tokenizer.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        return ModelResponse(text=text, model_name=self.config.model_name, raw=None)
