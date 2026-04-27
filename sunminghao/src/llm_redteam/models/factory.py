from __future__ import annotations

from typing import Any

from llm_redteam.models.base_client import ClientConfig
from llm_redteam.models.open_source_client import OpenSourceClient


def build_open_source_client(model_cfg: dict[str, Any], logger=None) -> OpenSourceClient:
    runtime_cfg = model_cfg.get("open_source_runtime", {})
    model_name = str(model_cfg.get("open_source_models", ["llama-2-7b-chat"])[0])
    timeout = float(runtime_cfg.get("timeout", 60.0))
    max_retries = int(runtime_cfg.get("max_retries", 3))
    model_id = str(runtime_cfg.get("model_id", model_name))
    trust_remote_code = bool(runtime_cfg.get("trust_remote_code", False))
    use_mock_when_unavailable = bool(runtime_cfg.get("use_mock_when_unavailable", True))
    device_map = runtime_cfg.get("device_map", "auto")
    torch_dtype = str(runtime_cfg.get("torch_dtype", "auto"))

    config = ClientConfig(model_name=model_name, timeout=timeout, max_retries=max_retries)
    model = None
    tokenizer = None
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
        if torch_dtype != "auto":
            try:
                import torch

                model_kwargs["torch_dtype"] = getattr(torch, torch_dtype)
            except Exception:
                if logger:
                    logger.warning("Failed to parse torch_dtype=%s, fallback to auto.", torch_dtype)
        if device_map:
            model_kwargs["device_map"] = device_map
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        if logger:
            logger.info("Loaded open-source model: model_id=%s as model_name=%s", model_id, model_name)
    except Exception as e:
        if logger:
            logger.warning("Open-source model load failed, fallback=%s, err=%s", use_mock_when_unavailable, e)
    return OpenSourceClient(
        config=config,
        model=model,
        tokenizer=tokenizer,
        use_mock_when_unavailable=use_mock_when_unavailable,
    )

