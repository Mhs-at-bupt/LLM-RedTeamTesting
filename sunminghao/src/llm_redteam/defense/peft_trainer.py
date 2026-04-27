from __future__ import annotations

from dataclasses import dataclass

from llm_redteam.utils.config import load_yaml


@dataclass
class PeftTrainConfig:
    base_model: str
    output_dir: str
    train_file: str
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    learning_rate: float = 2e-5
    batch_size: int = 4
    epochs: int = 3
    max_length: int = 2048


def load_peft_config(path: str) -> PeftTrainConfig:
    cfg = load_yaml(path)
    return PeftTrainConfig(**cfg)


def train_peft_lora(config_path: str) -> None:
    """Training skeleton (not auto-run)."""
    cfg = load_peft_config(config_path)
    print("PEFT/LoRA training skeleton loaded.")
    print(f"Base model: {cfg.base_model}")
    print(f"Train file: {cfg.train_file}")
    print(f"Output adapter dir: {cfg.output_dir}")
    print("Integrate transformers+peft trainer here when running full experiments.")

