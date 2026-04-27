from llm_redteam.utils.config import load_yaml


def test_all_main_configs_load() -> None:
    files = [
        "configs/attack_config.yaml",
        "configs/model_config.yaml",
        "configs/defense_config.yaml",
        "configs/experiment_config.yaml",
        "configs/peft_config.yaml",
    ]
    for f in files:
        cfg = load_yaml(f)
        assert isinstance(cfg, dict)
        assert len(cfg) > 0

