from __future__ import annotations

import importlib
import os
import platform
import sys
from pathlib import Path

from llm_redteam.utils.config import load_yaml


def _check_python() -> tuple[bool, str]:
    ok = sys.version_info >= (3, 10)
    return ok, f"Python {platform.python_version()} (required >= 3.10)"


def _check_module(name: str) -> tuple[bool, str]:
    try:
        importlib.import_module(name)
        return True, f"Module '{name}' is available"
    except Exception as e:
        return False, f"Module '{name}' import failed: {e}"


def _check_path(path: str) -> tuple[bool, str]:
    p = Path(path)
    return p.exists(), f"Path '{path}' exists={p.exists()}"


def _print_check(name: str, ok: bool, detail: str) -> None:
    status = "OK" if ok else "FAIL"
    print(f"[{status}] {name}: {detail}")


def main() -> int:
    checks: list[tuple[str, bool, str]] = []
    py_ok, py_msg = _check_python()
    checks.append(("python", py_ok, py_msg))

    for mod in [
        "yaml",
        "numpy",
        "pandas",
        "torch",
        "transformers",
        "sentence_transformers",
        "openai",
        "requests",
        "pytest",
    ]:
        ok, msg = _check_module(mod)
        checks.append((f"dependency:{mod}", ok, msg))

    attack_cfg = load_yaml("configs/attack_config.yaml")
    model_cfg = load_yaml("configs/model_config.yaml")
    exp_cfg = load_yaml("configs/experiment_config.yaml")
    def_cfg = load_yaml("configs/defense_config.yaml")
    _ = attack_cfg, def_cfg

    dataset = str(exp_cfg.get("dataset", "data/advbench/harmful_behaviors.csv"))
    checks.append(("dataset", *_check_path(dataset)))
    checks.append(("config:attack", *_check_path("configs/attack_config.yaml")))
    checks.append(("config:model", *_check_path("configs/model_config.yaml")))
    checks.append(("config:defense", *_check_path("configs/defense_config.yaml")))

    model_id = model_cfg.get("open_source_runtime", {}).get("model_id", "")
    if model_id:
        checks.append(("open_source_model_id", True, f"Configured model_id='{model_id}'"))
    else:
        checks.append(
            (
                "open_source_model_id",
                False,
                "No 'open_source_runtime.model_id' set in configs/model_config.yaml",
            )
        )

    for key in ["OPENAI_API_KEY", "GEMINI_API_KEY", "DEEPSEEK_API_KEY", "XAI_API_KEY"]:
        masked = "***set***" if os.getenv(key) else "<empty>"
        checks.append((f"env:{key}", True, f"value={masked} (optional for open-source mode)"))

    any_fail = False
    for name, ok, detail in checks:
        _print_check(name, ok, detail)
        any_fail = any_fail or (not ok)
    print()
    if any_fail:
        print("Environment check finished with failures. Please fix FAIL items first.")
        return 1
    print("Environment check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

