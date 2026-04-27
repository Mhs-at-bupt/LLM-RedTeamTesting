# LLM-RedTeamTesting

An Enhanced AutoDAN Framework for Automated Red Team Testing Against Large Language Models.

## Ethical and Safety Notice

- This repository is for academic safety research only.
- No operational harmful prompts are included in the public repository.
- API keys should be stored in `.env` and never committed.
- Some datasets and private experimental logs are excluded from the repository.

## Repository Structure

Core package is in `src/llm_redteam/` with modules:
- `attacks/`
- `models/`
- `evaluation/`
- `defense/`
- `experiments/`
- `utils/`

Supporting files:
- `configs/`, `scripts/`, `tests/`, `docs/`, `results/`, `data/`.

Single core workflow entry:

```bash
python -m llm_redteam.cli <subcommand> ...
```

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Environment Check

```bash
python -m llm_redteam.cli check-env
```

If `open_source_model_id` is `FAIL`, set `configs/model_config.yaml -> open_source_runtime.model_id`
to a local/remote HuggingFace model ID you can load.

## Environment Variables

Copy and edit `.env.example`:

```bash
OPENAI_API_KEY=
GEMINI_API_KEY=
DEEPSEEK_API_KEY=
XAI_API_KEY=
HF_TOKEN=
```

## Run Attack Experiments

```bash
python scripts/run_attack.py
```

or:

```bash
python -m llm_redteam.cli attack --config configs/attack_config.yaml
```

## Run Ablation Experiments

```bash
python scripts/run_ablation.py
```

## Run Defense Evaluation

```bash
python scripts/run_defense.py
```

or:

```bash
python -m llm_redteam.cli defense --config configs/defense_config.yaml
```

## Build PEFT Training Data

```bash
python scripts/build_peft_data.py --input results/runs/latest/results.jsonl
```

## Reproduce Paper Tables

```bash
python scripts/export_results.py
```

## Consolidate Historical Results

```bash
python scripts/collect_results.py --project-root . --output-dir results/collected
```

or:

```bash
python -m llm_redteam.cli collect-results --project-root . --output-dir results/collected
```

## One-Core Restructure (Includes sematicDAN)

Dry-run:

```bash
python scripts/restructure_core.py --project-root .
```

Apply:

```bash
python -m llm_redteam.cli restructure-core --project-root . --apply
```

## Citation / Acknowledgement

Please cite this repository and related AutoDAN / defense papers when using this code.

## Contact / Status

- Status: active research codebase under engineering refactor.
- Contact: repository owner/maintainer.
