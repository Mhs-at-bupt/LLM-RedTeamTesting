# Experiment Protocol

## Setup

- Configure model and attack settings in `configs/*.yaml`.
- Use fixed random seed for reproducibility.
- Keep private logs under `results/private/`.

## Models

- Open-source: `llama-2-7b-chat`, `gemma-2-9b`, `qwen3-8b-instruct`.
- Closed-source: `gpt-4.1`, `grok-4.1-fast-reason`, `deepseek-v3.1`.

## Metrics

- Attack success rate (ASR)
- Average semantic fitness
- Average query count
- Runtime statistics

## Result Formats

- Attack: `results.jsonl`, `summary.csv`, `config_used.yaml`, `logs.txt`.
- Defense: `defense_results.csv`, `defense_summary.md`.
- Export: LaTeX table files under `results/`.

