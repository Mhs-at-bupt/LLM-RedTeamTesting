# Result Management

## Goal

Unify scattered historical outputs into one canonical location:
- `results/collected/`

## What gets collected

- JSON/JSONL/CSV under `results/**`
- legacy result files from:
  - `sematicDAN/results/**`
  - `goal_weak/**`
  - `mutimutation/**`

## How to run

```bash
python scripts/collect_results.py --project-root . --output-dir results/collected
```

or:

```bash
python -m llm_redteam.cli collect-results --project-root . --output-dir results/collected
```

## Outputs

- `results/collected/raw/<category>/...`
- `results/collected/manifest.json`
- `results/collected/manifest.csv`
- `results/collected/summary.json`

## Notes

- Duplicate files are removed by SHA-256 hash.
- Temporary files like `~$*.json` are ignored.
- Private or sensitive artifacts should still be moved to `results/private/`.

