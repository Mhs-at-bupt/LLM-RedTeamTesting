# Restructure Status

## New Primary Entry

- Package code: `src/llm_redteam/`
- Configs: `configs/*.yaml`
- Scripts: `scripts/*.py`
- Tests: `tests/*.py`

## Legacy/Historical Artifacts (kept for traceability)

- Legacy evaluation scripts at repository root (e.g. `autodan_hga_eval*.py`, `check_asr.py`).
- Legacy utility implementations under `utils/` and `sematicDAN/utils/`.
- Historical result directories (nested) under `results/`, `sematicDAN/results/`, `goal_weak/`, `mutimutation/`.

## Canonical Workflow

1. Run experiments through `python -m llm_redteam.cli ...`.
2. Save new outputs under `results/runs/<timestamp>/`.
3. Consolidate legacy and current outputs with:
   - `python scripts/collect_results.py --project-root . --output-dir results/collected`

## Publication Guidance

- Keep only sanitized examples in public repo.
- Move sensitive/raw artifacts to `results/private/`.
- Avoid publishing full harmful prompt/response logs.

