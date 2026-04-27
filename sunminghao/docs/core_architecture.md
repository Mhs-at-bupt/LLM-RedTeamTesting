# Core Architecture (Single Entry)

This repository now uses one core architecture:

- Core package: `src/llm_redteam/`
- Unified run entry: `python -m llm_redteam.cli ...`
- Canonical data path: `data/`
- Canonical result path: `results/`

Legacy folders/files (for traceability) are archived into:

- `legacy_archive/dirs/`
- `legacy_archive/files/`

## Restructure command

Dry-run (recommended first):

```bash
python scripts/restructure_core.py --project-root .
```

Apply real changes:

```bash
python scripts/restructure_core.py --project-root . --apply
```

Equivalent CLI:

```bash
python -m llm_redteam.cli restructure-core --project-root . --apply
```

## Includes sematicDAN

`sematicDAN/` is treated as legacy implementation and will be archived under
`legacy_archive/dirs/sematicDAN`, while useful results are consolidated to
`results/collected/`.

