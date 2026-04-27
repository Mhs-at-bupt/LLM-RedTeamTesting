# Consolidated Results

This folder is the single location for useful extracted historical outputs.

- `raw/`: copied raw result files grouped by category.
- `manifest.json` / `manifest.csv`: full mapping from original path to collected path.
- `summary.json`: category counts and total size.

Build/update this folder with:

```bash
python scripts/collect_results.py --project-root . --output-dir results/collected
```

