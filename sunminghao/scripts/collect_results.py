from __future__ import annotations

import argparse
from pathlib import Path

from llm_redteam.experiments.result_collector import collect_useful_results


def main(project_root: str = ".", output_dir: str = "results/collected") -> None:
    records = collect_useful_results(project_root=project_root, output_dir=output_dir, copy_files=True)
    print(f"Collected {len(records)} files into {Path(project_root) / output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("collect_results")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output-dir", default="results/collected")
    args = parser.parse_args()
    main(args.project_root, args.output_dir)

