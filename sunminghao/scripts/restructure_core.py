from __future__ import annotations

import argparse

from llm_redteam.experiments.core_restructure import restructure_to_core


def main(project_root: str = ".", archive_dir: str = "legacy_archive", output_dir: str = "results/collected", apply: bool = False) -> None:
    report = restructure_to_core(
        project_root=project_root,
        archive_dir=archive_dir,
        output_dir=output_dir,
        dry_run=not apply,
    )
    mode = "APPLY" if apply else "DRY-RUN"
    print(
        f"[{mode}] collected={report.collected_files}, moved_dirs={len(report.moved_dirs)}, "
        f"moved_files={len(report.moved_files)}, copied_data={len(report.copied_data)}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("restructure_core")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--archive-dir", default="legacy_archive")
    parser.add_argument("--output-dir", default="results/collected")
    parser.add_argument("--apply", action="store_true", help="Apply changes. Without this flag, only dry-run report is generated.")
    args = parser.parse_args()
    main(args.project_root, args.archive_dir, args.output_dir, args.apply)

