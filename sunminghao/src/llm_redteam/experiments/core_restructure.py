from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from llm_redteam.experiments.result_collector import collect_useful_results


LEGACY_DIRS = ["sematicDAN", "goal_weak", "mutimutation"]
LEGACY_FILES = [
    "1.py",
    "api.py",
    "autodan_hga_eval.py",
    "autodan_hga_eval_goal.py",
    "autodan_hga_eval_gemma.py",
    "autodan_hga_eval_mutimutation.py",
    "check_asr.py",
    "count_success.py",
    "get_responses.py",
    "test.py",
]


@dataclass
class RestructureReport:
    collected_files: int
    moved_dirs: list[str]
    moved_files: list[str]
    copied_data: list[str]
    dry_run: bool


def _safe_move(src: Path, dst: Path, dry_run: bool) -> None:
    if not src.exists():
        return
    if dry_run:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        # Avoid overwrite to keep traceability if user has previous archives.
        suffix = 1
        while True:
            candidate = dst.with_name(f"{dst.name}_{suffix}")
            if not candidate.exists():
                dst = candidate
                break
            suffix += 1
    shutil.move(str(src), str(dst))


def _safe_copy(src: Path, dst: Path, dry_run: bool) -> bool:
    if not src.exists():
        return False
    if dst.exists():
        return False
    if dry_run:
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def restructure_to_core(
    project_root: str | Path,
    archive_dir: str = "legacy_archive",
    output_dir: str = "results/collected",
    dry_run: bool = True,
) -> RestructureReport:
    root = Path(project_root).resolve()
    archive_root = root / archive_dir

    collected = collect_useful_results(root, output_dir=output_dir, copy_files=not dry_run)

    moved_dirs: list[str] = []
    for d in LEGACY_DIRS:
        src = root / d
        if src.exists():
            dst = archive_root / "dirs" / d
            _safe_move(src, dst, dry_run=dry_run)
            moved_dirs.append(str(src))

    moved_files: list[str] = []
    for f in LEGACY_FILES:
        src = root / f
        if src.exists():
            dst = archive_root / "files" / f
            _safe_move(src, dst, dry_run=dry_run)
            moved_files.append(str(src))

    copied_data: list[str] = []
    if _safe_copy(
        root / "sematicDAN" / "advbench" / "harmful_behaviors.csv",
        root / "data" / "advbench" / "harmful_behaviors.csv",
        dry_run=dry_run,
    ):
        copied_data.append("data/advbench/harmful_behaviors.csv")
    if _safe_copy(
        root / "sematicDAN" / "assets" / "autodan_initial_prompt.txt",
        root / "assets" / "autodan_initial_prompt.txt",
        dry_run=dry_run,
    ):
        copied_data.append("assets/autodan_initial_prompt.txt")

    report = RestructureReport(
        collected_files=len(collected),
        moved_dirs=moved_dirs,
        moved_files=moved_files,
        copied_data=copied_data,
        dry_run=dry_run,
    )

    report_path = root / "docs" / "core_restructure_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(asdict(report), ensure_ascii=False, indent=2), encoding="utf-8")
    return report

