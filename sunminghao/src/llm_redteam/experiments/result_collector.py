from __future__ import annotations

import csv
import hashlib
import json
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class CollectedResultRecord:
    source_path: str
    target_path: str
    category: str
    size_bytes: int
    sha256: str


USEFUL_PATTERNS = [
    "**/results/**/*.json",
    "**/results/**/*.jsonl",
    "**/results/**/*.csv",
    "goal_weak/**/*.json",
    "mutimutation/**/*.json",
    "sematicDAN/results/**/*.json",
    "legacy_archive/**/*.json",
    "legacy_archive/**/*.jsonl",
    "legacy_archive/**/*.csv",
]

IGNORE_KEYWORDS = [
    "/results/private/",
    "\\results\\private\\",
    "~$",
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _categorize(path: Path) -> str:
    p = str(path).lower()
    if "autodan_hga_gemma" in p or "gemma" in p:
        return "autodan_hga_gemma"
    if "autodan_hga" in p:
        return "autodan_hga"
    if "autodan_ga" in p:
        return "autodan_ga"
    if "mutimutation" in p:
        return "mutimutation"
    if "semantic" in p or "semetic" in p:
        return "semantic"
    return "other"


def collect_useful_results(
    project_root: str | Path,
    output_dir: str | Path = "results/collected",
    copy_files: bool = True,
) -> list[CollectedResultRecord]:
    root = Path(project_root).resolve()
    out = (root / output_dir).resolve() if not Path(output_dir).is_absolute() else Path(output_dir).resolve()
    raw_dir = out / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    seen_hash: dict[str, Path] = {}
    records: list[CollectedResultRecord] = []

    all_files: list[Path] = []
    for pat in USEFUL_PATTERNS:
        all_files.extend(root.glob(pat))

    unique_files = sorted({p.resolve() for p in all_files if p.is_file()})
    for src in unique_files:
        src_str = str(src)
        if any(k in src_str for k in IGNORE_KEYWORDS):
            continue
        if src.name.startswith("~$"):
            continue
        cat = _categorize(src)
        digest = _sha256(src)
        if digest in seen_hash:
            continue
        seen_hash[digest] = src

        target = raw_dir / cat / src.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if copy_files:
            shutil.copy2(src, target)
        size = src.stat().st_size
        records.append(
            CollectedResultRecord(
                source_path=str(src),
                target_path=str(target),
                category=cat,
                size_bytes=size,
                sha256=digest,
            )
        )

    manifest_json = out / "manifest.json"
    manifest_csv = out / "manifest.csv"
    summary_json = out / "summary.json"
    out.mkdir(parents=True, exist_ok=True)

    manifest_json.write_text(
        json.dumps([asdict(r) for r in records], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with manifest_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["source_path", "target_path", "category", "size_bytes", "sha256"],
        )
        w.writeheader()
        w.writerows([asdict(r) for r in records])

    summary = {
        "total_files": len(records),
        "total_size_bytes": sum(r.size_bytes for r in records),
        "category_count": {},
    }
    for r in records:
        summary["category_count"][r.category] = summary["category_count"].get(r.category, 0) + 1
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    return records
