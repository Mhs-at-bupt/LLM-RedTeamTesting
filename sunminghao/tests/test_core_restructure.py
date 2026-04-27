from pathlib import Path

from llm_redteam.experiments.core_restructure import restructure_to_core


def test_restructure_to_core_dry_run(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    (project / "sematicDAN" / "advbench").mkdir(parents=True, exist_ok=True)
    (project / "sematicDAN" / "advbench" / "harmful_behaviors.csv").write_text("goal,target\nx,y\n", encoding="utf-8")
    (project / "mutimutation").mkdir(parents=True, exist_ok=True)
    (project / "mutimutation" / "a.json").write_text('{"x":1}', encoding="utf-8")
    (project / "autodan_hga_eval.py").write_text("print('legacy')\n", encoding="utf-8")

    report = restructure_to_core(project_root=project, dry_run=True)
    assert report.dry_run is True
    assert (project / "sematicDAN").exists()
    assert (project / "autodan_hga_eval.py").exists()
    assert (project / "docs" / "core_restructure_report.json").exists()


def test_restructure_to_core_apply(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    (project / "sematicDAN" / "assets").mkdir(parents=True, exist_ok=True)
    (project / "sematicDAN" / "assets" / "autodan_initial_prompt.txt").write_text("seed", encoding="utf-8")
    (project / "goal_weak").mkdir(parents=True, exist_ok=True)
    (project / "goal_weak" / "a.json").write_text('{"x":1}', encoding="utf-8")
    (project / "check_asr.py").write_text("print('legacy')\n", encoding="utf-8")

    report = restructure_to_core(project_root=project, dry_run=False)
    assert report.dry_run is False
    assert not (project / "sematicDAN").exists()
    assert not (project / "check_asr.py").exists()
    assert (project / "legacy_archive" / "dirs").exists()
    assert (project / "legacy_archive" / "files").exists()

