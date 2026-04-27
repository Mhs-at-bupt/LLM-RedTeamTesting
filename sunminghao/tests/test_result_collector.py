from pathlib import Path

from llm_redteam.experiments.result_collector import collect_useful_results


def test_collect_useful_results_basic(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    (project / "results" / "autodan_ga").mkdir(parents=True, exist_ok=True)
    (project / "results" / "autodan_ga" / "a.json").write_text('{"k":1}', encoding="utf-8")
    (project / "results" / "mutimutation").mkdir(parents=True, exist_ok=True)
    (project / "results" / "mutimutation" / "~$tmp.json").write_text("x", encoding="utf-8")

    records = collect_useful_results(project_root=project, output_dir="results/collected", copy_files=True)
    assert len(records) == 1
    assert records[0].category == "autodan_ga"
    assert (project / "results" / "collected" / "manifest.json").exists()
    assert (project / "results" / "collected" / "summary.json").exists()

