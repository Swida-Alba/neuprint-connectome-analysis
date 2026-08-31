"""Focused wiring checks for the Morphology Comparison sub-tab."""

from pathlib import Path

from ui.runner import ScriptRunner


def test_runner_resolves_comparison_output_marker(tmp_path):
    """The morphology comparison completion log must populate the Output
    Files panel."""
    run_folder = (Path(tmp_path) /
                  "morphology_comparison_MCNS_aMe12_aMe10_20260901_120000")
    run_folder.mkdir()
    (run_folder / "report.html").write_text("<html></html>", encoding="utf-8")

    runner = ScriptRunner()
    runner._run_logs = [
        ("stdout",
         f"[MorphologyProfileComparer] Output: {run_folder}"),
    ]

    assert runner._extract_output_folder(str(tmp_path)) == str(run_folder)
    assert runner._resolve_scan_dir(str(tmp_path)) == str(run_folder)


def test_comparison_run_folder_prefix_matches_scan_regex():
    """The run-folder prefix must be whitelisted in the scan-dir fallback."""
    assert ScriptRunner._RUN_FOLDER_PREFIX_RE.match(
        "morphology_comparison_MCNS_aMe12_20260901_120000")
