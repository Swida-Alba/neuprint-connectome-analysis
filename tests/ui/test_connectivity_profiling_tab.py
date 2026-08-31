"""Focused wiring checks for the connectivity Comparison sub-tab."""

from pathlib import Path

from ui.tabs import connectivity as connectivity_tab
from ui.runner import ScriptRunner


def test_comparison_output_dir_prefers_selected_subtab_value(monkeypatch):
    fallback_calls = []

    def fallback(scope):
        fallback_calls.append(scope)
        return "/tmp/inherited-profiling"

    monkeypatch.setattr(connectivity_tab, "get_tab_output_dir", fallback)

    assert (
        connectivity_tab._resolve_comparison_output_dir(" /tmp/selected ")
        == "/tmp/selected"
    )
    assert fallback_calls == []
    assert (
        connectivity_tab._resolve_comparison_output_dir("")
        == "/tmp/inherited-profiling"
    )
    assert (
        connectivity_tab._resolve_comparison_output_dir("   ")
        == "/tmp/inherited-profiling"
    )
    assert fallback_calls == ["connectivity_profiling", "connectivity_profiling"]


def test_runner_resolves_comparison_output_marker(tmp_path):
    """The comparison completion log must populate the Output Files panel."""
    run_folder = Path(tmp_path) / "profiling_MCNS_aMe_20260814_175350"
    run_folder.mkdir()
    (run_folder / "report.html").write_text("<html></html>", encoding="utf-8")

    runner = ScriptRunner()
    runner._run_logs = [
        ("stdout", f"[ConnectivityProfileComparer] Output: {run_folder}"),
    ]

    assert runner._extract_output_folder(str(tmp_path)) == str(run_folder)
    assert runner._resolve_scan_dir(str(tmp_path)) == str(run_folder)
