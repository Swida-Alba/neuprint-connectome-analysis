"""Tests for the OutputPanel-hosted result previews."""

import sys
from pathlib import Path

from nicegui import Client, ui
from nicegui.page import page

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ui.components.output_panel import OutputPanel


def _build_panel(page_path: str):
    client = Client(page(page_path))
    with client:
        panel = OutputPanel("Test Output")
        panel.create()
    return client, panel


def test_previews_section_hidden_until_a_successful_run():
    client, panel = _build_panel("/output-panel-previews-idle")
    assert panel.previews_section.visible is False


def test_successful_run_reveals_previews(tmp_path):
    client, panel = _build_panel("/output-panel-previews-run")
    results = tmp_path / "results"
    results.mkdir()
    (results / "bodyid_results.csv").write_text("a,b\n1,2\n",
                                                encoding="utf-8")

    panel._show_result_previews(
        {"returncode": 0, "output_folder": str(tmp_path)}, "find_homologs")

    assert panel.previews_section.visible is True
    expansions = [child for child in panel.previews_container.default_slot.children
                  if isinstance(child, ui.expansion)]
    assert len(expansions) == 4


def test_failed_run_leaves_previews_hidden(tmp_path):
    client, panel = _build_panel("/output-panel-previews-failed")
    panel._show_result_previews(
        {"returncode": 1, "output_folder": str(tmp_path)}, "find_homologs")
    assert panel.previews_section.visible is False


def test_cancelled_run_leaves_previews_hidden(tmp_path):
    client, panel = _build_panel("/output-panel-previews-cancelled")
    panel._show_result_previews(
        {"returncode": 0, "cancelled": True,
         "output_folder": str(tmp_path)}, "find_homologs")
    assert panel.previews_section.visible is False


def test_tool_without_previews_stays_hidden(tmp_path):
    client, panel = _build_panel("/output-panel-previews-flylight")
    panel._show_result_previews(
        {"returncode": 0, "output_folder": str(tmp_path)},
        "flylight_download")
    assert panel.previews_section.visible is False


def test_clear_hides_previews_again(tmp_path):
    client, panel = _build_panel("/output-panel-previews-clear")
    panel._show_result_previews(
        {"returncode": 0, "output_folder": str(tmp_path)}, "find_homologs")
    assert panel.previews_section.visible is True

    panel.clear()

    assert panel.previews_section.visible is False
