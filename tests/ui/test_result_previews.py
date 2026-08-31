"""Regression tests for the result-preview UI lifecycle."""

import sys
from pathlib import Path

from nicegui import Client, ui
from nicegui.page import page

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ui.components.result_previews import (
    PREVIEW_MAX_FILES,
    add_result_previews,
    clear_result_previews,
)
from ui.output_guide import preview_views


def _expansions(container):
    return [child for child in container.default_slot.children
            if isinstance(child, ui.expansion)]


def test_preview_views_come_from_the_registry():
    """Each tool's preview set is derived from TOOL_GUIDE_SPECS flags."""
    homolog_patterns = [v["pattern"] for v in preview_views("find_homologs")]
    assert homolog_patterns == [
        "results/bodyid_results.csv",
        "results/type_summary.csv",
        "results/type_level_results.csv",
        "results/morph_similarity.csv",
    ]
    assert all(p.endswith(".csv") for p in homolog_patterns)
    # Tools without tabular results register no previews.
    assert preview_views("flylight_download") == []
    assert preview_views("no_such_tool") == []


def test_result_previews_replace_previous_run_widgets():
    """A second render keeps one current preview set."""
    client = Client(page("/result-previews-replace-test"))
    with client:
        container = ui.column()

    views = preview_views("find_similar_morphology")
    assert views

    add_result_previews(None, container, "find_similar_morphology")
    assert len(_expansions(container)) == len(views)

    add_result_previews(None, container, "find_similar_morphology")
    assert len(_expansions(container)) == len(views)


def test_clear_result_previews_removes_previous_run_widgets():
    """Starting a new run can remove previews before it begins."""
    client = Client(page("/result-previews-clear-test"))
    with client:
        container = ui.column()

    add_result_previews(None, container, "find_similar_morphology")
    clear_result_previews(container)

    assert container.default_slot.children == []


def test_missing_files_render_placeholder(tmp_path):
    """Views whose files a run did not produce show a muted placeholder."""
    client = Client(page("/result-previews-missing-test"))
    with client:
        container = ui.column()

    add_result_previews(tmp_path, container, "find_homologs")

    expansions = _expansions(container)
    assert len(expansions) == len(preview_views("find_homologs"))
    for expansion in expansions:
        labels = [child for child in expansion.default_slot.children
                  if isinstance(child, ui.label)]
        assert any("Not available for this run." in label.text
                   for label in labels)


def test_registered_csv_is_rendered_as_table(tmp_path):
    """A produced result file is previewed as a top-N table."""
    results = tmp_path / "results"
    results.mkdir()
    (results / "bodyid_results.csv").write_text(
        "source_bodyId,target_type,jaccard\n1,Foo,0.5\n2,Bar,0.25\n",
        encoding="utf-8")

    client = Client(page("/result-previews-render-test"))
    with client:
        container = ui.column()

    add_result_previews(tmp_path, container, "find_homologs")

    first = _expansions(container)[0]
    tables = [child for child in first.default_slot.children
              if isinstance(child, ui.table)]
    assert len(tables) == 1
    assert tables[0].rows[0]["target_type"] == "Foo"
    # The other homolog views were not written for this run.
    for expansion in _expansions(container)[1:]:
        assert not [child for child in expansion.default_slot.children
                    if isinstance(child, ui.table)]


def test_glob_view_caps_rendered_files(tmp_path):
    """A pattern matching many files renders at most PREVIEW_MAX_FILES."""
    for dataset in ("MCNS", "FAFB", "HEMI", "MANC", "OLOB", "V626"):
        (tmp_path / f"{dataset}_types.csv").write_text(
            "type,max_score\nFoo,0.9\n", encoding="utf-8")

    client = Client(page("/result-previews-glob-test"))
    with client:
        container = ui.column()

    add_result_previews(tmp_path, container, "nb_find_neuron")

    glob_view = next(expansion for expansion in _expansions(container)
                     if "Type aggregates" in expansion.text)
    tables = [child for child in glob_view.default_slot.children
              if isinstance(child, ui.table)]
    assert len(tables) == PREVIEW_MAX_FILES
    labels = [child.text for child in glob_view.default_slot.children
              if isinstance(child, ui.label)]
    assert any("more matching files" in text for text in labels)
