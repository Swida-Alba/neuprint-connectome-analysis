"""Compact previews of a run's primary result tables.

Every run tab shares the OutputPanel; after a successful run it renders a
top-N preview of each table registered for that tool in
``TOOL_GUIDE_SPECS`` (ui/output_guide.py, entries flagged ``preview``).
Find Homologs, for example, previews ``bodyid_results.csv``,
``type_summary.csv`` and ``type_level_results.csv``.

`add_result_previews` renders those tables under the output panel so the
primary results are always visible after a run.
"""
import fnmatch
from pathlib import Path

import pandas as pd
from nicegui import ui

from ..output_guide import preview_views

PREVIEW_ROWS = 12

# A registered pattern may match several files (e.g. per-dataset exports);
# cap how many are rendered inside one expander.
PREVIEW_MAX_FILES = 4


def clear_result_previews(container=None) -> None:
    """Remove the preview widgets for the previous run from *container*."""
    if container is not None:
        container.clear()


def _matching_files(output_folder: Path, pattern: str) -> list:
    """Sorted relative paths in the run folder matching *pattern*."""
    matches = []
    for path in output_folder.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(output_folder).as_posix()
        if fnmatch.fnmatch(rel, pattern):
            matches.append(rel)
    return sorted(matches)


def _render_csv_table(path: Path) -> None:
    """One top-N ui.table preview of *path* (bounded read)."""
    try:
        df = pd.read_csv(path, nrows=PREVIEW_ROWS)
    except Exception as e:  # noqa: BLE001 — a bad file must not break the run
        ui.label(f"Could not read {path.name}: {e}").classes(
            "text-caption drocat-muted")
        return
    if df.empty:
        ui.label("No rows.").classes("text-caption drocat-muted")
        return
    columns = [{"name": c, "label": c, "field": c, "align": "left"}
               for c in df.columns]
    rows = df.where(pd.notna(df), None).to_dict("records")
    ui.table(columns=columns, rows=rows, row_key=None) \
        .classes("w-full").props("dense flat binary-state-sorting")


def add_result_previews(output_folder, container=None, tool_name=None) -> list:
    """Replace the target with a top-N preview of each registered result view.

    *tool_name* selects the preview definitions via
    `ui.output_guide.preview_views`; every matched file is rendered inside
    its view's expander (capped at PREVIEW_MAX_FILES files per view).

    The target is intentionally replaced rather than appended to.  A tab
    keeps its NiceGUI component tree alive across runs, so appending here
    would retain every previous run's preview accordions.  Returns the
    rendered views (empty when the tool registers no previews).
    """
    views = preview_views(tool_name) if tool_name else []
    folder = Path(output_folder) if output_folder else None
    target = container
    if target is None:
        target = ui.column().classes("w-full")
    else:
        target.clear()
    if not views:
        return []
    with target:
        for view in views:
            fname = Path(view["pattern"]).name
            with ui.expansion(
                    f"{view['title']} — {fname}",
                    caption=view["description"]).classes(
                "w-full").props("header-class='text-caption'"):
                rels = (_matching_files(folder, view["pattern"])
                        if folder is not None else [])
                if not rels:
                    ui.label("Not available for this run.").classes(
                        "text-caption drocat-muted")
                    continue
                for rel in rels[:PREVIEW_MAX_FILES]:
                    ui.label(rel).classes("text-caption drocat-muted")
                    _render_csv_table(folder / rel)
                if len(rels) > PREVIEW_MAX_FILES:
                    ui.label(
                        f"{len(rels) - PREVIEW_MAX_FILES} more matching "
                        f"files — open the output folder to browse."
                    ).classes("text-caption drocat-muted")
                ui.label(f"Top {PREVIEW_ROWS} rows of the run's results — "
                         f"open the file for the full list.").classes(
                    "text-caption drocat-muted")
    return views
