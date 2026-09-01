"""NiceGUI viewer for a locally cached neuron index."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Callable, List

from nicegui import ui

from ..config import PROJECT_ROOT
from ..neuron_index import (
    collect_zero_hit_matches,
    load_cached_neuron_index,
    neuron_index_path,
    query_match_group_subtypes,
    query_neuron_index,
)


# A one-character query can legitimately produce thousands of deduplicated
# names. Sending every match row to Quasar at once overwhelms the websocket
# even though the underlying server-side query is bounded and responsive.
# Keep the full membership maps for exact selection, but render a fixed page
# of match details; the panel pager still exposes every matched name.
MATCH_GROUP_PAGE_SIZE = 50
# One gesture can produce a value-click, a QTable selection event, and a
# delayed table update. Cover the maximum scroll-settle plus notification
# lifetime so that these events can never start a second focus animation.
FOCUS_DEDUP_SECONDS = 3.2

_ALIAS_MAPPER_PREWARM_STARTED = False


def _prewarm_alias_mapper() -> None:
    """Load the cross-dataset type mapper in a background thread, once.

    The alias panel uses it for zero-hit searches.  Loading it lazily at
    the first zero-hit search would freeze that refresh for seconds (the
    male-cns + FAFB + BANC tables are read and indexed), so the viewer
    starts the load as soon as an index is displayed.
    """
    global _ALIAS_MAPPER_PREWARM_STARTED
    if _ALIAS_MAPPER_PREWARM_STARTED:
        return
    _ALIAS_MAPPER_PREWARM_STARTED = True

    def _load() -> None:
        try:
            from comparison.cross_dataset_type_mapper import get_type_mapper

            get_type_mapper()
        except Exception:
            # A failed prewarm only delays the expansion to the first
            # zero-hit search, where collect_alias_matches retries.
            pass

    threading.Thread(
        target=_load, daemon=True, name="drocat-alias-mapper-prewarm"
    ).start()


def _normalized_focus_keys(keys) -> tuple[str, ...]:
    """Return stable, non-empty focus keys without changing their order."""
    result: list[str] = []
    seen: set[str] = set()
    for key in keys or ():
        value = str(key or "").strip()
        if value and value not in seen:
            result.append(value)
            seen.add(value)
    return tuple(result)


def _dataset_values(value) -> List[str]:
    """Normalize a single- or multi-dataset getter result."""
    if value is None:
        return []
    if isinstance(value, str):
        values = [value]
    else:
        try:
            values = list(value)
        except TypeError:
            values = [value]
    result = []
    for item in values:
        text = str(item or "").strip()
        if text and text not in result:
            result.append(text)
    return result


def _column_label(column: str) -> str:
    if column == "bodyId":
        return "Body ID"
    return column.replace("_", " ").strip().title()


def _relative_source(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _query_preview_values(getter: Callable[[], object] | None) -> List[str]:
    """Read the current query values for the viewer's compact preview."""
    if getter is None:
        return []
    try:
        value = getter()
    except Exception:
        return []
    if value is None:
        return []
    if isinstance(value, (str, int, float)):
        values = [value]
    else:
        try:
            values = list(value)
        except TypeError:
            values = [value]
    result = []
    seen = set()
    for item in values:
        text = str(item or "").strip()
        if text and text not in seen:
            result.append(text)
            seen.add(text)
    return result


def _render_missing_cache(content, dataset: str, path: Path) -> None:
    with content:
        ui.icon("database", color="orange").classes("text-4xl")
        ui.label("This dataset is not cached locally yet.").classes("text-subtitle1 font-bold")
        ui.label(
            f"The cached neuron index was not found at {_relative_source(path)}. "
            "The viewer does not open or stream the original dataset file."
        ).classes("text-body2")
        ui.label(
            "To make the index available, either run the selected analysis once "
            "(where the workflow supports first-run cache creation), or pull the "
            "dataset metadata and connections from the Settings tab."
        ).classes("text-body2")
        ui.label("Recommended UI flow").classes("font-bold text-primary mt-2")
        ui.label(
            "Open Settings → Dataset Cache, choose "
            f"{dataset}, then click Pull Dataset Metadata, followed by "
            "Pull Complete Connections for the connection cache. "
            "The connection pull is resumable."
        ).classes("text-body2")
        ui.link(
            "Open dataset-cache instructions",
            "docs/ui_guides/settings.html",
            new_tab=True,
        ).classes("drocat-doc-link")
        ui.label("Command-line alternative (from the project root)").classes(
            "font-bold text-primary mt-2"
        )
        ui.code(
            f"python src/build_connection_cache.py {dataset}",
            language="bash",
        ).classes("w-full")
        ui.label(
            "NeuPrint datasets need a configured token. FlyWire datasets must be "
            "prepared locally first; follow the matching preparation guide in Settings."
        ).classes("text-caption drocat-muted")


def _render_index(
    content,
    dataset: str,
    *,
    header_meta=None,
    query_values_getter: Callable[[], object] | None = None,
    query_selection: Callable[[List[str]], object] | None = None,
    query_resolution: Callable[[List[str]], object] | None = None,
    query_remove: Callable[[str], object] | None = None,
    query_edit: Callable[[str], object] | None = None,
    add_to_query: Callable[[List[str]], object] | None = None,
    query_label: str = "Current query",
    defer_apply: bool = False,
    apply_holder: dict | None = None,
) -> None:
    """Render the current dataset's index or its cache-missing state."""
    content.clear()
    if header_meta is not None:
        header_meta.clear()
    path = neuron_index_path(dataset)
    try:
        index = load_cached_neuron_index(dataset)
    except FileNotFoundError:
        _render_missing_cache(content, dataset, path)
        return
    except Exception as exc:
        with content:
            ui.icon("error", color="red").classes("text-4xl")
            ui.label("The cached neuron index could not be opened.").classes(
                "text-subtitle1 font-bold"
            )
            ui.label(str(exc)).classes("text-body2 drocat-err")
            ui.label(
                "Use Settings → Dataset Cache → Force rebuild, then open this viewer again."
            ).classes("text-caption drocat-muted")
        return

    columns = list(index.columns)
    if not columns:
        with content:
            ui.label("The cached neuron index is empty.").classes("text-body2 drocat-warn")
        return

    # The alias panel needs the type mapper; start its one-time load now so
    # a later zero-hit search does not stall on it.
    _prewarm_alias_mapper()

    if header_meta is not None:
        with header_meta:
            ui.badge(
                f"{index.frame.height:,} indexed rows", color="primary"
            ).props("outline")
            ui.label(f"Source: {_relative_source(index.path)}").classes(
                "text-caption drocat-muted drocat-neuron-source"
            )
            if index.enriched:
                ui.label(
                    "metadata-enriched"
                ).classes("text-caption drocat-muted drocat-neuron-enriched")

    with content:
        if query_values_getter is not None:
            with ui.element("div").classes("w-full drocat-neuron-intro-row"):
                with ui.element("section").classes("drocat-neuron-query-preview"):
                    query_preview_state = {"expanded": False}

                    def toggle_query_preview() -> None:
                        query_preview_state["expanded"] = not query_preview_state[
                            "expanded"
                        ]
                        if query_preview_state["expanded"]:
                            query_preview_scroll.classes(
                                add="drocat-neuron-query-preview-expanded",
                                remove="drocat-neuron-query-preview-collapsed",
                            )
                            query_preview_toggle.text = "Collapse"
                        else:
                            query_preview_scroll.classes(
                                add="drocat-neuron-query-preview-collapsed",
                                remove="drocat-neuron-query-preview-expanded",
                            )
                            query_preview_toggle.text = "Expand"
                        query_preview_toggle.update()

                    with ui.row().classes("w-full items-center justify-between gap-2"):
                        with ui.row().classes("items-center gap-2"):
                            ui.icon("playlist_add_check", color="primary").classes("text-lg")
                            ui.label(
                                f"{'Selected' if defer_apply else 'Current query'} · {query_label}"
                            ).classes(
                                "text-subtitle2 font-bold"
                            )
                        with ui.row().classes("items-center gap-1"):
                            query_preview_toggle = ui.button(
                                "Expand", on_click=toggle_query_preview
                            ).props("flat dense").classes(
                                "drocat-query-preview-expand-btn"
                            )
                            ui.badge(
                                "selected" if defer_apply else "mirrors input",
                                color="primary",
                            ).props("outline")
                    with ui.element("div").classes(
                        "w-full drocat-neuron-query-preview-list "
                        "drocat-neuron-query-preview-collapsed"
                    ) as query_preview_scroll:
                        query_preview = ui.row().classes(
                            "w-full items-center gap-1 flex-wrap"
                        )
                    query_preview_empty = ui.label(
                        "No values in the query yet. Select a match or body row to add one."
                    ).classes("text-caption drocat-muted mt-1")

                    def refresh_query_preview() -> None:
                        # Deferred mode shows the in-panel pending selection; the
                        # live mode mirrors the owning input's committed values.
                        values = (
                            selected_query_values()
                            if defer_apply
                            else _query_preview_values(query_values_getter)
                        )
                        query_preview.clear()
                        query_preview_empty.set_visibility(not values)
                        query_preview_toggle.set_visibility(bool(values))
                        with query_preview:
                            for value in values:
                                with ui.element("div").classes(
                                    "drocat-neuron-query-chip-wrap"
                                ) as query_chip:
                                    # Keep the value label's historic class so
                                    # the preview remains easy to inspect and
                                    # compatible with existing UI tests.
                                    ui.label(value).classes(
                                        "drocat-neuron-query-chip"
                                    )
                                    query_chip.on(
                                        "dblclick",
                                        lambda _event=None, v=value: edit_query_value(v),
                                    ).tooltip("Double-click to edit this query value")
                                    if query_remove is not None:
                                        ui.button(
                                            icon="close",
                                            on_click=lambda v=value: remove_query_value(v),
                                        ).props("flat round dense").classes(
                                            "drocat-neuron-query-chip-remove"
                                        ).tooltip("Remove from this query")

                    def edit_query_value(value: str) -> None:
                        """Remove the value from viewer selection, then edit it."""
                        remove_query_value(value)
                        if query_edit is not None:
                            query_edit(value)
                        refresh_query_preview()
                ui.label(
                    "Search returns all matches across bodyId, type, instance, and useful "
                    "type/taxonomy fields: strict case-sensitive prefixes come first, "
                    "followed by case-insensitive substring matches. Choose a target "
                    "column and match mode to apply that rule directly to this search "
                    "box; leave it unset for the global search. Numeric input is verified "
                    "against bodyId. Match details also keeps a secondary matched name "
                    "when the same row matches in another field. Select a matched name "
                    "to select every body sharing it, or select individual body rows to "
                    "add their body IDs."
                ).classes("text-caption drocat-muted drocat-neuron-search-help")
        else:
            ui.label(
                "Search returns all matches across bodyId, type, instance, and useful "
                "type/taxonomy fields: strict case-sensitive prefixes come first, "
                "followed by case-insensitive substring matches. Choose a target "
                "column and match mode to apply that rule directly to this search "
                "box; leave it unset for the global search. Numeric input is verified "
                "against bodyId."
            ).classes("text-caption drocat-muted drocat-neuron-search-help")

        with ui.row().classes(
            "w-full items-end gap-2 flex-wrap drocat-neuron-search-toolbar"
        ):
            search_input = ui.input(
                "Search identities & taxonomy",
                placeholder="e.g. aMe12 or 5813",
            ).props("outlined clearable input-debounce=180").classes(
                "flex-grow drocat-input drocat-neuron-search-field"
            )
            filter_options = {"__none__": "No column filter"}
            filter_options.update({column: _column_label(column) for column in columns})
            target_column = ui.select(
                options=filter_options,
                value="__none__",
                label="Target column",
            ).props("outlined").classes(
                "drocat-select drocat-neuron-search-field"
            ).style("min-width: 150px")
            filter_operator = ui.select(
                options={
                    "contains": "Contains",
                    "prefix": "Starts with",
                    "suffix": "Ends with",
                    "exact": "Exact",
                    "regex": "Regex",
                },
                value="contains",
                label="Match mode",
            ).props("outlined").classes(
                "drocat-select drocat-neuron-search-field"
            ).style("min-width: 140px")
            sort_options = {"__match_value__": "Matched value (default)"}
            sort_options.update({column: _column_label(column) for column in columns})
            sort_column = ui.select(
                options=sort_options,
                value="__match_value__",
                label="Sort by",
            ).props("outlined").classes(
                "drocat-select drocat-neuron-search-field"
            ).style("min-width: 150px")
            direction = ui.select(
                options={"asc": "Ascending", "desc": "Descending"},
                value="asc",
                label="Order",
            ).props("outlined").classes(
                "drocat-select drocat-neuron-search-field"
            ).style("min-width: 140px")
            page_size = ui.select(
                options={25: "25 / page", 50: "50 / page", 100: "100 / page", 200: "200 / page"},
                value=50,
                label="Rows",
            ).props("outlined").classes(
                "drocat-select drocat-neuron-search-field"
            ).style("min-width: 120px")

        filter_operator.set_enabled(False)

        initial = query_neuron_index(index, page_size=50)
        match_columns = [
            {
                "name": "match_column",
                "label": "Matched by",
                "field": "match_column",
                "align": "left",
                "classes": "drocat-neuron-match-by",
                "headerClasses": "drocat-neuron-match-by",
                "style": "width: 135px; min-width: 135px",
                "headerStyle": "width: 135px; min-width: 135px",
                "sortable": False,
            },
            {
                "name": "match_value",
                "label": "Matched value",
                "field": "match_value",
                "align": "left",
                "classes": "drocat-neuron-match-value",
                "headerClasses": "drocat-neuron-match-value",
                "style": "width: 190px; min-width: 190px",
                "headerStyle": "width: 190px; min-width: 190px",
                "sortable": False,
            },
            {
                "name": "body_count",
                "label": "Rows",
                "field": "body_count",
                "align": "right",
                "classes": "drocat-neuron-match-count",
                "headerClasses": "drocat-neuron-match-count",
                "style": "width: 64px; min-width: 64px",
                "headerStyle": "width: 64px; min-width: 64px",
                "sortable": False,
            },
        ]
        table_columns = [
            *[
                {
                    "name": column,
                    "label": _column_label(column),
                    "field": column,
                    # Full-index sorting is controlled above; enabling Quasar's
                    # client-side header sort here would sort only the current page.
                    "sortable": False,
                }
                for column in columns
            ],
        ]
        # The match panel is intentionally compact: one row per deduplicated
        # matched value. It is aligned with the metadata panel at the top, but
        # does not reserve blank rows for every repeated bodyId.
        selected_match_values: set[str] = set()
        selected_match_order: List[str] = []
        selected_match_members: dict[str, set[str]] = {}
        selected_match_body_ids: dict[str, tuple[str, ...]] = {}
        selected_body_ids: dict[str, str] = {}
        # Expanded subtype panels, keyed by the match-group value. Entries
        # survive query re-runs (paging, focus jumps) so a click on a member
        # body does not collapse the panel the user is working in. ``members``
        # keeps the payload from being reused when a later search changes the
        # membership of a same-named group.
        subtype_expansions: dict[str, dict] = {}
        # Lazily-computed complete key->bodyId map for the current query, used
        # only by the full metadata table's select-all across all pages.
        full_table_all_keys: dict[str, str] | None = None
        current_rows = list(initial.rows)
        match_groups_all = list(initial.match_groups)
        current_groups = match_groups_all[:MATCH_GROUP_PAGE_SIZE]
        match_state = {"page": 1}
        match_header_selection = {"all_visible": False}
        current_group_body_ids = {
            str(key): tuple(values)
            for key, values in initial.match_group_body_ids.items()
        }
        match_group_related = {
            str(key): tuple(values)
            for key, values in initial.match_group_related.items()
        }
        match_group_primary = {
            str(key): tuple(values)
            for key, values in initial.match_group_primary.items()
        }
        group_members = {
            str(key): set(values)
            for key, values in initial.match_group_members.items()
        }
        match_table = None
        table = None
        selection_status = None
        match_status = None
        match_page_position = None
        match_previous_button = None
        match_next_button = None
        query_callback = query_selection or add_to_query

        # Subtype expansion is only meaningful below a coarse taxonomy entry.
        # A type, instance, or bodyId match is already the leaf identity.
        subtype_leaf_columns = {"bodyId", "type", "instance"}

        def stamp_match_group_flags() -> None:
            """Attach expansion eligibility and persisted panel state."""
            can_expand = "type" in columns
            for group in match_groups_all:
                group["__can_expand"] = (
                    can_expand
                    and str(group.get("match_column_key", "") or "")
                    not in subtype_leaf_columns
                )
            for group in match_groups_all:
                key = str(group.get("__match_group_key", "") or "")
                entry = subtype_expansions.get(key)
                if entry is None:
                    continue
                if set(group_members.get(key, ())) != entry["members"]:
                    continue
                group["__subtypes"] = entry["display"]
                group["__expanded"] = entry["expanded"]
        stamp_match_group_flags()

        def effective_body_keys() -> set[str]:
            keys = set(selected_body_ids)
            for value in selected_match_values:
                keys.update(selected_match_members.get(value, set()))
                keys.update(group_members.get(value, set()))
            return keys

        def current_query_kwargs() -> dict:
            """Return the search/filter/sort kwargs shared by page and key queries."""
            requested_sort = sort_column.value
            if requested_sort == "__match_value__":
                # Ascending matched-value order is the implicit default. Preserve
                # an explicitly chosen descending direction for that same sort.
                requested_sort = (
                    "__match_value__" if direction.value == "desc" else None
                )
            return {
                "search": search_input.value or "",
                "search_column": target_column.value,
                "search_operator": filter_operator.value,
                "sort_by": requested_sort,
                "descending": direction.value == "desc",
            }

        def current_full_table_keys() -> dict[str, str]:
            """Lazily materialise every matching key->bodyId across all pages."""
            nonlocal full_table_all_keys
            if full_table_all_keys is None:
                full_table_all_keys = dict(
                    query_neuron_index(
                        index,
                        **current_query_kwargs(),
                        page=1,
                        page_size=50,
                        include_all_keys=True,
                    ).all_keys
                )
            return full_table_all_keys

        def related_match_values(value: str) -> List[str]:
            """Return one direct primary/secondary selection bundle.

            The backend intentionally does not return transitive connected
            components here.  Two independent primary names may share a
            taxonomy spelling on different rows; walking a graph would make
            selecting one primary unexpectedly select the other one too.
            """
            value = str(value or "").strip()
            if not value:
                return []
            return [
                linked
                for linked in match_group_related.get(value, (value,))
                if str(linked or "").strip()
            ]

        def match_member_keys(value: str) -> List[str]:
            """Return the exact table rows belonging to one clicked match."""
            keys: List[str] = []
            for linked_value in related_match_values(value):
                members = (
                    selected_match_members.get(linked_value, set())
                    or group_members.get(linked_value, ())
                )
                for member in members:
                    member = str(member or "").strip()
                    if member and member not in keys:
                        keys.append(member)
            return keys

        def remember_match(value: str, *, expand: bool = True) -> None:
            """Persist a selected name and its verified membership.

            Match rows are replaced whenever the search changes, so the
            selection cannot depend on the current page or current search
            result. The exact body-ID snapshot is retained for execution even
            when the selected name is no longer visible in the new search.
            """
            value = str(value or "").strip()
            if not value:
                return
            linked_values = related_match_values(value) if expand else [value]
            for linked_value in linked_values:
                if linked_value not in selected_match_values:
                    selected_match_values.add(linked_value)
                selected_match_members[linked_value] = set(
                    group_members.get(
                        linked_value,
                        selected_match_members.get(linked_value, set()),
                    )
                )
                selected_match_body_ids[linked_value] = tuple(
                    current_group_body_ids.get(
                        linked_value,
                        selected_match_body_ids.get(linked_value, ()),
                    )
                )
                for primary_value in match_group_primary.get(
                    linked_value, (linked_value,)
                ):
                    if (
                        primary_value
                        and primary_value not in selected_match_order
                    ):
                        selected_match_order.append(primary_value)

        def forget_match(value: str) -> None:
            value = str(value or "").strip()
            linked_values = set(related_match_values(value))
            if not linked_values:
                linked_values = {value}
            for linked_value in linked_values:
                selected_match_values.discard(linked_value)
                selected_match_members.pop(linked_value, None)
                selected_match_body_ids.pop(linked_value, None)
            selected_match_order[:] = [
                value for value in selected_match_order
                if value not in linked_values
            ]

        def remember_subtype(
            value: str,
            member_keys,
            body_ids,
        ) -> None:
            """Record one expanded subtype with its exact membership.

            Subtypes reuse the named-selection maps so query chips, body-ID
            resolution, chip removal, and cross-search persistence behave
            exactly like a match group picked from the panel.
            """
            value = str(value or "").strip()
            if not value:
                return
            selected_match_values.add(value)
            selected_match_members[value] = {
                str(key) for key in member_keys or () if str(key or "").strip()
            }
            selected_match_body_ids[value] = tuple(
                str(body_id) for body_id in body_ids or () if str(body_id or "").strip()
            )
            if value not in selected_match_order:
                selected_match_order.append(value)

        def row_body_id(row) -> str:
            """Return a verified, query-safe body ID for an individual row."""
            value = str(row.get("bodyId", "") or "").strip()
            if not value:
                return ""
            integer, dot, fraction = value.partition(".")
            if dot and integer.isdigit() and fraction and set(fraction) == {"0"}:
                return integer
            return value

        def selected_query_values() -> List[str]:
            values: List[str] = []
            seen: set[str] = set()
            # Keep the user's selection order even when a later search no
            # longer displays an earlier selected match group.
            for value in selected_match_order:
                if value in selected_match_values and value not in seen:
                    values.append(value)
                    seen.add(value)
            for row in current_rows:
                key = str(row.get("__neuron_key", "") or "")
                value = str(selected_body_ids.get(key, "") or "").strip()
                if value and value not in seen:
                    values.append(value)
                    seen.add(value)
            for value in selected_body_ids.values():
                value = str(value or "").strip()
                if value and value not in seen:
                    values.append(value)
                    seen.add(value)
            return values

        def selected_query_body_ids() -> List[str]:
            """Resolve selections to exact body IDs for query execution.

            The compact match panel displays human-readable names, but a name
            can occur in multiple metadata columns. Passing the resolved IDs
            separately prevents the eventual query from re-running a priority
            string lookup and selecting a different column by accident.
            """
            values: List[str] = []
            seen: set[str] = set()
            for value in selected_match_order:
                if value not in selected_match_values:
                    continue
                for body_id in selected_match_body_ids.get(value, ()):
                    body_id = str(body_id or "").strip()
                    if body_id and body_id not in seen:
                        values.append(body_id)
                        seen.add(body_id)
            for row in current_rows:
                key = str(row.get("__neuron_key", "") or "")
                if key not in selected_body_ids:
                    continue
                body_id = row_body_id(row)
                if body_id and body_id not in seen:
                    values.append(body_id)
                    seen.add(body_id)
            for body_id in selected_body_ids.values():
                body_id = str(body_id or "").strip()
                if body_id and body_id not in seen:
                    values.append(body_id)
                    seen.add(body_id)
            return values

        # Deferred apply (layer editor): hold the selection in the panel, then
        # commit it once when the dialog closes. Live mode pushes on every toggle.
        if defer_apply:
            def apply_pending_selection() -> None:
                if query_selection is not None:
                    query_selection(selected_query_values())
                if query_resolution is not None:
                    query_resolution(selected_query_body_ids())
            if apply_holder is not None:
                apply_holder["fn"] = apply_pending_selection
        if query_values_getter is not None:
            refresh_query_preview()

        def sync_query_selection() -> None:
            if not defer_apply:
                if query_callback is not None:
                    query_callback(selected_query_values())
                if query_resolution is not None:
                    query_resolution(selected_query_body_ids())
            if query_values_getter is not None:
                refresh_query_preview()

        def update_selection_status() -> None:
            if selection_status is not None:
                selected_count = len(selected_match_values) + len(selected_body_ids)
                selection_status.text = f"{selected_count} selected"
                selection_status.update()

        def refresh_table_selection() -> None:
            # Expansion checkboxes render from row data, so their checked
            # flags must be re-stamped before the table pushes an update.
            for group in current_groups:
                display = group.get("__subtypes")
                if not display:
                    continue
                for subtype in display.get("subtypes", ()):
                    subtype["selected"] = (
                        str(subtype.get("match_value", "") or "").strip()
                        in selected_match_values
                    )
            if match_table is not None:
                visible_match_rows = [
                    row for row in current_groups
                    if row.get("match_role") != "secondary"
                    and str(row.get("match_value", "") or "") in selected_match_values
                ]
                if match_header_selection["all_visible"]:
                    selectable_match_values = {
                        str(row.get("match_value", "") or "").strip()
                        for row in current_groups
                        if row.get("match_role") != "secondary"
                    }
                    if (
                        selectable_match_values
                        and selectable_match_values.issubset(selected_match_values)
                    ):
                        # Secondary rows are display-only and have no row
                        # checkbox. Keep them in QTable's internal selection
                        # only after a header select-all so Quasar can show
                        # the header as fully checked.
                        visible_match_rows.extend(
                            row for row in current_groups
                            if row.get("match_role") == "secondary"
                        )
                match_table.selected = visible_match_rows
                match_table.update()
            if table is not None:
                active = effective_body_keys()
                table.selected = [
                    row for row in current_rows
                    if str(row.get("__neuron_key", "") or "") in active
                ]
                table.update()
            update_selection_status()

        def handle_match_selection(event) -> None:
            visible_row_keys = {
                str(row.get("__match_group_key", "") or "")
                for row in current_groups
            }
            selected_row_keys = {
                str(row.get("__match_group_key", "") or "")
                for row in list(getattr(event, "selection", []) or [])
            }
            match_header_selection["all_visible"] = bool(
                visible_row_keys and visible_row_keys.issubset(selected_row_keys)
            )
            previously_selected = set(selected_match_values)
            visible_rows = [
                row for row in current_groups
                if row.get("match_role") != "secondary"
            ]
            visible_values = {
                str(row.get("match_value", "") or "").strip()
                for row in visible_rows
            }
            selected_rows = [
                row for row in list(getattr(event, "selection", []) or [])
                if row.get("match_role") != "secondary"
            ]
            raw_selected_values = {
                str(row.get("match_value", "") or "").strip()
                for row in selected_rows
            }
            selected_values: set[str] = set()
            for value in raw_selected_values:
                selected_values.update(related_match_values(value))
            # The table emits only the current result rows. Update those rows
            # while preserving selections made in an earlier search.
            for value in visible_values - selected_values:
                forget_match(value)
            for row in selected_rows:
                remember_match(str(row.get("match_value", "") or "").strip())
            # A matched-name selection includes every body in that group,
            # including rows on later pages. The full table mirrors the
            # current page of that selection when it is visible.
            sync_query_selection()
            # Selection and clicking a matched value use the same focus path:
            # resolve the first exact member, compute its sorted data page,
            # then scroll the actual metadata row into view.
            # QTable reports the complete selected set, not the row that was
            # just clicked. Focus the newly selected row's members so adding
            # aMe26 after aMe1/aMe13 does not jump back to the first group.
            new_values = [
                value for value in raw_selected_values
                if value not in previously_selected
            ]
            focus_value = new_values[-1] if new_values else (
                next(iter(raw_selected_values), "")
            )
            focus_keys = match_member_keys(focus_value)
            if selected_rows and focus_keys:
                request_focus(focus_keys, anchor_key=focus_keys[0])
            else:
                refresh_table_selection()

        def handle_match_toggle(event) -> None:
            """Apply one checkbox toggle without trusting stale QTable state.

            The match table is re-rendered after every query and selection
            change.  Binding the slot checkbox directly with ``v-model`` can
            therefore make a second click report the previous selection back
            to the server.  The slot emits the requested next state instead;
            this handler updates the persistent selection sets first and then
            refreshes the table from those sets.
            """
            args = getattr(event, "args", None)
            if not isinstance(args, dict):
                return
            row = args.get("row")
            if not isinstance(row, dict):
                return
            if row.get("match_role") == "secondary":
                return
            match_header_selection["all_visible"] = False
            value = str(row.get("match_value", "") or "").strip()
            if not value:
                return
            if bool(args.get("selected")):
                remember_match(value)
            else:
                forget_match(value)
            sync_query_selection()
            if bool(args.get("selected")):
                focus_keys = match_member_keys(value)
                if focus_keys:
                    request_focus(focus_keys, anchor_key=focus_keys[0])
                    return
            refresh_table_selection()

        def build_subtype_entry(group) -> dict:
            """Compute and cache the subtype panel for one match group.

            The server-side entry keeps the full selection payload (member
            keys and body IDs); only the compact display copy is attached to
            the row so a 500-subtype expansion never ships its whole
            membership over the websocket.
            """
            key = str(group.get("__match_group_key", "") or "")
            members = tuple(sorted(group_members.get(key, ())))
            payload = query_match_group_subtypes(index, members)
            display = {
                "subtypes": [
                    {
                        "match_value": subtype["match_value"],
                        "body_count": subtype["body_count"],
                        "selected": subtype["match_value"] in selected_match_values,
                    }
                    for subtype in payload["subtypes"]
                ],
                "total_types": payload["total_types"],
                "truncated": payload["truncated"],
            }
            entry = {
                "members": set(members),
                "payload": payload,
                "display": display,
                "expanded": False,
            }
            subtype_expansions[key] = entry
            return entry

        def handle_match_expand_toggle(event) -> None:
            key = str(getattr(event, "args", "") or "").strip()
            if not key:
                return
            group = next(
                (
                    candidate
                    for candidate in match_groups_all
                    if str(candidate.get("__match_group_key", "") or "") == key
                ),
                None,
            )
            if group is None or not group.get("__can_expand"):
                return
            entry = subtype_expansions.get(key)
            if entry is None or set(group_members.get(key, ())) != entry["members"]:
                entry = build_subtype_entry(group)
            entry["expanded"] = not entry["expanded"]
            group["__subtypes"] = entry["display"]
            group["__expanded"] = entry["expanded"]
            match_table.update_rows(current_groups)

        def handle_subtype_toggle(event) -> None:
            """Apply one expanded-subtype checkbox toggle."""
            args = getattr(event, "args", None)
            if not isinstance(args, dict):
                return
            group_key = str(args.get("group", "") or "").strip()
            value = str(args.get("value", "") or "").strip()
            if not value:
                return
            if bool(args.get("selected")):
                subtype = None
                entry = subtype_expansions.get(group_key)
                if entry is not None:
                    subtype = next(
                        (
                            candidate
                            for candidate in entry["payload"]["subtypes"]
                            if str(candidate.get("match_value", "")) == value
                        ),
                        None,
                    )
                if subtype is not None:
                    remember_subtype(
                        value,
                        subtype.get("member_keys", ()),
                        subtype.get("body_ids", ()),
                    )
                else:
                    remember_match(value)
            else:
                forget_match(value)
            sync_query_selection()
            refresh_table_selection()
            match_table.update_rows(current_groups)

        def handle_body_selection(event) -> None:
            visible = {
                str(row.get("__neuron_key", "") or ""): row
                for row in current_rows
            }
            selected_keys = {
                str(row.get("__neuron_key", "") or "")
                for row in list(getattr(event, "selection", []) or [])
            }
            # Unchecking one body row breaks a whole-name selection. The
            # remaining checked rows become individual selections.
            for value in list(selected_match_values):
                visible_group = (
                    selected_match_members.get(value, set())
                    | group_members.get(value, set())
                ) & set(visible)
                if visible_group and not visible_group.issubset(selected_keys):
                    forget_match(value)

            active_group_keys = set()
            for value in selected_match_values:
                active_group_keys.update(group_members.get(value, set()))
            for key, row in visible.items():
                if key not in selected_keys:
                    selected_body_ids.pop(key, None)
                elif key not in active_group_keys:
                    value = row_body_id(row)
                    if value:
                        selected_body_ids[key] = value
            sync_query_selection()
            refresh_table_selection()

        def remove_query_value(value: str) -> None:
            """Remove one value from the mirrored/Selected preview and selection state."""
            value = str(value or "").strip()
            if not value:
                return
            forget_match(value)
            for key, body_id in list(selected_body_ids.items()):
                if str(body_id or "").strip() == value:
                    selected_body_ids.pop(key, None)
            sync_query_selection()
            # In live mode the 'x' also removes from the owning input; in
            # deferred mode it only prunes the pending in-panel selection.
            if not defer_apply and query_remove is not None:
                query_remove(value)
            if query_values_getter is not None:
                refresh_query_preview()
            refresh_table_selection()

        def handle_full_table_select_all(_event=None) -> None:
            """Select or deselect the current query's rows across all pages.

            The header checkbox only reports a click; the server decides the
            direction by checking whether the whole result set is already
            selected, so a partially-selected page always resolves to
            "select-all" and a fully-selected result toggles back off.
            Because selections persist across searches, deselecting only
            removes the rows currently matching the query rather than wiping
            selections made against an earlier query.
            """
            keys = current_full_table_keys()
            if not keys:
                return
            if set(keys).issubset(effective_body_keys()):
                for key in keys:
                    selected_body_ids.pop(key, None)
            else:
                selected_body_ids.update(keys)
            sync_query_selection()
            refresh_table_selection()

        with ui.element("div").classes("w-full drocat-neuron-results-layout"):
            with ui.element("section").classes("drocat-neuron-match-panel"):
                with ui.row().classes("w-full items-center justify-between gap-2"):
                    with ui.row().classes("items-center gap-2"):
                        ui.icon("manage_search", color="primary").classes("text-lg")
                        ui.label("Match details").classes("text-subtitle2 font-bold")
                    with ui.row().classes("items-center gap-2"):
                        if query_callback is not None:
                            selection_status = ui.label("0 selected").classes(
                                "text-caption drocat-muted"
                            )
                with ui.row().classes(
                    "w-full items-center justify-between gap-2 flex-wrap drocat-neuron-panel-toolbar"
                ):
                    match_status = ui.label("No matched names").classes(
                        "text-caption drocat-muted flex-grow"
                    )
                    match_page_position = ui.label("Page 1 of 1").classes(
                        "text-caption drocat-muted"
                    )
                    with ui.row().classes("items-center gap-1"):
                        match_previous_button = ui.button(
                            "Previous matches", icon="chevron_left"
                        ).props("flat dense")
                        match_next_button = ui.button(
                            "Next matches", icon="chevron_right"
                        ).props("flat dense")
                match_table = ui.table(
                    rows=current_groups,
                    columns=match_columns,
                    row_key="__match_group_key",
                    selection="multiple",
                    on_select=handle_match_selection,
                    pagination=None,
                ).classes("w-full drocat-neuron-match-table")
                # The custom body slot adds the selection cell explicitly.
                # Render the matching header cell explicitly as well so it
                # uses the same width and alignment as the row checkboxes.
                match_table.add_slot(
                    "header",
                    r"""
                    <q-tr :props="props">
                      <q-th auto-width class="drocat-neuron-match-select-cell">
                        <q-checkbox
                          v-model="props.selected"
                          :indeterminate="props.selected === null"
                          dense
                        />
                      </q-th>
                      <q-th
                        v-for="col in props.cols"
                        :key="col.name"
                        :props="props"
                        :class="col.headerClasses"
                        :style="col.headerStyle"
                      >
                        {{ col.label }}
                      </q-th>
                    </q-tr>
                    """,
                )
                match_table.add_slot(
                    "body",
                    r"""
                    <q-tr
                      :props="props"
                      :class="{
                        'drocat-neuron-match-secondary-row': props.row.match_role === 'secondary',
                      }"
                    >
                      <q-td auto-width class="drocat-neuron-match-select-cell">
                      <q-checkbox
                        v-if="props.row.match_role !== 'secondary'"
                        :model-value="props.selected"
                        dense
                        @click.stop="$parent.$emit('match-selection-toggle', { row: props.row, selected: !props.selected })"
                      />
                      </q-td>
                      <q-td key="match_column" :props="props" class="drocat-neuron-match-by">
                        <div class="drocat-neuron-match-source">
                          <q-btn
                            v-if="props.row.__can_expand"
                            flat dense round size="xs"
                            class="drocat-neuron-match-expand-btn"
                            :icon="props.row.__expanded ? 'expand_less' : 'expand_more'"
                            @click.stop="$parent.$emit('match-expand-toggle', props.row.__match_group_key)"
                          />
                          <q-icon
                            v-if="props.row.match_role === 'secondary'"
                            name="arrow_right_alt"
                            class="drocat-neuron-match-secondary-arrow"
                            size="18px"
                          />
                          {{ props.row.match_column }}
                        </div>
                      </q-td>
                      <q-td key="match_value" :props="props">
                        <div class="drocat-neuron-match-value-line">
                          <q-btn
                            flat dense no-caps
                            class="drocat-neuron-match-jump"
                            :label="props.row.match_value"
                            @click.stop="$parent.$emit('match-value-click', props.row)"
                          />
                        </div>
                        <div
                          v-if="props.row.first_body_id"
                          class="drocat-neuron-match-first"
                        >first bodyId: {{ props.row.first_body_id }}</div>
                      </q-td>
                      <q-td key="body_count" :props="props" class="text-right">
                        {{ props.row.body_count }}
                      </q-td>
                    </q-tr>
                    <q-tr
                      v-if="props.row.__expanded && props.row.__subtypes"
                      class="drocat-neuron-match-subtype-panel-row"
                    >
                      <q-td colspan="4" class="drocat-neuron-match-subtype-cell">
                        <div class="drocat-neuron-match-subtype-head">
                          {{ props.row.__subtypes.total_types }}
                          {{ props.row.__subtypes.total_types === 1 ? 'type' : 'types' }}
                          in {{ props.row.match_value }}
                        </div>
                        <div class="drocat-neuron-match-subtype-list">
                          <div
                            v-for="subtype in props.row.__subtypes.subtypes"
                            :key="subtype.match_value"
                            class="drocat-neuron-match-subtype-item"
                          >
                            <q-checkbox
                              :model-value="!!subtype.selected"
                              dense
                              @click.stop="$parent.$emit('match-subtype-toggle', { group: props.row.__match_group_key, value: subtype.match_value, selected: !subtype.selected })"
                            />
                            <span class="drocat-neuron-match-subtype-name">{{ subtype.match_value }}</span>
                            <span class="drocat-neuron-match-subtype-count">{{ subtype.body_count }}</span>
                          </div>
                          <div
                            v-if="!props.row.__subtypes.subtypes.length"
                            class="drocat-neuron-match-subtype-note"
                          >
                            No type values in this entry.
                          </div>
                          <div
                            v-if="props.row.__subtypes.truncated"
                            class="drocat-neuron-match-subtype-note"
                          >
                            Showing the first {{ props.row.__subtypes.subtypes.length }} of
                            {{ props.row.__subtypes.total_types }} types. Refine by searching the type name.
                          </div>
                        </div>
                      </q-td>
                    </q-tr>
                    """,
                )
            with ui.element("section").classes("drocat-neuron-full-panel"):
                with ui.row().classes("w-full items-center gap-2"):
                    ui.icon("table_view", color="primary").classes("text-lg")
                    ui.label("Full neuron metadata").classes("text-subtitle2 font-bold")
                ui.label(
                    "Scroll horizontally to inspect every retained metadata field."
                ).classes("text-caption drocat-muted")
                with ui.row().classes(
                    "w-full items-center justify-between gap-2 flex-wrap drocat-neuron-panel-toolbar"
                ):
                    page_info = ui.label("").classes(
                        "text-caption drocat-muted flex-grow"
                    )
                    no_results = ui.label(
                        "No rows match the current search/filter."
                    ).classes("text-caption drocat-warn")
                    page_position = ui.label("").classes(
                        "text-caption drocat-muted"
                    )
                    with ui.row().classes("items-center gap-1"):
                        previous_button = ui.button(
                            "Previous page", icon="chevron_left"
                        ).props("flat dense")
                        next_button = ui.button(
                            "Next page", icon="chevron_right"
                        ).props("flat dense")
                with ui.element("div").classes("w-full drocat-data-viewer-scroll"):
                    table = ui.table(
                        rows=initial.rows,
                        columns=table_columns,
                        row_key="__neuron_key",
                        selection="multiple",
                        on_select=handle_body_selection,
                        pagination=None,
                    ).classes("w-full drocat-data-viewer-table")
                    # Keep the table's default selection checkbox at the left
                    # while using a body slot for the row-specific highlight.
                    table.add_slot(
                        "body",
                        r"""
                        <q-tr
                          :data-neuron-key="props.row.__neuron_key"
                          :class="{
                            'drocat-neuron-selected-row': props.selected,
                          }"
                        >
                          <q-td auto-width>
                            <q-checkbox v-model="props.selected" dense />
                          </q-td>
                          <q-td
                            v-for="col in props.cols"
                            :key="col.name"
                            :props="props"
                            :class="{
                              'drocat-neuron-hit-cell': (
                                props.row.match_column_keys || [props.row.match_column_key]
                              ).includes(col.name),
                              'drocat-neuron-secondary-hit-cell': (
                                props.row.secondary_match_column_keys || []
                              ).includes(col.name),
                            }"
                            :data-match-column="(
                              props.row.match_column_keys || [props.row.match_column_key]
                            ).includes(col.name) ? col.name : null"
                            :data-match-role="(
                              props.row.secondary_match_column_keys || []
                            ).includes(col.name) ? 'secondary' : null"
                          >
                            <span
                              v-if="props.row.__highlighted_cells && props.row.__highlighted_cells[col.name]"
                              v-html="props.row.__highlighted_cells[col.name]"
                            ></span>
                            <span v-else>{{ props.row[col.field] }}</span>
                          </q-td>
                        </q-tr>
                        """,
                    )
                    # The header select-all mirrors the row checkboxes but
                    # selects every matching row across all pages, not just the
                    # current page. The checkbox is view-only (``:model-value``)
                    # so the server can drive the selection and refresh the whole
                    # table from the persistent key set.
                    table.add_slot(
                        "header",
                        r"""
                        <q-tr :props="props">
                          <q-th auto-width class="drocat-neuron-select-cell">
                            <q-checkbox
                              :model-value="props.selected"
                              :indeterminate="props.selected === null"
                              dense
                              @click.stop="$parent.$emit('full-table-select-all')"
                            />
                          </q-th>
                          <q-th
                            v-for="col in props.cols"
                            :key="col.name"
                            :props="props"
                          >
                            {{ col.label }}
                          </q-th>
                        </q-tr>
                        """,
                    )

        state = {"page": initial.page, "page_size": 50}

        # Expanded cross-dataset search panel: shown only when the local
        # query returns zero rows. Content is strictly informational —
        # other-dataset rows are never merged into this table or selection.
        with ui.element("section").classes(
            "w-full drocat-neuron-alias-panel"
        ) as alias_section:
            alias_container = ui.element("div").classes("w-full")
        alias_section.set_visibility(False)

        def _search_local_alias(name: str) -> None:
            """Refill the local search with an alias found in this dataset."""
            if str(search_input.value or "").strip() == name:
                refresh(reset_page=True)
            else:
                search_input.set_value(name)

        def render_alias_matches() -> None:
            try:
                matches = collect_zero_hit_matches(
                    dataset, str(search_input.value or "").strip()
                )
            except Exception:
                alias_section.set_visibility(False)
                return
            native = matches.get("native", [])
            mapped = matches.get("mapped", [])
            native_useful = any(
                entry.get("types") or entry.get("labels") for entry in native
            )
            mapped_useful = any(
                entry["outcome"] == "matched" and entry["candidates"]
                for entry in mapped
            )
            useful = native_useful or mapped_useful
            alias_section.set_visibility(useful)
            if not useful:
                alias_container.clear()
                return
            alias_container.clear()
            with alias_container:
                ui.separator()
                with ui.row().classes("w-full items-center gap-2 flex-wrap"):
                    ui.icon("travel_explore", color="warning").classes("text-lg")
                    ui.label(
                        "No rows here. Cross-dataset matches — informational "
                        "only, please double check."
                    ).classes("text-subtitle2 font-bold")

                def _annotation_text(ann) -> str:
                    if not ann:
                        return "— no counterpart in this dataset"
                    if ann["kind"] == "one of N":
                        return "— here: one of " + ", ".join(ann["targets"])
                    if ann["kind"] == "renamed":
                        return "— here: maps to '" + ann["targets"][0] + "'"
                    if ann["kind"] == "same name":
                        return "— same name in this dataset"
                    return f"— here: {ann['kind']} {', '.join(ann['targets'])}"

                if native_useful:
                    with ui.row().classes("w-full items-center gap-2 flex-wrap"):
                        ui.label(
                            "Type-name matches in other datasets "
                            "(name-similar — not necessarily the same type):"
                        ).classes("text-caption font-bold drocat-muted")
                    for entry in native:
                        if not (entry.get("types") or entry.get("labels")):
                            continue
                        with ui.row().classes(
                            "w-full items-start gap-2 flex-wrap "
                            "drocat-neuron-alias-row"
                        ):
                            ui.badge(entry["dataset"]).props("outline")
                            with ui.element("div").classes("flex-grow"):
                                for cand in entry.get("types", []):
                                    text = (
                                        f"'{cand['name']}' "
                                        f"({cand['count']:,} neurons) "
                                        + _annotation_text(cand.get("mapped"))
                                    )
                                    ui.label(text).classes("text-caption")
                                if entry.get("types_truncated"):
                                    ui.label(
                                        f"+{entry['types_truncated']} more types"
                                    ).classes("text-caption drocat-muted")
                                for label in entry.get("labels", []):
                                    ui.label(
                                        f"label '{label['label']}' · "
                                        f"{label['column']} "
                                        f"({label['count']:,} neurons)"
                                    ).classes("text-caption")
                                    covered = [
                                        f"'{t['name']}' ({t['count']:,}) "
                                        + _annotation_text(t.get("mapped"))
                                        for t in label.get("types", [])
                                    ]
                                    if covered:
                                        ui.label(
                                            "    types under this label: "
                                            + "; ".join(covered)
                                        ).classes("text-caption drocat-muted")
                                    if label.get("types_truncated"):
                                        ui.label(
                                            f"    +{label['types_truncated']} "
                                            "more types under this label"
                                        ).classes("text-caption drocat-muted")
                                if entry.get("labels_truncated"):
                                    ui.label(
                                        f"+{entry['labels_truncated']} more labels"
                                    ).classes("text-caption drocat-muted")

                if mapped_useful:
                    with ui.row().classes("w-full items-center gap-2 flex-wrap"):
                        ui.label(
                            "Auto type mapping:"
                        ).classes("text-caption font-bold drocat-muted")
                    for entry in mapped:
                        if entry["outcome"] != "matched" or not entry["candidates"]:
                            continue
                        with ui.row().classes(
                            "w-full items-start gap-2 flex-wrap "
                            "drocat-neuron-alias-row"
                        ):
                            ui.badge(
                                entry["dataset"]
                                + (" (this dataset)" if entry["is_selected"] else "")
                            ).props("outline")
                            for cand in entry["candidates"]:
                                text = f"'{cand['name']}' — {cand['kind']}"
                                if cand.get("aggregates"):
                                    text += (
                                        "; a match also covers: "
                                        + ", ".join(cand["aggregates"])
                                    )
                                if cand.get("count") is not None:
                                    text += f" ({cand['count']:,} neurons)"
                                ui.label(text).classes("text-caption")
                                if entry["is_selected"]:
                                    ui.button(
                                        f"Search '{cand['name']}' here",
                                        icon="search",
                                    ).props("flat dense").on_click(
                                        lambda _e=None, name=cand["name"]:
                                        _search_local_alias(name)
                                    )

                matched_datasets = {
                    entry["dataset"]
                    for entry in mapped
                    if entry["outcome"] == "matched" and entry["candidates"]
                }
                matched_datasets.update(
                    entry["dataset"]
                    for entry in native
                    if entry.get("types") or entry.get("labels")
                )
                unknown = [
                    entry["dataset"]
                    for entry in mapped
                    if entry["dataset"] not in matched_datasets
                ]
                if unknown:
                    ui.label(
                        "No known counterpart in: " + ", ".join(unknown)
                    ).classes("text-caption drocat-muted")

        # A QTable gesture may emit both a value-click and a selection event.
        # Coalesce those duplicate events by their exact anchor while still
        # allowing a different matched entry to be selected immediately.
        focus_request = {"requested_at": 0.0, "anchor": ""}

    def scroll_to_table_rows(focus_keys, anchor_key: str | None = None) -> None:
        """Focus every visible member row after the table finishes scrolling.

        ``scrollIntoView({behavior: 'smooth'})`` is asynchronous. Starting the
        animation on the same tick makes the shade disappear while the row is
        still moving, and repeated NiceGUI/QTable events can queue a second
        flash. A client-side duplicate cooldown and settle poll make one click
        produce one post-scroll flash. Rows outside the current page are
        intentionally ignored by the DOM lookup; the server first moves the
        page to the first member, so every member on that focused page is
        shaded together.
        """
        keys = list(_normalized_focus_keys(focus_keys))
        if not keys:
            return
        encoded_keys = json.dumps(keys)
        anchor = str(anchor_key or keys[0]).strip() or keys[0]
        encoded_anchor = json.dumps(anchor)
        focus_dedup_ms = int(FOCUS_DEDUP_SECONDS * 1000)
        ui.run_javascript(
            f"""
            setTimeout(() => {{
                const rawKeys = {encoded_keys};
                const keys = Array.from(new Set(rawKeys)).sort();
                const anchor = {encoded_anchor};
                const root = document.querySelector('.drocat-neuron-full-panel .drocat-data-viewer-scroll');
                if (!root) return;
                const state = window.__drocatNeuronFocusState || (window.__drocatNeuronFocusState = {{
                    token: 0, signature: '', requestedAt: 0, blockedUntil: 0, timer: null
                }});
                const signature = anchor;
                const now = performance.now();
                // Coalesce duplicate click/selection events for the same
                // matched entry, but allow a deliberate selection of a
                // different entry immediately. The anchor is the exact row
                // the user selected, not the first previously selected row.
                if (now < state.blockedUntil && state.signature === signature) return;
                state.signature = signature;
                state.requestedAt = now;
                state.blockedUntil = now + {focus_dedup_ms};
                state.token += 1;
                const token = state.token;
                if (state.timer) window.clearTimeout(state.timer);
                root.querySelectorAll('.drocat-neuron-focus-flash').forEach(row =>
                    row.classList.remove('drocat-neuron-focus-flash'));

                let attempts = 0;
                const locate = () => Array.from(root.querySelectorAll('tr[data-neuron-key]'))
                    .filter(row => keys.includes(row.dataset.neuronKey));
                const findRows = () => {{
                    if (state.token !== token) return;
                    const rows = locate();
                    if (!rows.length && attempts++ < 15) {{
                        window.setTimeout(findRows, 60);
                        return;
                    }}
                    if (!rows.length) return;
                    const anchorRow = rows.find(row => row.dataset.neuronKey === anchor) || rows[0];
                    anchorRow.scrollIntoView({{ behavior: 'smooth', block: 'center', inline: 'nearest' }});
                    let lastRect = '';
                    let stableFrames = 0;
                    const started = performance.now();
                    const flash = () => {{
                        if (state.token !== token) return;
                        root.querySelectorAll('.drocat-neuron-focus-flash').forEach(row =>
                            row.classList.remove('drocat-neuron-focus-flash'));
                        rows.forEach(row => {{
                            row.classList.remove('drocat-neuron-focus-flash');
                            void row.offsetWidth;
                            row.classList.add('drocat-neuron-focus-flash');
                        }});
                        state.timer = window.setTimeout(() => {{
                            if (state.token !== token) return;
                            rows.forEach(row => row.classList.remove('drocat-neuron-focus-flash'));
                            state.timer = null;
                        }}, 1400);
                    }};
                    const waitForSettle = () => {{
                        if (state.token !== token) return;
                        const rect = anchorRow.getBoundingClientRect();
                        const currentRect = `${{Math.round(rect.top)}}:${{Math.round(rect.left)}}`;
                        stableFrames = currentRect === lastRect ? stableFrames + 1 : 0;
                        lastRect = currentRect;
                        if ((stableFrames >= 3 && performance.now() - started >= 120)
                            || performance.now() - started >= 1200) {{
                            flash();
                            return;
                        }}
                        window.requestAnimationFrame(waitForSettle);
                    }};
                    window.requestAnimationFrame(waitForSettle);
                }};
                findRows();
            }}, 80);
            """
        )

    def render_match_page() -> None:
        """Render one bounded match-detail page without re-running the query."""
        total = len(match_groups_all)
        pages = max(1, (total + MATCH_GROUP_PAGE_SIZE - 1) // MATCH_GROUP_PAGE_SIZE)
        match_state["page"] = max(1, min(match_state["page"], pages))
        start = (match_state["page"] - 1) * MATCH_GROUP_PAGE_SIZE
        end = min(total, start + MATCH_GROUP_PAGE_SIZE)
        current_groups[:] = match_groups_all[start:end]
        match_table.update_rows(current_groups)
        if total:
            match_status.text = (
                f"Showing {start + 1:,}–{end:,} of {total:,} matched names"
            )
        else:
            match_status.text = "No matched names"
        match_status.update()
        match_page_position.text = (
            f"Page {match_state['page']:,} of {pages:,}"
        )
        match_page_position.update()
        match_previous_button.set_enabled(match_state["page"] > 1)
        match_next_button.set_enabled(match_state["page"] < pages)

    def refresh(
        _event=None,
        *,
        reset_page: bool = False,
        focus_key: str | None = None,
        focus_keys=None,
        focus_anchor_key: str | None = None,
    ):
        nonlocal full_table_all_keys
        if reset_page:
            state["page"] = 1
            match_state["page"] = 1
        try:
            current_page_size = int(page_size.value or 50)
        except (TypeError, ValueError):
            current_page_size = 50
        state["page_size"] = current_page_size
        # Selecting every row needs the complete result set; fetching it here
        # would make every keystroke build the whole key map, so it stays lazy.
        full_table_all_keys = None
        def run_query(requested_page: int, requested_focus_key: str | None = None):
            return query_neuron_index(
                index,
                **current_query_kwargs(),
                page=requested_page,
                page_size=current_page_size,
                focus_key=requested_focus_key,
            )

        result = run_query(state["page"], focus_key)
        if focus_key and result.focus_page and result.focus_page != result.page:
            # The first pass computes the sorted position; the second fetches
            # only the page containing that position.
            result = run_query(result.focus_page)
        state["page"] = result.page
        current_rows[:] = list(result.rows)
        match_groups_all[:] = list(result.match_groups)
        current_group_body_ids.clear()
        current_group_body_ids.update({
            str(key): tuple(values)
            for key, values in result.match_group_body_ids.items()
        })
        match_group_related.clear()
        match_group_related.update({
            str(key): tuple(values)
            for key, values in result.match_group_related.items()
        })
        match_group_primary.clear()
        match_group_primary.update({
            str(key): tuple(values)
            for key, values in result.match_group_primary.items()
        })
        group_members.clear()
        group_members.update({
            str(key): set(values)
            for key, values in result.match_group_members.items()
        })
        stamp_match_group_flags()
        render_match_page()
        table.update_rows(current_rows)
        if focus_keys is None and focus_key:
            focus_keys = (focus_key,)
        # Restore selections after replacing the rows. On a different page,
        # only matching visible rows are checked; the underlying selection
        # sets still retain entries selected on other pages.
        refresh_table_selection()
        if focus_keys:
            scroll_to_table_rows(
                focus_keys,
                anchor_key=focus_anchor_key or focus_key,
            )
        page_position.text = f"Page {result.page:,} of {result.pages:,}"
        page_position.update()
        previous_button.set_enabled(result.page > 1)
        next_button.set_enabled(result.page < result.pages)
        if result.total:
            start = (result.page - 1) * result.page_size + 1
            end = min(result.total, result.page * result.page_size)
            page_info.text = f"Showing {start:,}–{end:,} of {result.total:,} matching rows"
        else:
            page_info.text = "0 matching rows"
        page_info.update()
        no_results.set_visibility(result.total == 0)
        if result.total == 0:
            render_alias_matches()
        else:
            alias_section.set_visibility(False)

    def reset_and_refresh(_event=None):
        refresh(reset_page=True)

    def request_focus(focus_keys, anchor_key: str | None = None) -> None:
        """Run one page-jump/focus request for one user action.

        NiceGUI can deliver a matched-value click and the QTable selection
        update independently. Suppress duplicate focus requests for the same
        exact anchor during the current scroll/breathe interval, while a
        different selected entry gets its own focus immediately.
        """
        normalized = tuple(_normalized_focus_keys(focus_keys))
        if not normalized:
            refresh_table_selection()
            return
        now = time.monotonic()
        anchor = str(anchor_key or normalized[0]).strip()
        if anchor not in normalized:
            anchor = normalized[0]
        if (
            now - focus_request["requested_at"] < FOCUS_DEDUP_SECONDS
            and focus_request["anchor"] == anchor
        ):
            refresh_table_selection()
            return
        focus_request["requested_at"] = now
        focus_request["anchor"] = anchor
        refresh(
            focus_key=anchor,
            focus_keys=normalized,
            focus_anchor_key=anchor,
        )

    def handle_match_value_click(event):
        row = getattr(event, "args", None)
        if not isinstance(row, dict):
            return
        value = str(row.get("match_value", "") or "").strip()
        # ``group_members`` contains the private table-row keys (body ID plus
        # source ordinal), not display body IDs. Sort the set so a multi-row
        # click chooses the same anchor every time.
        member_keys = tuple(sorted(group_members.get(value, ())))
        if member_keys:
            request_focus(member_keys, anchor_key=member_keys[0])

    search_input.on_value_change(reset_and_refresh)
    def handle_filter_column_change(event):
        has_target = target_column.value not in {None, "", "__none__"}
        filter_operator.set_enabled(has_target)
        reset_and_refresh(event)

    target_column.on_value_change(handle_filter_column_change)
    filter_operator.on_value_change(reset_and_refresh)
    sort_column.on_value_change(reset_and_refresh)
    direction.on_value_change(reset_and_refresh)
    page_size.on_value_change(reset_and_refresh)

    def change_page(delta):
        state["page"] = max(1, state["page"] + delta)
        refresh()

    def change_match_page(delta):
        match_state["page"] = max(1, match_state["page"] + delta)
        render_match_page()
        refresh_table_selection()

    previous_button.on_click(lambda: change_page(-1))
    next_button.on_click(lambda: change_page(1))
    match_previous_button.on_click(lambda: change_match_page(-1))
    match_next_button.on_click(lambda: change_match_page(1))
    match_table.on("match-value-click", handle_match_value_click)
    match_table.on("match-selection-toggle", handle_match_toggle)
    match_table.on("match-expand-toggle", handle_match_expand_toggle)
    match_table.on("match-subtype-toggle", handle_subtype_toggle)
    table.on("full-table-select-all", handle_full_table_select_all)
    refresh()


def create_neuron_index_viewer_link(
    dataset_getter: Callable[[], object],
    *,
    label: str = "See available neurons",
    on_open: Callable[[], object] | None = None,
    query_values_getter: Callable[[], object] | None = None,
    query_selection: Callable[[List[str]], object] | None = None,
    query_resolution: Callable[[List[str]], object] | None = None,
    query_remove: Callable[[str], object] | None = None,
    query_edit: Callable[[str], object] | None = None,
    add_to_query: Callable[[List[str]], object] | None = None,
    query_label: str = "Current query",
    defer_apply: bool = False,
):
    """Create a link-like control that opens the cached-index viewer.

    ``dataset_getter`` is evaluated at click time, so changing the dataset in
    a tool tab immediately changes the viewer target.  A multi-dataset getter
    (the Cross-Dataset tab) gets a dataset selector inside the dialog.
    When query callbacks are supplied, the match panel supports multi-select
    and synchronizes selected matched values with the owning query input.
    ``query_remove`` makes the mirrored query preview editable by removing
    one value at a time; ``query_edit`` lets a double-click return that value
    to the owning chip editor.

    With ``defer_apply=True`` (the layer editor), matching values are NOT
    pushed to the caller on every toggle. Instead they accumulate in the
    panel's "Selected" preview (which shows ``selected_query_values()`` and is
    renamed from "Current query"/"mirrors input") and are committed once, when
    the dialog closes via its OK or 'x' button. The preview chips' 'x' then only
    prunes the pending selection rather than the caller's value list.

    ``on_open`` runs immediately before the viewer resolves its dataset. It is
    useful for callers that need to start a fresh selection session without
    changing the viewer's query-selection semantics.
    """
    dialog = ui.dialog()
    # Deferred apply: the OK / 'x' buttons commit the latest pending selection
    # once, then close. ``_render_index`` (re)sets the apply callback per open.
    _apply_holder: dict = {"fn": None}

    def finish() -> None:
        try:
            if _apply_holder["fn"] is not None:
                _apply_holder["fn"]()
        finally:
            dialog.close()

    with dialog:
        with ui.card().classes(
            "w-[min(98vw,1800px)] max-w-none drocat-neuron-viewer-card"
        ):
            with ui.row().classes(
                "w-full items-center justify-between gap-2 drocat-neuron-dialog-header"
            ):
                with ui.row().classes("items-center gap-2 min-w-0 flex-grow"):
                    title = ui.label("Available neurons").classes(
                        "text-h6 drocat-neuron-dialog-title"
                    )
                    header_meta = ui.row().classes(
                        "items-center gap-2 flex-wrap min-w-0 drocat-neuron-header-meta"
                    )
                # OK finishes the selection and exits the panel; the 'x' is a
                # secondary dismiss. Both commit any deferred (pending) selection.
                ui.button(
                    "OK", icon="check",
                    on_click=finish,
                ).props("color=primary dense").classes("drocat-neuron-ok")
                ui.button(icon="close", on_click=finish).props("flat round dense")
            dataset_picker_slot = ui.row().classes("w-full items-center")
            content = ui.column().classes(
                "w-full gap-2 drocat-neuron-viewer-content"
            )

    def open_viewer():
        if on_open is not None:
            try:
                on_open()
            except Exception:
                # Opening the viewer should remain usable even when a caller's
                # session bookkeeping is being torn down during a tab switch.
                pass
        try:
            datasets = _dataset_values(dataset_getter())
        except Exception as exc:
            datasets = []
            error = str(exc)
        else:
            error = ""

        dataset_picker_slot.clear()
        content.clear()
        if error:
            title.text = "Available neurons"
            with content:
                ui.label(f"Could not determine the selected dataset: {error}").classes(
                    "text-body2 drocat-err"
                )
        elif not datasets:
            title.text = "Available neurons"
            with content:
                ui.label("Choose a dataset before opening the neuron index.").classes(
                    "text-body2 drocat-warn"
                )
        else:
            if len(datasets) > 1:
                with dataset_picker_slot:
                    picker = ui.select(
                        options=datasets,
                        value=datasets[0],
                        label="Dataset to view",
                    ).props("outlined").classes("drocat-select").style("min-width: 280px")
                picker.on_value_change(lambda event: _open_dataset(str(event.value)))
            _open_dataset(datasets[0])
        dialog.open()

    def _open_dataset(dataset: str):
        title.text = f"Available neurons · {dataset}"
        title.update()
        _render_index(
            content,
            dataset,
            header_meta=header_meta,
            query_values_getter=query_values_getter,
            query_selection=query_selection,
            query_resolution=query_resolution,
            query_remove=query_remove,
            query_edit=query_edit,
            add_to_query=add_to_query,
            query_label=query_label,
            defer_apply=defer_apply,
            apply_holder=_apply_holder,
        )

    link = ui.button(label, icon="search", on_click=open_viewer).props(
        "flat dense no-caps"
    ).classes("drocat-inline-link")
    # Expose the dialog for component-level tests and for callers that want
    # to close it after changing tabs.
    link.neuron_index_dialog = dialog
    return link
