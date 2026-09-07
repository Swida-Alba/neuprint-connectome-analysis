"""Round 2 — the standalone Type Mapping entrance (composed global view).

"Type Mapping" beside the Cross-Dataset tab's dataset selector opens a
preview dialog that maps the search across EVERY ordered pair of the
selected datasets (the global search) and renders one composed
N-dataset type-level graph plus per-pair cards and CSV exports.
Strictly informational — nothing leaks into the analysis selection
(spec: _plan/plan-type-mapping-round2-entrance-composed-view.md).
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable, Dict, List

from nicegui import ui

from .banner import push_banner
from .common import neuron_list_input

logger = logging.getLogger(__name__)

COMPOSED_NODE_CAP = 80


def create_type_mapping_entry(get_datasets: Callable[[], list]):
    """The entrance button + preview dialog (spec §3).

    Returns the button element; the tab wires ``refresh_state()`` into
    the dataset selector so the disabled state follows the selection
    (disabled with a tooltip until >= 2 selected datasets have cached
    neuron indexes).
    """
    from ..neuron_index import load_cached_neuron_index

    state: Dict[str, Any] = {"pair_flows": {}, "pools": {}, "meta": {},
                             "composed": None, "datasets": [],
                             "summary": [], "orphans": {}}

    def _ready() -> bool:
        datasets = list(get_datasets() or [])
        return len(datasets) >= 2 and all(
            load_cached_neuron_index(ds) is not None for ds in datasets)

    # Same fixed window as the 'See available neurons' viewer: the backdrop
    # never dismisses it, the card is viewport-bounded with internal scroll,
    # and the corner 'x' is the only way out.
    dialog = ui.dialog().props("persistent")
    with dialog, \
            ui.card().classes(
                "w-[min(98vw,1800px)] max-w-none drocat-neuron-viewer-card"
            ):
        with ui.row().classes(
            "w-full items-center justify-between gap-2 drocat-neuron-dialog-header"
        ):
            ui.label("Auto type mapping preview").classes(
                "text-h6 drocat-neuron-dialog-title")
            ui.button(icon="close", on_click=dialog.close).props(
                "flat round dense")
        with ui.column().classes("w-full gap-2 drocat-neuron-viewer-content"):
            ui.label(
                "Preview of the auto type mapping the cross-dataset analysis "
                "will use — informational only, please double check."
            ).classes("text-caption drocat-muted")
            def _suggest(text: str):
                """Dataset-aware suggestions — the SAME staged semantics as
                the tab's query boxes: types first, widening to the metadata
                columns when no type matched, so typing 'circadian' offers
                'circadian_clock' (FAFB) and 'circadian_neuron' (BANC) here
                too."""
                from ..type_suggestions import dataset_aware_suggestions

                return dataset_aware_suggestions(
                    text, list(get_datasets() or []), "auto", limit=None)

            search = neuron_list_input(
                label="Types to map",
                placeholder="e.g. APDN3, aMe.* — one query per chip",
                unit_label="query",
                show_upload=False,
                suggestions=_suggest,
                # Standalone history: the panel's Recent/Frequent list reads
                # and writes its own store, so panel searches never mix with
                # the analysis tabs' neuron-query history.
                history_kind="type_mapping",
                # History rows annotated like suggestion rows: the matched
                # column and dataset(s) ("type · flywire_FAFB_v783") from
                # the current selection's local pools.
                history_hint_datasets=lambda: list(get_datasets() or []),
                hint="One query per chip. The filter modes match the standard "
                     "query (exact / starts with / contains / ends with / "
                     "regex); dataset-aware type suggestions from the "
                     "selected datasets appear as you type. This box keeps "
                     "its own query history, separate from the tabs.",
            ).classes("w-full")
            with ui.row():
                # lazy dispatch: _run_click is defined below the dialog build
                # (NiceGUI schedules the returned coroutine as a task)
                search_btn = ui.button("Search mappings", icon="search",
                                       on_click=lambda: _run_click())
            # Loading notice: painted BEFORE the heavy search leaves the
            # event loop (the first search warms a large index and can take
            # a minute or two — silent freezing looked like a hang).
            loading_row = ui.row().classes("items-center gap-2")
            with loading_row:
                ui.spinner("dots", size="md", color="primary")
                ui.label("Loading the selected datasets' type indexes and "
                         "mapping — the first search can take a minute or "
                         "two. Please wait …").classes(
                    "text-caption drocat-muted")
            loading_row.set_visibility(False)
            notes_label = ui.label("").classes("text-caption drocat-muted")
            results = ui.column().classes("w-full")

    def _deliver(html: str, name: str) -> None:
        ui.download.content(html, name, "text/html")
        push_banner(
            f"{name} — check your browser's default downloads folder. "
            "Informational only, please double check.")

    def _deliver_flows(src: str, tgt: str, flows, pools,
                       kind: str, variant: str, stamp: str) -> None:
        from comparison.mapping_visualization import (
            render_bridge_linker_html,
            render_mapping_network_html,
            render_mapping_sankey_html,
        )
        foreign = tgt.replace(":", "_")
        name = f"mapping_{kind}_{variant}_{foreign}_{stamp}.html"
        orphans = (state.get("orphans") or {}).get((src, tgt))
        if kind == "sankey":
            html = render_mapping_sankey_html(flows, pools=pools,
                                              variant=variant,
                                              orphans=orphans)
        elif variant == "linker":
            html = render_bridge_linker_html(
                flows, source_dataset=src, target_dataset=tgt, pools=pools)
        else:
            html = render_mapping_network_html(flows, pools=pools,
                                               orphans=orphans)
        if not html:
            ui.notify("Nothing to visualize.", type="info")
            return
        _deliver(html, name)

    def _deliver_pair_csv(src: str, tgt: str, flows, pools,
                          stamp: str) -> None:
        from comparison.mapping_visualization import build_bridges_csv

        text = build_bridges_csv(flows, pools=pools)
        if not text:
            ui.notify("Nothing to export.", type="info")
            return
        name = f"bridges_{src.replace(':', '_')}_{tgt.replace(':', '_')}_{stamp}.csv"
        ui.download.content(text, name, "text/csv")
        push_banner(
            f"{name} — check your browser's default downloads folder. "
            "Informational only, please double check.")

    def _pair_card(src: str, tgt: str, flows, pools: dict) -> None:
        from comparison.cross_dataset_type_mapper import bridge_linker_text

        stamp = time.strftime("%Y%m%d_%H%M%S")
        rows = []
        for flow in flows:
            info = bridge_linker_text(flow.get("bridges") or [], src, tgt,
                                      flow.get("foreign_type", ""))
            pool = pools.get((flow.get("source_type"),
                              flow.get("foreign_type"))) or {}
            # §dedupe (user 2026-09-07): granularity ("n to m") and
            # coverage ("covered n of m") carried the SAME two numbers —
            # one Coverage column now uses the pool's own
            # ``covered <target pool> of <target type total>`` (which
            # also stays filled when one side's pool is empty, the old
            # blank-when-either-side-empty case).
            cov = pool.get("coverage") or "—"
            rows.append([
                flow.get("source_type", ""),
                flow.get("foreign_type", ""),
                f"{flow.get('source_count') or 0} → "
                f"{flow.get('foreign_count') or 0}",
                info["text"], cov,
            ])
        ui.table(
            columns=[
                {"name": "name", "label": "Type", "field": "name",
                 "align": "left"},
                {"name": "foreign", "label": "Mapped to", "field": "foreign",
                 "align": "left"},
                {"name": "counts", "label": "Neurons", "field": "counts",
                 "align": "left"},
                {"name": "map_used", "label": "Map used", "field": "map_used",
                 "align": "left"},
                {"name": "cov", "label": "Pool coverage", "field": "cov",
                 "align": "left"},
            ],
            rows=[dict(zip(("name", "foreign", "counts", "map_used",
                             "cov"), r)) for r in rows],
        ).classes("w-full")
        with ui.row().classes("flex-wrap"):
            ui.button("Sankey (type-level)",
                      on_click=lambda: _deliver_flows(
                          src, tgt, flows, pools, "sankey", "type", stamp))
            ui.button("Sankey (linker)",
                      on_click=lambda: _deliver_flows(
                          src, tgt, flows, pools, "sankey", "linker", stamp))
            ui.button("Network",
                      on_click=lambda: _deliver_flows(
                          src, tgt, flows, pools, "network", "type", stamp))
            ui.button("Linker paths",
                      on_click=lambda: _deliver_flows(
                          src, tgt, flows, pools, "network", "linker", stamp))
            ui.button("Export bridges (CSV)",
                      on_click=lambda: _deliver_pair_csv(
                          src, tgt, flows, pools, stamp))

    def _render_results() -> None:
        results.clear()
        pair_flows = state["pair_flows"]
        pools = state["pools"]
        with results:
            composed = state.get("composed")
            if composed:
                with ui.row().classes("items-center gap-2 flex-wrap"):
                    stamp = time.strftime("%Y%m%d_%H%M%S")
                    ui.button("Composed view (HTML)", icon="account_tree",
                              on_click=lambda: _deliver(
                                  composed,
                                  f"mapping_composed_{stamp}.html"))
                    ui.button("Export bridges (CSV) — all pairs",
                              on_click=lambda: _deliver_combined_csv(stamp))
            else:
                ui.label("No composed graph (nothing mapped across the "
                         "selected datasets).").classes(
                    "text-caption drocat-muted")
            for note in state["meta"].get("notes", []):
                ui.label(note).classes("text-caption drocat-muted")
            summary = state.get("summary") or []
            if summary:
                ui.table(
                    columns=[
                        {"name": "dataset", "label": "Dataset",
                         "field": "dataset", "align": "left"},
                        {"name": "types", "label": "Matched types",
                         "field": "types", "align": "left"},
                        {"name": "neurons", "label": "Neurons",
                         "field": "neurons", "align": "left"},
                        {"name": "pairs", "label": "Mapped pairs",
                         "field": "pairs", "align": "left"},
                        {"name": "mapped_types", "label": "Mapped types",
                         "field": "mapped_types", "align": "left"},
                        {"name": "mapped_neurons",
                         "label": "Mapped neurons (per pair)",
                         "field": "mapped_neurons", "align": "left"},
                        {"name": "unmapped", "label": "Unmapped (orphans)",
                         "field": "unmapped", "align": "left"},
                    ],
                    rows=summary,
                ).classes("w-full")
            else:
                ui.label("No mappings found for the search across the "
                         "selected datasets.").classes(
                    "text-caption drocat-muted")
            orphan_all = state.get("orphans") or {}
            if orphan_all:
                with ui.expansion(
                        f"Orphan types — no mapped counterpart "
                        f"({sum(len(v) for v in orphan_all.values())})",
                        icon="link_off").classes("w-full"):
                    for (src, tgt), entries in sorted(orphan_all.items()):
                        names = ", ".join(
                            f"{e['type']} ({e['count']})"
                            for e in entries)
                        ui.label(
                            f"{src} → {tgt}: {names}").classes(
                            "text-caption drocat-muted")
            for (src, tgt), flows in sorted(
                    pair_flows.items(),
                    key=lambda kv: (-len(kv[1]), kv[0])):
                with ui.expansion(
                        f"{src} → {tgt} · {len(flows)} mapped pairs",
                        icon="compare_arrows").classes("w-full"):
                    _pair_card(src, tgt, flows, pools)
            if not pair_flows:
                ui.label("No mappings found for the search across the "
                         "selected datasets.").classes(
                    "text-caption drocat-muted")

    def _deliver_combined_csv(stamp: str) -> None:
        from comparison.mapping_visualization import (
            build_bridges_csv,
            infer_bridge_columns,
        )

        pair_flows = state["pair_flows"]
        pools = state["pools"]
        # One uniform width for the whole file: the UNION of every
        # pair's bridge columns. Per-pair headers differ (each pair only
        # carries the linker columns its bridges use), so stripping
        # headers and concatenating produced ragged rows (the
        # Tablecruncher "N different row lengths" report).
        union: List[str] = []
        for flows in pair_flows.values():
            for column in infer_bridge_columns(flows):
                if column not in union:
                    union.append(column)
        parts: List[str] = []
        for (src, tgt), flows in sorted(pair_flows.items()):
            text = build_bridges_csv(flows, pools=pools,
                                     bridge_columns=union)
            if not text:
                continue
            lines = text.splitlines()
            if parts:
                lines = lines[1:]  # shared header already written
            parts.append("\n".join(lines))
        if not parts:
            ui.notify("Nothing to export.", type="info")
            return
        name = f"bridges_all_pairs_{stamp}.csv"
        ui.download.content("\n".join(parts), name, "text/csv")
        push_banner(
            f"{name} — check your browser's default downloads folder. "
            "Informational only, please double check.")

    def _pool_pair(flows, src, tgt, indexes, pools):
        """Pool bodyIds per mapped pair via the preferred chain (§12)."""
        from ..neuron_index import pool_bridge_body_ids
        from comparison.cross_dataset_type_mapper import (
            preferred_bridge_chain,
            standardize_bridge,
        )

        for flow in flows:
            chains = [c for c in (flow.get("bridges") or [])
                      if c and c[-1].get("value") == flow.get("foreign_type")]
            chain = preferred_bridge_chain(
                chains or flow.get("bridges") or [], src, tgt)
            if chain is None:
                continue
            key = (flow.get("source_type"), flow.get("foreign_type"))
            if key in pools:
                continue
            foreign_index = indexes.get(tgt)
            try:
                pools[key] = pool_bridge_body_ids(
                    src, tgt,
                    standardize_bridge(chain, src, tgt),
                    flow.get("source_type"),
                    flow.get("foreign_type"),
                    indexes={src: indexes.get(src), tgt: foreign_index}
                    if foreign_index is not None else None)
            except Exception:
                # one un-poolable pair only loses the pooled-count hover,
                # but the failure is logged, not silent
                logger.warning(
                    "pool_bridge_body_ids failed for %r (%s → %s)",
                    key, src, tgt, exc_info=True)
                continue

    def _set_loading(on: bool) -> None:
        """Show/hide the loading notice and freeze the Search button."""
        loading_row.set_visibility(on)
        if on:
            search_btn.disable()
        else:
            search_btn.enable()

    async def _run_click() -> None:
        """Paint the loading notice FIRST, then search off the event loop.

        The heavy work runs in a worker thread via ``run.io_bound`` (it
        touches no NiceGUI elements), so the spinner keeps animating
        while the first search warms the type index; the results render
        back on the event loop when it completes.
        """
        queries, datasets, mode = _collect_queries()
        if not queries:
            ui.notify("Enter a type search first.", type="warning")
            return
        if len(datasets) < 2:
            ui.notify("Select at least 2 datasets with cached neuron "
                      "indexes.", type="warning")
            return
        _set_loading(True)
        try:
            # yield once so the browser paints the notice before the
            # heavy search starts
            await asyncio.sleep(0.05)
            loop = asyncio.get_running_loop()
            outcome = await loop.run_in_executor(
                None, lambda: _compute(queries, datasets, mode))
            _apply(outcome)
            _record_panel_history(queries, datasets, outcome)
        except Exception:
            logger.exception(
                "type mapping search failed (queries=%r, datasets=%r)",
                queries, datasets)
            ui.notify("The type mapping search failed — please double "
                      "check the query and try again.", type="negative")
        finally:
            _set_loading(False)

    def _record_panel_history(queries, datasets, outcome) -> None:
        """Record the searched chips in the panel's OWN history store.

        Only confirmed hits are recorded, mirroring the shared neuron
        history's rule: at least one queried type must have matched a
        dataset (the per-dataset summary's matched-type count). Failed or
        zero-hit searches never pollute the Recent/Frequent list.
        """
        matched = sum(int(row.get("types") or 0)
                      for row in (outcome.get("summary") or []))
        if not matched:
            return
        try:
            from ..type_mapping_history import record as _record_history

            _record_history([str(q) for q in queries],
                            datasets=[str(d) for d in datasets])
        except Exception:
            # history is a convenience, never an error
            logger.warning("type mapping history record failed",
                           exc_info=True)

    def _collect_queries() -> tuple:
        mode, chips = search.get_value()
        queries = [str(q).strip() for q in chips if str(q).strip()]
        datasets = [d for d in (get_datasets() or [])
                    if load_cached_neuron_index(d) is not None]
        return queries, datasets, mode

    def _compute(queries, datasets, mode) -> Dict[str, Any]:
        """The heavy global search — NO NiceGUI calls (worker thread)."""
        from ..neuron_index import (
            collect_native_type_matches,
            count_types_in_index,
            enrich_native_type_matches,
            mapped_type_targets,
            resolve_type_matches,
        )
        from comparison.cross_dataset_type_mapper import get_type_mapper
        from comparison.mapping_visualization import (
            build_mapping_flows,
            dedupe_mirrored_pairs,
            origin_seeded_flows,
            render_composed_mapping_html,
        )

        try:
            mapper = get_type_mapper()
        except Exception:
            mapper = None

        indexes = {ds: load_cached_neuron_index(ds) for ds in datasets}
        # §12: resolve every chip under the ACTIVE FILTER MODE (exact /
        # startswith / contains / endswith / regex) against each selected
        # dataset's type column; no-hit chips fall back to the staged
        # native search (labels) below.
        resolved = resolve_type_matches(queries, mode, datasets, indexes)
        origins = resolved["origins"]
        notes = list(resolved["notes"])

        pair_flows: Dict[tuple, list] = {}
        pools: Dict[tuple, Dict[str, Any]] = {}

        # Explicit modes: map FROM each origin dataset (the query lives
        # where it matched) INTO every other selected dataset.
        for origin in sorted(origins):
            o_types = origins[origin]
            source_counts = count_types_in_index(indexes[origin], o_types)
            for target in datasets:
                if target == origin:
                    continue
                flows = origin_seeded_flows(
                    origin, o_types, target, source_counts=source_counts)
                if not flows:
                    continue
                ends = sorted({f["foreign_type"] for f in flows})
                f_counts = count_types_in_index(indexes[target], ends)
                for flow in flows:
                    flow["foreign_count"] = f_counts.get(
                        flow["foreign_type"], 0)
                pair_flows[(origin, target)] = \
                    pair_flows.get((origin, target), []) + flows
                _pool_pair(flows, origin, target, indexes, pools)

        # Zero-hit fallback chips: the previous staged native sweep
        # (substring types + taxonomy labels → pooled nodes).
        for chip in resolved["fallback_chips"]:
            for ds in datasets:
                index = indexes[ds]
                entries_by_foreign: Dict[str, list] = {}
                for entry in collect_native_type_matches(
                        ds, chip, datasets=datasets, uncapped=True):
                    foreign = entry.get("dataset", "")
                    if foreign == ds or foreign not in datasets:
                        continue
                    entries_by_foreign.setdefault(foreign, []).append(entry)
                flat = [e for entries in entries_by_foreign.values()
                        for e in entries]
                if flat:
                    enrich_native_type_matches(flat, ds)
                for foreign, entries in entries_by_foreign.items():
                    source_counts = count_types_in_index(index, [
                        t for e in entries
                        for t in e.get("mapped_type_names", [])])
                    flows = build_mapping_flows(entries, ds,
                                                source_counts=source_counts)
                    if not flows:
                        continue
                    pair_flows[(ds, foreign)] = \
                        pair_flows.get((ds, foreign), []) + flows
                    _pool_pair(flows, ds, foreign, indexes, pools)

        # One canonical entry per unordered pair (§12 mirror dedupe).
        pair_flows = dedupe_mirrored_pairs(pair_flows, origins.keys())

        # W3 orphans: queried/expanded types with NO mapped counterpart
        # in a specific target dataset stay visible — one orphan entry
        # per (origin, target) with the type's own neuron count.
        orphans: Dict[tuple, List[Dict[str, Any]]] = {}
        for origin in sorted(origins):
            o_types = origins[origin]
            flowed_sources = {
                f.get("source_type")
                for (s, _t), fl in pair_flows.items() if s == origin
                for f in fl
            }
            for target in datasets:
                if target == origin:
                    continue
                missing = [t for t in o_types if t not in flowed_sources]
                if not missing:
                    continue
                counts = count_types_in_index(indexes[origin], missing)
                orphans[(origin, target)] = [
                    {"dataset": origin, "type": t,
                     "count": counts.get(t, 0), "target": target}
                    for t in missing]

        # Per-dataset summary strip (fig 2): matched types, neurons,
        # mapped pairs, mapped types, mapped neurons, unmapped.
        # §backend unification (user 2026-09-07): the received types come
        # from the SAME shared resolution as the 'See available neurons'
        # auto-initiated type mapping (``mapped_type_targets``), and
        # mapped_neurons counts each target's neurons ONCE — the old
        # per-flow Σ foreign_count double-counted shared targets
        # (s-LNv / 5thsLNv_LNd6 ×2, SMP227 ×3), which turned 219 unique
        # male-cns neurons into 243.
        recv_types_by_ds: Dict[str, set] = {ds: set() for ds in datasets}
        if mapper is not None and getattr(mapper, '_loaded', False):
            for origin in sorted(origins):
                for target in datasets:
                    if target == origin:
                        continue
                    for otype in origins[origin]:
                        ann = mapped_type_targets(
                            mapper, otype, origin, target)
                        if ann:
                            recv_types_by_ds[target].update(
                                ann.get('targets') or [])
        # the flow ends are the bridge half of the same resolution
        for (s, t), fl in pair_flows.items():
            recv_types_by_ds.setdefault(t, set()).update(
                f.get('foreign_type') or '' for f in fl
                if f.get('foreign_type'))

        summary = []
        for ds in datasets:
            matched = origins.get(ds, [])
            neurons = (sum(count_types_in_index(
                indexes[ds], matched).values()) if matched else 0)
            pairs = sum(len(fl) for (s, t), fl in pair_flows.items()
                        if s == ds or t == ds)
            recv_types = recv_types_by_ds.get(ds, set())
            recv_neurons = (sum(count_types_in_index(
                indexes[ds], sorted(recv_types)).values())
                if recv_types else 0)
            issued_types = {f.get("foreign_type") or ""
                            for (s, t), fl in pair_flows.items()
                            if s == ds for f in fl}
            unmapped = sum(len(v) for (s, _t), v in orphans.items()
                           if s == ds)
            summary.append({
                "dataset": ds,
                "types": len(matched),
                "neurons": neurons,
                "pairs": pairs,
                "mapped_types": len(recv_types | issued_types),
                "mapped_neurons": recv_neurons,
                "unmapped": unmapped,
            })

        html, meta = render_composed_mapping_html(
            pair_flows, node_cap=COMPOSED_NODE_CAP)
        meta = dict(meta or {})
        meta["notes"] = notes + list(meta.get("notes", []))
        return {"pair_flows": pair_flows, "pools": pools, "meta": meta,
                "composed": html, "datasets": datasets,
                "summary": summary, "orphans": orphans}

    def _apply(outcome: Dict[str, Any]) -> None:
        """Apply the search outcome and render results (event loop)."""
        state.update(**outcome)
        notes_label.set_text(
            "   ".join(outcome["meta"].get("notes", [])))
        _render_results()

    button = ui.button("Type Mapping", icon="hub", on_click=dialog.open)

    def refresh_state() -> None:
        if _ready():
            button.enable()
            button.tooltip("Preview the auto type mapping across the "
                           "selected datasets (informational only)")
        else:
            button.disable()
            button.tooltip("Select at least 2 datasets with cached neuron "
                           "indexes to preview the type mapping")

    button.search_container = search  # type: ignore[attr-defined]
    button.refresh_state = refresh_state  # type: ignore[attr-defined]
    refresh_state()
    return button
