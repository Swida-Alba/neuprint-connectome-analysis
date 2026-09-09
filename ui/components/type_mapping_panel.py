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

# Quasar table cells render nowrap by default, so a long Map used /
# Maps-to cell stretches its table until the later columns overflow the
# viewport.  These column helpers cap the width and force wrapping so
# ALL columns stay visible and long values continue on the next line
# (user 2026-09-07).
_WRAP = "white-space: normal; overflow-wrap: anywhere;"


def _col(name: str, label: str, field: str, *,
         max_w=None, min_w=None) -> Dict[str, Any]:
    """A QTable column def with wrapping and an optional width cap."""
    bounds = ((f"max-width: {max_w}px;" if max_w else "")
              + (f"min-width: {min_w}px;" if min_w else ""))
    return {
        "name": name, "label": label, "field": field, "align": "left",
        "headerStyle": _WRAP + bounds,
        "style": _WRAP + bounds,
    }


def _pool_mapping_pair(flows, src, tgt, indexes, pools) -> None:
    """Pool bodyIds for one type-mapping pair.

    BodyIds are coverage evidence only.  The type-level mapping has already
    been resolved by the mapper; this helper merely computes the independent
    endpoint pools used to report ``m of n`` coverage in the rendered card.
    """
    from ..neuron_index import resolve_prioritized_bridge_pool
    from comparison.mapping_visualization import mapping_pool_key

    for flow in flows:
        chains = [c for c in (flow.get("bridges") or [])
                  if c and c[-1].get("value") == flow.get("foreign_type")]
        chains = chains or flow.get("bridges") or []
        key = mapping_pool_key(
            src, tgt, flow.get("source_type"), flow.get("foreign_type"))
        if key in pools:
            continue
        index_kw = {
            dataset: index for dataset, index in (
                (src, indexes.get(src)), (tgt, indexes.get(tgt)))
            if index is not None
        }
        result = resolve_prioritized_bridge_pool(
            src, tgt, chains,
            flow.get("source_type"), flow.get("foreign_type"),
            indexes=index_kw)
        for attempt in result.get("attempts") or []:
            if attempt.get("status") in {"unsupported", "error"}:
                linker_text = "; ".join(
                    f"{item.get('column', '')}="
                    f"{item.get('raw_value', '')}"
                    f"[{item.get('canonical_value', '')}]"
                    for item in attempt.get("linker_values") or [])
                logger.warning(
                    "unsupported bridge chain dropped for %r (%s → %s; "
                    "rank %s; linkers %s; source pool %s; target pool %s; "
                    "%s)", key, src, tgt, attempt.get("rank"),
                    linker_text or "none", attempt.get("source_pool_size"),
                    attempt.get("target_pool_size"),
                    attempt.get("reason") or attempt.get("status"))
        if result.get("resolution_status") != "supported":
            continue
        pools[key] = result


def _compute_type_mapping(queries, datasets, mode) -> Dict[str, Any]:
    """Compute the global type mapping without touching NiceGUI state.

    This is deliberately a module-level, pickle-safe callback.  A cold
    mapper load uses it in a dedicated spawned process so pandas parsing and
    the Python graph build cannot starve the websocket event loop.
    """
    from ..neuron_index import (
        collect_native_type_matches,
        count_types_in_index,
        enrich_native_type_matches,
        _load_cross_match_index,
        _load_coverage_index,
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

    # Type mapping is a type-level operation. Keep bodyId/linker columns out
    # of the resolver and its counts; they are only needed later to annotate
    # a rendered type edge with coverage.
    indexes = {ds: _load_cross_match_index(ds) for ds in datasets}
    if any(indexes.get(ds) is None for ds in datasets):
        raise RuntimeError(
            "one or more selected type indexes could not be read")

    # Coverage is intentionally separate from type matching. These compact
    # projections contain only bodyId, type, and linker columns used to
    # report an m-of-n coverage subset.
    coverage_indexes: Dict[str, Any] = {}

    def _coverage_indexes_for(src: str, tgt: str) -> Dict[str, Any]:
        selected: Dict[str, Any] = {}
        for dataset in (src, tgt):
            if dataset not in coverage_indexes:
                coverage_indexes[dataset] = _load_coverage_index(dataset)
            index = coverage_indexes[dataset]
            if index is not None:
                selected[dataset] = index
        return selected

    # §12: resolve every chip under the ACTIVE FILTER MODE (exact /
    # startswith / contains / endswith / regex) against each selected
    # dataset's type column; no-hit chips fall back to the staged native
    # search (labels) below.
    resolved = resolve_type_matches(queries, mode, datasets, indexes)
    origins = resolved["origins"]
    origin_matches = resolved.get("origin_matches", {})
    notes = list(resolved["notes"])

    pair_flows: Dict[tuple, list] = {}
    pools: Dict[tuple, Dict[str, Any]] = {}

    # Explicit modes: map FROM each origin dataset (the query lives where it
    # matched) INTO every other selected dataset.
    for origin in sorted(origins):
        o_types = origins[origin]
        source_counts = count_types_in_index(indexes[origin], o_types)
        for target in datasets:
            if target == origin:
                continue
            flows = origin_seeded_flows(
                origin, o_types, target, source_counts=source_counts,
                matched_origins=origin_matches.get(origin))
            if not flows:
                continue
            ends = sorted({f["foreign_type"] for f in flows})
            f_counts = count_types_in_index(indexes[target], ends)
            for flow in flows:
                flow["foreign_count"] = f_counts.get(
                    flow["foreign_type"], 0)
            pair_flows[(origin, target)] = \
                pair_flows.get((origin, target), []) + flows
            _pool_mapping_pair(
                flows, origin, target,
                _coverage_indexes_for(origin, target), pools)

    # Zero-hit fallback chips: the staged native sweep (substring types +
    # taxonomy labels → pooled nodes).
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
                _pool_mapping_pair(
                    flows, ds, foreign,
                    _coverage_indexes_for(ds, foreign), pools)

    # One canonical entry per unordered pair (§12 mirror dedupe).
    pair_flows = dedupe_mirrored_pairs(pair_flows, origins.keys())

    # W3 orphans: queried/expanded types with NO mapped counterpart in a
    # specific target dataset stay visible.
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

    # Per-dataset summary strip.  Mapped neurons count each target type once;
    # the old per-flow sum double-counted shared targets.
    recv_types_by_ds: Dict[str, set] = {ds: set() for ds in datasets}
    if mapper is not None and getattr(mapper, "_loaded", False):
        for origin in sorted(origins):
            for target in datasets:
                if target == origin:
                    continue
                for otype in origins[origin]:
                    ann = mapped_type_targets(
                        mapper, otype, origin, target)
                    if ann:
                        recv_types_by_ds[target].update(
                            ann.get("targets") or [])
    # The flow ends are the bridge half of the same resolution.
    for (s, t), fl in pair_flows.items():
        recv_types_by_ds.setdefault(t, set()).update(
            f.get("foreign_type") or "" for f in fl
            if f.get("foreign_type"))

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
        issued_sources = {f.get("source_type") or ""
                          for (s, _t), fl in pair_flows.items()
                          if s == ds for f in fl
                          if f.get("source_type")}
        issued_neurons = (sum(count_types_in_index(
            indexes[ds], sorted(issued_sources)).values())
                          if issued_sources else 0)
        unmapped = sum(len(v) for (s, _t), v in orphans.items()
                       if s == ds)
        summary.append({
            "dataset": ds,
            "types": len(matched),
            "neurons": neurons,
            "pairs": pairs,
            "mapped_types": len(recv_types | issued_sources),
            "mapped_neurons": recv_neurons,
            "issued_neurons": issued_neurons,
            "unmapped": unmapped,
        })

    html, meta = render_composed_mapping_html(
        pair_flows, pools=pools, node_cap=COMPOSED_NODE_CAP)
    meta = dict(meta or {})
    meta["notes"] = notes + list(meta.get("notes", []))
    return {"pair_flows": pair_flows, "pools": pools, "meta": meta,
            "composed": html, "datasets": datasets,
            "summary": summary, "orphans": orphans}


def create_type_mapping_entry(get_datasets: Callable[[], list]):
    """The entrance button + preview dialog (spec §3).

    Returns the button element; the tab wires ``refresh_state()`` into
    the dataset selector so the disabled state follows the selection
    (disabled with a tooltip until >= 2 selected datasets have cached
    neuron indexes).
    """
    from ..neuron_index import neuron_index_path

    state: Dict[str, Any] = {"pair_flows": {}, "pools": {}, "meta": {},
                             "composed": None, "datasets": [],
                             "summary": [], "orphans": {}}

    def _ready() -> bool:
        datasets = list(get_datasets() or [])
        return len(datasets) >= 2 and all(
            neuron_index_path(ds).is_file() for ds in datasets)

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

            search_action: Dict[str, Any] = {}

            def _add_search_action() -> None:
                # The Search button lives IN the input row (right of the
                # Match-by select) so query box, filter and action share
                # one toolbar row (user 2026-09-07). self-stretch keeps it
                # at the row's height; lazy dispatch: _run_click is defined
                # below the dialog build (NiceGUI schedules the returned
                # coroutine as a task).
                search_action["button"] = ui.button(
                    "Search mappings", icon="search",
                    on_click=lambda: _run_click()
                ).classes("self-stretch")

            search = neuron_list_input(
                label="Types to map",
                placeholder="e.g. APDN3, aMe.* — one query per chip",
                unit_label="query",
                show_upload=False,
                suggestions=_suggest,
                # Same height as the query box, and the Search button
                # slots in right of it — the chip input narrows to make
                # room (user 2026-09-07).
                filter_dense=False,
                input_actions=_add_search_action,
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
            search_btn = search_action["button"]
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

        text = build_bridges_csv(flows, pools=pools, extended=True)
        if not text:
            ui.notify("Nothing to export.", type="info")
            return
        name = f"mapping_{src.replace(':', '_')}_{tgt.replace(':', '_')}_{stamp}.csv"
        ui.download.content(text, name, "text/csv")
        push_banner(
            f"{name} — check your browser's default downloads folder. "
            "Informational only, please double check.")

    def _pair_card(src: str, tgt: str, flows, pools: dict) -> None:
        from comparison.cross_dataset_type_mapper import bridge_linker_text
        from comparison.mapping_visualization import (
            format_pool_side,
            get_mapping_pool,
        )
        from utils.naming_utils import dataset_abbrev

        stamp = time.strftime("%Y%m%d_%H%M%S")
        src_code = dataset_abbrev(src) or src
        tgt_code = dataset_abbrev(tgt) or tgt
        rows = []
        for flow in flows:
            s_type = flow.get("source_type", "")
            f_type = flow.get("foreign_type", "")
            info = bridge_linker_text(flow.get("bridges") or [], src, tgt,
                                      f_type)
            pool = get_mapping_pool(pools, flow)
            s_total = int(flow.get("source_count") or 0)
            t_total = int(flow.get("foreign_count") or 0)
            # each linker annotated with its own pooled bodyId count on
            # its home side (user 2026-09-07: "each linker corresponding
            # neuron number")
            per_linker = {(l.get("column"), l.get("value")): l
                          for l in (pool.get("per_linker") or [])}
            parts = []
            for entry in info["entries"]:
                text = entry["text"]
                linker = per_linker.get((entry["column"], entry["value"]))
                if linker and linker.get("body_ids"):
                    text += (f" · {len(linker['body_ids']):,} "
                             f"{dataset_abbrev(linker.get('home', '')) or '?'}"
                             " bodyIds")
                parts.append(text)
            selected_chain = pool.get("selected_chain") if pool else None
            selected_text = ""
            if selected_chain:
                selected_text = bridge_linker_text(
                    [selected_chain], src, tgt, f_type).get("text") or ""
            all_valid_chains = (pool.get("valid_chains") if pool else None)
            all_valid_text = ""
            if all_valid_chains:
                all_valid_text = bridge_linker_text(
                    all_valid_chains, src, tgt, f_type).get("text") or ""
            if selected_text and all_valid_text and (
                    all_valid_text != selected_text):
                map_used = (f"selected: {selected_text}; "
                            f"all valid evidence: {all_valid_text}")
            else:
                map_used = selected_text or (" + ".join(parts)
                                             if parts else (info["text"] or "—"))
            if flow.get("mapping_status") == "valid_split_evidence":
                map_used = (
                    "valid 1-to-N evidence (branch counts are non-exclusive) "
                    "— no single canonical target; "
                    f"{map_used}"
                )
            elif flow.get("mapping_status") == "conflict":
                map_used = "unresolved conflict — no automatic target; " \
                           f"{map_used}"
            # basis-aware per-side cells (user 2026-09-09): a measured
            # subset, the unconstrained full population, an unmeasurable
            # side and a missing pool must never look alike
            if pool:
                selected_cov = " · ".join((
                    format_pool_side(src_code, pool, "source"),
                    format_pool_side(tgt_code, pool, "target")))
                all_valid_cov = " · ".join((
                    format_pool_side(src_code, pool, "source", "all_valid"),
                    format_pool_side(tgt_code, pool, "target", "all_valid")))
                cov = selected_cov
                if all_valid_cov != selected_cov:
                    cov += f" · all-valid union: {all_valid_cov}"
            else:
                cov = "not pooled"
            if flow.get("mapping_status") == "valid_split_evidence":
                cov = "branch evidence (non-exclusive); " + cov
            rows.append([
                s_type, f_type,
                f"{s_total} {src_code} → {t_total} {tgt_code}",
                map_used, cov,
            ])
        # Quasar table cells nowrap by default: cap the long columns and
        # force wrapping so every column stays visible (user 2026-09-07)
        ui.table(
            columns=[
                _col("name", f"Type ({src_code})", "name", max_w=170),
                _col("foreign", f"Mapped to ({tgt_code})", "foreign",
                     max_w=170),
                _col("counts", "Neurons", "counts", min_w=150),
                _col("map_used", "Map used (per linker)", "map_used",
                     max_w=440),
                _col("cov", "Pool coverage (bodyIds)", "cov", min_w=210),
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
            ui.button("Network (type-level)",
                      on_click=lambda: _deliver_flows(
                          src, tgt, flows, pools, "network", "type", stamp))
            ui.button("Network (linker)",
                      on_click=lambda: _deliver_flows(
                          src, tgt, flows, pools, "network", "linker", stamp))
            ui.button("Export mapping",
                      on_click=lambda: _deliver_pair_csv(
                          src, tgt, flows, pools, stamp))

    def _coverage_panel(src: str, tgt: str, flows, pools: dict) -> None:
        """Bidirectional type-level coverage for ONE dataset pair.

        Rendered at the results' top level (user 2026-09-07: outside
        the dataset-pair card again, with the pair named in the title):
        forward = each queried type's total mapped number (1-to-N
        visible), backward = each receiving type and the sources
        converging on it (N-to-1 visible).  Both tables name their
        coverage columns by DATASET (user 2026-09-09: 'source/target
        side' read as flipped in the backward view).
        """
        from comparison.mapping_visualization import (
            build_type_coverage,
            mapping_pool_key,
        )

        # Coverage columns are named by the FULL dataset key (user
        # 2026-09-09): 'source/target side' read as flipped in the
        # backward view, and abbreviations collide for the two BANC
        # releases (both "BANC").
        pair_pools = {}
        for flow in flows:
            key = mapping_pool_key(
                src, tgt, flow.get("source_type"), flow.get("foreign_type"))
            if key in pools:
                pair_pools[key] = pools[key]
        coverage = build_type_coverage({(src, tgt): flows}, pair_pools)
        forward = coverage.get("forward") or []
        backward = coverage.get("reverse") or []
        if not (forward or backward):
            return
        with ui.expansion(
                f"Type coverage — {src} → {tgt} "
                f"(bidirectional, 1-to-N / N-to-1)",
                icon="swap_vert").classes("w-full"):
            ui.label(
                "Forward — each queried type: its neurons, the "
                "targets it maps to, and how many of its bodyIds "
                "carry the type-level evidence. Coverage shows both the "
                "selected bridge and the union of all independently valid "
                "bridge alternatives (pool of total, with the "
                "share in parentheses; a 1-to-N row's target total is the "
                "summed population of all mapped target types).  States: "
                "'not pooled' = no coverage pool at all, 'not measured' = "
                "the side's coverage index is unavailable, '0 of n "
                "(0.0%)' = measured zero.  No bodyId-to-bodyId pairing "
                "is inferred."
            ).classes("text-caption drocat-muted")
            ui.table(
                columns=[
                    _col("type", "Queried type", "type", max_w=170),
                    _col("dataset", "Dataset", "dataset", max_w=180),
                    _col("count", "Neurons", "count", min_w=90),
                    _col("maps_to", "Maps to", "maps_to", max_w=440),
                    _col("relationship", "Relationship",
                         "relationship", min_w=110),
                    _col("coverage_note", "Coverage interpretation",
                         "coverage_note", max_w=330),
                    _col("query_cov_selected",
                         f"{src} side (bodyIds) — selected bridge",
                         "query_cov_selected", min_w=190),
                    _col("query_cov",
                         f"{src} side (bodyIds) — all-valid union",
                         "query_cov", min_w=205),
                    _col("target_cov_selected",
                         f"{tgt} side (bodyIds) — selected bridge",
                         "target_cov_selected", min_w=190),
                    _col("target_cov",
                         f"{tgt} side (bodyIds) — all-valid union",
                         "target_cov", min_w=205),
                ],
                rows=forward,
            ).classes("w-full")
            ui.label(
                "Backward — each receiving type and the sources that "
                "map onto it: several sources make the N-to-1 "
                "explicit.  Same datasets as above — the coverage "
                "columns show selected-bridge and all-valid-union scopes "
                "by dataset (pool of total, percent "
                "in parentheses), with no bodyId pairing inferred; the "
                "same three states apply.  Split branches and overlapping "
                "bodyId evidence are non-exclusive."
            ).classes("text-caption drocat-muted")
            ui.table(
                columns=[
                    _col("type", "Receiving type", "type", max_w=170),
                    _col("dataset", "Dataset", "dataset", max_w=180),
                    _col("count", "Neurons", "count", min_w=90),
                    _col("mapped_from", "Mapped from", "mapped_from",
                         max_w=440),
                    _col("relationship", "Relationship",
                         "relationship", min_w=110),
                    _col("coverage_note", "Coverage interpretation",
                         "coverage_note", max_w=330),
                    _col("source_cov_selected",
                         f"{src} side (bodyIds) — selected bridge",
                         "source_cov_selected", min_w=190),
                    _col("source_cov",
                         f"{src} side (bodyIds) — all-valid union",
                         "source_cov", min_w=205),
                    _col("target_cov_selected",
                         f"{tgt} side (bodyIds) — selected bridge",
                         "target_cov_selected", min_w=190),
                    _col("target_cov",
                         f"{tgt} side (bodyIds) — all-valid union",
                         "target_cov", min_w=205),
                ],
                rows=backward,
            ).classes("w-full")

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
                    ui.button("Export mapping — all pairs (CSV)",
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
                         "label": "Mapped neurons (received)",
                         "field": "mapped_neurons", "align": "left"},
                        {"name": "issued_neurons",
                         "label": "Queried neurons (issued)",
                         "field": "issued_neurons", "align": "left"},
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
                _coverage_panel(src, tgt, flows, pools)
                with ui.expansion(
                        f"{src} → {tgt} · {len(flows)} mapped pairs",
                        icon="compare_arrows").classes("w-full"):
                    _pair_card(src, tgt, flows, pools)
            if not pair_flows:
                ui.label("No mappings found for the search across the "
                         "selected datasets.").classes(
                    "text-caption drocat-muted")

    def _deliver_combined_csv(stamp: str) -> None:
        from comparison.mapping_visualization import build_bridges_csv

        pair_flows = state["pair_flows"]
        pools = state["pools"]
        # One fixed column set for every pair, so the all-pairs file is a
        # plain header + rows concatenation (the old per-pair pivoted
        # bridge-<column> fields needed a union-of-columns hack to keep
        # uniform field counts).
        parts: List[str] = []
        for (src, tgt), flows in sorted(pair_flows.items()):
            text = build_bridges_csv(flows, pools=pools, extended=True)
            if not text:
                continue
            lines = text.splitlines()
            if parts:
                lines = lines[1:]  # shared header already written
            parts.append("\n".join(lines))
        if not parts:
            ui.notify("Nothing to export.", type="info")
            return
        name = f"mapping_all_pairs_{stamp}.csv"
        ui.download.content("\n".join(parts), name, "text/csv")
        push_banner(
            f"{name} — check your browser's default downloads folder. "
            "Informational only, please double check.")

    def _set_loading(on: bool) -> None:
        """Show/hide the loading notice and freeze the Search button."""
        loading_row.set_visibility(on)
        if on:
            search_btn.disable()
        else:
            search_btn.enable()

    async def _run_click() -> None:
        """Paint the loading notice FIRST, then search off the event loop.

        Warm searches use the existing serialized worker-thread path.  A
        cold mapper load uses the dedicated spawned process so pandas and the
        Python graph build cannot starve the websocket event loop.
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
            from ..neuron_index import (
                is_type_mapper_loaded,
                run_cross_dataset_scan_in_process,
                run_serialized_cross_dataset_scan,
            )
            if not is_type_mapper_loaded():
                outcome = await run_cross_dataset_scan_in_process(
                    _compute_type_mapping, queries, datasets, mode)
            else:
                loop = asyncio.get_running_loop()
                outcome = await loop.run_in_executor(
                    None,
                    lambda: run_serialized_cross_dataset_scan(
                        _compute, queries, datasets, mode),
                )
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
                    if neuron_index_path(d).is_file()]
        return queries, datasets, mode

    def _compute(queries, datasets, mode) -> Dict[str, Any]:
        """Compatibility wrapper for the shared process-safe computation."""
        return _compute_type_mapping(queries, datasets, mode)

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
