"""Interactive figures for the auto type mapping bridges.

Consumes the derivation chains produced by
``CrossDatasetTypeMapper.get_type_bridges`` and renders the type-level
mapping graph through the vispath machinery (the dedicated ``mapping``
left-to-right preset layout): current dataset types on the left, foreign
dataset types in the middle joined by direct per-pair edges, and the
matched query entries on the right converging the foreign types they
cover — so 1-to-1, 1-to-N, and N-to-1 are visible in the node geometry.

The bridge derivation (columns · via) is never a node: it lives on the
pair-edge hover labels next to the per-side neuron counts, and in the
exported CSV. Both builders are pure: they take flows (mapped pairs +
bridge chains + neuron counts) and return figures/graphs; writing files
and opening the browser is the caller's job.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import networkx as nx

from utils.naming_utils import dataset_abbrev


def format_bridge(bridge: List[Dict[str, str]]) -> str:
    """Render one bridge chain with type-identity endpoints, e.g.

    ``PLP080[MCNS·type] → PLP080[MCNS·flywireType] →
    PLP080[FAFB·additional_type(s)] → APDN3[FAFB·type]``.

    Semantics: the endpoints are always the two datasets' ``type``
    identities; a crosswalk hop renders the crosswalk-cell value in its
    true home column (``flywireType`` is a male-cns metadata column);
    an annotation hop renders the CELL entry (the ``via`` — the old
    name listed on the reached type's rows).  Dataset sources use the
    4-char display abbreviations (MCNS/FAFB/BANC/HEMI/MANC).  A
    same-name middle hop renders its identity pass-through.
    Alternative bridges for one pair are kept separate by the caller
    (one hover line or one ``maps via #N`` label each).
    """
    from comparison.cross_dataset_type_mapper import CROSSWALK_COLUMNS, hop_home

    # The source dataset anchors crosswalk-hop attribution (their physical
    # column lives in the source's rows); it is NOT used to re-attribute
    # type hops — those keep their own namespace.
    source_dataset = bridge[0].get("dataset", "") if bridge else ""
    parts = []
    for index, hop in enumerate(bridge):
        # True home: only crosswalk columns physically live in the source
        # dataset's rows; type/annotation hops belong to the namespace
        # recorded on the hop (a type hop is NEVER re-attributed to the
        # source — that duplicated endpoints like DN1pA[MCNS·type] →
        # DN1pA[MCNS·type] on transitive same-name chains).
        home = (source_dataset
                if hop["column"] in CROSSWALK_COLUMNS
                else hop_home(hop, source_dataset))
        abbr = dataset_abbrev(home)
        is_last = index == len(bridge) - 1
        if index == 0 or hop["column"] == "type":
            parts.append(f"{hop['value']}[{abbr}·{hop['column']}]")
            continue
        if is_last and hop.get("via") and hop["column"] != "type":
            # annotation edge: the CELL entry is the via (the old name
            # listed on the reached type's rows); append the reached
            # type identity as the endpoint hop.
            entry = hop.get("via") or hop["value"]
            parts.append(f"{entry}[{abbr}·{hop['column']}]")
            parts.append(f"{hop['value']}[{abbr}·type]")
            continue
        if (is_last and index > 0
                and hop["column"] in CROSSWALK_COLUMNS):
            # crosswalk arrival: the walk landed on the target through the
            # source's crosswalk cell — append the reached type identity
            # so every bridge text ends with a `type` endpoint.
            parts.append(f"{hop['value']}[{abbr}·{hop['column']}]")
            target_abbr = dataset_abbrev(hop.get("dataset", ""))
            parts.append(f"{hop['value']}[{target_abbr}·type]")
            continue
        parts.append(f"{hop['value']}[{abbr}·{hop['column']}]")
    return " → ".join(parts)


def build_bridge_texts(bridges, limit: int = 2) -> List[str]:
    """Formatted bridge chains for one pair (at most ``limit``)."""
    return [format_bridge(b) for b in (bridges or [])[:limit] if b]


def build_mapping_flows(entries, source_dataset: str,
                        source_counts: Optional[Dict[str, int]] = None):
    """Flatten expansion entries into mapped flows with bridge chains.

    ``entries`` are enriched native-match entries (see
    ``ui.neuron_index.collect_native_type_matches`` /
    ``enrich_native_type_matches``).  Each flow is one mapped pair
    (current-dataset type ↔ foreign dataset type) carrying every bridge
    chain that derives it plus the native match origin.

    ``source_counts`` maps current-dataset type names to their real neuron
    counts in the current dataset's index (see
    ``ui.neuron_index.count_types_in_index``). When supplied,
    ``source_count`` is that per-side number; ``foreign_count`` always is
    the foreign type's count in the foreign dataset. Without the map the
    source side falls back to the foreign count.
    """
    counts = source_counts or {}
    flows: List[Dict[str, Any]] = []
    seen_pairs = set()
    for entry in entries:
        foreign = entry.get("dataset", "")
        matched = entry.get("matched_written") or ""

        from comparison.cross_dataset_type_mapper import standardize_bridge

        def _chains_for(bridges_by_target, foreign_type):
            """Chains of one pair, filtered to the flow's foreign type.

            ``get_type_bridges`` returns every chain that REACHES the
            foreign namespace (including renames onward to other
            primaries, e.g. DN1pA → DN2) — only the chains that END at
            this flow's foreign type describe this pair.
            """
            return [
                chain for chains in bridges_by_target.values()
                for chain in (chains or [])
                if chain and chain[-1].get("value") == foreign_type
            ]

        def _flows_for(covered_name, covered_count, matched_origin,
                       bridges_by_target):
            """Emit one flow per local counterpart of a foreign type.

            A foreign type frequently matches twice (as a native type AND
            under a taxonomy label) — the pair flows are identical, so
            only the first occurrence is kept.  Each flow carries ONLY
            its own target's chains (the per-target chain lists must not
            be mixed — pooling and hovers attribute them per pair),
            ordered most-informative first: the chain with the most
            direct (registry) linkers leads, so a same-name pair's
            primary hover is its *Type/annotation VERIFICATION, not the
            bare name-equality chain.
            """
            for target in sorted((bridges_by_target or {}).keys()):
                key = (target, foreign, covered_name)
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                chains = _chains_for(
                    {target: (bridges_by_target or {}).get(target)},
                    covered_name)

                def _order(chain):
                    linkers = standardize_bridge(
                        chain, source_dataset, foreign)
                    direct = sum(1 for l in linkers
                                 if l["kind"] == "linker"
                                 and not l["indirect"])
                    total = len(linkers)
                    # bare name-equality chains sink below ANY linker
                    # chain (see preferred_bridge_chain)
                    return (0 if total else 1, -direct, total, len(chain))

                chains.sort(key=_order)
                flows.append({
                    "source_dataset": source_dataset,
                    "target_dataset": foreign,
                    "source_type": target,
                    "source_count": counts.get(target),
                    "foreign_type": covered_name,
                    "foreign_count": covered_count,
                    "matched_origin": matched_origin,
                    "bridges": chains,
                })

        for cand in (entry.get("types_all") or entry.get("types", [])):
            _flows_for(
                cand["name"], cand.get("count"),
                f"type · '{matched}'" if matched else "type",
                cand.get("bridges_by_target") or {})
        for label in (entry.get("labels_all") or entry.get("labels", [])):
            matched_origin = (
                f"{label['column']} · '{label.get('matched_written') or label['label']}'"
            )
            for covered in (label.get("covered_all")
                            or label.get("types", [])):
                _flows_for(
                    covered["name"], covered.get("count"), matched_origin,
                    covered.get("bridges_by_target") or {})
    return flows


def build_mapping_network_graph(flows, *, layer_gap: int = 380,
                                row_gap: int = 70) -> "nx.DiGraph":
    """Layered left-to-right type-mapping network, bridges hidden.

    Edges are the per-pair type mappings themselves, so 1-to-1, 1-to-N,
    and N-to-1 are visible in the node geometry.  The matched query entry
    (taxonomy label hits) sits at the rightmost layer and converges the
    foreign types it covers.  The bridge derivation (columns · via) is
    carried exclusively on the pair edges (``bridge_texts`` attribute)
    together with the per-side neuron counts — never as a node.

    Layer 0 = current dataset types, layer 1 = foreign dataset types,
    layer 2 = matched query entries.  Pair-edge weight is the source
    side's neuron count when known (falls back to the foreign count);
    coverage-edge weight is the foreign count, summing to the entry
    label's total.
    """
    graph = nx.DiGraph()
    ordered = sorted(
        flows,
        key=lambda f: -(f.get("foreign_count") or f.get("source_count") or 0),
    )
    entry_cover: Dict[str, Dict[str, Any]] = {}
    for flow in ordered:
        foreign_count = int(flow.get("foreign_count") or 0)
        source_count = int(flow.get("source_count") or 0)
        origin_column, _, origin_value = flow.get("matched_origin", "").partition(" · ")
        origin_value = origin_value.strip("'")

        src_id = f"0|{flow.get('source_dataset', '')}|{flow.get('source_type', '')}"
        tgt_id = f"1|{flow.get('target_dataset', '')}|{flow.get('foreign_type', '')}"

        pair_weight = source_count or max(1, foreign_count)
        graph.add_node(src_id, node_type="source",
                       label=flow.get("source_type", ""),
                       title=(f"{flow.get('source_type', '')} · "
                              f"{flow.get('source_dataset', '')} "
                              f"({pair_weight} neurons)"),
                       position={"x": 0, "y": 0})
        graph.add_node(tgt_id, node_type="target",
                       label=flow.get("foreign_type", ""),
                       title=(f"{flow.get('foreign_type', '')} · "
                              f"{flow.get('target_dataset', '')} "
                              f"({foreign_count or pair_weight} neurons)"),
                       position={"x": 1, "y": 0})
        # the pair edge IS the mapping; its hover label carries the bridge
        # derivation and both sides' neuron counts
        bridge_texts = build_bridge_texts(flow.get("bridges"))
        if graph.has_edge(src_id, tgt_id):
            existing = graph[src_id][tgt_id]
            existing["weight"] += pair_weight
            existing["bridge_texts"] = list(dict.fromkeys(
                existing.get("bridge_texts", []) + bridge_texts))
        else:
            graph.add_edge(src_id, tgt_id, weight=pair_weight,
                           title=f"{pair_weight} neurons",
                           bridge_texts=bridge_texts,
                           source_count=source_count,
                           foreign_count=foreign_count,
                           source_dataset=flow.get("source_dataset", ""),
                           target_dataset=flow.get("target_dataset", ""))

        # the query hit converges the foreign types it covers (rightmost)
        if origin_column and origin_column != "type":
            entry_id = (f"2|{flow.get('target_dataset', '')}|"
                        f"{flow.get('matched_origin', '') or 'matched'}")
            graph.add_node(entry_id, node_type="entry",
                           label=flow.get("matched_origin", "") or "matched",
                           position={"x": 2, "y": 0})
            cover = entry_cover.setdefault(
                entry_id, {"types": set(), "neurons": 0})
            cover["types"].add(flow.get("foreign_type", ""))
            cover["neurons"] += foreign_count or pair_weight
            if graph.has_edge(tgt_id, entry_id):
                graph[tgt_id][entry_id]["weight"] += foreign_count or pair_weight
            else:
                graph.add_edge(tgt_id, entry_id,
                               weight=foreign_count or pair_weight,
                               title=f"{foreign_count or pair_weight} neurons",
                               bridge_texts=[],
                               target_dataset=flow.get("target_dataset", ""))

    # Entry hover titles name what they cover (coverage, not derivation).
    for entry_id, cover in entry_cover.items():
        graph.nodes[entry_id]["title"] = (
            f"{graph.nodes[entry_id].get('label', '')} — covers "
            f"{len(cover['types'])} types, {cover['neurons']:,} neurons"
        )

    _assign_layered_positions(graph, layer_gap=layer_gap, row_gap=row_gap)
    return graph


def _layered_y_positions(graph, node_layer: Dict[str, int],
                         row_gap: int) -> Dict[str, float]:
    """Barycenter ordering + even y spacing for an explicit node→layer map.

    Two alternating sweeps order each layer by the mean row of its
    neighbours in the adjacent layers (ties keep heavier nodes first), so
    sources sharing a target end up adjacent and edge crossings are
    minimized; y positions come out evenly spaced per layer.
    """
    max_layer = max(node_layer.values(), default=0)
    order: Dict[int, List[str]] = {}
    for node, layer in node_layer.items():
        order.setdefault(layer, []).append(node)

    def weight_of(node: str) -> int:
        total = sum(d.get("weight", 0)
                    for _, _, d in graph.in_edges(node, data=True))
        total += sum(d.get("weight", 0)
                     for _, _, d in graph.out_edges(node, data=True))
        return total

    def neighbors_in(node: str, layer: int) -> List[str]:
        return [n for n in set(graph.predecessors(node))
                | set(graph.successors(node))
                if node_layer.get(n) == layer]

    def barycenter(node: str, layer: int) -> float:
        rows = [order[other].index(nbr)
                for other in (layer - 1, layer + 1) if other in order
                for nbr in neighbors_in(node, other) if nbr in order[other]]
        return sum(rows) / len(rows) if rows else float("inf")

    for _sweep in range(2):
        for layer in (list(range(max_layer + 1))
                      + list(range(max_layer, -1, -1))):
            order[layer].sort(
                key=lambda n: (barycenter(n, layer), -weight_of(n)))

    positions: Dict[str, float] = {}
    for layer, nodes in order.items():
        for index, node in enumerate(nodes):
            positions[node] = -index * row_gap
    return positions


def _assign_layered_positions(graph, *, layer_gap: int, row_gap: int) -> None:
    """Uniform per-layer spacing with barycenter ordering.

    Layers derive from the leading ``layer|`` prefix of the node ids.
    """
    node_layer = {n: int(str(n).split("|", 1)[0]) for n in graph.nodes}
    ys = _layered_y_positions(graph, node_layer, row_gap)
    for node, layer in node_layer.items():
        graph.nodes[node]["position"] = {
            "x": layer * layer_gap, "y": ys[node]
        }


def render_mapping_network_html(flows, *,
                                title: str = "Auto type mapping bridges"
                                ) -> Optional[str]:
    """Render the type-level mapping network to an HTML string.

    Uses the vispath machinery with the dedicated layered ``mapping``
    layout (types left, foreign types middle, query entries right,
    physics off, nodes draggable).  Nothing is written to the
    repository — the temp render file lives in the system temp dir.
    Returns the HTML text, or None when vispath is unavailable.
    """
    graph = build_mapping_network_graph(flows)
    if not graph.nodes:
        return None

    # Hover source info per node: {dataset code: "title"} — plain
    # type · dataset · count titles; the bridge derivation stays on the
    # pair edges.
    node_dataset_info: Dict[str, Dict[str, str]] = {}
    dataset_legend: Dict[str, str] = {}
    for node, data in graph.nodes(data=True):
        hop_dataset = str(node).split("|")[1] if "|" in str(node) else ""
        # 4-char abbreviation (MCNS/FAFB/...; version suffix on family
        # collision) — one-char codes are ambiguous as datasets grow.
        code = dataset_abbrev(hop_dataset) or "?"
        node_dataset_info[node] = {
            code: str(data.get("title") or data.get("label", ""))
        }
        dataset_legend[code] = hop_dataset

    # Pair-edge hover labels: one maps-via label per bridge chain plus the
    # per-side neuron counts. Keys stay {key:val}-safe (no ';' inside
    # values) for the exported edge-list CSV round trip.
    edge_labels: Dict[tuple, Dict[str, str]] = {}
    for src, tgt, data in graph.edges(data=True):
        labels: Dict[str, str] = {}
        texts = data.get("bridge_texts") or []
        for index, text in enumerate(texts, start=1):
            key = f"maps via #{index}" if len(texts) > 1 else "maps via"
            labels[key] = text
        source_count = data.get("source_count")
        if source_count:
            labels["source neurons"] = (
                f"{source_count} · {data.get('source_dataset', '')}")
        foreign_count = data.get("foreign_count")
        if foreign_count:
            labels["foreign neurons"] = (
                f"{foreign_count} · {data.get('target_dataset', '')}")
        if labels:
            edge_labels[(src, tgt)] = labels

    return _vispath_html(
        graph, edge_labels=edge_labels,
        node_dataset_info=node_dataset_info,
        dataset_legend=dataset_legend)


def write_mapping_network_html(flows, output_path: str, *,
                               open_browser: bool = False,
                               title: str = "Auto type mapping bridges"):
    """Write the interactive Cytoscape network HTML for the mapped types.

    Thin file wrapper around ``render_mapping_network_html`` (kept for
    the analyzer/CLI callers).  Returns the output path, or None when
    vispath is unavailable or there is nothing to draw.
    """
    text = render_mapping_network_html(flows, title=title)
    if text is None:
        return None
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return output_path


def _import_vispath():
    """Import VisualizePath, extending sys.path with the subproject."""
    import os
    import sys

    try:
        from vispath_pkg.vispath import VisualizePath
    except ImportError:
        repo_src = os.path.abspath(os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "..", "vispath-subproject", "src"))
        if repo_src not in sys.path:
            sys.path.append(repo_src)
        try:
            from vispath_pkg.vispath import VisualizePath
        except Exception:
            return None
    return VisualizePath


def _render_mapping_graph(graph, output_path: str, *, open_browser: bool = False,
                          edge_labels: Optional[Dict[tuple, Dict[str, str]]] = None,
                          node_dataset_info: Optional[Dict[str, Dict[str, str]]] = None,
                          dataset_legend: Optional[Dict[str, str]] = None,
                          edge_weight_label: str = "neurons",
                          linker_colors: Optional[Dict[str, str]] = None):
    """Render one mapping DiGraph through the vispath cytoscape renderer.

    Raises ``_VispathUnavailable`` when the vispath package cannot be
    imported.  ``linker_colors`` (column → color) is emitted as a legend
    note so colored linker nodes stay explainable.
    """
    import os

    VisualizePath = _import_vispath()
    if VisualizePath is None:
        raise _VispathUnavailable()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    # VisualizePath.__init__ requires pathway data we do not have; build the
    # instance without it and set only the attributes the cytoscape renderer
    # reads.
    class _RendererVisualizer(VisualizePath):
        """Bare visualizer: every attribute a real __init__ would set falls
        back to its signature default, so the cytoscape renderer never hits
        a missing attribute while the pathway-data loading is skipped."""

        _SAFE_FALLBACKS = {
            "conn_df": None, "showfig": False, "verbose": False,
            "network_layout": "mapping",
            "node_color": ["#5b8cff", "#94a3b8"],
            "edge_color": "#64748b",
            "custom_edge_colors": {}, "custom_node_colors": {},
            "edge_labels": {}, "nt_type": None,
            "generate_empty_network": False, "sheet_name": None,
            "min_edge_width": 1, "max_edge_width": 8,
            "min_font_size": 8, "max_font_size": 16,
            "min_node_size": 8, "max_node_size": 40,
            "edge_width_scale": "log", "edge_width_factor": 1.0,
            "edge_width_log_base": None, "edge_opacity": 0.6,
            "highlight_opacity": 1.0, "highlight_color": "#f59e0b",
            "source_opacity": 1.0, "intermediate_opacity": 1.0,
            "target_opacity": 1.0, "link_color": "rgba(100, 100, 100, 0.5)",
            "straight_reciprocal_edges": False,
            "color_edges_by_nt": False, "color_nodes_by_nt": False,
            "separate_hemispheres": False,
            "hemisphere_desaturate_side": None,
            "hemisphere_desaturate_factor": 0.4,
            "edge_weight_label": "synapses",
        }

        def __getattr__(self, name):
            fallbacks = _RendererVisualizer._SAFE_FALLBACKS
            if name in fallbacks:
                import copy

                return copy.deepcopy(fallbacks[name])
            import inspect

            parameter = inspect.signature(VisualizePath.__init__).parameters.get(name)
            if parameter is not None and parameter.default is not inspect._empty:
                return parameter.default
            raise AttributeError(name)

    visualizer = _RendererVisualizer.__new__(_RendererVisualizer)
    visualizer.path_file = None
    visualizer.output_folder = os.path.dirname(os.path.abspath(output_path))
    visualizer.verbose = False
    visualizer.network_layout = "mapping"
    visualizer.source_color = "#5b8cff"
    visualizer.intermediate_color = "#94a3b8"
    visualizer.target_color = "#22c55e"
    visualizer.node_color = ["#5b8cff", "#94a3b8"]
    visualizer.edge_color = "#64748b"
    visualizer.node_dataset_info = node_dataset_info
    visualizer.dataset_legend = dataset_legend
    visualizer.edge_labels = edge_labels
    # the mapping weights are neuron counts, not synapses
    visualizer.edge_weight_label = "neurons"
    visualizer.linker_colors = linker_colors or {}
    visualizer._plot_cytoscape_network(
        graph, output_path, layout="mapping", open_browser=open_browser
    )
    return output_path


def _vispath_html(graph, *, edge_labels=None, node_dataset_info=None,
                  dataset_legend=None) -> Optional[str]:
    """Render a mapping graph to an HTML string.

    The vispath renderer is path-based, so the render goes to a temp
    file in the system temp dir and is read back — nothing lands in the
    repository (the viewer delivers artifacts as browser downloads).
    Returns None when vispath is unavailable.
    """
    import os
    import tempfile

    with tempfile.TemporaryDirectory(prefix="drocat_mapping_") as tmp:
        path = os.path.join(tmp, "mapping.html")
        try:
            _render_mapping_graph(
                graph, path, open_browser=False,
                edge_labels=edge_labels,
                node_dataset_info=node_dataset_info,
                dataset_legend=dataset_legend,
                edge_weight_label="neurons")
        except _VispathUnavailable:
            return None
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read()


class _VispathUnavailable(ImportError):
    """The vispath package could not be imported."""


def build_bridge_linker_graph(flows, *, source_dataset: str,
                              target_dataset: str,
                              pools: Optional[Dict[tuple, Dict[str, Any]]] = None,
                              layer_gap: int = 380,
                              row_gap: int = 70) -> "nx.DiGraph":
    """Detailed linker graph of the standardized bridges.

    Paths run source types → linker nodes → target types, where each
    linker node is one standardized hop (``standardize_bridge``) colored
    by its matched column (``LINKER_COLORS``); same-name pass hops
    collapse into the edges (no node).  ``pools`` optionally maps
    ``(source_type, foreign_type)`` → ``pool_bridge_body_ids`` results;
    when present, linker-node hover titles carry the pooled bodyId
    counts per side.  Layer 0 = current dataset types, layers 1..k = the
    linkers in registry order, last layer = target types.
    """
    from comparison.cross_dataset_type_mapper import (
        LINKER_COLORS,
        standardize_bridge,
    )

    graph = nx.DiGraph()
    pools = pools or {}
    max_linkers = 0

    def _endpoint(side: int, dataset: str, type_name: str, count) -> str:
        node_id = f"{side}|{dataset}|{type_name}"
        title = (f"{type_name} · {dataset} "
                 f"({count or 0} neurons)")
        graph.add_node(
            node_id,
            node_type="source" if side == 0 else "target",
            label=type_name, title=title,
            position={"x": 0, "y": 0})
        return node_id

    ordered = sorted(
        flows,
        key=lambda f: -(f.get("foreign_count") or f.get("source_count") or 0),
    )
    for flow in ordered:
        source_type = flow.get("source_type", "")
        foreign_type = flow.get("foreign_type", "")
        count = max(1, flow.get("foreign_count")
                    or flow.get("source_count") or 1)
        src_id = _endpoint(0, flow.get("source_dataset", ""), source_type,
                           flow.get("source_count"))
        tgt_id = _endpoint(1, flow.get("target_dataset", ""), foreign_type,
                           flow.get("foreign_count"))
        pool = pools.get((source_type, foreign_type)) or {}
        pool_note = ""
        if pool:
            pool_note = (
                f" — pool: {len(pool.get('source_body_ids', []))} "
                f"bodyIds ({dataset_abbrev(source_dataset)}) / "
                f"{len(pool.get('target_body_ids', []))} "
                f"bodyIds ({dataset_abbrev(target_dataset)})")

        for chain in (flow.get("bridges") or [])[:2]:
            linkers = [l for l in standardize_bridge(
                chain, source_dataset, target_dataset)
                if l.get("kind") == "linker"]
            max_linkers = max(max_linkers, len(linkers))
            chain_text = format_bridge(chain)
            previous = src_id
            for order_index, linker in enumerate(linkers, start=1):
                node_id = f"L|{linker['column']}|{linker['value']}"
                title = (f"{linker['column']} · {linker['value']} "
                         f"[{dataset_abbrev(linker['home'])}]{pool_note}")
                graph.add_node(
                    node_id, node_type="linker",
                    home_dataset=linker.get("home", ""),
                    label=linker["value"], title=title,
                    color=LINKER_COLORS.get(linker["column"], "#94a3b8"),
                    position={"x": order_index, "y": 0})
                if not graph.has_edge(previous, node_id):
                    graph.add_edge(previous, node_id, weight=count,
                                   title=f"{count} neurons",
                                   bridge_texts=[chain_text])
                previous = node_id
            if not graph.has_edge(previous, tgt_id):
                graph.add_edge(previous, tgt_id, weight=count,
                               title=f"{count} neurons", bridge_texts=[])

    if not graph.nodes:
        return graph

    # Layer 0 = source types, layers 1..k = linkers, last layer = targets.
    target_layer = max(1, max_linkers) + 1
    node_layer: Dict[str, int] = {}
    for node, data in graph.nodes(data=True):
        if data.get("node_type") == "linker":
            node_layer[node] = int(data["position"]["x"])
        elif data.get("node_type") == "source":
            node_layer[node] = 0
        else:
            node_layer[node] = target_layer
    ys = _layered_y_positions(graph, node_layer, row_gap)
    for node, layer in node_layer.items():
        graph.nodes[node]["position"] = {
            "x": layer * layer_gap, "y": ys[node]}
    return graph


def render_bridge_linker_html(flows, *, source_dataset: str,
                              target_dataset: str,
                              pools: Optional[Dict[tuple,
                                                   Dict[str, Any]]] = None,
                              title: str = "Standardized bridge linkers"
                              ) -> Optional[str]:
    """Render the standardized linker paths to an HTML string.

    Source types → colored linkers → target types; nothing is written
    to the repository.  Returns the HTML text, or None when vispath is
    unavailable or there is nothing to draw.
    """
    graph = build_bridge_linker_graph(
        flows, source_dataset=source_dataset, target_dataset=target_dataset,
        pools=pools)
    if not graph.nodes:
        return None

    node_dataset_info: Dict[str, Dict[str, str]] = {}
    dataset_legend: Dict[str, str] = {}
    for node, data in graph.nodes(data=True):
        # linker nodes carry their home dataset explicitly (their node id's
        # second segment is the COLUMN — abbreviating that produced the
        # confusing FLYW/ADDI codes); dataset nodes keep the id segment.
        hop_dataset = (data.get("home_dataset")
                       or (str(node).split("|")[1]
                           if "|" in str(node) else ""))
        # 4-char abbreviation (MCNS/FAFB/...; version suffix on family
        # collision) — one-char codes are ambiguous as datasets grow.
        code = dataset_abbrev(hop_dataset) or "?"
        node_dataset_info[node] = {
            code: str(data.get("title") or data.get("label", ""))
        }
        dataset_legend[code] = hop_dataset

    edge_labels: Dict[tuple, Dict[str, str]] = {}
    for src, tgt, data in graph.edges(data=True):
        labels: Dict[str, str] = {}
        for index, text in enumerate(data.get("bridge_texts") or [], start=1):
            key = f"bridge #{index}" if len(data["bridge_texts"]) > 1 else "bridge"
            labels[key] = text
        if labels:
            edge_labels[(src, tgt)] = labels

    return _vispath_html(
        graph, edge_labels=edge_labels,
        node_dataset_info=node_dataset_info,
        dataset_legend=dataset_legend)


def write_bridge_linker_html(flows, output_path: str, *,
                             source_dataset: str, target_dataset: str,
                             pools: Optional[Dict[tuple,
                                                  Dict[str, Any]]] = None,
                             open_browser: bool = False,
                             title: str = "Standardized bridge linkers"):
    """Write the interactive Cytoscape HTML of the standardized linker
    paths.  Thin file wrapper around ``render_bridge_linker_html``.
    Returns the output path, or None when vispath is unavailable or
    there is nothing to draw.
    """
    text = render_bridge_linker_html(
        flows, source_dataset=source_dataset, target_dataset=target_dataset,
        pools=pools, title=title)
    if text is None:
        return None
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return output_path


def build_mapping_type_sankey_figure(flows, *, pools: Optional[Dict[tuple,
                                     Dict[str, Any]]] = None,
                                     max_flows: int = 500):
    """Type-level sankey: source types → target types, no linker bands.

    The categorization-free companion of the linker-band sankey
    (``build_mapping_sankey_figure``): one aggregated ribbon per
    (source type, target type) pair, value = the smaller pooled bodyId
    count (the matched granularity), hover carries the bridge text and
    the per-side pool sizes.  Same cap + CSV-notice as the linker
    version (the vispath default ``edgeN_limit`` of 500); drag-
    adjustable (plotly snap arrangement).
    """
    import plotly.graph_objects as go

    pools = pools or {}
    ordered = sorted(
        flows,
        key=lambda f: -(f.get("foreign_count") or f.get("source_count") or 0),
    )
    overflow = max(0, len(ordered) - max_flows)
    if overflow:
        ordered = ordered[:max_flows]

    labels: List[str] = []
    colors: List[str] = []
    node_index: Dict[str, int] = {}
    link_source: List[int] = []
    link_target: List[int] = []
    link_value: List[int] = []
    link_hover: List[str] = []
    link_seen: Dict[tuple, int] = {}

    def _node(key: str, label: str, color: str) -> int:
        if key not in node_index:
            node_index[key] = len(labels)
            labels.append(label)
            colors.append(color)
        return node_index[key]

    def _link(src: int, tgt: int, value: int, hover: str) -> None:
        key = (src, tgt)
        if key in link_seen:
            index = link_seen[key]
            link_value[index] += value
            if hover not in link_hover[index]:
                link_hover[index] += f"<br>{hover}"
            return
        link_seen[key] = len(link_source)
        link_source.append(src)
        link_target.append(tgt)
        link_value.append(value)
        link_hover.append(hover)

    for flow in ordered:
        source_type = flow.get("source_type", "")
        foreign_type = flow.get("foreign_type", "")
        source_ds = flow.get("source_dataset", "")
        target_ds = flow.get("target_dataset", "")
        count = max(1, flow.get("foreign_count")
                    or flow.get("source_count") or 1)
        pool = pools.get((source_type, foreign_type)) or {}
        src_count = (len(pool.get("source_body_ids", []))
                     if pool.get("source_body_ids") else count)
        tgt_count = (len(pool.get("target_body_ids", []))
                     if pool.get("target_body_ids") else count)
        src_id = _node(
            f"src|{source_ds}|{source_type}",
            f"{source_type} · {dataset_abbrev(source_ds)}", "#5b8cff")
        tgt_id = _node(
            f"tgt|{target_ds}|{foreign_type}",
            f"{foreign_type} · {dataset_abbrev(target_ds)}", "#22c55e")
        texts = [format_bridge(c)
                 for c in (flow.get("bridges") or [])[:2]]
        hover = ("<br>".join(texts) + "<br>"
                 f"pool: {src_count} → {tgt_count} bodyIds")
        _link(src_id, tgt_id, min(src_count, tgt_count) or 1, hover)

    if not labels:
        return None

    note = (f" (+{overflow} more flows — the full mapping is in the CSV "
            f"export)" if overflow else "")
    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(
            pad=15,
            thickness=20,
            label=labels,
            color=colors,
            line=dict(color="black", width=0.5),
        ),
        link=dict(
            source=link_source,
            target=link_target,
            value=link_value,
            hovertemplate="%{customdata}<extra></extra>",
            customdata=link_hover,
        ),
    )])
    fig.update_layout(
        title_text=("Auto type mapping — type-level flows" + note),
        font_size=11,
        height=max(420, min(1800, 60 * len(labels))),
    )
    return fig


def build_mapping_sankey_figure(flows, *, pools: Optional[Dict[tuple,
                                Dict[str, Any]]] = None,
                                max_flows: int = 500):
    """Native sankey (plotly ``go.Sankey``) over the standardized linker
    flows — the "better categorization" companion to the vispath
    type-level network (§9C.3 of the bodyId-bridge plan).

    Bands run source types → the standardized linkers (``flywireType``,
    ``additional_type(s)``, … in chain order) → target types.  Ribbon
    values are the pooled bodyId counts per bridge when ``pools``
    ((source_type, foreign_type) → ``pool_bridge_body_ids`` result) is
    supplied, else the flow's neuron counts.  Linker nodes are colored
    per ``LINKER_COLORS``.  Flows are capped at ``max_flows`` (the
    vispath default ``edgeN_limit`` of 500) for readability; the title
    notes that the full mapping lives in the CSV export.  Nodes stay
    drag-adjustable (plotly snap arrangement).
    """
    import plotly.graph_objects as go
    from comparison.cross_dataset_type_mapper import (
        LINKER_COLORS,
        standardize_bridge,
    )

    pools = pools or {}
    ordered = sorted(
        flows,
        key=lambda f: -(f.get("foreign_count") or f.get("source_count") or 0),
    )
    overflow = max(0, len(ordered) - max_flows)
    if overflow:
        ordered = ordered[:max_flows]

    labels: List[str] = []
    colors: List[str] = []
    node_index: Dict[str, int] = {}

    def _node(key: str, label: str, color: str) -> int:
        if key not in node_index:
            node_index[key] = len(labels)
            labels.append(label)
            colors.append(color)
        return node_index[key]

    link_source: List[int] = []
    link_target: List[int] = []
    link_value: List[int] = []
    link_hover: List[str] = []
    link_seen = {}

    def _link(src: int, tgt: int, value: int, hover: str) -> None:
        key = (src, tgt)
        if key in link_seen:
            index = link_seen[key]
            link_value[index] += value
            if hover not in link_hover[index]:
                link_hover[index] += f"<br>{hover}"
            return
        link_seen[key] = len(link_source)
        link_source.append(src)
        link_target.append(tgt)
        link_value.append(value)
        link_hover.append(hover)

    max_linkers = 0
    for flow in ordered:
        source_type = flow.get("source_type", "")
        foreign_type = flow.get("foreign_type", "")
        source_ds = flow.get("source_dataset", "")
        target_ds = flow.get("target_dataset", "")
        count = max(1, flow.get("foreign_count")
                    or flow.get("source_count") or 1)
        pool = pools.get((source_type, foreign_type)) or {}
        src_count = (len(pool.get("source_body_ids", []))
                     if pool.get("source_body_ids") else count)
        tgt_count = (len(pool.get("target_body_ids", []))
                     if pool.get("target_body_ids") else count)

        src_id = _node(
            f"0|{source_ds}|{source_type}",
            f"{source_type} · {dataset_abbrev(source_ds)}", "#5b8cff")
        tgt_id = _node(
            f"9|{target_ds}|{foreign_type}",
            f"{foreign_type} · {dataset_abbrev(target_ds)}", "#22c55e")

        for chain in (flow.get("bridges") or [])[:2]:
            bridge_text = format_bridge(chain)
            linkers = [l for l in standardize_bridge(
                chain, source_ds, target_ds) if l.get("kind") == "linker"]
            max_linkers = max(max_linkers, len(linkers))
            previous = src_id
            previous_value = src_count
            for order_index, linker in enumerate(linkers, start=1):
                abbr = dataset_abbrev(linker["home"])
                node_id = (f"{order_index}|{linker['home']}|"
                           f"{linker['column']}|{linker['value']}")
                node = _node(
                    node_id,
                    f"{linker['value']} · {abbr} [{linker['column']}]",
                    LINKER_COLORS.get(linker["column"], "#94a3b8"))
                hop_count = tgt_count if order_index == len(linkers) \
                    else src_count
                hover = (f"{bridge_text}<br>"
                         f"pool: {previous_value} → {hop_count} bodyIds")
                _link(previous, node, min(previous_value, hop_count) or 1,
                      hover)
                previous = node
                previous_value = hop_count
            hover = (f"{bridge_text}<br>"
                     f"pool: {previous_value} → {tgt_count} bodyIds")
            _link(previous, tgt_id, tgt_count or 1, hover)

    if not labels:
        return None

    note = (f" (+{overflow} more flows — the full mapping is in the CSV "
            f"export)" if overflow else "")
    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(
            pad=15,
            thickness=20,
            label=labels,
            color=colors,
            line=dict(color="black", width=0.5),
        ),
        link=dict(
            source=link_source,
            target=link_target,
            value=link_value,
            hovertemplate="%{customdata}<extra></extra>",
            customdata=link_hover,
        ),
    )])
    fig.update_layout(
        title_text=("Auto type mapping — standardized linker flows"
                    + note),
        font_size=11,
        height=max(420, min(1800, 60 * len(labels))),
    )
    return fig
