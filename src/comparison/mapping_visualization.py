"""Interactive figures for the auto type mapping bridges.

Consumes the derivation chains produced by
``CrossDatasetTypeMapper.get_type_bridges`` and renders through the
vispath machinery: the type-level network (dagre layout — no custom
layer map; current dataset types on the left, foreign dataset types in
the middle joined by direct per-pair edges, matched query entries on
the right converging the foreign types they cover) and the layered
Sankey (``create_sankey`` backend with the shared interactive control
panel — user-adjustable node/edge colors).

The bridge derivation (columns · via) is never a node: it lives on the
pair-edge hover labels next to the per-side neuron counts, and in the
exported CSV. Both builders are pure: they take flows (mapped pairs +
bridge chains + neuron counts) and return figures/graphs; writing files
and opening the browser is the caller's job.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

from utils.naming_utils import dataset_abbrev

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-dataset node groups (§13): every mapping network view groups and
# colors its nodes BY DATASET — one shared color per dataset — with the
# color-keyed legend rendered by vispath.  The palette is stable across
# artifacts and sessions; unseen dataset codes get a deterministic
# fallback color.
DATASET_GROUP_COLORS = {
    "MCNS": "#2563eb",
    "FAFB": "#16a34a",
    "BANC": "#dc2626",
    "HEMI": "#9333ea",
    "MANC": "#ea580c",
    "FLYW": "#0d9488",
}
_FALLBACK_GROUP_COLORS = ("#0d9488", "#ca8a04", "#7c3aed", "#db2777",
                          "#0284c7", "#65a30d", "#c2410c", "#475569")


def dataset_group_color(code: str) -> str:
    """Stable color for one 4-char dataset code (§13)."""
    code = str(code or "?").upper()
    if code in DATASET_GROUP_COLORS:
        return DATASET_GROUP_COLORS[code]
    index = sum(ord(ch) for ch in code) % len(_FALLBACK_GROUP_COLORS)
    return _FALLBACK_GROUP_COLORS[index]


def mapping_pool_key(source_dataset: str, target_dataset: str,
                     source_type: str, foreign_type: str) -> Tuple[str, str,
                                                                    str, str]:
    """Return the direction- and dataset-scoped key for one bridge pool.

    Type names are not globally unique: ``l-LNv`` occurs in FAFB, BANC and
    other namespaces, and a bidirectional result can contain the same
    ``(source_type, foreign_type)`` tuple in both directions.  Pool state must
    therefore carry both dataset endpoints as well as the type endpoints.
    """
    return (str(source_dataset or ""), str(target_dataset or ""),
            str(source_type or ""), str(foreign_type or ""))


def get_mapping_pool(pools: Optional[Dict[tuple, Dict[str, Any]]],
                     flow: Dict[str, Any]) -> Dict[str, Any]:
    """Look up a flow's bodyId pool, accepting the legacy two-part key.

    The two-part fallback keeps the renderer/API compatible with callers that
    construct small synthetic pool dictionaries directly.  The UI stores the
    four-part key so real multi-pair results cannot cross-contaminate.
    """
    if not pools:
        return {}
    full = mapping_pool_key(
        flow.get("source_dataset", ""), flow.get("target_dataset", ""),
        flow.get("source_type", ""), flow.get("foreign_type", ""))
    pool = pools.get(full)
    if pool is not None:
        return pool
    legacy = (flow.get("source_type", ""), flow.get("foreign_type", ""))
    return pools.get(legacy) or {}


def format_coverage(count: int, total: Optional[int]) -> str:
    """Human-readable one-side coverage, e.g. ``1,655 of 1,683 (98.3%)``.

    Thousands separators plus a one-decimal percentage so ``x of y`` never
    needs mental division.  An unknown total (the side's coverage index was
    unavailable) yields ``not measured`` — distinct from a measured zero,
    which renders ``0 of 168 (0.0%)``.  ``not pooled`` (no pool at all)
    stays a caller decision.
    """
    if total is None:
        return "not measured"
    if total <= 0:
        return f"{count:,} of 0"
    pct = 100.0 * count / total
    return f"{count:,} of {total:,} ({pct:.1f}%)"


_POOL_BASE_FULL = "full population"
_POOL_BASE_UNMEASURED = "unmeasured"


def format_pool_side(code: str, pool: Dict[str, Any], side: str,
                     scope: str = "selected") -> str:
    """Basis-aware one-side coverage cell for the pair card.

    ``side`` is ``'source'`` or ``'target'``.  ``scope`` is ``'selected'``
    (the first supported bridge) or ``'all_valid'`` (the union of every
    independently supported alternative).  Renders the four pool
    states so they can never be confused:

    * measured subset   — ``FB: 1,655 of 1,683 (98.3%)``
    * full population   — ``FB: all 1,683`` (unconstrained side; the
      chain names the type identity, so nothing was filtered)
    * unmeasured        — ``FB: not measured`` (coverage index absent)
    * missing pool      — ``not pooled`` (caller passes an empty pool)
    """
    if not pool:
        return "not pooled"
    prefix = "all_valid_" if scope == "all_valid" else ""
    ids_key = f"{prefix}{side}_body_ids"
    ids = pool.get(ids_key)
    if ids is None:
        ids = pool.get(f"{side}_body_ids") or []
    basis = pool.get(f"{prefix}{side}_basis")
    if basis is None:
        basis = pool.get(f"{side}_basis")
    if basis == _POOL_BASE_UNMEASURED:
        return f"{code}: not measured"
    total = pool.get(f"{prefix}{side}_type_total")
    if total is None:
        total = pool.get(f"{side}_type_total")
    if basis == _POOL_BASE_FULL:
        return f"{code}: all {total:,}" if total else f"{code}: all {len(ids):,}"
    return f"{code}: {format_coverage(len(ids), total)}"


def _dataset_groups(graph) -> List[Dict[str, str]]:
    """Tag every node with its dataset group; return the group list (§13).

    Nodes carry their dataset in the ``home_dataset`` attribute (linker
    nodes) or in a ``|``-joined id — the second segment for the pair /
    composed views (``0|<dataset>|<type>``), the first for the source-map
    ids (``<dataset>|type``).  Sets ``graph.nodes[n]['group']`` to the
    4-char dataset code and returns the vispath ``node_groups`` list
    (``[{name, label, color}]``) sorted by code for stable rendering.
    """

    def _node_dataset(node: str, data: Dict[str, Any]) -> str:
        ds = str(data.get("home_dataset") or "")
        if ds:
            return ds
        parts = str(node).split("|")
        if len(parts) > 2:
            return parts[1]
        if len(parts) == 2:
            return parts[0]
        return ""

    codes: Dict[str, str] = {}
    for node, data in graph.nodes(data=True):
        ds = _node_dataset(node, data)
        code = dataset_abbrev(ds) or "?"
        graph.nodes[node]["group"] = code
        codes.setdefault(code, ds)
    return [{"name": code, "label": code,
             "color": dataset_group_color(code)}
            for code, _ds in sorted(codes.items())]


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
            # type identity as the endpoint hop.  When the composed
            # annotation landing already emitted the identical band
            # (the W1 landing hop carries the same token in the same
            # namespace+column), skip the duplicate entry — the endpoint
            # identity still closes the chain.
            entry = hop.get("via") or hop["value"]
            entry_band = f"{entry}[{abbr}·{hop['column']}]"
            if not parts or parts[-1] != entry_band:
                parts.append(entry_band)
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
    try:
        from comparison.cross_dataset_type_mapper import get_type_mapper

        mapper = get_type_mapper()
    except Exception:
        mapper = None
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
                try:
                    decision = mapper.get_mapping_decision(
                        target, source_dataset, foreign,
                        include_bridges=False)
                except Exception:
                    decision = {}
                mapping_status = decision.get("status", "mapped")
                if mapping_status == "conflict":
                    # Bridge discovery may retain a same-name or other
                    # diagnostic chain, but a scoped vote conflict is not an
                    # accepted mapping and must not contribute flow counts or
                    # coverage.  The caller can surface it as an orphan or a
                    # conflict note instead.
                    continue
                if mapping_status == "unmapped":
                    mapping_status = ("valid_split_evidence"
                                      if len((bridges_by_target or {})) > 1
                                      else "bridged")
                flows.append({
                    "source_dataset": source_dataset,
                    "target_dataset": foreign,
                    "source_type": target,
                    "source_count": counts.get(target),
                    "foreign_type": covered_name,
                    "foreign_count": covered_count,
                    "matched_origin": matched_origin,
                    "bridges": chains,
                    "mapping_status": mapping_status,
                    "mapping_relationship": decision.get("relationship"),
                    "mapping_target_types": decision.get("target_types", []),
                    "mapping_conflicts": decision.get("conflicts", []),
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


def origin_seeded_flows(origin_dataset: str, matched_types, target_dataset: str,
                        *, source_counts: Optional[Dict[str, int]] = None,
                        foreign_counts: Optional[Dict[str, int]] = None,
                        matched_origins: Optional[Dict[str, Any]] = None,
                        max_chains_per_flow: int = 8,
                        max_types: int = 500) -> List[Dict[str, Any]]:
    """Map one ORIGIN dataset's matched types into a target dataset (§12).

    Exact / literal / regex chip matches are resolved per dataset first;
    this seeds the mapping FROM the origin dataset through the existing
    mapper (``get_type_bridges`` — every chain already passes
    ``BRIDGE_SOURCE_MAP`` licensing and ``bridge_is_valid``), emitting
    the SAME flow dicts as ``build_mapping_flows`` so the per-pair
    cards, CSV and composed graph are unchanged downstream.
    """
    from comparison.cross_dataset_type_mapper import get_type_mapper

    mapper = get_type_mapper()
    if mapper is None or not getattr(mapper, "_loaded", False):
        return []
    if (not origin_dataset or not target_dataset
            or origin_dataset == target_dataset):
        return []
    key_o = mapper._get_type_mapping_key(origin_dataset)
    key_t = mapper._get_type_mapping_key(target_dataset)
    if key_o == key_t:  # family versions share one registry namespace
        return []

    counts_o = source_counts or {}
    counts_t = foreign_counts or {}
    types = list(dict.fromkeys(
        str(t).strip() for t in (matched_types or []) if str(t).strip()))
    if max_types and len(types) > max_types:
        types = types[:max_types]

    flows: List[Dict[str, Any]] = []
    seen: set = set()

    def _matched_origin(type_name: str) -> str:
        records = (matched_origins or {}).get(type_name) or []
        if isinstance(records, dict):
            records = [records]
        if records:
            record = sorted(
                records,
                key=lambda item: (
                    str(item.get("column", "")),
                    str(item.get("value", "")),
                ),
            )[0]
            column = str(record.get("column", "") or "").strip()
            value = str(record.get("value", "") or type_name).strip()
            if column and column != "type":
                return f"{column} · '{value}'"
        return f"type · '{type_name}'"

    for type_name in types:
        try:
            chains = mapper.get_type_bridges(
                type_name, origin_dataset, target_dataset, max_bridges=0)
        except Exception:
            # one unmappable type must not sink the sweep, but the
            # failure is logged, not silent
            logger.warning(
                "origin_seeded_flows: bridges for %r (%s → %s) failed",
                type_name, origin_dataset, target_dataset, exc_info=True)
            chains = []
        by_end: Dict[str, List] = {}
        for chain in chains or []:
            if not chain:
                continue
            end = str(chain[-1].get("value", ""))
            if end:
                by_end.setdefault(end, []).append(chain)
        for foreign_type in sorted(by_end):
            pair = (type_name, foreign_type)
            if pair in seen:
                continue
            seen.add(pair)
            try:
                decision = mapper.get_mapping_decision(
                    type_name, origin_dataset, target_dataset,
                    include_bridges=False)
            except Exception:
                decision = {}
            mapping_status = decision.get("status", "mapped")
            if mapping_status == "conflict":
                # Keep unresolved BANC/annotation conflicts out of accepted
                # flow and coverage totals even when a bare same-name chain
                # is also discoverable.
                continue
            if mapping_status == "unmapped":
                mapping_status = ("valid_split_evidence"
                                  if len(by_end) > 1 else "bridged")
            flows.append({
                "source_dataset": origin_dataset,
                "target_dataset": target_dataset,
                "source_type": type_name,
                "foreign_type": foreign_type,
                "source_count": int(counts_o.get(type_name) or 0),
                "foreign_count": int(counts_t.get(foreign_type) or 0),
                "matched_origin": _matched_origin(type_name),
                "bridges": by_end[foreign_type][:max_chains_per_flow],
                # Type-level policy is kept beside the evidence chains so
                # renderers can distinguish a valid split from a rejected
                # vote conflict without inspecting an unscoped type name.
                "mapping_status": mapping_status,
                "mapping_relationship": decision.get("relationship"),
                "mapping_target_types": decision.get("target_types", []),
                "mapping_conflicts": decision.get("conflicts", []),
            })
    return flows


def dedupe_mirrored_pairs(pair_flows: Dict[tuple, list],
                          origin_datasets=()) -> Dict[tuple, list]:
    """Collapse mirrored mapping results to ONE canonical edge (§12).

    Dedupe granularity is the unordered TYPE PAIR inside an unordered
    dataset pair: the two derivation directions of one equivalence
    collapse to a single flow — origin-source direction preferred, then
    more verified chains, then the larger source-side neuron count, then
    the longer bridge evidence (lexicographic ids last, for
    determinism).  Type pairs that only the REVERSE direction found are
    KEPT: an asymmetric route must not vanish with a dropped
    dataset-pair direction (crosswalk evidence reads male-cns → FAFB,
    but the FAFB↔male-cns equivalence must still render from the other
    side).

    The surviving edge direction is the DERIVATION direction (where the
    evidence chain reads); the graph shows the equivalence.
    """
    from comparison.cross_dataset_type_mapper import standardize_bridge

    origins = set(origin_datasets or ())

    def _verified(flow) -> int:
        src = flow.get("source_dataset", "")
        tgt = flow.get("target_dataset", "")
        count = 0
        for chain in (flow.get("bridges") or [])[:2]:
            try:
                if any(l.get("kind") == "linker" for l in
                       standardize_bridge(chain, src, tgt)):
                    count += 1
            except Exception:
                continue
        return count

    best: Dict[tuple, tuple] = {}
    for (src_ds, tgt_ds), flows in (pair_flows or {}).items():
        for flow in flows or []:
            s_type = str(flow.get("source_type", ""))
            t_type = str(flow.get("foreign_type", ""))
            if not s_type or not t_type:
                continue
            key = (tuple(sorted((src_ds, tgt_ds))),
                   tuple(sorted((s_type, t_type))))
            score = (
                1 if src_ds in origins else 0,
                _verified(flow),
                int(flow.get("source_count") or 0),
                len(flow.get("bridges") or []),
                src_ds, s_type,
            )
            prev = best.get(key)
            if prev is None or score > prev[0]:
                best[key] = (score, src_ds, tgt_ds, flow)

    result: Dict[tuple, list] = {}
    for _key, (_score, src_ds, tgt_ds, flow) in best.items():
        result.setdefault((src_ds, tgt_ds), []).append(flow)
    for flows in result.values():
        flows.sort(key=lambda f: (f.get("source_type", ""),
                                  f.get("foreign_type", "")))
    return result


def _endpoint_pool_counts(pools, *, include_datasets: bool = False,
                          scope: str = "all_valid") -> tuple:
    """Pooled bodyId counts per type on each side (user report).

    Returns ``(source_counts, target_counts)`` — type name → the UNIQUE
    pooled bodyId count for that type on that side (the UNION across
    every pair the type participates in).  An N-to-1 target pools a
    DIFFERENT subset of its bodyIds per counterpart — male-cns CL125 /
    SLP249 (4 each) and PLP080 / SLP250 (2 each) bridge onto disjoint
    subsets of FAFB APDN3's 12 bodyIds — so the old max-across-pairs
    aggregation reported the largest single subset ("pool 4 bodyIds"
    next to "12 neurons"); the union reports what the mapping actually
    reaches.  Nodes carry this on their hover next to the index neuron
    count, because the bodyId count is what the mapped granularity
    actually rests on.
    """
    src_ids: Dict[Any, set] = {}
    tgt_ids: Dict[Any, set] = {}
    for key, pool in (pools or {}).items():
        if len(key) == 4:
            s_ds, t_ds, s_type, f_type = key
        else:
            s_ds, t_ds = "", ""
            s_type, f_type = key
        source_key = ((s_ds, s_type) if include_datasets
                      else s_type)
        target_key = ((t_ds, f_type) if include_datasets
                      else f_type)
        # Node hovers describe the complete independently supported evidence
        # for the type.  Pair-edge/ribbon weights remain selected-chain
        # values via ``pair_flow_weight``; keeping these scopes separate is
        # what makes an MCNS split visible as 6/6 without inflating each
        # branch edge.
        if scope == "selected":
            source_values = pool.get("source_body_ids")
            target_values = pool.get("target_body_ids")
        else:
            source_values = pool.get("all_valid_source_body_ids")
            if source_values is None:
                source_values = pool.get("source_body_ids")
            target_values = pool.get("all_valid_target_body_ids")
            if target_values is None:
                target_values = pool.get("target_body_ids")
        if source_values:
            src_ids.setdefault(source_key, set()).update(
                str(b) for b in source_values)
        if target_values:
            tgt_ids.setdefault(target_key, set()).update(
                str(b) for b in target_values)
    return ({t: len(v) for t, v in src_ids.items()},
            {t: len(v) for t, v in tgt_ids.items()})


def pair_flow_weight(flow, pool: Optional[Dict[str, Any]] = None) -> int:
    """The per-pair coverage weight — ONE formula for every artifact.

    The Sankey ribbon, the network pair edge, the linker-path edges and
    the composed-graph edges all render the same type-level bridge, so they
    must all carry the same number.  Per side: the independent pooled bodyId
    count when a coverage pool exists, else that side's neuron count (a side
    with no data falls back to the other side's), collapsed by min.  This is
    a conservative visual coverage weight, not a count of matched bodyId
    pairs; no bodyId-to-bodyId correspondence is inferred.
    """
    source_count = int(flow.get("source_count") or 0)
    foreign_count = int(flow.get("foreign_count") or 0)
    pool = pool or {}
    src_side = (len(pool.get("source_body_ids") or [])
                or source_count or foreign_count or 1)
    tgt_side = (len(pool.get("target_body_ids") or [])
                or foreign_count or source_count or 1)
    return max(1, min(src_side, tgt_side))


def _pool_title_suffix(count: int, selected_count: Optional[int] = None) -> str:
    if not count:
        return ""
    text = f" · pool {count:,} bodyIds (all-valid union"
    if selected_count is not None and selected_count != count:
        text += f"; selected bridge {selected_count:,}"
    else:
        text += "; selected bridge"
    return text + ")"


def build_mapping_network_graph(flows, *,
                                pools: Optional[Dict[tuple,
                                                     Dict[str, Any]]] = None,
                                orphans: Optional[List[Dict[str, Any]]] = None
                                ) -> "nx.DiGraph":
    """Layered left-to-right type-mapping network, bridges hidden.

    Edges are the per-pair type mappings themselves, so 1-to-1, 1-to-N,
    and N-to-1 are visible in the node geometry.  The matched query entry
    (taxonomy label hits) sits at the rightmost layer and converges the
    foreign types it covers.  The bridge derivation (columns · via) is
    carried exclusively on the pair edges (``bridge_texts`` attribute)
    together with the per-side neuron counts — never as a node.

    Layer 0 = current dataset types, layer 1 = foreign dataset types,
    layer 2 = matched query entries.  Pair-edge weight is the shared
    ``pair_flow_weight`` (independent endpoint coverage / min of the two
    sides) — the same number the Sankey ribbon draws for the type bridge;
    coverage-edge
    weight is the foreign count, summing to the entry label's total.
    With ``pools``, node hovers also carry the pooled bodyId count of
    the type on its side.

    ``orphans`` (W3) are queried types with NO mapped counterpart in a
    target dataset: ``{'dataset', 'type', 'count', 'target'}``.  Each is
    added as an ISOLATED node (no edges) titled with the missing-target
    note, so an unmapped type stays visible instead of silently
    disappearing.
    """
    pools = pools or {}
    dataset_scoped = any(len(key) == 4 for key in pools)
    src_pool_ids, tgt_pool_ids = _endpoint_pool_counts(
        pools, include_datasets=dataset_scoped)
    src_selected_pool_ids, tgt_selected_pool_ids = _endpoint_pool_counts(
        pools, include_datasets=dataset_scoped, scope="selected")
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

        # the pair edge carries the SHARED per-pair flow weight (the same
        # number the Sankey ribbon draws) — never the source type's whole
        # count duplicated onto every edge it participates in
        pool = get_mapping_pool(pools, flow)
        pair_weight = pair_flow_weight(flow, pool)
        graph.add_node(src_id, node_type="source",
                       label=flow.get("source_type", ""),
                       title=(f"{flow.get('source_type', '')} · "
                              f"{flow.get('source_dataset', '')} "
                              f"({source_count or pair_weight} neurons)"
                              + _pool_title_suffix(
                                  src_pool_ids.get((
                                      flow.get("source_dataset", ""),
                                      flow.get("source_type", "")), 0)
                                  if dataset_scoped else
                                  src_pool_ids.get(
                                      flow.get("source_type", ""), 0),
                                  src_selected_pool_ids.get((
                                      flow.get("source_dataset", ""),
                                      flow.get("source_type", "")), 0)
                                  if dataset_scoped else
                                  src_selected_pool_ids.get(
                                      flow.get("source_type", ""), 0))))
        graph.add_node(tgt_id, node_type="target",
                       label=flow.get("foreign_type", ""),
                       title=(f"{flow.get('foreign_type', '')} · "
                              f"{flow.get('target_dataset', '')} "
                              f"({foreign_count or pair_weight} neurons)"
                              + _pool_title_suffix(
                                  tgt_pool_ids.get((
                                      flow.get("target_dataset", ""),
                                      flow.get("foreign_type", "")), 0)
                                  if dataset_scoped else
                                  tgt_pool_ids.get(
                                      flow.get("foreign_type", ""), 0),
                                  tgt_selected_pool_ids.get((
                                      flow.get("target_dataset", ""),
                                      flow.get("foreign_type", "")), 0)
                                  if dataset_scoped else
                                  tgt_selected_pool_ids.get(
                                      flow.get("foreign_type", ""), 0))))
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
                           label=flow.get("matched_origin", "") or "matched")
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

    # W3 orphans: queried types with no mapped counterpart in a target
    # dataset stay VISIBLE as isolated nodes (degree 0) with the
    # missing-target note on their hover — never silently dropped.
    for orphan in (orphans or []):
        ds = str(orphan.get("dataset", ""))
        otype = str(orphan.get("type", ""))
        target = str(orphan.get("target", ""))
        if not ds or not otype:
            continue
        node = f"{otype}|{ds}"
        count = int(orphan.get("count") or 0)
        target_code = dataset_abbrev(target) or target
        if node in graph:
            graph.nodes[node]["orphan"] = True
            graph.nodes[node]["title"] = (
                f"{otype} — no mapped counterpart in "
                f"{target_code} ({count:,} neurons)")
        else:
            graph.add_node(node, label=otype, orphan=True, neurons=count,
                           title=(f"{otype} — no mapped counterpart in "
                                  f"{target_code} ({count:,} neurons)"))

    # Entry hover titles name what they cover (coverage, not derivation).
    for entry_id, cover in entry_cover.items():
        graph.nodes[entry_id]["title"] = (
            f"{graph.nodes[entry_id].get('label', '')} — covers "
            f"{len(cover['types'])} types, {cover['neurons']:,} neurons"
        )

    return graph


def render_mapping_network_html(flows, *,
                                title: str = "Auto type mapping bridges",
                                pools: Optional[Dict[tuple,
                                                     Dict[str, Any]]] = None,
                                orphans: Optional[List[Dict[str, Any]]] = None
                                ) -> Optional[str]:
    """Render the type-level mapping network to an HTML string.

    Uses the vispath machinery with the dagre layout (types left,
    foreign types middle, query entries right, nodes draggable).
    ``pools`` adds the pooled bodyId counts to the node hovers.
    ``orphans`` adds isolated unmapped-type nodes (W3).
    Nothing is written to the repository — the temp render file lives
    in the system temp dir.  Returns the HTML text, or None when
    vispath is unavailable.
    """
    graph = build_mapping_network_graph(flows, pools=pools,
                                        orphans=orphans)
    if not graph.nodes:
        return None

    # Hover source info per node: {dataset code: "title"} — plain
    # type · dataset · count titles; the bridge derivation stays on the
    # pair edges.
    node_dataset_info: Dict[str, Dict[str, str]] = {}
    dataset_legend: Dict[str, str] = {}
    ds_counts: Dict[str, int] = {}
    for node, data in graph.nodes(data=True):
        hop_dataset = str(node).split("|")[1] if "|" in str(node) else ""
        # 4-char abbreviation (MCNS/FAFB/...; version suffix on family
        # collision) — one-char codes are ambiguous as datasets grow.
        code = dataset_abbrev(hop_dataset) or "?"
        node_dataset_info[node] = {
            code: str(data.get("title") or data.get("label", ""))
        }
        dataset_legend[code] = hop_dataset
        ds_counts[code] = ds_counts.get(code, 0) + 1
    # §layout unification: header chips render CODE: full (count)
    legend_meta = {code: {"count": count}
                   for code, count in ds_counts.items()}

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
        dataset_legend=dataset_legend,
        dataset_legend_meta=legend_meta,
        node_groups=_dataset_groups(graph), layout="dagre")


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
                          dataset_legend_meta: Optional[Dict[str,
                                                             Dict[str, Any]]] = None,
                          edge_weight_label: str = "neurons",
                          linker_colors: Optional[Dict[str, str]] = None,
                          node_groups: Optional[List[Dict[str, str]]] = None,
                          layout: str = "dagre"):
    """Render one mapping DiGraph through the vispath cytoscape renderer.

    Raises ``_VispathUnavailable`` when the vispath package cannot be
    imported.  ``linker_colors`` (column → color) is emitted as a legend
    note so colored linker nodes stay explainable.  ``layout`` selects
    the cytoscape layout (dagre for the type-level network, mapping for
    the linker paths).
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
            "edge_label_font_size": 9,
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
            "dagre_rank_dir": "TB",
            "node_groups": [],
            "dataset_legend_meta": {},
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
    visualizer.network_layout = layout
    # Mapping artifacts read source → target LEFT-TO-RIGHT (types left,
    # foreign types middle/right); the connectome hierarchy default TB
    # turned every mapping view into a top-down fan (the broken-layout
    # report).
    visualizer.dagre_rank_dir = "LR" if layout == "dagre" else "TB"
    visualizer.source_color = "#5b8cff"
    visualizer.intermediate_color = "#94a3b8"
    visualizer.target_color = "#22c55e"
    visualizer.node_color = ["#5b8cff", "#94a3b8"]
    visualizer.edge_color = "#64748b"
    visualizer.node_dataset_info = node_dataset_info
    visualizer.dataset_legend = dataset_legend
    # §layout unification: per-code {'count', 'color'} — dataset chips
    # render ``CODE: full (count)`` and linker-column chips carry their
    # LINKER_COLORS swatch; the duplicated dynamic group chip for a code
    # present here is skipped by the vispath legend.
    visualizer.dataset_legend_meta = dataset_legend_meta or {}
    visualizer.edge_labels = edge_labels
    # the mapping weights are neuron counts, not synapses
    visualizer.edge_weight_label = "neurons"
    visualizer.linker_colors = linker_colors or {}
    # declared per-dataset node groups (§13) — drive the group buttons,
    # the color/opacity dropdown and the color-keyed legend
    visualizer.node_groups = list(node_groups or [])
    visualizer._plot_cytoscape_network(
        graph, output_path, layout=layout, open_browser=open_browser
    )
    return output_path


def _vispath_html(graph, *, edge_labels=None, node_dataset_info=None,
                  dataset_legend=None, dataset_legend_meta=None,
                  node_groups=None,
                  layout: str = "dagre") -> Optional[str]:
    """Render a mapping graph to an HTML string.

    The vispath renderer is path-based, so the render goes to a temp
    file in the system temp dir and is read back — nothing lands in the
    repository (the viewer delivers artifacts as browser downloads).
    ``layout`` selects the cytoscape layout (dagre positions the
    type-mapping network natively — no custom layer map needed).
    ``node_groups`` declares the per-dataset groups (§13).  Returns None
    when vispath is unavailable.
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
                dataset_legend_meta=dataset_legend_meta,
                node_groups=node_groups,
                edge_weight_label="neurons", layout=layout)
        except _VispathUnavailable:
            return None
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read()


class _VispathUnavailable(ImportError):
    """The vispath package could not be imported."""


def _layered_y_positions(graph, node_layer: Dict[str, int],
                         row_gap: int) -> Dict[str, float]:
    """Barycenter ordering + even y spacing for an explicit node→layer map.

    Used by the linker-path view (which keeps the ``mapping`` preset
    layout); the type-level network positions itself with dagre instead.
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


def build_bridge_linker_graph(flows, *, source_dataset: str,
                              target_dataset: str,
                              pools: Optional[Dict[tuple, Dict[str, Any]]] = None,
                              layer_gap: int = 380,
                              row_gap: int = 70) -> "nx.DiGraph":
    """Detailed linker graph of the standardized bridges.

    Paths run source types → linker nodes → target types, where each
    linker node is one standardized hop (``standardize_bridge``) colored
    by its matched column (``LINKER_COLORS``); same-name pass hops
    collapse into the edges (no node).  ``pools`` optionally maps the
    direction- and dataset-scoped
    ``(source_dataset, target_dataset, source_type, foreign_type)`` key to
    ``pool_bridge_body_ids`` results; the legacy two-part key is also
    accepted for programmatic callers;
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
    dataset_scoped = any(len(key) == 4 for key in pools)
    src_pool_ids, tgt_pool_ids = _endpoint_pool_counts(
        pools, include_datasets=dataset_scoped)
    src_selected_pool_ids, tgt_selected_pool_ids = _endpoint_pool_counts(
        pools, include_datasets=dataset_scoped, scope="selected")

    ordered = sorted(
        flows,
        key=lambda f: -(f.get("foreign_count") or f.get("source_count") or 0),
    )

    # §layout (user 2026-09-06): ONE COLUMN PER BRIDGE LINKER COLUMN —
    # never per-chain hop order.  The canonical column order is the first
    # appearance walking the chains source→target (chains are canonical,
    # so MCNS↔FAFB lays out flywireType then additional_type(s): the
    # four-column MCNS type → flywireType → additional_type(s) → FAFB
    # type shape).  Layering by per-chain hop order instead let a
    # 1-linker chain place its additional_type(s) node in the SAME
    # column as another chain's flywireType nodes, and a shared linker
    # node's position was overwritten by whichever chain processed last.
    observed_columns: List[str] = []
    for flow in ordered:
        pool = get_mapping_pool(pools, flow)
        chains = (pool.get("valid_chains")
                  if pool.get("valid_chains") is not None
                  else (flow.get("bridges") or [])[:2])
        for chain in chains:
            for linker in (l for l in standardize_bridge(
                    chain, source_dataset, target_dataset)
                    if l.get("kind") == "linker"):
                if linker["column"] not in observed_columns:
                    observed_columns.append(linker["column"])
    from comparison.cross_dataset_type_mapper import BRIDGE_STANDARD
    registry = (
        BRIDGE_STANDARD.get((source_dataset, target_dataset))
        or BRIDGE_STANDARD.get((target_dataset, source_dataset))
        or ()
    )
    column_order = [
        column for column, _home in registry if column in observed_columns
    ]
    column_order.extend(
        column for column in observed_columns if column not in column_order
    )
    column_layer = {column: index + 1
                    for index, column in enumerate(column_order)}

    def _endpoint(side: int, dataset: str, type_name: str, count) -> str:
        node_id = f"{side}|{dataset}|{type_name}"
        all_counts = src_pool_ids if side == 0 else tgt_pool_ids
        selected_counts = (src_selected_pool_ids
                           if side == 0 else tgt_selected_pool_ids)
        key = (dataset, type_name) if dataset_scoped else type_name
        title = (f"{type_name} · {dataset} "
                 f"({count or 0} neurons)"
                 + _pool_title_suffix(
                     all_counts.get(key, 0), selected_counts.get(key, 0)))
        graph.add_node(
            node_id,
            node_type="source" if side == 0 else "target",
            label=type_name, title=title,
            position={"x": 0, "y": 0})
        return node_id

    for flow in ordered:
        source_type = flow.get("source_type", "")
        foreign_type = flow.get("foreign_type", "")
        # the shared per-pair flow weight — the same number the Sankey
        # ribbon and the type-level network edge draw for this pair
        pool = get_mapping_pool(pools, flow)
        count = pair_flow_weight(flow, pool)
        src_id = _endpoint(0, flow.get("source_dataset", ""), source_type,
                           flow.get("source_count"))
        tgt_id = _endpoint(1, flow.get("target_dataset", ""), foreign_type,
                           flow.get("foreign_count"))
        selected_linkers = {
            (str(linker.get("column", "")),
             str(linker.get("value", "")),
             str(linker.get("home", ""))): linker
            for linker in (pool.get("per_linker") or [])
        }
        all_valid_linkers = {
            (str(linker.get("column", "")),
             str(linker.get("value", "")),
             str(linker.get("home", ""))): linker
            for linker in (pool.get("all_valid_per_linker") or [])
        }
        chains = (pool.get("valid_chains")
                  if pool.get("valid_chains") is not None
                  else (flow.get("bridges") or [])[:2])
        for chain in chains:
            linkers = [l for l in standardize_bridge(
                chain, source_dataset, target_dataset)
                if l.get("kind") == "linker"]
            chain_text = format_bridge(chain)
            previous = src_id
            for linker in linkers:
                node_id = f"L|{linker['column']}|{linker['value']}"
                marker = (str(linker.get("column", "")),
                          str(linker.get("value", "")),
                          str(linker.get("home", "")))
                selected_entry = selected_linkers.get(marker)
                all_valid_entry = all_valid_linkers.get(marker)
                own_pool_note = ""
                if selected_entry is not None:
                    own_pool_note = (
                        f" — selected linker pool: "
                        f"{len(selected_entry.get('body_ids') or []):,} "
                        f"bodyIds ({dataset_abbrev(linker.get('home', ''))})")
                elif all_valid_entry is not None:
                    own_pool_note = (
                        f" — valid alternative linker pool: "
                        f"{len(all_valid_entry.get('body_ids') or []):,} "
                        f"bodyIds ({dataset_abbrev(linker.get('home', ''))})")
                if (all_valid_entry is not None and selected_entry is not None
                        and len(all_valid_entry.get('body_ids') or []) !=
                        len(selected_entry.get('body_ids') or [])):
                    own_pool_note += (
                        f"; all-valid union "
                        f"{len(all_valid_entry.get('body_ids') or []):,}")
                title = (f"{linker['column']} · {linker['value']} "
                         f"[{dataset_abbrev(linker['home'])}]"
                         f"{own_pool_note}")
                graph.add_node(
                    node_id, node_type="linker",
                    home_dataset=linker.get("home", ""),
                    label=linker["value"], title=title,
                    color=LINKER_COLORS.get(linker["column"], "#94a3b8"),
                    position={"x": column_layer[linker["column"]], "y": 0})
                if node_id == previous:
                    continue  # self-alias hops fold onto the same node
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

    # Layer 0 = source types, layers 1..k = the linker COLUMNS in canonical
    # order, last layer = target types — one column per linker column, no
    # mixing.
    target_layer = len(column_layer) + 1
    node_layer: Dict[str, int] = {}
    linker_homes: Dict[str, str] = {}
    for node, data in graph.nodes(data=True):
        if data.get("node_type") == "linker":
            node_layer[node] = column_layer[
                str(node).split("|")[1]]
            linker_homes.setdefault(str(node).split("|")[1],
                                    data.get("home_dataset", ""))
        elif data.get("node_type") == "source":
            node_layer[node] = 0
        else:
            node_layer[node] = target_layer
    ys = _layered_y_positions(graph, node_layer, row_gap)
    for node, layer in node_layer.items():
        graph.nodes[node]["position"] = {
            "x": layer * layer_gap, "y": ys[node]}
    graph.graph["linker_columns"] = column_order
    graph.graph["linker_homes"] = linker_homes
    return graph


_LINKER_LEGEND_DESC = {
    "flywireType": "male-cns → FAFB crosswalk column (fT)",
    "hemibrainType": "male-cns crosswalk column (hT)",
    "mancType": "male-cns crosswalk column (mT)",
    "fafb_cell_type": "BANC curated FAFB label",
    "malecns_cell_type": "BANC curated MCNS label (mct)",
    "hemibrain_cell_type": "BANC curated hemibrain label",
    "manc_cell_type": "BANC curated MANC label",
    "banc_release_crosswalk": "BANC root_626 ↔ root_888 relation",
    "release_alias": "explicit MCNS release-name alias",
    "additional_type(s)": "FAFB additional-name cells (aT)",
    "Alternative Cell Type(s)":
        "BANC Alternative Cell Type(s) cells (ACT)",
}


def _collect_linker_columns(
        flows, pools: Optional[Dict[tuple, Dict[str, Any]]] = None
        ) -> List[tuple]:
    """(column, home-dataset, distinct-value count) in chain order."""
    from comparison.cross_dataset_type_mapper import standardize_bridge

    columns: List[str] = []
    homes: Dict[str, str] = {}
    values: Dict[str, set] = {}
    for flow in flows or []:
        s = flow.get("source_dataset", "")
        t = flow.get("target_dataset", "")
        pool = get_mapping_pool(pools, flow)
        chains = (pool.get("valid_chains")
                  if pool.get("valid_chains") is not None
                  else (flow.get("bridges") or [])[:2])
        for chain in chains:
            for linker in (l for l in standardize_bridge(chain, s, t)
                           if l.get("kind") == "linker"):
                if linker["column"] not in columns:
                    columns.append(linker["column"])
                    homes[linker["column"]] = linker.get("home", "")
                    values[linker["column"]] = set()
                values[linker["column"]].add(linker["value"])
    return [(column, homes[column], len(values[column]))
            for column in columns]


def _linker_legend_html(columns_with_homes: List[tuple]) -> str:
    """Bridge-linker legend in the unified ``NAME: full (count)`` format.

    The Sankey backend has no dataset-legend header, so the linker
    entries render as an injected note — same chip shape as the
    vispath header legend: LINKER_COLORS swatch, bold column name, the
    home dataset's full name, and the distinct linker-value count.
    """
    from comparison.cross_dataset_type_mapper import LINKER_COLORS

    items = []
    for column, home, count in columns_with_homes:
        color = LINKER_COLORS.get(column, "#94a3b8")
        desc = _LINKER_LEGEND_DESC.get(column, f"{column} cells")
        items.append(
            '<span style="display:inline-flex;align-items:center;gap:6px;'
            'margin-right:14px;white-space:nowrap" '
            f'title="{desc}">'
            '<span style="display:inline-block;width:9px;height:9px;'
            'border-radius:50%;background:' + color + ';"></span>'
            f'<span style="font-weight:bold;color:#666">{column}:</span> '
            f'<span style="font-size:11px">{home or "?"}</span>'
            f'<span style="font-size:10px;color:#64748b">({count})</span>'
            '</span>')
    if not items:
        return ""
    return (
        '<div style="position:fixed;bottom:8px;left:8px;z-index:9999;'
        'background:rgba(255,255,255,0.94);border:1px solid #cbd5e1;'
        'border-radius:8px;padding:8px 12px;max-width:96vw;'
        'font:12px/1.6 -apple-system,Segoe UI,sans-serif;color:#0f172a">'
        '<b>Bridge linkers</b> (one column per linker): ' + " ".join(items)
        + "</div>")


def _inject_body_note(html: str, note: str) -> str:
    """Insert a fixed-position note right after the artifact's <body>."""
    import re

    return re.sub(r"<body[^>]*>", lambda m: m.group(0) + note, html, count=1)


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
    ds_counts: Dict[str, int] = {}
    linker_counts: Dict[str, int] = {}
    for node, data in graph.nodes(data=True):
        # linker nodes carry their home dataset explicitly (their node id's
        # second segment is the COLUMN — abbreviating that produced the
        # confusing FLYW/ADDI codes); dataset nodes keep the id segment.
        is_linker = data.get("node_type") == "linker"
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
        if is_linker:
            # §layout: linker columns are legend entries of their own —
            # counted separately so they never inflate a dataset count
            linker_column = str(node).split("|")[1]
            linker_counts[linker_column] = \
                linker_counts.get(linker_column, 0) + 1
        else:
            ds_counts[code] = ds_counts.get(code, 0) + 1

    # §layout unification (user 2026-09-07): the linker columns join the
    # existing MCNS/FAFB header chips in the same CODE: full (count)
    # format, with their LINKER_COLORS swatch; the dynamic group chip for
    # a code present here is skipped by the vispath legend (no duplicate
    # header row).
    from comparison.cross_dataset_type_mapper import LINKER_COLORS

    linker_homes = graph.graph.get("linker_homes", {})
    for column, count in linker_counts.items():
        dataset_legend[column] = linker_homes.get(column, "")
    legend_meta: Dict[str, Dict[str, Any]] = {
        code: {"count": count} for code, count in ds_counts.items()}
    for column, count in linker_counts.items():
        entry = {"count": count}
        color = LINKER_COLORS.get(column)
        if color:
            entry["color"] = color
        legend_meta[column] = entry

    edge_labels: Dict[tuple, Dict[str, str]] = {}
    for src, tgt, data in graph.edges(data=True):
        labels: Dict[str, str] = {}
        for index, text in enumerate(data.get("bridge_texts") or [], start=1):
            key = f"bridge #{index}" if len(data["bridge_texts"]) > 1 else "bridge"
            labels[key] = text
        if labels:
            edge_labels[(src, tgt)] = labels

    html = _vispath_html(
        graph, edge_labels=edge_labels,
        node_dataset_info=node_dataset_info,
        dataset_legend=dataset_legend,
        dataset_legend_meta=legend_meta,
        node_groups=_dataset_groups(graph),
        layout="mapping")
    return html


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


def build_mapping_sankey_paths(flows, *, pools: Optional[Dict[tuple,
                               Dict[str, Any]]] = None,
                               variant: str = "linker",
                               max_flows: int = 500,
                               orphans: Optional[List[Dict[str, Any]]] = None
                               ) -> List[tuple]:
    """Path rows for the vispath-backend mapping Sankey.

    Returns ``[(node_names, hop_weights), ...]`` — one row per rendered
    chain.  ``variant="type"`` aggregates each type bridge into a single
    source→target hop at the conservative endpoint-coverage weight (the
    smaller independent pooled bodyId count); ``variant="linker"`` walks
    the standardized linker bands (``flywireType``, ``additional_type(s)`` …
    in chain order).  Ribbon weights are the independent endpoint coverage
    counts per bridge when
    ``pools`` (direction- and dataset-scoped keys, with a legacy
    ``(source_type, foreign_type)`` fallback) is supplied, else the flow's
    neuron counts.  Flows are
    capped at ``max_flows`` (the vispath default ``edgeN_limit`` of
    500); the full mapping lives in the CSV export.

    ``orphans`` (W3) are unmapped queried types — each renders as a
    single-node row (its own neuron count as the stub weight) with the
    missing-target note, so the Sankey shows what did NOT map.
    """
    from comparison.cross_dataset_type_mapper import standardize_bridge

    pools = pools or {}
    ordered = sorted(
        flows,
        key=lambda f: -(f.get("foreign_count") or f.get("source_count") or 0),
    )
    ordered = ordered[:max_flows]

    rows: List[tuple] = []
    seen_paths = set()
    for flow in ordered:
        source_ds = flow.get("source_dataset", "")
        target_ds = flow.get("target_dataset", "")
        source_type = flow.get("source_type", "")
        foreign_type = flow.get("foreign_type", "")
        # the shared per-pair flow weight — the pooled granularity (min
        # of the two sides), CONSTANT along the whole path: a shared
        # target must not flatten fan-in ribbons to its own (identical)
        # count, and the network edge for this pair draws the SAME number
        pool = get_mapping_pool(pools, flow)
        flow_weight = pair_flow_weight(flow, pool)
        src_name = f"{source_type} · {dataset_abbrev(source_ds)}"
        tgt_name = f"{foreign_type} · {dataset_abbrev(target_ds)}"

        if variant == "type":
            key = (src_name, tgt_name)
            if key not in seen_paths:
                seen_paths.add(key)
                rows.append(([src_name, tgt_name], [flow_weight]))
            continue

        chains = (pool.get("valid_chains")
                  if pool.get("valid_chains") is not None
                  else (flow.get("bridges") or [])[:2])
        for chain in chains:
            linkers = [l for l in standardize_bridge(
                chain, source_ds, target_ds) if l.get("kind") == "linker"]
            names = [src_name]
            for linker in linkers:
                names.append(
                    f"{linker['value']} · {dataset_abbrev(linker['home'])}"
                    f" [{linker['column']}]")
            names.append(tgt_name)
            weights = [flow_weight] * (len(names) - 1)
            key = tuple(names)
            if key not in seen_paths:
                seen_paths.add(key)
                rows.append((names, weights))

    # W3 orphans: single-node rows so unmapped types stay visible in the
    # Sankey (the stub weight is the orphan's own neuron count).
    for orphan in (orphans or []):
        ds = str(orphan.get("dataset", ""))
        otype = str(orphan.get("type", ""))
        target = str(orphan.get("target", ""))
        if not ds or not otype:
            continue
        name = (f"{otype} · {dataset_abbrev(ds)} — no counterpart in "
                f"{dataset_abbrev(target) or target}")
        key = (name,)
        if key not in seen_paths:
            seen_paths.add(key)
            rows.append(([name], [max(1, int(orphan.get("count") or 1))]))
    return rows


def build_type_coverage(pair_flows,
                        pools: Optional[Dict[tuple, Dict[str, Any]]] = None
                        ) -> Dict[str, List[Dict[str, Any]]]:
    """Bidirectional type-level coverage rows for the mapping results.

    Two views over the same mapped pairs (user 2026-09-07):

    * ``forward`` — one row per QUERIED (source) type: its neuron count,
      the target types it maps to, the relationship (``1-to-N`` when the
      query fans out), how many of its OWN bodyIds the mapping reaches
      in total (union over its pairs) and how many target-side bodyIds
      the targets cover.
    * ``reverse`` — one row per RECEIVING (target) type: the source
      types that converge on it — several sources make an N-to-1
      explicit (three FAFB types mapping onto male-cns ``SMP227``) —
      with both sides' coverage.  The UI renders this view under the
      heading "Backward", and both tables name their coverage columns
      by DATASET (``<dataset> side (bodyIds)``) so "backward" is never
      misread as swapping which dataset each column measures.

    Coverage cells are pool-based ``x of y (pct)`` bodyIds counts.  A
    side's denominator is the population of the endpoint types involved:
    neuron indexes carry one row per bodyId, so the per-target totals of
    a 1-to-N row sum exactly the distinct neurons of the mapped target
    types (types are disjoint body sets).  Pairs without pools
    contribute nothing and a row without any pooled pair reads ``not
    pooled``; a side whose pools could not be measured (``source_basis``
    / ``target_basis`` ``unmeasured``) contributes nothing to ITS side
    and reads ``not measured`` — never a fake ``0 of n``.  Pure: no UI,
    no I/O.
    """
    pools = pools or {}
    forward: Dict[tuple, Dict[str, Any]] = {}
    reverse: Dict[tuple, Dict[str, Any]] = {}
    for (src_ds, tgt_ds), flows in (pair_flows or {}).items():
        for flow in flows or []:
            s_type = flow.get("source_type", "")
            f_type = flow.get("foreign_type", "")
            if not s_type or not f_type:
                continue
            s_count = int(flow.get("source_count") or 0)
            f_count = int(flow.get("foreign_count") or 0)
            pool = get_mapping_pool(
                pools,
                {**flow,
                 "source_dataset": flow.get("source_dataset") or src_ds,
                 "target_dataset": flow.get("target_dataset") or tgt_ds})
            s_ids = set(pool.get("source_body_ids") or [])
            t_ids = set(pool.get("target_body_ids") or [])
            s_all_ids = set(
                pool.get("all_valid_source_body_ids")
                if pool.get("all_valid_source_body_ids") is not None
                else s_ids)
            t_all_ids = set(
                pool.get("all_valid_target_body_ids")
                if pool.get("all_valid_target_body_ids") is not None
                else t_ids)
            pooled = bool(pool)
            src_measurable = pooled and pool.get(
                "source_basis") != "unmeasured"
            tgt_measurable = pooled and pool.get(
                "target_basis") != "unmeasured"
            src_all_measurable = pooled and pool.get(
                "all_valid_source_basis", pool.get("source_basis")) != \
                "unmeasured"
            tgt_all_measurable = pooled and pool.get(
                "all_valid_target_basis", pool.get("target_basis")) != \
                "unmeasured"

            fwd = forward.setdefault((src_ds, s_type), {
                "dataset": src_ds, "type": s_type, "count": s_count,
                "targets": {}, "any_pooled": False,
            })
            fwd["count"] = max(fwd["count"], s_count)
            fwd["any_pooled"] = fwd["any_pooled"] or pooled
            entry = fwd["targets"].setdefault(
                (tgt_ds, f_type),
                {"s_ids": set(), "t_ids": set(), "t_total": 0,
                 "all_s_ids": set(), "all_t_ids": set(),
                 "all_s_overlap_ids": set(),
                 "all_t_overlap_ids": set(),
                 "pooled": False, "s_measured": False, "t_measured": False,
                 "all_s_measured": False, "all_t_measured": False})
            entry["s_ids"] |= s_ids
            entry["t_ids"] |= t_ids
            entry["all_s_ids"] |= s_all_ids
            entry["all_t_ids"] |= t_all_ids
            entry["all_s_overlap_ids"] |= set(
                pool.get("all_valid_source_overlap_body_ids") or [])
            entry["all_t_overlap_ids"] |= set(
                pool.get("all_valid_target_overlap_body_ids") or [])
            entry["t_total"] = max(entry["t_total"], f_count)
            entry["pooled"] = entry["pooled"] or pooled
            entry["s_measured"] = entry["s_measured"] or src_measurable
            entry["t_measured"] = entry["t_measured"] or tgt_measurable
            entry["all_s_measured"] = (entry["all_s_measured"]
                                        or src_all_measurable)
            entry["all_t_measured"] = (entry["all_t_measured"]
                                        or tgt_all_measurable)
            rev = reverse.setdefault((tgt_ds, f_type), {
                "dataset": tgt_ds, "type": f_type, "count": f_count,
                "sources": {}, "any_pooled": False,
            })
            rev["count"] = max(rev["count"], f_count)
            rev["any_pooled"] = rev["any_pooled"] or pooled
            sentry = rev["sources"].setdefault(
                (src_ds, s_type),
                {"s_ids": set(), "t_ids": set(), "s_total": 0,
                 "all_s_ids": set(), "all_t_ids": set(),
                 "all_s_overlap_ids": set(),
                 "all_t_overlap_ids": set(),
                 "pooled": False, "s_measured": False, "t_measured": False,
                 "all_s_measured": False, "all_t_measured": False})
            sentry["s_ids"] |= s_ids
            sentry["t_ids"] |= t_ids
            sentry["all_s_ids"] |= s_all_ids
            sentry["all_t_ids"] |= t_all_ids
            sentry["all_s_overlap_ids"] |= set(
                pool.get("all_valid_source_overlap_body_ids") or [])
            sentry["all_t_overlap_ids"] |= set(
                pool.get("all_valid_target_overlap_body_ids") or [])
            sentry["s_total"] = max(sentry["s_total"], s_count)
            sentry["pooled"] = sentry["pooled"] or pooled
            sentry["s_measured"] = sentry["s_measured"] or src_measurable
            sentry["t_measured"] = sentry["t_measured"] or tgt_measurable
            sentry["all_s_measured"] = (sentry["all_s_measured"]
                                         or src_all_measurable)
            sentry["all_t_measured"] = (sentry["all_t_measured"]
                                         or tgt_all_measurable)
    def _cov(union: set, total: int, pooled: bool, measured: bool) -> str:
        if not pooled:
            return "not pooled"
        if not measured:
            return "not measured"
        return format_coverage(len(union), total)

    forward_rows: List[Dict[str, Any]] = []
    for row in forward.values():
        groups: Dict[str, List[str]] = {}
        query_union: set = set()
        target_union: set = set()
        selected_query_union: set = set()
        selected_target_union: set = set()
        all_query_branches: List[set] = []
        all_target_branches: List[set] = []
        selected_query_branches: List[set] = []
        selected_target_branches: List[set] = []
        all_query_overlap: set = set()
        all_target_overlap: set = set()
        has_valid_alternatives = False
        target_total = 0
        query_measured = False
        target_measured = False
        selected_query_measured = False
        selected_target_measured = False
        for (t_ds, f_type), entry in sorted(row["targets"].items()):
            code = dataset_abbrev(t_ds) or t_ds
            groups.setdefault(code, []).append(f_type)
            if (entry["all_s_ids"] != entry["s_ids"]
                    or entry["all_t_ids"] != entry["t_ids"]):
                has_valid_alternatives = True
            if not entry["pooled"]:
                continue
            if entry["s_measured"]:
                selected_query_union |= entry["s_ids"]
                selected_query_branches.append(set(entry["s_ids"]))
                selected_query_measured = True
            if entry["t_measured"]:
                selected_target_union |= entry["t_ids"]
                selected_target_branches.append(set(entry["t_ids"]))
                selected_target_measured = True
            if entry["all_s_measured"]:
                query_union |= entry["all_s_ids"]
                all_query_branches.append(set(entry["all_s_ids"]))
                all_query_overlap.update(entry.get("all_s_overlap_ids", set()))
                query_measured = True
            if entry["all_t_measured"]:
                target_union |= entry["all_t_ids"]
                all_target_branches.append(set(entry["all_t_ids"]))
                all_target_overlap.update(entry.get("all_t_overlap_ids", set()))
                target_total += entry["t_total"]
                target_measured = True
        # A legacy pool has no all-valid fields; in that case the all-valid
        # view intentionally falls back to the selected view.
        if not query_measured and selected_query_measured:
            query_union = selected_query_union
            query_measured = True
        if not target_measured and selected_target_measured:
            target_union = selected_target_union
            target_total = sum(
                entry["t_total"] for entry in row["targets"].values()
                if entry["pooled"] and entry["t_measured"])
            target_measured = True

        def _overlap_ids(branches: List[set]) -> set:
            from collections import Counter

            counts = Counter()
            for branch in branches:
                counts.update(branch)
            return {body_id for body_id, count in counts.items()
                    if count > 1}

        query_overlap_all_valid_ids = (
            _overlap_ids(all_query_branches) | all_query_overlap)
        target_overlap_all_valid_ids = (
            _overlap_ids(all_target_branches) | all_target_overlap)
        query_overlap_all_valid = len(query_overlap_all_valid_ids)
        target_overlap_all_valid = len(target_overlap_all_valid_ids)
        query_overlap_selected = len(_overlap_ids(selected_query_branches))
        target_overlap_selected = len(_overlap_ids(selected_target_branches))
        coverage_notes = []
        if len(row["targets"]) > 1:
            coverage_notes.append("branch evidence is non-exclusive")
        if has_valid_alternatives:
            coverage_notes.append(
                "all-valid union includes supported alternative bridges")
        if query_overlap_selected or target_overlap_selected:
            coverage_notes.append(
                "selected overlap: "
                f"{query_overlap_selected} source bodyIds, "
                f"{target_overlap_selected} target bodyIds")
        if query_overlap_all_valid or target_overlap_all_valid:
            coverage_notes.append(
                "all-valid overlap: "
                f"{query_overlap_all_valid} source bodyIds, "
                f"{target_overlap_all_valid} target bodyIds")
        coverage_note = "; ".join(coverage_notes)
        forward_rows.append({
            "dataset": row["dataset"],
            "type": row["type"],
            "count": row["count"],
            "maps_to": " · ".join(
                f"{code}: {', '.join(names)}"
                for code, names in sorted(groups.items())),
            "targets": len(row["targets"]),
            "relationship": ("1-to-N" if len(row["targets"]) > 1
                             else "1-to-1"),
            "query_cov": _cov(query_union, row["count"],
                              row["any_pooled"], query_measured),
            "target_cov": _cov(target_union, target_total,
                               row["any_pooled"], target_measured),
            "query_cov_selected": _cov(
                selected_query_union, row["count"], row["any_pooled"],
                selected_query_measured),
            "target_cov_selected": _cov(
                selected_target_union,
                sum(entry["t_total"] for entry in row["targets"].values()
                    if entry["pooled"] and entry["t_measured"]),
                row["any_pooled"], selected_target_measured),
            "query_cov_all_valid": _cov(
                query_union, row["count"], row["any_pooled"], query_measured),
            "target_cov_all_valid": _cov(
                target_union, target_total, row["any_pooled"],
                target_measured),
            "query_overlap_selected": query_overlap_selected,
            "target_overlap_selected": target_overlap_selected,
            "query_overlap_all_valid": query_overlap_all_valid,
            "target_overlap_all_valid": target_overlap_all_valid,
            "coverage_note": coverage_note,
        })
    forward_rows.sort(key=lambda r: (-int(r["count"] or 0),
                                     str(r["type"])))

    reverse_rows: List[Dict[str, Any]] = []
    for row in reverse.values():
        groups: Dict[str, List[str]] = {}
        source_union: set = set()
        source_total = 0
        target_union: set = set()
        selected_source_union: set = set()
        selected_target_union: set = set()
        selected_source_total = 0
        all_source_branches: List[set] = []
        all_target_branches: List[set] = []
        selected_source_branches: List[set] = []
        selected_target_branches: List[set] = []
        all_source_overlap: set = set()
        all_target_overlap: set = set()
        has_valid_alternatives = False
        source_measured = False
        target_measured = False
        selected_source_measured = False
        selected_target_measured = False
        for (s_ds, s_type), entry in sorted(row["sources"].items()):
            code = dataset_abbrev(s_ds) or s_ds
            groups.setdefault(code, []).append(s_type)
            if (entry["all_s_ids"] != entry["s_ids"]
                    or entry["all_t_ids"] != entry["t_ids"]):
                has_valid_alternatives = True
            if not entry["pooled"]:
                continue
            if entry["s_measured"]:
                selected_source_union |= entry["s_ids"]
                selected_source_branches.append(set(entry["s_ids"]))
                selected_source_total += entry["s_total"]
                selected_source_measured = True
            if entry["t_measured"]:
                selected_target_union |= entry["t_ids"]
                selected_target_branches.append(set(entry["t_ids"]))
                selected_target_measured = True
            if entry["all_s_measured"]:
                source_union |= entry["all_s_ids"]
                all_source_branches.append(set(entry["all_s_ids"]))
                all_source_overlap.update(entry.get("all_s_overlap_ids", set()))
                source_total += entry["s_total"]
                source_measured = True
            if entry["all_t_measured"]:
                target_union |= entry["all_t_ids"]
                all_target_branches.append(set(entry["all_t_ids"]))
                all_target_overlap.update(entry.get("all_t_overlap_ids", set()))
                target_measured = True
        if not source_measured and selected_source_measured:
            source_union = selected_source_union
            source_total = selected_source_total
            source_measured = True
        if not target_measured and selected_target_measured:
            target_union = selected_target_union
            target_measured = True

        def _overlap_ids(branches: List[set]) -> set:
            from collections import Counter

            counts = Counter()
            for branch in branches:
                counts.update(branch)
            return {body_id for body_id, count in counts.items()
                    if count > 1}

        source_overlap_all_valid_ids = (
            _overlap_ids(all_source_branches) | all_source_overlap)
        target_overlap_all_valid_ids = (
            _overlap_ids(all_target_branches) | all_target_overlap)
        source_overlap_all_valid = len(source_overlap_all_valid_ids)
        target_overlap_all_valid = len(target_overlap_all_valid_ids)
        source_overlap_selected = len(_overlap_ids(selected_source_branches))
        target_overlap_selected = len(_overlap_ids(selected_target_branches))
        coverage_notes = []
        if len(row["sources"]) > 1:
            coverage_notes.append("branch evidence is non-exclusive")
        if has_valid_alternatives:
            coverage_notes.append(
                "all-valid union includes supported alternative bridges")
        if source_overlap_selected or target_overlap_selected:
            coverage_notes.append(
                "selected overlap: "
                f"{source_overlap_selected} source bodyIds, "
                f"{target_overlap_selected} target bodyIds")
        if source_overlap_all_valid or target_overlap_all_valid:
            coverage_notes.append(
                "all-valid overlap: "
                f"{source_overlap_all_valid} source bodyIds, "
                f"{target_overlap_all_valid} target bodyIds")
        coverage_note = "; ".join(coverage_notes)
        reverse_rows.append({
            "dataset": row["dataset"],
            "type": row["type"],
            "count": row["count"],
            "mapped_from": " · ".join(
                f"{code}: {', '.join(names)}"
                for code, names in sorted(groups.items())),
            "sources": len(row["sources"]),
            "relationship": ("N-to-1" if len(row["sources"]) > 1
                             else "1-to-1"),
            "source_cov": _cov(source_union, source_total,
                               row["any_pooled"], source_measured),
            "target_cov": _cov(target_union, row["count"],
                               row["any_pooled"], target_measured),
            "source_cov_selected": _cov(
                selected_source_union, selected_source_total,
                row["any_pooled"], selected_source_measured),
            "target_cov_selected": _cov(
                selected_target_union, row["count"], row["any_pooled"],
                selected_target_measured),
            "source_cov_all_valid": _cov(
                source_union, source_total, row["any_pooled"],
                source_measured),
            "target_cov_all_valid": _cov(
                target_union, row["count"], row["any_pooled"],
                target_measured),
            "source_overlap_selected": source_overlap_selected,
            "target_overlap_selected": target_overlap_selected,
            "source_overlap_all_valid": source_overlap_all_valid,
            "target_overlap_all_valid": target_overlap_all_valid,
            "coverage_note": coverage_note,
        })
    reverse_rows.sort(key=lambda r: (0 if r["relationship"] == "N-to-1"
                                     else 1, -int(r["count"] or 0),
                                     str(r["type"])))
    return {"forward": forward_rows, "reverse": reverse_rows}


def render_mapping_sankey_html(flows, *, pools: Optional[Dict[tuple,
                               Dict[str, Any]]] = None,
                               variant: str = "linker",
                               max_flows: int = 500,
                               orphans: Optional[List[Dict[str, Any]]] = None
                               ) -> Optional[str]:
    """Layered mapping Sankey through the vispath backend.

    Uses ``create_sankey`` — the generation that embeds the shared
    interactive control panel (per-node color pickers, edge
    color/opacity sliders, metric toggles, PNG/SVG export), so node and
    edge colors stay user-adjustable.  Path rows come from
    ``build_mapping_sankey_paths``; the render goes to a temp dir and
    is read back as an HTML string (nothing is written to the
    repository).  When the flow cap trims flows, the artifact carries a
    floating "+N more flows" notice pointing at the CSV export (the
    full mapping is never silently dropped).  ``orphans`` (W3) render
    as single-node stub rows.  Returns None when vispath
    is unavailable or there is nothing to draw.
    """
    rows = build_mapping_sankey_paths(
        flows, pools=pools, variant=variant, max_flows=max_flows,
        orphans=orphans)
    if not rows:
        return None
    VisualizePath = _import_vispath()
    if VisualizePath is None:
        return None

    import os
    import tempfile

    import pandas as pd

    total_flows = len([f for f in (flows or []) if f])
    overflow = max(0, total_flows - max_flows)

    frame = pd.DataFrame({
        "path_block": [" -> ".join(names) for names, _w in rows],
        "weights": [weights for _n, weights in rows],
    })
    with tempfile.TemporaryDirectory(prefix="drocat_sankey_") as tmp:
        visualizer = VisualizePath(
            path_file=frame, output_folder=tmp, showfig=False,
            verbose=False,
            source_color="#5b8cff", intermediate_color="#94a3b8",
            target_color="#22c55e", link_color="rgba(100, 100, 100, 0.4)",
            edge_weight_label="neurons")
        try:
            path = visualizer.visualize_sankey()
        except Exception:
            logger.warning("mapping sankey render failed",
                           exc_info=True)
            return None
        if not path or not os.path.isfile(path):
            return None
        with open(path, "r", encoding="utf-8") as handle:
            html = handle.read()
    notes = []
    if overflow:
        notes.append(
            '<div style="position:fixed;top:8px;left:8px;z-index:9999;'
            'background:rgba(255,255,255,0.94);border:1px solid #cbd5e1;'
            'border-radius:8px;padding:8px 12px;'
            'font:12px/1.45 -apple-system,Segoe UI,sans-serif;color:#0f172a">'
            f'<b>+{overflow} more flows not drawn</b> (cap {max_flows}) — '
            'the full mapping is in the CSV export.</div>')
    if variant == "linker":
        legend = _linker_legend_html(_collect_linker_columns(flows, pools))
        if legend:
            notes.append(legend)
    if any(
            (get_mapping_pool(pools, flow).get("valid_chain_count", 0) > 1)
            for flow in (flows or []) if flow):
        notes.append(
            '<div style="position:fixed;bottom:8px;left:8px;z-index:9999;'
            'background:rgba(255,255,255,0.94);border:1px solid #cbd5e1;'
            'border-radius:8px;padding:8px 12px;'
            'font:12px/1.45 -apple-system,Segoe UI,sans-serif;color:#0f172a">'
            '<b>Alternative bridge evidence retained.</b> Ribbon weights use '
            'the selected bridge and are not summed; see Type coverage for '
            'the all-valid union and overlap.</div>')
    if notes:
        html = _inject_body_note(html, "".join(notes))
    return html


def build_source_map_graph() -> "nx.DiGraph":
    """The valid-bridge network of ``BRIDGE_SOURCE_MAP`` (§9I).

    Nodes are (namespace, column) pairs — one ``type`` identity node per
    mapping namespace plus every licensed metadata column; edges are
    the licensed transitions (same-name identity between the type
    nodes, one edge per map target).  Hover titles carry the licensing
    rule, so the graph IS the documentation of which bridges exist.
    """
    from comparison.cross_dataset_type_mapper import (
        ANNOTATION_COLUMNS,
        CROSSWALK_COLUMNS,
        BRIDGE_IDENTITY,
        BRIDGE_SOURCE_MAP,
    )

    graph = nx.DiGraph()

    def _abbr(key: str) -> str:
        return dataset_abbrev(key) or key

    def _type_node(key: str) -> str:
        node = f"{key}|type"
        if not graph.has_node(node):
            graph.add_node(
                node, node_type="type", label=f"type · {_abbr(key)}",
                title=(f"Primary type identity of {_abbr(key)} — "
                       "same-name matches connect every dataset"))
        return node

    for (home, column), targets in BRIDGE_SOURCE_MAP.items():
        if home == BRIDGE_IDENTITY:
            continue
        home_type = _type_node(home)
        column_node = f"{home}|{column}"
        target_text = ", ".join(
            _abbr(t) for t in sorted(targets))
        if column in CROSSWALK_COLUMNS:
            title = (
                f"male-cns crosswalk column — routes to {target_text}. "
                "Valid in a bridge only when a routed dataset is an "
                "ENDPOINT of the pair (no hemibrainType on "
                "male-cns↔BANC bridges).")
        elif column in ANNOTATION_COLUMNS:
            title = (
                f"Additional-name cells on the {_abbr(home)} rows — "
                f"lands in {target_text} only.")
        else:
            title = f"Licensed bridge column of {_abbr(home)}."
        graph.add_node(column_node, node_type="linker",
                       label=f"{column} · {_abbr(home)}",
                       title=title)
        if not graph.has_edge(home_type, column_node):
            graph.add_edge(home_type, column_node, weight=1,
                           title=f"{_abbr(home)} rows carry the column",
                           bridge_texts=[])
        for target in sorted(targets):
            graph.add_edge(column_node, _type_node(target), weight=1,
                           title=title, bridge_texts=[])

    # same-name identity: the type nodes connect pairwise, both ways
    type_nodes = [n for n, d in graph.nodes(data=True)
                  if d.get("node_type") == "type"]
    for index, a in enumerate(type_nodes):
        for b in type_nodes[index + 1:]:
            graph.add_edge(a, b, weight=1,
                           title="same-name identity", bridge_texts=[])
            graph.add_edge(b, a, weight=1,
                           title="same-name identity", bridge_texts=[])
    return graph


def render_source_map_network_html(
        title: str = "Valid type-mapping bridges (BRIDGE_SOURCE_MAP)",
) -> Optional[str]:
    """Interactive HTML network of the valid bridges (§9I).

    Renders ``build_source_map_graph()`` through the vispath dagre
    pipeline and prepends a floating panel with the licensing rules,
    linked to ``docs/AUTO_TYPE_MAPPING.md`` and the regeneration
    command — the HTML, the code constant, and the docs all describe
    the same map.  Returns None when vispath is unavailable.
    """
    graph = build_source_map_graph()
    node_dataset_info: Dict[str, Dict[str, str]] = {}
    dataset_legend: Dict[str, str] = {}
    for node, data in graph.nodes(data=True):
        home = str(node).split("|", 1)[0]
        code = dataset_abbrev(home) or "?"
        node_dataset_info[node] = {
            code: str(data.get("title") or data.get("label", ""))}
        dataset_legend[code] = home
    html = _vispath_html(graph, node_dataset_info=node_dataset_info,
                         dataset_legend=dataset_legend,
                         node_groups=_dataset_groups(graph),
                         layout="dagre")
    if not html:
        return None
    import re

    panel = (
        '<div style="position:fixed;top:8px;left:8px;z-index:9999;'
        'max-width:460px;background:rgba(255,255,255,0.94);border:1px '
        'solid #cbd5e1;border-radius:8px;padding:10px 12px;'
        'font:12px/1.45 -apple-system,Segoe UI,sans-serif;color:#0f172a">'
        f'<b>{title}</b><br>'
        'Every derivation edge is licensed by '
        '<code>BRIDGE_SOURCE_MAP</code> — valid bridges are derived from '
        'the source map, not by guessing.<br>'
        '1. Same-name <code>type</code> identity connects any two '
        'datasets.<br>'
        '2. male-cns crosswalk columns route to their own family only: '
        '<code>flywireType</code> → FAFB, '
        '<code>hemibrainType</code> → hemibrain, <code>mancType</code> '
        '→ manc. BANC uses direct curated label columns and the narrow '
        '<code>banc_release_crosswalk</code>; it is never a connector.<br>'
        '3. <code>additional_type(s)</code> maps to FAFB only; '
        '<code>Alternative Cell Type(s)</code> only within BANC.<br>'
        'Docs: <code>docs/AUTO_TYPE_MAPPING.md</code> · Regenerate: '
        '<code>python scripts/render_source_map_network.py</code>'
        '</div>')
    return re.sub(r"<body[^>]*>", lambda m: m.group(0) + panel, html,
                  count=1)


def _order_component(component, ds_types, pair_flows):
    """§10.1: order one component's datasets by affinity.

    Start at the dataset with the largest mapped-flow volume; then
    repeatedly append the dataset sharing the most same-name types with
    the ordered prefix (ties: flow volume, then name) — converge–diverge
    hubs dissolve because same-name partners sit adjacent.
    """

    def flow_volume(a, b):
        return len(pair_flows.get((a, b), [])) + len(pair_flows.get((b, a), []))

    remaining = set(component)
    ordered = [sorted(remaining, key=lambda d: (
        -sum(flow_volume(d, o) for o in remaining if o != d), d))[0]]
    remaining.discard(ordered[0])
    while remaining:
        known_types: set = set()
        for d in ordered:
            known_types.update(ds_types.get(d, set()))

        def affinity(d):
            same = len(ds_types.get(d, set()) & known_types)
            volume = sum(flow_volume(d, o) for o in ordered)
            return (-same, -volume, d)

        nxt = sorted(remaining, key=affinity)[0]
        ordered.append(nxt)
        remaining.discard(nxt)
    return ordered


def build_composed_mapping_graph(pair_flows, *, pools=None,
                                 node_cap: int = 80):
    """Composed N-dataset type-level graph (Round 2, spec §4).

    ``pair_flows`` maps ``(source_dataset, target_dataset)`` to the
    ``build_mapping_flows`` product for that pair — the global search's
    per-pair results.  Nodes are ``<layer>|<dataset>|<type>`` with the
    layer taken from the component's affinity-ordered datasets (§10.1);
    per-pair edges carry the maps-via texts and per-side neuron counts;
    label-query pooled nodes (``entry`` group) sit on their owning
    dataset fed by the types they cover; every type node's hover lists
    its cross-dataset matches (§10.2).  Beyond ``node_cap`` nodes,
    all-same-name types are hidden first (§6) and reported via
    ``meta['notes']``.  Returns ``(graph, meta)``.
    """
    from comparison.cross_dataset_type_mapper import standardize_bridge

    pools = pools or {}
    dataset_scoped = any(len(key) == 4 for key in pools)
    src_pool_ids, tgt_pool_ids = _endpoint_pool_counts(
        pools, include_datasets=dataset_scoped)
    src_selected_pool_ids, tgt_selected_pool_ids = _endpoint_pool_counts(
        pools, include_datasets=dataset_scoped, scope="selected")

    graph = nx.DiGraph()
    ds_types: Dict[str, set] = {}
    edges: List[tuple] = []
    entry_specs: Dict[tuple, Dict[str, Any]] = {}
    for (src_ds, tgt_ds), flows in (pair_flows or {}).items():
        for flow in flows or []:
            src_type = flow.get("source_type", "")
            tgt_type = flow.get("foreign_type", "")
            if not src_type or not tgt_type:
                continue
            ds_types.setdefault(src_ds, set()).add(src_type)
            ds_types.setdefault(tgt_ds, set()).add(tgt_type)
            src_count = int(flow.get("source_count") or 0)
            tgt_count = int(flow.get("foreign_count") or 0)
            linker_bearing = any(
                any(l.get("kind") == "linker"
                    for l in standardize_bridge(c, src_ds, tgt_ds))
                for c in (flow.get("bridges") or [])[:2])
            pool = get_mapping_pool(pools, {
                **flow,
                "source_dataset": flow.get("source_dataset") or src_ds,
                "target_dataset": flow.get("target_dataset") or tgt_ds,
            })
            edges.append(((src_ds, src_type), (tgt_ds, tgt_type), {
                # Use the same selected-chain coverage weight as the
                # pair-level network and Sankey.  When no pool is supplied,
                # pair_flow_weight retains the historical type-count
                # fallback for programmatic callers.
                "weight": pair_flow_weight(flow, pool),
                "bridge_texts": build_bridge_texts(flow.get("bridges")),
                "linker_bearing": linker_bearing,
                "source_count": src_count,
                "foreign_count": tgt_count,
                "source_dataset": src_ds,
                "target_dataset": tgt_ds,
            }))
            origin = flow.get("matched_origin", "")
            origin_column, _, origin_value = origin.partition(" · ")
            origin_value = origin_value.strip("'")
            if origin_column and origin_column != "type":
                spec = entry_specs.setdefault(
                    (tgt_ds, origin),
                    {"label": origin, "types": {}, "neurons": 0})
                spec["types"][tgt_type] = spec["types"].get(tgt_type, 0) + \
                    tgt_count
                spec["neurons"] += tgt_count

    # connected components over the datasets that have mapped pairs
    ds_graph = nx.Graph()
    for src_ds, tgt_ds in (pair_flows or {}):
        ds_graph.add_edge(src_ds, tgt_ds)
    orders: Dict[str, int] = {}
    ds_component_size: Dict[str, int] = {}
    components: List[Dict[str, Any]] = []
    for component in sorted(nx.connected_components(ds_graph),
                            key=lambda c: sorted(c)):
        ordered = _order_component(component, ds_types, pair_flows)
        components.append({"datasets": ordered})
        for layer, ds in enumerate(ordered):
            orders[ds] = layer
            ds_component_size[ds] = len(ordered)

    def _node_id(ds: str, type_name: str) -> str:
        return f"{orders[ds]}|{ds}|{type_name}"

    def _role(ds: str) -> str:
        layer = orders[ds]
        size = ds_component_size[ds]
        return "source" if layer == 0 else (
            "target" if layer == size - 1 else "intermediate")

    # per-node side counts (a type can be source-side in one pair and
    # target-side in another)
    node_src: Dict[tuple, int] = {}
    node_tgt: Dict[tuple, int] = {}
    matches: Dict[tuple, List[tuple]] = {}
    for src, tgt, attrs in edges:
        if attrs["source_count"]:
            node_src[src] = max(node_src.get(src, 0), attrs["source_count"])
        if attrs["foreign_count"]:
            node_tgt[tgt] = max(node_tgt.get(tgt, 0), attrs["foreign_count"])
        matches.setdefault(src, []).append(tgt)
        matches.setdefault(tgt, []).append(src)

    def _ensure_node(ds: str, type_name: str, count: int) -> None:
        """Create the labeled type node unless it already exists.

        Every EDGE endpoint must get a real node even when both its
        counts are 0: networkx would otherwise auto-create an
        attribute-less node on add_edge and the renderer falls back to
        the raw ``<layer>|<dataset>|<type>`` id as the label (the BANC
        v888 DN1pE report — the name resolves in the mapper's
        v626-keyed BANC namespace but has no rows in the selected
        v888 table, so its count is 0).
        """
        nid = _node_id(ds, type_name)
        if graph.has_node(nid):
            return
        title = (f"{type_name} · {dataset_abbrev(ds)} "
                 f"({count or 0} neurons)")
        pool_count = max(
            src_pool_ids.get((ds, type_name), 0)
            if dataset_scoped else src_pool_ids.get(type_name, 0),
            tgt_pool_ids.get((ds, type_name), 0)
            if dataset_scoped else tgt_pool_ids.get(type_name, 0),
        )
        selected_pool_count = max(
            src_selected_pool_ids.get((ds, type_name), 0)
            if dataset_scoped else src_selected_pool_ids.get(type_name, 0),
            tgt_selected_pool_ids.get((ds, type_name), 0)
            if dataset_scoped else tgt_selected_pool_ids.get(type_name, 0),
        )
        if pool_count:
            title += _pool_title_suffix(pool_count, selected_pool_count)
        matched = sorted({(m_ds, m_t) for m_ds, m_t in matches.get(
            (ds, type_name), []) if m_ds != ds})
        if matched:
            title += " — matched: " + ", ".join(
                f"{m_t} ({dataset_abbrev(m_ds)})" for m_ds, m_t in matched)
        graph.add_node(nid, node_type=_role(ds),
                       label=type_name, title=title)

    for (ds, type_name), _meta in sorted(node_src.items()
                                         | node_tgt.items()):
        count = max(node_src.get((ds, type_name), 0),
                    node_tgt.get((ds, type_name), 0))
        _ensure_node(ds, type_name, count)

    # zero-count endpoints of real mappings still render (see
    # _ensure_node)
    for src, tgt, attrs in edges:
        _ensure_node(src[0], src[1], attrs["source_count"])
        _ensure_node(tgt[0], tgt[1], attrs["foreign_count"])

    for src, tgt, attrs in edges:
        sid = _node_id(src[0], src[1])
        tid = _node_id(tgt[0], tgt[1])
        if graph.has_edge(sid, tid):
            existing = graph[sid][tid]
            existing["weight"] += attrs["weight"]
            existing["bridge_texts"] = list(dict.fromkeys(
                existing.get("bridge_texts", []) + attrs["bridge_texts"]))
        else:
            graph.add_edge(sid, tid, title=f"{attrs['weight']} neurons",
                           **attrs)

    # label-query pooled nodes: one per owning dataset per matched label,
    # fed by the covered types (§1.B′)
    for (ds, origin), spec in sorted(entry_specs.items()):
        node_id = f"E|{ds}|{origin}"
        graph.add_node(node_id, node_type="entry", label=origin,
                       title=(f"{origin} — covers {len(spec['types'])} types, "
                              f"{spec['neurons']:,} neurons"))
        for type_name, count in sorted(spec["types"].items()):
            nid = _node_id(ds, type_name)
            if graph.has_node(nid):
                graph.add_edge(nid, node_id, weight=count or 1,
                               title=f"{count or 1} neurons",
                               bridge_texts=[])

    # composed-graph scoping (§6): beyond the node cap hide all-same-name
    # types first — bare name echoes carry no mapping information, while
    # linker-bearing types and pooled-fed types are never hidden
    notes: List[str] = []
    hidden = 0
    while graph.number_of_nodes() > node_cap:
        candidates = []
        for nid, data in graph.nodes(data=True):
            if data.get("node_type") not in ("source", "intermediate",
                                             "target"):
                continue
            nbrs = list(graph.successors(nid)) + list(graph.predecessors(nid))
            if not nbrs or any(graph.nodes[o].get("node_type") == "entry"
                               for o in nbrs):
                continue
            label = data.get("label")
            if not all(graph.nodes[o].get("label") == label for o in nbrs):
                continue
            touched = list(graph.in_edges(nid)) + list(graph.out_edges(nid))
            if any(graph[u][v].get("linker_bearing") for u, v in touched):
                continue
            candidates.append(nid)
        if not candidates:
            break
        for nid in candidates:
            graph.remove_node(nid)
            hidden += 1
            if graph.number_of_nodes() <= node_cap:
                break
    if hidden:
        notes.append(f"+{hidden} same-name types hidden")

    meta = {"components": components, "notes": notes,
            "hidden_same_name": hidden, "node_cap": node_cap}
    return graph, meta


def render_composed_mapping_html(pair_flows, *, pools=None,
                                 node_cap: int = 80,
                                 title: str = "Composed type mapping"):
    """Render the composed N-dataset mapping to an HTML string (Round 2).

    Returns ``(html, meta)`` — html is None when vispath is unavailable
    or nothing is mapped; meta carries component orders and scoping
    notes for the popup.
    """
    graph, meta = build_composed_mapping_graph(
        pair_flows, pools=pools, node_cap=node_cap)
    if not graph.nodes:
        return None, meta

    node_dataset_info: Dict[str, Dict[str, str]] = {}
    dataset_legend: Dict[str, str] = {}
    for node, data in graph.nodes(data=True):
        ds = str(node).split("|")[1] if str(node).count("|") >= 1 else ""
        if node.startswith("E|"):
            ds = str(node).split("|")[1]
        code = dataset_abbrev(ds) or "?"
        node_dataset_info[node] = {
            code: str(data.get("title") or data.get("label", ""))}
        dataset_legend[code] = ds

    edge_labels: Dict[tuple, Dict[str, str]] = {}
    for src, tgt, data in graph.edges(data=True):
        labels: Dict[str, str] = {}
        for index, text in enumerate(data.get("bridge_texts") or [], start=1):
            key = f"maps via #{index}" if len(data["bridge_texts"]) > 1 \
                else "maps via"
            labels[key] = text
        if labels:
            edge_labels[(src, tgt)] = labels

    html = _vispath_html(graph, edge_labels=edge_labels,
                         node_dataset_info=node_dataset_info,
                         dataset_legend=dataset_legend,
                         node_groups=_dataset_groups(graph),
                         layout="dagre")
    return html, meta


def build_bridges_csv(flows, *, pools=None, extended: bool = False) -> Optional[str]:
    """Mapping CSV for one pair's flows (user 2026-09-09 redesign).

    One row per (source type, target type) pair with EXPLICIT endpoints —
    ``source_dataset``/``target_dataset`` so the all-pairs export is
    self-contained — plus the matched source entry and its column, the
    rendered preferred bridge and its standardized linker columns,
    machine-readable per-side pool coverage, and the FULL per-type
    bodyId populations (``source_body_ids`` / ``target_body_ids``: every
    bodyId of the mapped type in its OWN dataset, one brace-wrapped
    comma-separated list per cell so spreadsheet delimiters never split it
    — listed per type,
    never paired across datasets).  The column set is fixed,
    so per-pair and all-pairs files share one header and the old
    union-of-bridge-columns concatenation hack is gone.  Uniform field
    counts, proper quoting.  Returns None when there is nothing to
    export.  ``extended=True`` adds selected/all-valid coverage scopes,
    resolver status, ranks, and the selected/all-valid pool IDs.  The
    default keeps the historical 20-column contract for downstream callers;
    the Type Mapping UI requests the extended form explicitly.
    """
    import csv as _csv
    import io

    from comparison.cross_dataset_type_mapper import (
        bridge_linker_text,
        preferred_bridge_chain,
        standardize_bridge,
    )

    pools = pools or {}
    flows = [f for f in (flows or []) if f]
    if not flows:
        return None

    def _body_ids_cell(ids) -> str:
        # Brace-wrapped comma-separated IDs are ONE CSV field while matching
        # the documented export contract.  A pool without the key
        # (legacy/synthetic) stays an empty cell.
        return "" if ids is None else "{" + ", ".join(
            str(body_id) for body_id in ids) + "}"

    legacy_header = [
        "source_dataset", "source_entry", "matched_column", "source_type",
        "target_dataset", "target_type", "relationship",
        "source_neurons", "target_neurons",
        "bridge", "bridge_columns", "mapping_origin",
        "source_pool", "source_total", "target_pool", "target_total",
        "source_body_ids", "target_body_ids",
        "pool_coverage", "pool_coverage_basis",
    ]
    extended_header = [
        "source_dataset", "source_entry", "matched_column", "source_type",
        "target_dataset", "target_type", "relationship",
        "source_neurons", "target_neurons",
        "selected_bridge", "bridge", "bridge_columns", "mapping_origin",
        "mapping_status", "selected_bridge_rank", "valid_bridge_count",
        "selected_linker_values", "selected_linker_canonical_values",
        "unsupported_attempts",
        "source_pool", "source_total", "target_pool", "target_total",
        "all_valid_source_pool", "all_valid_source_total",
        "all_valid_target_pool", "all_valid_target_total",
        "source_body_ids", "target_body_ids",
        "all_valid_source_body_ids", "all_valid_target_body_ids",
        "pool_coverage", "pool_coverage_basis", "coverage_overlap",
        "coverage_scope",
    ]
    header = extended_header if extended else legacy_header
    targets_by_source: Dict[str, set] = {}
    for flow in flows:
        targets_by_source.setdefault(
            str(flow.get("source_type", "")), set()).add(
            str(flow.get("foreign_type", "")))

    buffer = io.StringIO()
    writer = _csv.writer(buffer, quoting=_csv.QUOTE_MINIMAL)
    writer.writerow(header)
    for flow in flows:
        src_ds = flow.get("source_dataset", "")
        tgt_ds = flow.get("target_dataset", "")
        chains = flow.get("bridges") or []
        src_type = flow.get("source_type", "")
        foreign = flow.get("foreign_type", "")
        chain = preferred_bridge_chain(chains, src_ds, tgt_ds)
        info = bridge_linker_text(chains, src_ds, tgt_ds, foreign)
        pool = get_mapping_pool(pools, flow)
        selected_chain = pool.get("selected_chain") if pool else chain
        linkers = [
            linker for linker in standardize_bridge(
                selected_chain or [], src_ds, tgt_ds)
            if linker.get("kind") == "linker"
        ] if selected_chain else []
        selected_info = bridge_linker_text(
            [selected_chain] if selected_chain else [],
            src_ds, tgt_ds, foreign)
        # The matched entry: the source type itself when the match ran
        # through the type column, otherwise the label/annotation value
        # that matched (recorded in matched_origin as "<column> · 'v'").
        origin = flow.get("matched_origin", "")
        origin_column, _, origin_value = origin.partition(" · ")
        if origin_column and origin_column != "type":
            matched_column = origin_column
            source_entry = origin_value.strip("'")
        elif origin_column == "type":
            matched_column = "type"
            source_entry = origin_value.strip("'") or src_type
        else:
            matched_column = "type"
            source_entry = src_type
        relationship = ("1-to-N"
                        if len(targets_by_source.get(src_type, ())) > 1
                        else "1-to-1")
        # Keep one compact human-readable field, but expose BOTH
        # independent sides.  ``coverage`` remains the historical
        # target-side alias for old callers and synthetic fixtures.
        source_coverage = pool.get("source_coverage") or ""
        target_coverage = pool.get("target_coverage") or pool.get(
            "coverage") or ""
        if source_coverage and target_coverage:
            pool_coverage = (
                f"source {source_coverage}; target {target_coverage}")
        else:
            pool_coverage = target_coverage or source_coverage
        source_total = pool.get("source_type_total")
        target_total = pool.get("target_type_total")
        all_source_ids = pool.get("all_valid_source_body_ids")
        if all_source_ids is None:
            all_source_ids = pool.get("source_body_ids")
        all_target_ids = pool.get("all_valid_target_body_ids")
        if all_target_ids is None:
            all_target_ids = pool.get("target_body_ids")
        all_source_total = pool.get("all_valid_source_type_total")
        if all_source_total is None:
            all_source_total = source_total
        all_target_total = pool.get("all_valid_target_type_total")
        if all_target_total is None:
            all_target_total = target_total
        all_source_coverage = (
            format_coverage(len(all_source_ids or []), all_source_total)
            if all_source_total is not None else "")
        all_target_coverage = (
            format_coverage(len(all_target_ids or []), all_target_total)
            if all_target_total is not None else "")
        if all_source_coverage and all_target_coverage and (
                all_source_coverage != source_coverage
                or all_target_coverage != target_coverage):
            pool_coverage += (
                f"; all-valid union source {all_source_coverage}; "
                f"target {all_target_coverage}")
        coverage_basis = pool.get("coverage_basis") or ""
        if pool:
            all_source_basis = pool.get(
                "all_valid_source_basis", pool.get("source_basis", ""))
            all_target_basis = pool.get(
                "all_valid_target_basis", pool.get("target_basis", ""))
            coverage_basis = (
                f"selected: {pool.get('source_basis', '')} / "
                f"{pool.get('target_basis', '')}; all-valid: "
                f"{all_source_basis} / {all_target_basis}; "
                f"{coverage_basis}")
        selected_linker_values = "; ".join(
            str(linker.get("raw_value", linker.get("value", "")))
            for linker in linkers)
        selected_linker_canonical_values = "; ".join(
            str(linker.get("canonical_value", linker.get("value", "")))
            for linker in linkers)
        unsupported_attempts = "; ".join(
            f"#{attempt.get('rank')}: {attempt.get('reason') or attempt.get('status')}"
            for attempt in ((pool.get("attempts") or []) if pool else [])
            if not attempt.get("supported"))
        coverage_overlap = ""
        if pool:
            coverage_overlap = (
                f"source {pool.get('all_valid_source_overlap_count', 0)}; "
                f"target {pool.get('all_valid_target_overlap_count', 0)}")
        legacy_row = [
            src_ds,
            source_entry,
            matched_column,
            src_type,
            tgt_ds,
            foreign,
            relationship,
            flow.get("source_count") or 0,
            flow.get("foreign_count") or 0,
            info["text"],
            "; ".join(linker["column"] for linker in linkers),
            "same name" if not linkers else "mapped",
            pool.get("source_pool_size", ""),
            source_total if source_total is not None else "",
            pool.get("target_pool_size", ""),
            target_total if target_total is not None else "",
            # FULL per-type populations (user 2026-09-09): every bodyId of
            # the mapped type in its OWN dataset — never a cross-dataset
            # bodyId pairing.
            _body_ids_cell(pool.get("source_type_body_ids")),
            _body_ids_cell(pool.get("target_type_body_ids")),
            pool_coverage,
            pool.get("coverage_basis") or "",
        ]
        extended_row = [
            src_ds,
            source_entry,
            matched_column,
            src_type,
            tgt_ds,
            foreign,
            relationship,
            flow.get("source_count") or 0,
            flow.get("foreign_count") or 0,
            selected_info["text"],
            info["text"],
            "; ".join(linker["column"] for linker in linkers),
            "same name" if not linkers else "mapped",
            flow.get("mapping_status") or "mapped",
            pool.get("selected_chain_rank", ""),
            pool.get("valid_chain_count", ""),
            selected_linker_values,
            selected_linker_canonical_values,
            unsupported_attempts,
            pool.get("source_pool_size", ""),
            source_total if source_total is not None else "",
            pool.get("target_pool_size", ""),
            target_total if target_total is not None else "",
            pool.get("all_valid_source_pool_size", "") if pool else "",
            all_source_total if all_source_total is not None else "",
            pool.get("all_valid_target_pool_size", "") if pool else "",
            all_target_total if all_target_total is not None else "",
            _body_ids_cell(pool.get("source_type_body_ids")),
            _body_ids_cell(pool.get("target_type_body_ids")),
            _body_ids_cell(all_source_ids),
            _body_ids_cell(all_target_ids),
            pool_coverage,
            coverage_basis,
            coverage_overlap,
            pool.get("coverage_scope") or "selected bridge",
        ]
        writer.writerow(extended_row if extended else legacy_row)
    return buffer.getvalue()
