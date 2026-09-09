"""Unit tests for the Round 2 mapping graph (spec §4/§6/§7).

Synthetic pair flows only — the mapping graph builder's layer naming,
per-component dataset ordering (§10.1), matched-type hovers (§10.2),
pooled label nodes, the scoping cap (§6) and the bridges CSV contract
(§7) are validated without touching the real datasets.
"""

from comparison.mapping_visualization import (
    build_bridges_csv,
    build_composed_mapping_graph,
    render_composed_mapping_html,
)

MCNS = 'male-cns:v1.0'
FAFB = 'flywire_FAFB_v783'
BANC = 'banc_v626'


def _flow(s, st, t, tt, sc, fc, origin='type', linkers=True, **extra):
    hops = [
        {'dataset': s, 'column': 'type', 'value': st},
        {'dataset': 'male-cns:v1.0', 'column': 'flywireType', 'value': 'LMTe01'},
        {'dataset': t, 'column': 'type', 'value': tt},
    ] if linkers else [
        {'dataset': s, 'column': 'type', 'value': st},
        {'dataset': t, 'column': 'type', 'value': tt},
    ]
    flow = {'source_dataset': s, 'target_dataset': t, 'source_type': st,
            'foreign_type': tt, 'source_count': sc, 'foreign_count': fc,
            'matched_origin': origin, 'bridges': [hops]}
    flow.update(extra)
    return flow


def test_composed_graph_layers_roles_and_hovers():
    pair_flows = {
        (MCNS, FAFB): [_flow(MCNS, 'T1', FAFB, 'T1', 4, 4),
                       _flow(MCNS, 'T2', FAFB, 'T3', 2, 3)],
        (FAFB, BANC): [_flow(FAFB, 'T3', BANC, 'T3', 3, 5),
                       _flow(FAFB, 'T3', BANC, 'T4', 3, 1, linkers=False)],
        (BANC, MCNS): [_flow(BANC, 'T4', MCNS, 'T4', 1, 1)],
    }
    graph, meta = build_composed_mapping_graph(pair_flows)
    # one connected component of three datasets, deterministic order
    assert len(meta['components']) == 1
    order = meta['components'][0]['datasets']
    assert sorted(order) == sorted({MCNS, FAFB, BANC})
    # roles follow the component positions
    roles = {ds: ('source' if i == 0 else
                  'target' if i == len(order) - 1 else 'intermediate')
             for i, ds in enumerate(order)}
    for layer, ds in enumerate(order):
        nid = f'{layer}|{ds}|'
        nodes = [n for n in graph.nodes if str(n).startswith(nid)]
        assert nodes
        for n in nodes:
            assert graph.nodes[n]['node_type'] == roles[ds]
    # hovers list the cross-dataset matches (§10.2), own dataset excluded
    t2 = [n for n, d in graph.nodes(data=True) if d['label'] == 'T2'][0]
    assert 'matched: T3 (FAFB)' in graph.nodes[t2]['title']
    t4 = [n for n, d in graph.nodes(data=True) if d['label'] == 'T4'][0]
    title = graph.nodes[t4]['title']
    assert 'T3' in title and 'T4' in title
    # pair edges carry the maps-via texts
    assert any(d.get('bridge_texts') for _, _, d in graph.edges(data=True))


def test_composed_component_order_prefers_same_name():
    # A shares X and X2 with B (two same-name pairs); B shares only Y
    # with C — §10.1 puts A adjacent to B even though volumes tie.
    pair_flows = {
        ('ds_a', 'ds_b'): [_flow('ds_a', 'X', 'ds_b', 'X', 1, 1),
                           _flow('ds_a', 'X2', 'ds_b', 'X2', 1, 1)],
        ('ds_b', 'ds_c'): [_flow('ds_b', 'Y', 'ds_c', 'Y', 1, 1)],
    }
    _graph, meta = build_composed_mapping_graph(pair_flows)
    assert meta['components'][0]['datasets'] == ['ds_b', 'ds_a', 'ds_c']


def test_composed_pooled_label_node():
    pair_flows = {
        (FAFB, MCNS): [
            _flow(FAFB, 'T1', MCNS, 'M1', 2, 2,
                  origin="cell_type · 'circadian_clock'",
                  origin_dataset=FAFB, origin_column='cell_type',
                  origin_value='circadian_clock', origin_label=
                  "cell_type · 'circadian_clock'", origin_type='T1',
                  origin_count=2),
            _flow(FAFB, 'T1', MCNS, 'M2', 2, 3,
                  origin="cell_type · 'circadian_clock'",
                  origin_dataset=FAFB, origin_column='cell_type',
                  origin_value='circadian_clock', origin_label=
                  "cell_type · 'circadian_clock'", origin_type='T1',
                  origin_count=2),
            _flow(FAFB, 'T2', MCNS, 'M1', 1, 3,
                  origin="cell_type · 'circadian_clock'",
                  origin_dataset=FAFB, origin_column='cell_type',
                  origin_value='circadian_clock', origin_label=
                  "cell_type · 'circadian_clock'", origin_type='T2',
                  origin_count=1),
        ],
        (FAFB, BANC): [
            _flow(FAFB, 'T1', BANC, 'B1', 2, 4,
                  origin="cell_type · 'circadian_clock'",
                  origin_dataset=FAFB, origin_column='cell_type',
                  origin_value='circadian_clock', origin_label=
                  "cell_type · 'circadian_clock'", origin_type='T1',
                  origin_count=2),
            _flow(FAFB, 'T2', BANC, 'B2', 1, 5,
                  origin="cell_type · 'circadian_clock'",
                  origin_dataset=FAFB, origin_column='cell_type',
                  origin_value='circadian_clock', origin_label=
                  "cell_type · 'circadian_clock'", origin_type='T2',
                  origin_count=1),
        ],
    }
    graph, _meta = build_composed_mapping_graph(pair_flows)
    entries = [n for n, d in graph.nodes(data=True)
               if d['node_type'] == 'entry']
    assert len(entries) == 1
    entry = entries[0]
    assert entry == f"E|{FAFB}|cell_type · 'circadian_clock'"
    assert 'circadian_clock' in graph.nodes[entry]['label']
    assert graph.nodes[entry]['home_dataset'] == FAFB
    assert graph.nodes[entry]['origin_column'] == 'cell_type'
    # The query entry is attached to the two unique origin types, even though
    # T1 fans out to multiple target types and two target datasets.
    successors = list(graph.successors(entry))
    assert {graph.nodes[n]['label'] for n in successors} == {'T1', 'T2'}
    assert not list(graph.predecessors(entry))
    assert all(n.startswith(f'0|{FAFB}|') for n in successors)
    assert 'covers 2 types, 3 neurons' in graph.nodes[entry]['title']
    assert not any(
        d['node_type'] == 'entry' and n.split('|')[1] in {MCNS, BANC}
        for n, d in graph.nodes(data=True))


def _entry_flows(**overrides):
    """Three circadian_clock flows: T1→M1/M2, T2→M1 with origin metadata."""
    def _meta(origin_type, origin_count):
        base = dict(
            origin="cell_type · 'circadian_clock'",
            origin_dataset=FAFB, origin_column='cell_type',
            origin_value='circadian_clock',
            origin_label="cell_type · 'circadian_clock'",
            origin_type=origin_type, origin_count=origin_count)
        base.update(overrides)
        return base
    return [
        _flow(FAFB, 'T1', MCNS, 'M1', 2, 2, **_meta('T1', 2)),
        _flow(FAFB, 'T1', MCNS, 'M2', 2, 3, **_meta('T1', 2)),
        _flow(FAFB, 'T2', MCNS, 'M1', 1, 3, **_meta('T2', 1)),
    ]


def test_pair_network_query_entry_owns_origin_side():
    """§14: the per-pair type-level network attaches a taxonomy query
    entry to its ORIGIN dataset and the origin-side source types — the
    same contract the composed graph already follows (§13)."""
    from comparison.mapping_visualization import build_mapping_network_graph

    graph = build_mapping_network_graph(_entry_flows())
    entries = [n for n, d in graph.nodes(data=True)
               if d['node_type'] == 'entry']
    assert entries == [f"E|{FAFB}|cell_type · 'circadian_clock'"]
    entry = entries[0]
    assert graph.nodes[entry]['home_dataset'] == FAFB
    assert graph.nodes[entry]['origin_column'] == 'cell_type'
    assert graph.nodes[entry]['origin_value'] == 'circadian_clock'
    # the entry touches only origin-side source types, in the entry →
    # source direction (the entry is the query input)
    successors = list(graph.successors(entry))
    assert {graph.nodes[n]['label'] for n in successors} == {'T1', 'T2'}
    assert all(n.startswith(f'0|{FAFB}|') for n in successors)
    assert not list(graph.predecessors(entry))
    # hover coverage counts UNIQUE origin types and source-side neurons,
    # never the target-side received counts
    assert 'covers 2 types, 3 neurons' in graph.nodes[entry]['title']
    # no entry keyed under the target dataset
    assert not any(
        d['node_type'] == 'entry' and n.split('|')[1] == MCNS
        for n, d in graph.nodes(data=True))
    # the real pair-mapping edges survive unchanged (T1→M1, T1→M2, T2→M1)
    pair_edges = {(u, v) for u, v in graph.edges()
                  if not graph[u][v].get('entry_edge')}
    assert pair_edges == {(f'0|{FAFB}|T1', f'1|{MCNS}|M1'),
                          (f'0|{FAFB}|T1', f'1|{MCNS}|M2'),
                          (f'0|{FAFB}|T2', f'1|{MCNS}|M1')}
    assert any(graph[u][v]['bridge_texts'] for u, v in pair_edges)


def test_pair_network_entry_falls_back_to_source_dataset():
    """§14: a legacy flow with only the matched_origin display string keys
    its entry under the flow's source (origin) dataset."""
    from comparison.mapping_visualization import build_mapping_network_graph

    graph = build_mapping_network_graph(
        [_flow(FAFB, 'T1', MCNS, 'M1', 2, 2,
               origin="cell_type · 'legacy_label'")])
    entries = [n for n, d in graph.nodes(data=True)
               if d['node_type'] == 'entry']
    assert entries == [f"E|{FAFB}|cell_type · 'legacy_label'"]
    assert all(n.startswith(f'0|{FAFB}|')
               for n in graph.successors(entries[0]))


def test_pair_network_entry_attaches_to_target_side_origin():
    """Native-match flows (the viewer's expanded search, and the panel's
    fallback chips) run searched → foreign with the matched column on the
    FOREIGN side: the entry must attach there and count the foreign
    population (user 2026-09-10: the viewer's circadian_clock artifact
    was mis-owned by MCNS, hovering 'covers 40 types, 219 neurons' — the
    received side)."""
    from comparison.mapping_visualization import build_mapping_network_graph

    def _native_flow(local, foreign_type, count, foreign_count):
        return _flow(
            MCNS, local, FAFB, foreign_type, count, foreign_count,
            origin="cell_type · 'circadian_clock'",
            origin_dataset=FAFB, origin_column='cell_type',
            origin_value='circadian_clock',
            origin_label="cell_type · 'circadian_clock'",
            origin_type=foreign_type, origin_count=foreign_count)

    graph = build_mapping_network_graph([
        _native_flow('M1', 's-CPDN3A', 3, 38),
        _native_flow('M2', 's-CPDN3C', 2, 32),
        _native_flow('M1', 's-CPDN3D', 3, 37),
    ])
    entries = [n for n, d in graph.nodes(data=True)
               if d['node_type'] == 'entry']
    assert entries == [f"E|{FAFB}|cell_type · 'circadian_clock'"]
    entry = entries[0]
    assert graph.nodes[entry]['home_dataset'] == FAFB
    successors = list(graph.successors(entry))
    assert {graph.nodes[n]['label']
            for n in successors} == {'s-CPDN3A', 's-CPDN3C', 's-CPDN3D'}
    # the origin side is presented LEFT (layer 0), so dagre ranks
    # query entry → FAFB types → searched MCNS types — the same natural
    # flow as the panel's origin-seeded exports
    assert all(n.startswith(f'0|{FAFB}|') for n in successors)
    assert not list(graph.predecessors(entry))
    # the hover counts the ORIGIN side's covered population (38+32+37),
    # never the searched side's received neurons
    assert 'covers 3 types, 107 neurons' in graph.nodes[entry]['title']
    assert not any(
        d['node_type'] == 'entry' and n.split('|')[1] == MCNS
        for n, d in graph.nodes(data=True))
    # the pair edges are drawn origin → counterpart (presentation
    # direction); per-side counts stay on the edge attrs
    pair_edges = {(u, v) for u, v in graph.edges()
                  if not graph[u][v].get('entry_edge')}
    assert pair_edges == {(f'0|{FAFB}|s-CPDN3A', f'1|{MCNS}|M1'),
                          (f'0|{FAFB}|s-CPDN3C', f'1|{MCNS}|M2'),
                          (f'0|{FAFB}|s-CPDN3D', f'1|{MCNS}|M1')}
    for u, v in pair_edges:
        data = graph[u][v]
        assert data['source_dataset'] == MCNS
        assert data['target_dataset'] == FAFB


def test_pair_network_type_query_has_no_entry():
    """A `type`-column query creates no entry node (§14 guard)."""
    from comparison.mapping_visualization import build_mapping_network_graph

    graph = build_mapping_network_graph(
        [_flow(MCNS, 'SMP227', FAFB, 's-CPDN3B', 6, 25)])
    assert not any(d['node_type'] == 'entry'
                   for _, d in graph.nodes(data=True))


def test_composed_cap_hides_same_name_first():
    pair_flows = {}
    for i in range(20):
        a, b = f'S{i}', f'S{i}'
        pair_flows[(MCNS, FAFB)] = pair_flows.get((MCNS, FAFB), []) + [
            _flow(MCNS, a, FAFB, b, 1, 1, linkers=False)]
    pair_flows[(MCNS, FAFB)].append(_flow(MCNS, 'L1', FAFB, 'L1', 4, 4))
    pair_flows[(MCNS, FAFB)].append(_flow(MCNS, 'L2', FAFB, 'L3', 2, 2))
    graph, meta = build_composed_mapping_graph(pair_flows, node_cap=12)
    assert meta['hidden_same_name'] > 0
    assert any('same-name types hidden' in n for n in meta['notes'])
    # the cap holds and linker-bearing types survive
    assert graph.number_of_nodes() <= 12
    remaining_labels = {d['label'] for _, d in graph.nodes(data=True)}
    assert {'L1', 'L2', 'L3'} <= remaining_labels


def test_composed_render_html():
    pair_flows = {
        (MCNS, FAFB): [_flow(MCNS, 'T1', FAFB, 'T1', 4, 4)],
    }
    html, meta = render_composed_mapping_html(pair_flows)
    assert html and 'cytoscape' in html.lower()
    assert '<title>Mapping graph</title>' in html
    empty, meta2 = render_composed_mapping_html({})
    assert empty is None


def test_bridges_csv_contract():
    flows = [_flow(MCNS, 'T1', FAFB, 'T1', 4, 4),
             _flow(MCNS, 'T2, X', FAFB, 'T3', 2, 3, linkers=False)]
    text = build_bridges_csv(flows, pools={
        ('T2, X', 'T3'): {'source_body_ids': [1, 2],
                          'target_body_ids': [3, 4, 5],
                          'source_coverage': 'covered 2 of 2 (100.0%)',
                          'target_coverage': 'covered 2 of 3 (66.7%)',
                          'source_type_body_ids': ['900', '901'],
                          'target_type_body_ids': ['400', '401', '402']}})
    lines = text.strip().splitlines()
    assert lines[0] == (
        'source_dataset,source_entry,matched_column,source_type,'
        'target_dataset,target_type,relationship,source_neurons,'
        'target_neurons,bridge,bridge_columns,mapping_origin,'
        'source_pool,source_total,target_pool,target_total,'
        'source_body_ids,target_body_ids,'
        'pool_coverage,pool_coverage_basis')
    # linker-bearing row: explicit endpoints, matched entry + column
    assert lines[1].startswith(
        'male-cns:v1.0,T1,type,T1,flywire_FAFB_v783,T1,1-to-1,4,4,')
    assert 'flywireType' in lines[1] and ',mapped,' in lines[1]
    # bare same-name row: no linker columns, same-name origin, pool
    # coverage + FULL per-type bodyId populations filled (quoting handles
    # the comma in the type name)
    assert '"T2, X"' in lines[2] and 'same name' in lines[2]
    assert 'source covered 2 of 2 (100.0%); target covered 2 of 3 (66.7%)' in lines[2]
    # bodyIds export as brace-wrapped comma-separated lists: ONE quoted CSV
    # field, so comma-containing values stay intact for spreadsheet readers.
    import csv as _csv
    import io as _io

    rows = list(_csv.reader(_io.StringIO(text)))
    assert len(rows[0]) == 20
    assert rows[2][16] == '{900, 901}'
    assert rows[2][17] == '{400, 401, 402}'
    # a pool without the per-type keys leaves the bodyId cells empty
    assert rows[1][16] == '' and rows[1][17] == ''

    taxonomy_text = build_bridges_csv([
        _flow(MCNS, 'T1', FAFB, 'T1', 4, 4,
              origin="cell_type · 'circadian_clock'")])
    taxonomy_reader = _csv.reader(_io.StringIO(taxonomy_text))
    next(taxonomy_reader)  # header
    taxonomy_row = next(taxonomy_reader)
    assert taxonomy_row[1:4] == ['circadian_clock', 'cell_type', 'T1']
    assert build_bridges_csv([]) is None


def test_bridges_csv_relationship_reflects_fan_in():
    """Pair rows label cardinality from BOTH fan directions: two source
    types converging on one target read N-to-1 instead of two 1-to-1 rows
    (user 2026-09-10: 5th-LNv and LNd_CRY+_ITP+ both map to MCNS
    5thsLNv_LNd6); forward fan-out stays 1-to-N and both directions give
    N-to-N."""
    import csv as _csv
    import io as _io

    flows = [
        _flow(MCNS, 'T1', FAFB, 'T1', 4, 4, linkers=False),
        _flow(MCNS, '5th-LNv', FAFB, 'Shared', 2, 4, linkers=False),
        _flow(MCNS, 'LNd_CRY+_ITP+', FAFB, 'Shared', 2, 4, linkers=False),
        _flow(MCNS, 'Splitter', FAFB, 'SplitA', 3, 1, linkers=False),
        _flow(MCNS, 'Splitter', FAFB, 'SplitB', 3, 1, linkers=False),
        _flow(MCNS, 'MeshA', FAFB, 'MeshX', 1, 1, linkers=False),
        _flow(MCNS, 'MeshA', FAFB, 'MeshY', 1, 1, linkers=False),
        _flow(MCNS, 'MeshB', FAFB, 'MeshX', 1, 1, linkers=False),
        _flow(MCNS, 'MeshB', FAFB, 'MeshY', 1, 1, linkers=False),
    ]
    rows = list(_csv.reader(_io.StringIO(build_bridges_csv(flows))))
    relationship = {(r[3], r[5]): r[6] for r in rows[1:]}
    assert relationship[('T1', 'T1')] == '1-to-1'
    assert relationship[('5th-LNv', 'Shared')] == 'N-to-1'
    assert relationship[('LNd_CRY+_ITP+', 'Shared')] == 'N-to-1'
    assert relationship[('Splitter', 'SplitA')] == '1-to-N'
    assert relationship[('Splitter', 'SplitB')] == '1-to-N'
    assert relationship[('MeshA', 'MeshX')] == 'N-to-N'
    assert relationship[('MeshA', 'MeshY')] == 'N-to-N'
    assert relationship[('MeshB', 'MeshX')] == 'N-to-N'
    assert relationship[('MeshB', 'MeshY')] == 'N-to-N'


def test_coverage_forward_rows_label_converging_sources_n_to_1():
    """Forward coverage rows escalate to N-to-1 when the target also
    receives other queried types; pure fan-out keeps 1-to-N."""
    from comparison.mapping_visualization import build_type_coverage

    pair_flows = {
        (MCNS, FAFB): [
            _flow(MCNS, '5th-LNv', FAFB, 'Shared', 2, 4, linkers=False),
            _flow(MCNS, 'LNd_CRY+_ITP+', FAFB, 'Shared', 2, 4,
                  linkers=False),
            _flow(MCNS, 'T1', FAFB, 'T1', 4, 4, linkers=False),
            _flow(MCNS, 'Splitter', FAFB, 'SplitA', 3, 1, linkers=False),
            _flow(MCNS, 'Splitter', FAFB, 'SplitB', 3, 1, linkers=False),
        ],
    }
    coverage = build_type_coverage(pair_flows)
    relationship = {r['type']: r['relationship']
                    for r in coverage['forward']}
    assert relationship['5th-LNv'] == 'N-to-1'
    assert relationship['LNd_CRY+_ITP+'] == 'N-to-1'
    assert relationship['T1'] == '1-to-1'
    assert relationship['Splitter'] == '1-to-N'


def test_zero_count_endpoints_still_render_labeled():
    """A mapped type with 0 rows in the SELECTED table (the BANC v888
    DN1pE report: the name resolves in the mapper's v626-keyed BANC
    namespace but the selected v888 table has none) must still render
    as a labeled node — networkx would otherwise auto-create an
    attribute-less node on add_edge and the vispath renderer falls back
    to the raw ``<layer>|<dataset>|<type>`` id as the label."""
    graph, _meta = build_composed_mapping_graph({
        (MCNS, FAFB): [_flow(MCNS, 'T1', FAFB, 'T1', 4, 4,
                             linkers=False)],
        (BANC, FAFB): [_flow(BANC, 'DN1pE', FAFB, 'DN1pE', 0, 4,
                             linkers=False)],
    })
    for nid, data in graph.nodes(data=True):
        assert data.get('label'), nid
    banc_node = next(d for n, d in graph.nodes(data=True)
                     if BANC in n and d.get('label') == 'DN1pE')
    assert '(0 neurons)' in banc_node['title']
    assert 'matched: DN1pE (FAFB)' in banc_node['title']


def test_combined_bridges_csv_uniform_width():
    """All-pairs combined export: one fixed-width schema for every pair —
    the per-pair CSVs are plain header + rows concatenations (the old
    pivoted bridge-<column> fields needed a union-of-columns hack to
    avoid the Tablecruncher ragged-rows bug)."""
    import csv
    import io

    pair_a = [_flow(MCNS, 'T1', FAFB, 'T1', 4, 4)]                # flywireType linker
    pair_b = [_flow(FAFB, 'T9', BANC, 'T9', 2, 2, linkers=False)]  # bare same-name

    headers = set()
    widths = set()
    for text in (build_bridges_csv(pair_a), build_bridges_csv(pair_b)):
        rows = list(csv.reader(io.StringIO(text)))
        headers.add(tuple(rows[0]))
        widths.update(len(r) for r in rows)
    assert len(headers) == 1
    assert widths == {20}
    assert headers.pop()[:2] == ('source_dataset', 'source_entry')


def test_extended_bridges_csv_exposes_selected_and_all_valid_scopes():
    """The UI export keeps selected evidence and supported alternatives distinct."""
    import csv
    import io

    flow = _flow(MCNS, 'T1', FAFB, 'T1', 4, 4)
    selected = [
        {'dataset': MCNS, 'column': 'type', 'value': 'T1'},
        {'dataset': MCNS, 'column': 'flywireType',
         'value': 'auto:LMTe01'},
        {'dataset': FAFB, 'column': 'type', 'value': 'T1'},
    ]
    alternate = [
        {'dataset': MCNS, 'column': 'type', 'value': 'T1'},
        {'dataset': MCNS, 'column': 'flywireType', 'value': 'LMTe02'},
        {'dataset': FAFB, 'column': 'type', 'value': 'T1'},
    ]
    flow['bridges'] = [selected, alternate]
    flow['mapping_status'] = 'valid_split_evidence'
    pool = {
        'selected_chain': selected,
        'valid_chains': [selected, alternate],
        'selected_chain_rank': 1,
        'valid_chain_count': 2,
        'source_body_ids': ['s1'],
        'target_body_ids': ['t1'],
        'source_type_body_ids': ['s1', 's2', 's3'],
        'target_type_body_ids': ['t1', 't2'],
        'all_valid_source_body_ids': ['s1', 's2'],
        'all_valid_target_body_ids': ['t1', 't2'],
        'source_pool_size': 1,
        'source_type_total': 3,
        'target_pool_size': 1,
        'target_type_total': 2,
        'all_valid_source_pool_size': 2,
        'all_valid_source_type_total': 3,
        'all_valid_target_pool_size': 2,
        'all_valid_target_type_total': 2,
        'source_basis': 'linker rows',
        'target_basis': 'full population',
        'all_valid_source_basis': 'union of supported bridge pools',
        'all_valid_target_basis': 'full population',
        'coverage_basis': 'independent endpoint pools; no bodyId pairing',
        'coverage_scope': 'selected bridge; all valid alternatives retained',
        'all_valid_source_overlap_count': 0,
        'all_valid_target_overlap_count': 0,
        'attempts': [
            {'rank': 0, 'supported': False, 'status': 'unsupported',
             'reason': 'target-side linker rows had no bodyIds'},
        ],
    }
    rows = list(csv.reader(io.StringIO(build_bridges_csv(
        [flow], pools={('T1', 'T1'): pool}, extended=True))))
    header = rows[0]
    row = dict(zip(header, rows[1]))
    assert len(header) == 35
    assert row['mapping_status'] == 'valid_split_evidence'
    assert row['selected_bridge_rank'] == '1'
    assert row['valid_bridge_count'] == '2'
    assert row['selected_linker_values'] == 'auto:LMTe01'
    assert row['selected_linker_canonical_values'] == 'LMTe01'
    assert row['all_valid_source_pool'] == '2'
    assert row['all_valid_target_pool'] == '2'
    assert row['coverage_scope'] == (
        'selected bridge; all valid alternatives retained')
    assert 'target-side linker rows had no bodyIds' in row[
        'unsupported_attempts']


def test_pair_flow_weight_shared_formula():
    """ONE per-pair weight formula for every artifact (user 2026-09-07).

    The network used to duplicate the SOURCE type's whole count onto
    every edge (all edges of a 12-neuron type showed "12"), the linker
    graph fell back foreign-first, and the Sankey used the pooled min —
    three artifacts, three numbers for the same pair."""
    from comparison.mapping_visualization import pair_flow_weight

    # no pool: min of the two sides (source-first fallback when one side
    # is unknown)
    assert pair_flow_weight(
        {'source_count': 12, 'foreign_count': 4}) == 4
    assert pair_flow_weight(
        {'source_count': 2, 'foreign_count': 12}) == 2
    assert pair_flow_weight({'source_count': 0, 'foreign_count': 12}) == 12
    assert pair_flow_weight({'foreign_count': 12}) == 12
    assert pair_flow_weight({}) == 1
    # pool present: the pooled bodyId granularity wins per side
    pool = {'source_body_ids': ['a', 'b'], 'target_body_ids': ['x']}
    assert pair_flow_weight(
        {'source_count': 12, 'foreign_count': 12}, pool) == 1
    # a one-sided pool falls back to that side's neuron count
    one_sided = {'source_body_ids': [], 'target_body_ids': ['x'] * 5}
    assert pair_flow_weight(
        {'source_count': 2, 'foreign_count': 12}, one_sided) == 2


def test_endpoint_pool_counts_union_not_max():
    """N-to-1 targets pool a DIFFERENT disjoint bodyId subset per
    counterpart — the hover count must be the UNION across the pairs,
    not the largest single subset (APDN3 showed "pool 4 bodyIds" next to
    "12 neurons" because CL125/SLP249 pool 4 each and PLP080/SLP250 2
    each)."""
    from comparison.mapping_visualization import _endpoint_pool_counts

    pools = {
        ('CL125', 'APDN3'): {
            'source_body_ids': ['a', 'b', 'c', 'd'],
            'target_body_ids': ['1', '2', '3', '4']},
        ('SLP249', 'APDN3'): {
            'source_body_ids': ['e', 'f', 'g', 'h'],
            'target_body_ids': ['5', '6', '7', '8']},
        ('PLP080', 'APDN3'): {
            'source_body_ids': ['i', 'j'],
            'target_body_ids': ['9', '10']},
        ('SLP250', 'APDN3'): {
            'source_body_ids': ['k', 'l'],
            'target_body_ids': ['11', '12']},
    }
    src, tgt = _endpoint_pool_counts(pools)
    assert tgt == {'APDN3': 12}  # union — the old max said 4
    assert src == {'CL125': 4, 'SLP249': 4, 'PLP080': 2, 'SLP250': 2}
    # overlapping pools count shared bodyIds once
    pools[('DN1', 'APDN3')] = {
        'source_body_ids': ['m'],
        'target_body_ids': ['12', '13']}
    _src, tgt = _endpoint_pool_counts(pools)
    assert tgt['APDN3'] == 13


def test_mapping_pools_are_scoped_by_dataset_direction_and_type():
    """The same type names in opposite directions must not swap coverage."""
    from comparison.mapping_visualization import (
        build_type_coverage,
        get_mapping_pool,
    )

    forward = _flow(FAFB, 'l-LNv', BANC, 'l-LNv', 8, 6)
    reverse = _flow(BANC, 'l-LNv', FAFB, 'l-LNv', 6, 8)
    pair_flows = {(FAFB, BANC): [forward], (BANC, FAFB): [reverse]}
    pools = {
        (FAFB, BANC, 'l-LNv', 'l-LNv'): {
            'source_body_ids': ['f1', 'f2'],
            'target_body_ids': ['b1'],
        },
        (BANC, FAFB, 'l-LNv', 'l-LNv'): {
            'source_body_ids': ['b1', 'b2', 'b3'],
            'target_body_ids': ['f1', 'f2', 'f3', 'f4'],
        },
    }

    assert get_mapping_pool(pools, forward)['source_body_ids'] == ['f1', 'f2']
    assert get_mapping_pool(pools, reverse)['source_body_ids'] == [
        'b1', 'b2', 'b3']

    coverage = build_type_coverage(pair_flows, pools)
    rows = {(row['dataset'], row['type']): row
            for row in coverage['forward']}
    assert rows[(FAFB, 'l-LNv')]['query_cov'] == '2 of 8 (25.0%)'
    assert rows[(FAFB, 'l-LNv')]['target_cov'] == '1 of 6 (16.7%)'
    assert rows[(BANC, 'l-LNv')]['query_cov'] == '3 of 6 (50.0%)'
    assert rows[(BANC, 'l-LNv')]['target_cov'] == '4 of 8 (50.0%)'


def test_type_coverage_forward_1_to_n_and_totals():
    """Forward coverage rows (user 2026-09-07): the queried type's own
    neuron count, its TOTAL mapped number (union of its per-pair pools),
    the relationship label, and per-side x-of-y coverage."""
    from comparison.mapping_visualization import build_type_coverage

    pair_flows = {(FAFB, MCNS): [
        _flow(FAFB, 'APDN3', MCNS, 'CL125', 12, 4),
        _flow(FAFB, 'APDN3', MCNS, 'PLP080', 12, 2),
        _flow(FAFB, 'APDN3', MCNS, 'SLP249', 12, 4),
        _flow(FAFB, 'APDN3', MCNS, 'SLP250', 12, 2),
    ]}
    pools = {
        ('APDN3', 'CL125'): {'source_body_ids': ['1', '2', '3', '4'],
                             'target_body_ids': ['a', 'b', 'c', 'd']},
        ('APDN3', 'PLP080'): {'source_body_ids': ['5', '6'],
                              'target_body_ids': ['e', 'f']},
        # SLP249 / SLP250 have NO pool entry (unpoolable pair): they
        # contribute nothing to the unions
    }
    coverage = build_type_coverage(pair_flows, pools)
    forward = coverage['forward']
    assert len(forward) == 1
    row = forward[0]
    assert (row['type'], row['dataset'], row['count']) == (
        'APDN3', FAFB, 12)
    assert row['relationship'] == '1-to-N'
    assert row['maps_to'] == 'MCNS: CL125, PLP080, SLP249, SLP250'
    # total mapped number: union of the pooled source-side subsets
    assert row['query_cov'] == '6 of 12 (50.0%)'
    # target-side coverage sums only the pooled pairs
    assert row['target_cov'] == '6 of 6 (100.0%)'

    reverse = coverage['reverse']
    by_type = {r['type']: r for r in reverse}
    assert by_type['CL125']['relationship'] == '1-to-1'
    assert by_type['CL125']['source_cov'] == '4 of 12 (33.3%)'
    assert by_type['CL125']['target_cov'] == '4 of 4 (100.0%)'
    assert by_type['SLP249']['source_cov'] == 'not pooled'


def test_type_coverage_reverse_fanout_label_and_unions():
    """User report (circadian_clock): three FAFB types map onto ONE
    male-cns type.  §12.1 user decision (2026-09-09): relationship cells
    follow the ROW SUBJECT's fan-out, so the backward row reads 1-to-N
    (read from the receiving type back to its sources) — never 1-to-1 —
    and carries both sides' coverage (source-side union vs the receiving
    type's own population)."""
    from comparison.mapping_visualization import build_type_coverage

    sources = [('A', 5, ['s1', 's2', 's3', 's4', 's5']),
               ('B', 7, ['s6', 's7', 's8', 's9', 's10', 's11', 's12']),
               ('C', 2, ['s13', 's14'])]
    pair_flows = {(FAFB, MCNS): [
        _flow(FAFB, name, MCNS, 'SMP227', count, 4)
        for name, count, _ids in sources]}
    pools = {
        (name, 'SMP227'): {
            'source_body_ids': ids,
            'target_body_ids': ['t1', 't2', 't3', 't4'] if i == 0 else [],
        }
        for i, (name, _count, ids) in enumerate(sources)}
    reverse = build_type_coverage(pair_flows, pools)['reverse']
    assert len(reverse) == 1
    row = reverse[0]
    assert (row['type'], row['dataset'], row['count']) == (
        'SMP227', MCNS, 4)
    assert row['relationship'] == '1-to-N'
    assert row['sources'] == 3
    assert row['mapped_from'] == 'FAFB: A, B, C'
    # source side: 14 pooled of 14 queried; target side: the union (4)
    # of the receiving type's 4 bodyIds
    assert row['source_cov'] == '14 of 14 (100.0%)'
    assert row['target_cov'] == '4 of 4 (100.0%)'
    # fan-out rows sort first
    assert reverse[0]['relationship'] == '1-to-N'
    # query-scoped rows carry no dataset-wide fields
    assert row['coverage_scope'] == 'query'
    assert 'incoming_source_count' not in row


def test_type_coverage_reverse_dataset_wide_incoming_context():
    """§12.3: a reverse context upgrades the backward row to the
    dataset-wide incoming scope — the full incoming family in
    mapped_from (active query marked), the relationship from the FULL
    family, coverage cells re-measured with the incoming population
    union as the source denominator, and the query-scoped slice
    preserved on query_scope_* fields."""
    from comparison.mapping_visualization import build_type_coverage

    sources = [('A', 5, ['s1', 's2', 's3', 's4', 's5']),
               ('B', 7, ['s6', 's7', 's8', 's9', 's10', 's11', 's12'])]
    pair_flows = {(FAFB, MCNS): [
        _flow(FAFB, name, MCNS, 'SMP227', count, 4)
        for name, count, _ids in sources]}
    pools = {
        ('A', 'SMP227'): {
            'source_body_ids': ['s1', 's2', 's3'],
            'target_body_ids': ['t1', 't2'],
        },
    }
    # the dataset-wide family adds source 'C' (not part of the query)
    contexts = {(MCNS, 'SMP227'): {
        'source_dataset': FAFB,
        'target_dataset': MCNS,
        'receiving_type': 'SMP227',
        'receiving_count': 4,
        'sources': [
            {'type': 'A', 'count': 5, 'pooled': True,
             'selected_source_pool_size': 3, 'selected_target_pool_size': 2,
             'all_valid_source_pool_size': 3, 'all_valid_target_pool_size': 2},
            {'type': 'B', 'count': 7, 'pooled': False,
             'selected_source_pool_size': 0, 'selected_target_pool_size': 0,
             'all_valid_source_pool_size': 0, 'all_valid_target_pool_size': 0},
            {'type': 'C', 'count': 8, 'pooled': True,
             'selected_source_pool_size': 2, 'selected_target_pool_size': 1,
             'all_valid_source_pool_size': 4, 'all_valid_target_pool_size': 3},
        ],
        'incoming_source_count': 3,
        'truncated': False,
        'source_population_total': 20,
        'pooled': True,
        'selected_source_union_ids': ['s1', 's2', 's3', 'c1', 'c2'],
        'all_valid_source_union_ids': ['s1', 's2', 's3', 'c1', 'c2',
                                       'c3', 'c4'],
        'selected_target_union_ids': ['t1', 't2', 't3'],
        'all_valid_target_union_ids': ['t1', 't2', 't3', 't4'],
        'source_overlap_selected_ids': [],
        'source_overlap_all_valid_ids': [],
        'target_overlap_selected_ids': [],
        'target_overlap_all_valid_ids': [],
        'selected_source_measured': True,
        'selected_target_measured': True,
        'all_valid_source_measured': True,
        'all_valid_target_measured': True,
    }}
    row = build_type_coverage(
        pair_flows, pools, reverse_contexts=contexts)['reverse'][0]
    assert row['relationship'] == '1-to-N'
    assert row['coverage_scope'] == 'dataset-wide incoming'
    assert row['incoming_source_count'] == 3
    assert row['active_query_sources'] == ['A', 'B']
    assert row['truncated'] is False
    # full incoming family listed (it lives on the SOURCE dataset side);
    # the active query members marked
    assert row['mapped_from'] == (
        'FAFB: A, B, C — active query: A, B')
    # source denominator is the incoming population union (5 + 7 + 8)
    assert row['source_cov_selected'] == '5 of 20 (25.0%)'
    assert row['source_cov'] == '7 of 20 (35.0%)'
    # target side keeps the receiving population as the denominator
    assert row['target_cov_selected'] == '3 of 4 (75.0%)'
    assert row['target_cov'] == '4 of 4 (100.0%)'
    # the query-scoped slice survives on the row
    assert row['query_scope_relationship'] == '1-to-N'
    assert row['query_scope_mapped_from'] == 'FAFB: A, B'
    # query-scope denominator counts only the measured sources (A)
    assert row['query_scope_source_cov_selected'] == '3 of 5 (60.0%)'
    assert 'dataset-wide incoming: 3 source types' in row['coverage_note']


def test_format_coverage_states():
    """The shared one-side coverage formatter: thousands separators, a
    one-decimal share, and the measured-zero vs unmeasured distinction."""
    from comparison.mapping_visualization import format_coverage

    assert format_coverage(1655, 1683) == '1,655 of 1,683 (98.3%)'
    assert format_coverage(2, 8) == '2 of 8 (25.0%)'
    assert format_coverage(4, 4) == '4 of 4 (100.0%)'
    assert format_coverage(0, 168) == '0 of 168 (0.0%)'   # measured zero
    assert format_coverage(0, 0) == '0 of 0'              # degenerate
    assert format_coverage(3, None) == 'not measured'     # side unknown


def test_type_coverage_unmeasured_side_reads_not_measured():
    """A pool whose side's coverage index was unavailable must render
    'not measured' — never a fake '0 of n' measured zero."""
    from comparison.mapping_visualization import build_type_coverage

    pair_flows = {(FAFB, MCNS): [
        _flow(FAFB, 'T1', MCNS, 'T9', 8, 12, linkers=False)]}
    pools = {
        ('T1', 'T9'): {
            'source_body_ids': [],
            'target_body_ids': [],
            'source_basis': 'unmeasured',
            'target_basis': 'linker rows',
        },
    }
    forward = build_type_coverage(pair_flows, pools)['forward'][0]
    assert forward['query_cov'] == 'not measured'
    # the target side measured a real zero: 0 of 12
    assert forward['target_cov'] == '0 of 12 (0.0%)'
