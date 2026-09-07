"""Unit tests for the Round 2 composed type-mapping view (spec §4/§6/§7).

Synthetic pair flows only — the composed graph builder's layer naming,
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


def _flow(s, st, t, tt, sc, fc, origin='type', linkers=True):
    hops = [
        {'dataset': s, 'column': 'type', 'value': st},
        {'dataset': 'male-cns:v1.0', 'column': 'flywireType', 'value': 'LMTe01'},
        {'dataset': t, 'column': 'type', 'value': tt},
    ] if linkers else [
        {'dataset': s, 'column': 'type', 'value': st},
        {'dataset': t, 'column': 'type', 'value': tt},
    ]
    return {'source_dataset': s, 'target_dataset': t, 'source_type': st,
            'foreign_type': tt, 'source_count': sc, 'foreign_count': fc,
            'matched_origin': origin, 'bridges': [hops]}


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
        (MCNS, FAFB): [
            _flow(MCNS, 'T1', FAFB, 'C1', 2, 2,
                  origin="cell_type · 'circadian_clock'"),
            _flow(MCNS, 'T2', FAFB, 'C2', 1, 3,
                  origin="cell_type · 'circadian_clock'"),
        ],
    }
    graph, _meta = build_composed_mapping_graph(pair_flows)
    entries = [n for n, d in graph.nodes(data=True)
               if d['node_type'] == 'entry']
    assert len(entries) == 1
    entry = entries[0]
    assert entry.startswith('E|')
    assert 'circadian_clock' in graph.nodes[entry]['label']
    # fed by BOTH covered types, cover stat in the hover
    preds = list(graph.predecessors(entry))
    assert len(preds) == 2
    assert 'covers 2 types' in graph.nodes[entry]['title']


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
    empty, meta2 = render_composed_mapping_html({})
    assert empty is None


def test_bridges_csv_contract():
    flows = [_flow(MCNS, 'T1', FAFB, 'T1', 4, 4),
             _flow(MCNS, 'T2, X', FAFB, 'T3', 2, 3, linkers=False)]
    text = build_bridges_csv(flows, pools={
        ('T2, X', 'T3'): {'source_body_ids': [1, 2],
                          'target_body_ids': [3, 4, 5],
                          'coverage': 'covered 2 of 3'}})
    lines = text.strip().splitlines()
    assert lines[0] == ('dataset,entry_kind,matched_column,name,foreign_type,'
                        'neuron_count,mapped_kind,mapped_to,map_used,'
                        'bridge-flywireType,pool_coverage')
    # linker-bearing row: bridge cell filled, quoting handles the comma
    assert 'male-cns:v1.0,type,type,T1,T1,4,mapped,T1,' in lines[1]
    # bare same-name row: empty bridge cell, pool coverage filled
    assert '"T2, X"' in lines[2] and 'same name' in lines[2]
    assert lines[2].endswith('covered 2 of 3')
    assert build_bridges_csv([]) is None


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
    """All-pairs combined export: one header, uniform field counts —
    pairs with different linker sets pad to the bridge-column union
    (the Tablecruncher ragged-rows bug)."""
    import csv
    import io

    from comparison.mapping_visualization import infer_bridge_columns

    pair_a = [_flow(MCNS, 'T1', FAFB, 'T1', 4, 4)]                # flywireType linker
    pair_b = [_flow(FAFB, 'T9', BANC, 'T9', 2, 2, linkers=False)]  # bare same-name
    union = list(dict.fromkeys(
        infer_bridge_columns(pair_a) + infer_bridge_columns(pair_b)))
    assert union == ['flywireType']

    widths = set()
    for text in (build_bridges_csv(pair_a, bridge_columns=union),
                 build_bridges_csv(pair_b, bridge_columns=union)):
        rows = list(csv.reader(io.StringIO(text)))
        assert {len(r) for r in rows} == {11}
        assert rows[0][-2:] == ['bridge-flywireType',
                                'pool_coverage']
        widths.add(len(rows[0]))
    assert len(widths) == 1
    # the bare pair pads the missing bridge column with an empty cell
    rows_b = list(csv.reader(io.StringIO(
        build_bridges_csv(pair_b, bridge_columns=union))))
    assert rows_b[1][9] == ''
    assert rows_b[1][-2:] == ['', '']


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
    assert row['query_cov'] == '6 of 12'
    # target-side coverage sums only the pooled pairs
    assert row['target_cov'] == '6 of 6'

    reverse = coverage['reverse']
    by_type = {r['type']: r for r in reverse}
    assert by_type['CL125']['relationship'] == '1-to-1'
    assert by_type['CL125']['source_cov'] == '4 of 12'
    assert by_type['CL125']['target_cov'] == '4 of 4'
    assert by_type['SLP249']['source_cov'] == 'not pooled'


def test_type_coverage_reverse_makes_n_to_1_explicit():
    """User report (circadian_clock): three FAFB types map onto ONE
    male-cns type — the reverse row must label it N-to-1 and carry both
    sides' coverage (source-side union vs summed source counts)."""
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
    assert row['relationship'] == 'N-to-1'
    assert row['sources'] == 3
    assert row['mapped_from'] == 'FAFB: A, B, C'
    # source side: 14 pooled of 14 queried; target side: the union (4)
    # of the receiving type's 4 bodyIds
    assert row['source_cov'] == '14 of 14'
    assert row['target_cov'] == '4 of 4'
    # N-to-1 rows sort first
    assert reverse[0]['relationship'] == 'N-to-1'
