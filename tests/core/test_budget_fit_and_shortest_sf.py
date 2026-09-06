"""§7.4 budget-fit search + §7.2 shortest-mode StrongestFirst regression
tests (plan: _plan/plan-pathfinding-doc-audit.md §7).

Budget-fit invariants:
- cap invariant: the floored cone never exceeds the budget;
- the fit search finds the brute-force-minimal fitting tier;
- floor_skipped when the full lossless-closed cone fits the cap;
- declined when no non-empty tier fits (single-tier cone);
- truncation at max_probes=1 degrades to the one-shot result.

Shortest StrongestFirst invariants:
- unbudgeted emission set == find_paths_shortest_backward (property);
- emission order is descending bottleneck;
- a bitten run is exactly "all min-hop paths with bottleneck >= tau".
"""

import random

import polars as pl

from coana import fit_edge_budget, apply_edge_budget_floor
from core.fast_graph import FastGraph


def _layer(edges, label='0->1'):
    return pl.DataFrame([{'bodyId_pre': u, 'bodyId_post': v, 'weight': w,
                          'conn_layer': label} for u, v, w in edges])


def _pairs(tables):
    out = set()
    for t in tables:
        if t.height:
            out |= set(zip(t['bodyId_pre'].to_list(),
                           t['bodyId_post'].to_list()))
    return out


def _cone(strong_pairs, weak_pairs=(), dead_pairs=()):
    """Two-edge live paths (S -> x -> T) plus optional dead branches."""
    edges = []
    for i, (w_pre, w_post) in enumerate(strong_pairs):
        edges.append(('S', f'A{i}', w_pre))
        edges.append((f'A{i}', 'T', w_post))
    for i, (w_pre, w_post) in enumerate(weak_pairs):
        edges.append(('S', f'C{i}', w_pre))
        edges.append((f'C{i}', 'T', w_post))
    for u, v, w in dead_pairs:
        edges.append((u, v, w))
    return edges


def _budget_graph():
    """One live path at w10, 10 live at w9 + 480 dead at w9, 15 live at w8,
    100 live at w5 (732 edges). The one-shot landing (w1 = 9) floors at
    w0 = 10 and keeps only 2 edges — 98% of the cap wasted on the
    boundary-tie mass the +1 excludes."""
    strong = [(10, 10)]
    weak = [(9, 9)] * 10 + [(8, 8)] * 15 + [(5, 5)] * 100
    dead = [(f'X{i}', f'Y{i}', 9) for i in range(480)]
    return _cone(strong, weak_pairs=weak, dead_pairs=dead)


# ---------------------------------------------------------------------------
# §7.4 budget-fit
# ---------------------------------------------------------------------------

def test_fit_floored_beats_one_shot():
    edges = _budget_graph()
    tables = [_layer(edges)]
    cap = 100
    one_out, one = apply_edge_budget_floor(tables, cap, ['S'], ['T'], 2)
    fit_out, fit = fit_edge_budget(tables, cap, ['S'], ['T'], 2)
    assert one['applied'] is True and fit['status'] == 'floored'

    one_kept = sum(t.height for t in one_out)
    fit_kept = sum(t.height for t in fit_out)
    # cap invariant
    assert fit_kept <= cap
    # the fit floor admits the boundary tier the +1 landing excluded
    assert fit_kept > one_kept
    assert fit['floor'] < one['floor']
    # the fit cone is a superset of the one-shot cone
    assert _pairs(one_out) <= _pairs(fit_out)
    assert fit['budget_fully_used'] or fit['truncated']


def test_fit_floor_skipped_when_closed_cone_fits():
    # 34 live edges + 10 dead (w2): total 44 > cap 40, but the closed cone
    # is 34 <= 40 — no floor may be applied.
    strong = [(6, 6), (5, 5)]
    weak = [(4, 4)] * 15
    dead = [(f'X{i}', f'Y{i}', 2) for i in range(10)]
    edges = _cone(strong, weak_pairs=weak, dead_pairs=dead)
    tables = [_layer(edges)]
    assert len(edges) == 44
    fit_out, fit = fit_edge_budget(tables, 40, ['S'], ['T'], 2)
    assert fit['status'] == 'no_floor_needed'
    assert fit['floor_skipped']
    kept = sum(t.height for t in fit_out)
    assert kept == 34


def test_fit_declined_single_tier():
    edges = []
    for i in range(10):
        edges.append(('S', f'A{i}', 5))
        edges.append((f'A{i}', 'T', 5))
    tables = [_layer(edges)]
    fit_out, fit = fit_edge_budget(tables, 10, ['S'], ['T'], 2)
    assert fit['status'] == 'declined'
    assert sum(t.height for t in fit_out) == len(edges)


def test_fit_truncated_degrades_to_one_shot():
    edges = _budget_graph()
    tables = [_layer(edges)]
    one_out, one = apply_edge_budget_floor(tables, 40, ['S'], ['T'], 2)
    fit_out, fit = fit_edge_budget(tables, 40, ['S'], ['T'], 2, max_probes=1)
    assert fit['status'] == 'floored' and one['applied'] is True
    assert fit['truncated']
    assert _pairs(fit_out) == _pairs(one_out)


# ---------------------------------------------------------------------------
# §7.2 shortest StrongestFirst
# ---------------------------------------------------------------------------

def _sf_graph():
    g = FastGraph()
    for u, v, w in [('S', 'A', 9), ('A', 'T', 9),
                    ('S', 'B', 5), ('B', 'T', 5),
                    ('S', 'C', 2), ('C', 'T', 2),
                    ('S', 'D', 7), ('D', 'T', 7)]:
        g.add_edge(u, v, w)
    return g


def test_shortest_sf_bitten_tau_set():
    g = _sf_graph()
    stats = {}
    paths = set(map(tuple, g.find_paths_shortest_strongest_first(
        ['T'], ['S'], 2, budget=2, stats=stats)))
    # four tied min-hop paths (bottlenecks 9 / 7 / 5 / 2) — budget 2 keeps
    # the two strongest, drains ties at tau=7, drops the rest.
    assert stats['emitted'] == 2
    assert stats['budget_bitten'] is True
    assert stats['tau'] == 7
    assert paths == {('S', 'A', 'T'), ('S', 'D', 'T')}


def test_shortest_sf_descending_order():
    g = _sf_graph()
    bns = [bn for bn, _p, _t in g.find_paths_shortest_strongest_first(
        ['T'], ['S'], 2)]
    assert bns == sorted(bns, reverse=True) and bns


def test_shortest_sf_unbitten_matches_backward():
    for seed in (0, 1, 2, 3, 4):
        rng = random.Random(seed)
        nodes = [f'n{i}' for i in range(10)]
        g = FastGraph()
        for u in nodes:
            for v in rng.sample(nodes, 4):
                if u != v:
                    g.add_edge(u, v, rng.randint(1, 9))
        sources, targets = nodes[:2], nodes[-2:]
        backward = set(map(tuple, g.find_paths_shortest_backward(
            targets, sources, 4, verbose=False)))
        sf = set(map(tuple, g.find_paths_shortest_strongest_first(
            targets, sources, 4, budget=None)))
        assert sf == backward, f'seed {seed}: set mismatch'
