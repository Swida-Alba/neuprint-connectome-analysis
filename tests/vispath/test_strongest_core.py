"""Shared strongest-selection core tests (§6c of the report-fixes plan).

The core is the single definition of "strongest" shared by the
StrongestFirst enumerator and the visualization edge/path selector.
"""

import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
for p in (PROJECT_ROOT, PROJECT_ROOT / "src", PROJECT_ROOT / "vispath-subproject" / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from vispath_pkg.strongest_core import (  # noqa: E402
    bottleneck,
    drain_budget,
    interior_strength,
    path_rank_key,
    selection_threshold,
    widest_path_backward,
)


_ADJ = {
    'S': {'A': 10, 'B': 4},
    'A': {'T': 7, 'B': 6},
    'B': {'T': 5},
    'T': {},
}


def test_widest_path_backward_values():
    W = widest_path_backward(_ADJ, {'T'}, 3)
    assert W[0]['T'] == float('inf')
    assert W[1]['A'] == 7          # A -> T
    assert W[1]['B'] == 5          # B -> T
    assert W[2]['S'] == 7          # S -> A -> T (min(10, 7))
    assert W[3]['S'] == 7


def test_fast_graph_delegates_to_core():
    from core.fast_graph import FastGraph
    import pandas as pd
    df = pd.DataFrame(
        [(u, v, w) for u, nbrs in
         {'S': {'A': 10, 'B': 4}, 'A': {'T': 7, 'B': 6}, 'B': {'T': 5}}.items()
         for v, w in nbrs.items()],
        columns=['pre', 'post', 'weight'])
    g = FastGraph()
    g.build_from_dataframe(df, 'pre', 'post', 'weight', store_edge_attrs=False)
    core_w = widest_path_backward(g.adj, {'T'}, 3)
    graph_w = g._widest_path_backward({'T'}, 3)
    assert core_w == graph_w


def test_path_rank_key_bottleneck_primary():
    # interior strength no longer outranks the full-path bottleneck (§6c)
    key = path_rank_key(weights=[1, 100, 60], path_probability=0.5)
    assert key[0] == 1 and key[1] == 100 and key[2] == 0.5
    assert path_rank_key([5, 5]) < path_rank_key([9, 9])


def test_selection_threshold():
    assert selection_threshold([5, 9, 2]) == 2
    assert selection_threshold([]) is None


def test_drain_budget_exact_tau_cutoff():
    ranked = [
        (10, ('S', 'T'), 'p1'),
        (10, ('S', 'T'), 'p2'),      # tie at tau — drained, not dropped
        (7, ('S', 'T'), 'p3'),
        (4, ('S', 'T'), 'p4'),
    ]
    selected, tau, bitten = drain_budget(ranked, budget=2, per_pair_k=None)
    # budget 2 + drained tie at tau=10 -> exactly the two strongest
    assert [ranked[i][1] for i in selected] == [('S', 'T'), ('S', 'T')]
    assert tau == 10 and bitten is True


def test_drain_budget_unbounded_selects_all():
    ranked = [(7, ('S', 'T'), 'a'), (3, ('S', 'T'), 'b')]
    selected, tau, bitten = drain_budget(ranked, budget=None)
    assert selected == [0, 1] and tau == 3 and bitten is False


def test_bottleneck_and_interior():
    assert bottleneck([10, 3, 50]) == 3
    assert interior_strength([10, 3, 50]) == 3
    # 2-hop paths have no strict interior -> falls back to the bottleneck
    assert interior_strength([10, 50]) == 10
