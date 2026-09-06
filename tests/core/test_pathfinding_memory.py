#!/usr/bin/env python
"""
Tests for the pathfinding memory optimizations (2026-08 audit, stage 1+2).

Covers:
  - Opt-in max_paths_bodyid path cap (truncation + warning; default unbounded)
  - FastGraph slim edge-attr mode (weights identical, no attr dicts)
  - Dense-pivot cell guard in the matrix exporters
  - The compact _ConnRowIndex map replacing dict-of-lists row indexes
  - Byte-budget eviction of the FindAllPath graph cache
  - _match_path_edges_to_layers Polars-join equivalence with the old sets
"""

import os
import sys
from pathlib import Path

import pandas as pd
import polars as pl
import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "vispath-subproject" / "src"))

import coana  # noqa: E402
from coana import (  # noqa: E402
    _ConnRowIndex,
    _DENSE_PIVOT_CELL_LIMIT,
    _FINDALLPATH_GRAPH_CACHE,
    _match_path_edges_to_layers,
)
from tests.core.test_pathfinding import _make_pipeline_fc  # noqa: E402
from vispath_pkg.fast_graph_core import FastGraph  # noqa: E402

# Diamond: two disjoint 2-hop paths S->A->T and S->B->T
_DIAMOND_EDGES = [("S", "A", 5), ("S", "B", 5), ("A", "T", 5), ("B", "T", 5)]


def _make_fc(**attrs):
    fc = object.__new__(coana.FindNeuronConnection)
    fc._vprint = lambda msg="", level="full", end="\n", flush=False: None
    fc.verbose_mode = "silent"
    fc.progress_events = False
    fc._warn_notes = []
    for key, value in attrs.items():
        setattr(fc, key, value)
    return fc


# =============================================================================
# Opt-in path cap
# =============================================================================

class TestPathCap:
    def setup_method(self):
        _FINDALLPATH_GRAPH_CACHE.clear()

    def teardown_method(self):
        _FINDALLPATH_GRAPH_CACHE.clear()

    def test_dataclass_default_is_none(self):
        assert coana.FindNeuronConnection.__dataclass_fields__[
            "max_paths_bodyid"].default is None

    def test_cap_routes_to_strongest_first(self, monkeypatch, tmp_path):
        # unequal weights so the budget genuinely drops the weaker route
        edges = [("S", "A", 5), ("A", "T", 3), ("S", "B", 2), ("B", "T", 6)]
        """Fix C (2026-09-04): a positive max_paths_bodyid routes through
        the StrongestFirst enumerator regardless of the selected algorithm
        — the arbitrary-order TRUNCATED cap no longer exists. The output is
        the strongest path(s) with the reported tau."""
        fc, _fetch_calls, _logs = _make_pipeline_fc(
            monkeypatch, tmp_path, edges, max_interlayer=2)
        fc.max_paths_bodyid = 1
        fc.FindAllPath()

        # strongest path kept; weakest alternative dropped at tau
        assert any("max_paths_bodyid" in note for note in fc._warn_notes)
        assert not any("TRUNCATED" in note for note in fc._warn_notes)
        type_csv = os.path.join(
            fc.allpath_folder, "src_to_tgt_allpaths_type.csv")
        assert os.path.exists(type_csv)
        assert len(pl.read_csv(type_csv)) == 1

    def test_default_run_is_unbounded(self, monkeypatch, tmp_path):
        fc, _fetch_calls, _logs = _make_pipeline_fc(
            monkeypatch, tmp_path, _DIAMOND_EDGES, max_interlayer=2)
        assert fc.max_paths_bodyid is None
        fc.FindAllPath()

        assert not any("max_paths_bodyid" in note for note in fc._warn_notes)
        type_csv = os.path.join(
            fc.allpath_folder, "src_to_tgt_allpaths_type.csv")
        assert len(pl.read_csv(type_csv)) == 2


# =============================================================================
# FastGraph slim edge-attr mode
# =============================================================================

class TestFastGraphSlimAttrs:
    def test_default_stores_attrs(self):
        G = FastGraph()
        G.build_from_dataframe(pl.DataFrame({
            "u": ["A", "B"], "v": ["B", "C"], "w": [2, 3]}),
            "u", "v", "w")
        assert G.edge_attrs[("A", "B")]["weight"] == 2
        assert G.store_edge_attrs is True

    def test_slim_build_keeps_weights_drops_attrs(self):
        G = FastGraph()
        # duplicate pair across "layers" must still sum to 5
        G.build_from_dataframe(pl.DataFrame({
            "u": ["A", "B", "A"], "v": ["B", "C", "B"], "w": [2, 3, 3]}),
            "u", "v", "w", store_edge_attrs=False)
        assert G.adj["A"]["B"] == 5
        assert G.adj["B"]["C"] == 3
        assert G.edge_attrs == {}
        assert G.number_of_edges() == 2

    def test_derived_graphs_of_slim_graph_work(self):
        G = FastGraph()
        G.build_from_dataframe(pl.DataFrame({
            "u": ["A", "B"], "v": ["B", "C"], "w": [2, 3]}),
            "u", "v", "w", store_edge_attrs=False)
        R = G.reverse(copy=False)
        assert R.adj["B"]["A"] == 2
        sub = G.subgraph(["A", "B"])
        assert sub.adj["A"]["B"] == 2
        assert sub.edge_attrs == {}


# =============================================================================
# Dense-pivot guard
# =============================================================================

class TestDensePivotGuard:
    def _frame(self, n):
        return pl.DataFrame({
            "bodyId_pre": [f"pre{i}" for i in range(n)],
            "bodyId_post": [f"post{j}" for j in range(n)],
            "weight": [1] * n,
            "nt_type": ["ACh"] * n,
        })

    def test_csv_guard_skips_dense_pivot(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setattr(coana, "_DENSE_PIVOT_CELL_LIMIT", 10)
        fc = _make_fc()
        fc._save_matrices_to_csv(self._frame(6), str(tmp_path), level="bodyId")
        # 6x6 = 36 cells > 10: nothing written, warning printed
        assert list(tmp_path.iterdir()) == []
        assert "exceeds the" in capsys.readouterr().out

    def test_csv_writes_within_budget(self, tmp_path):
        fc = _make_fc()
        fc._save_matrices_to_csv(self._frame(3), str(tmp_path), level="bodyId")
        assert (tmp_path / "conn_mat_bodyId_nt.csv").exists()

    def test_excel_guard_returns_before_touching_writer(self, monkeypatch):
        monkeypatch.setattr(coana, "_DENSE_PIVOT_CELL_LIMIT", 10)

        class _BoomWriter:
            def __getattr__(self, name):
                raise AssertionError("writer must not be touched")

        fc = _make_fc()
        # Must not raise despite the exploding writer: guard returns first
        fc._save_matrices_to_excel(self._frame(6).to_pandas(), _BoomWriter(),
                                   level="bodyId")


# =============================================================================
# Compact row index
# =============================================================================

class TestConnRowIndex:
    def test_from_groups_contract(self):
        index = _ConnRowIndex.from_groups([("1", [0, 1]), ("2", [2])])
        assert index["1"] == [0, 1]
        assert index["2"] == [2]
        assert index.get("1") == [0, 1]
        assert index.get("missing", []) == []
        assert "1" in index and "missing" not in index
        assert sorted(index.keys()) == ["1", "2"]
        assert len(index) == 2 and bool(index)
        assert all(isinstance(i, int) for i in index["1"])

    def test_missing_key_semantics(self):
        index = _ConnRowIndex.from_groups([])
        assert len(index) == 0 and not index
        with pytest.raises(KeyError):
            index["nope"]
        assert index.get("nope") is None

    def test_from_dict_and_repr(self):
        index = _ConnRowIndex.from_dict({"a": [3, 4]})
        assert index["a"] == [3, 4]
        assert "_ConnRowIndex" in repr(index)


# =============================================================================
# Graph cache byte budget
# =============================================================================

class TestGraphCacheBudget:
    def setup_method(self):
        _FINDALLPATH_GRAPH_CACHE.clear()

    def teardown_method(self):
        _FINDALLPATH_GRAPH_CACHE.clear()

    @staticmethod
    def _entry(rows=1000):
        return {
            "threshold": 1, "depth": 2, "complete": True,
            "all_connections": [pl.DataFrame({
                "bodyId_pre": ["a"] * rows,
                "bodyId_post": ["b"] * rows,
                "weight": [1] * rows,
            })],
            "layer_neurons": [set()],
            "all_neurons_in_network": set(),
        }

    def test_budget_evicts_oldest(self, monkeypatch):
        monkeypatch.setattr(coana, "_FINDALLPATH_CACHE_BUDGET_BYTES", 1)
        coana._findallpath_cache_put("k1", self._entry())
        coana._findallpath_cache_put("k2", self._entry())
        # one entry always survives, but the older one was evicted
        assert "k2" in _FINDALLPATH_GRAPH_CACHE
        assert "k1" not in _FINDALLPATH_GRAPH_CACHE

    def test_within_budget_keeps_entries(self, monkeypatch):
        monkeypatch.setattr(coana, "_FINDALLPATH_CACHE_BUDGET_BYTES",
                            1024 * 1024 * 1024)
        coana._findallpath_cache_put("k1", self._entry(rows=5))
        coana._findallpath_cache_put("k2", self._entry(rows=5))
        assert set(_FINDALLPATH_GRAPH_CACHE) == {"k1", "k2"}

    def test_estimator_counts_frames(self):
        assert coana._findallpath_cache_entry_bytes(self._entry(rows=50)) > 0


# =============================================================================
# Layer matching equivalence
# =============================================================================

class TestLayerMatchEquivalence:
    EDGES = {("a", "b"), ("b", "c"), ("x", "y")}
    LAYERS = [
        pl.DataFrame({
            "bodyId_pre": ["a", "b", "q"],
            "bodyId_post": ["b", "c", "r"],
            "weight": [1, 2, 3],
        }),
        pl.DataFrame({"bodyId_pre": [], "bodyId_post": []}),
        None,
    ]

    def test_matches_polars_pandas_and_int_frames(self):
        pandas_layer = pd.DataFrame({
            "bodyId_pre": ["x", "q"], "bodyId_post": ["y", "r"]})
        int_layer = pl.DataFrame({
            "bodyId_pre": [10, 20], "bodyId_post": [20, 30]})
        layers = self.LAYERS[:1] + [pandas_layer, int_layer]

        valid_by_layer, matched = _match_path_edges_to_layers(
            self.EDGES, layers)

        assert valid_by_layer[0] == {("a", "b"), ("b", "c")}
        assert valid_by_layer[1] == {("x", "y")}
        # int bodyId layer casts to Utf8: ('10', '20') must NOT match the
        # string pair ('10', '20') absent from the edge set
        assert valid_by_layer[2] == set()
        assert matched == {("a", "b"), ("b", "c"), ("x", "y")}

    def test_matches_reference_set_implementation(self):
        layers = [
            pl.DataFrame({
                "bodyId_pre": ["a", "b", "a"],
                "bodyId_post": ["b", "c", "b"],
            }),
        ]
        valid_by_layer, matched = _match_path_edges_to_layers(
            self.EDGES, layers)
        reference = coana._layer_table_edge_pairs(layers[0])
        assert valid_by_layer[0] == self.EDGES & reference
        assert matched == self.EDGES & reference

    def test_empty_edges_short_circuits(self):
        valid_by_layer, matched = _match_path_edges_to_layers(set(), [
            pl.DataFrame({"bodyId_pre": ["a"], "bodyId_post": ["b"]})])
        assert valid_by_layer == [set()]
        assert matched == set()
