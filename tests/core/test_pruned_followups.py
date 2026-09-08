"""Tests for the pruned-run follow-ups (plan-pruned-run-followups.md).

Covers W1 (bridge band dedupe), W4 (floor-aware re-enumeration flag in
the analyzer batch), W5 (applied = max(applied, floor)), W2 (taxonomy
chip resolution), W3 (orphan nodes/rows), and the untyped-drop
duplicate-column regression.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "ui"))

FAFB = "flywire_FAFB_v783"
MCNS = "male-cns:v1.0"


# ---------------------------------------------------------------------------
# W1 — bridge band dedupe (fig 1)
# ---------------------------------------------------------------------------

def test_format_bridge_composed_landing_has_no_duplicate_band():
    from comparison.mapping_visualization import format_bridge
    chain = [
        {"dataset": "banc_v626", "column": "type", "value": "LNd_c"},
        {"dataset": FAFB, "column": "additional_type(s)",
         "value": "LNd_c"},
        {"dataset": FAFB, "column": "additional_type(s)",
         "value": "LNd_CRY-", "via": "LNd_c"},
    ]
    text = format_bridge(chain)
    assert text.count("LNd_c[FAFB·additional_type(s)]") == 1
    assert text.endswith("LNd_CRY-[FAFB·type]")


def test_format_bridge_distinct_columns_keep_both_bands():
    from comparison.mapping_visualization import format_bridge
    chain = [
        {"dataset": MCNS, "column": "type", "value": "CL125"},
        {FAFB: FAFB, "column": "flywireType", "value": "LMTe01"},
        {"dataset": FAFB, "column": "additional_type(s)",
         "value": "APDN3", "via": "LMTe01"},
    ]
    text = format_bridge(chain)
    assert "LMTe01[MCNS·flywireType]" in text
    assert "LMTe01[FAFB·additional_type(s)]" in text


# ---------------------------------------------------------------------------
# W5 — applied = max(applied, floor)
# ---------------------------------------------------------------------------

def _analyzer_with_meta(meta, key_threshold=3):
    import comparison.comparison_analyzer as ca

    class _Params:
        def __init__(self):
            self._auto_type_mapper = None

    a = object.__new__(ca.ComparisonAnalyzer)
    a.parameters = _Params()
    a._path_run_meta = {("banc_v888", key_threshold): meta}
    return a


def test_applied_state_floored_complete_applies_floor():
    a = _analyzer_with_meta({
        "tau": None, "tau_canonical": None, "budget_bitten": False,
        "paths_complete": True, "skipped": False, "duplicate_of": None,
        "applied_folder": 3, "edge_weight_floor": 151.0,
    })
    applied, pruned, floor, source = a._applied_state_for("banc_v888", 3)
    assert applied == 151  # W5: the floor IS the applied threshold
    assert pruned is True and floor == 151.0
    assert source == "edge_budget"


def test_applied_state_floored_bitten_keeps_canonical():
    a = _analyzer_with_meta({
        "tau": 40.0, "tau_canonical": 39, "budget_bitten": True,
        "paths_complete": False, "skipped": False, "duplicate_of": None,
        "applied_folder": 3, "edge_weight_floor": 35.0,
    })
    applied, _pruned, _floor, _source = a._applied_state_for("banc_v888", 3)
    assert applied == 39  # canonical >= floor: unchanged


def test_applied_state_unfloored_unchanged():
    a = _analyzer_with_meta({
        "tau": 25.0, "tau_canonical": 24, "budget_bitten": True,
        "paths_complete": False, "skipped": False, "duplicate_of": None,
        "applied_folder": 10, "edge_weight_floor": None,
    }, key_threshold=10)
    applied, _pruned, _floor, _source = a._applied_state_for("banc_v888", 10)
    assert applied == 24


# ---------------------------------------------------------------------------
# W4 — the analyzer batch honors the orchestrator's re-enumeration flag
# ---------------------------------------------------------------------------

def test_replay_batch_reenumerates_flagged_thresholds(monkeypatch):
    import comparison.comparison_analyzer as ca

    calls = {"ran": []}
    saved = {}

    def fake_run_path_analysis(ds, threshold, verbose_mode="silent"):
        calls["ran"].append(threshold)
        return pd.DataFrame([{"type_pre": "A", "type_post": "B"}])

    a = object.__new__(ca.ComparisonAnalyzer)
    a.verbose = False

    def _log(msg, *arg, **kw):
        pass
    a._log = _log
    a.parameters = type("P", (), {})()
    a.parameters.pathfinding = "StrongestFirst"
    a.parameters.output_folder = None  # no saves
    a.parameters.skip_bodyId = True
    a.parameters.cache_only = True
    a.parameters.drop_untyped = True
    a.raw_results = {"banc_v888": {}}
    a._path_run_meta = {}
    a._path_taus = {}
    a._g_skip_allowed = lambda *args, **kw: False
    a.label_mapper = None
    a.run_path_analysis = fake_run_path_analysis
    # The orchestrator call is stubbed — the batch test exercises the
    # analyzer's result handling, not the FNC construction.
    a._run_multi_threshold_replay = (
        lambda ds, thresholds, verbose_mode="silent": results)

    results = {
        3: {"tau": 155.0, "tau_canonical": 155, "budget_bitten": False,
            "paths_complete": True, "replayed": False, "skipped": False,
            "duplicate_of": None, "applied_folder": 3,
            "edge_weight_floor": 151.0},
        10: {"replayed": False, "_reenumerate": True, "tau": None,
             "paths_complete": True, "skipped": False,
             "edge_weight_floor": None},
        30: {"tau": 160.0, "tau_canonical": 160, "budget_bitten": False,
             "paths_complete": True, "replayed": True, "skipped": False,
             "duplicate_of": None, "applied_folder": 30,
             "edge_weight_floor": 151.0},
    }
    meta_map = {}
    a._execute_replay_batch(
        "banc_v888", [3, 10, 30], 3, meta_map,
        lambda t, meta: None, prev_meta_init=None, prev_t_init=None)
    # t=10 was flagged below-the-floor: it ran individually.
    assert calls["ran"] == [10]
    assert not a.raw_results["banc_v888"][10].empty
    assert meta_map[10]["paths_complete"] is True


# ---------------------------------------------------------------------------
# W3 — orphan nodes / rows
# ---------------------------------------------------------------------------

def test_network_graph_orphan_isolated_node():
    from comparison.mapping_visualization import build_mapping_network_graph
    flows = [{
        "source_dataset": MCNS, "target_dataset": FAFB,
        "source_type": "Mapped1", "foreign_type": "Mapped1F",
        "source_count": 5, "foreign_count": 5,
        "matched_origin": "type", "bridges": [[
            {"dataset": MCNS, "column": "type", "value": "Mapped1"},
            {"dataset": FAFB, "column": "type", "value": "Mapped1F"}]],
    }]
    orphans = [{"dataset": MCNS, "type": "Lonely", "count": 7,
                "target": FAFB}]
    graph = build_mapping_network_graph(flows, orphans=orphans)
    node = "Lonely|male-cns:v1.0"
    assert node in graph
    assert graph.nodes[node].get("orphan") is True
    assert graph.degree(node) == 0  # isolated: no fabricated flow
    assert "no mapped counterpart" in graph.nodes[node]["title"]


def test_sankey_paths_orphan_stub_row():
    from comparison.mapping_visualization import build_mapping_sankey_paths
    rows = build_mapping_sankey_paths([], orphans=[
        {"dataset": MCNS, "type": "Lonely", "count": 7, "target": FAFB}])
    assert len(rows) == 1
    names, weights = rows[0]
    assert len(names) == 1 and "Lonely" in names[0]
    assert "no counterpart" in names[0]
    assert weights == [7]


# ---------------------------------------------------------------------------
# Untyped-drop duplicate-column regression (the live-test root bug)
# ---------------------------------------------------------------------------

def test_drop_untyped_tolerates_preexisting_provenance_columns():
    import comparison.comparison_analyzer as ca

    a = object.__new__(ca.ComparisonAnalyzer)
    a.verbose = False
    a.parameters = type("P", (), {})()
    a.parameters.drop_untyped = True
    a._untyped_dropped_records = []
    a._untyped_drop_stats = {}
    df = pd.DataFrame({
        "type_pre": ["L2", "L2"],
        "type_post": ["Tm2", "Unknown"],
        "dataset": ["banc_v888", "banc_v888"],
        "threshold": [10, 10],
    })
    out = a._drop_untyped_neurons("banc_v888", 10, df)
    assert len(out) == 1  # the Unknown row dropped, no exception
    assert len(a._untyped_dropped_records) == 1
