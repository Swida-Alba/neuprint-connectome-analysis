"""Budget-combination matrix cells (plan-pruned-run-followups.md §4b).

Analyzer-batch cells for the (Edge Budget × SF budget) interaction
space. The orchestrator is stubbed per cell with the outcome tuple the
coana layer produces; the assertions pin the ANALYZER's handling:
skip/alias decisions, re-enumeration of below-floor thresholds, folder
aliasing, and the never-skip-on-unknown-τ rule (M12).
"""

import json
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FAFB = "flywire_FAFB_v783"


def _bare_analyzer():
    import comparison.comparison_analyzer as ca

    a = object.__new__(ca.ComparisonAnalyzer)
    a.verbose = False
    a._log = lambda msg, *arg, **kw: None
    a.parameters = type("P", (), {})()
    a.parameters.pathfinding = "StrongestFirst"
    a.parameters.output_folder = None  # no saves
    a.parameters.skip_bodyId = True
    a.parameters.cache_only = True
    a.parameters.drop_untyped = True
    a.raw_results = {FAFB: {}}
    a._path_run_meta = {}
    a._path_taus = {}
    a._g_skip_allowed = lambda *args, **kw: False
    a.label_mapper = None
    return a


def _run_batch(a, results, pending, lowest=3):
    meta_map = {}

    def _note_actual(t, meta):
        meta_map[t] = dict(meta)  # mirrors _note_actual in the analyzer

    a._run_multi_threshold_replay = (
        lambda ds, thresholds, verbose_mode="silent": results)
    a._execute_replay_batch(
        FAFB, list(pending), lowest, meta_map,
        _note_actual, prev_meta_init=None, prev_t_init=None)
    return meta_map


def _meta(tau, canonical, bitten, complete, skipped=False, dup=None,
          folder=None, floor=None):
    return {"tau": tau, "tau_canonical": canonical,
            "budget_bitten": bitten, "paths_complete": complete,
            "replayed": True, "skipped": skipped, "duplicate_of": dup,
            "applied_folder": folder, "edge_weight_floor": floor}


def test_m4_floor_and_bite_reenumerates_below_floor_thresholds():
    """M4/M11: a floored+bitten t0 with w0 above later asked thresholds —
    those sit below the floor and must re-enumerate individually."""
    a = _bare_analyzer()
    ran = []
    a.run_path_analysis = (
        lambda ds, t, verbose_mode="silent": ran.append(t) or pd.DataFrame(
            [{"type_pre": "A", "type_post": "B"}]))
    results = {
        3: _meta(155.0, 155, False, True, folder=3, floor=151.0),
        10: {"replayed": False, "_reenumerate": True, "tau": None,
             "paths_complete": True, "skipped": False,
             "edge_weight_floor": None},
        30: _meta(160.0, 160, False, True, folder=30, floor=151.0),
    }
    meta_map = _run_batch(a, results, [3, 10, 30])
    assert ran == [10]  # below-floor asked: individually re-enumerated
    assert meta_map[10]["paths_complete"] is True


def test_m7_tau_above_all_asked_collapses_everything():
    """M7: landing tau above every other asked threshold — all later
    asked thresholds are materialized/aliased, nothing re-enumerates."""
    a = _bare_analyzer()
    ran = []
    a.run_path_analysis = (
        lambda ds, t, verbose_mode="silent": ran.append(t) or pd.DataFrame())
    tau = 90.0
    results = {
        3: _meta(tau, int(tau), True, False, folder=3),
        10: _meta(tau, int(tau), True, False, skipped=True, dup=3,
                  folder=3),
        20: _meta(tau, int(tau), True, False, skipped=True, dup=3,
                  folder=3),
        30: _meta(tau, int(tau), True, False, skipped=True, dup=3,
                  folder=3),
    }
    meta_map = _run_batch(a, results, [3, 10, 20, 30])
    assert ran == []
    assert all(meta_map[t]["skipped"] for t in (10, 20, 30))


def test_m8_tau_at_t0_no_collapse():
    """M8: landing tau == t0 — no collapse; every asked threshold
    materializes its own slice."""
    a = _bare_analyzer()
    results = {
        3: _meta(3.0, 3, True, False, folder=3),
        10: _meta(10.0, 10, False, True, folder=10),
    }
    meta_map = _run_batch(a, results, [3, 10])
    assert not meta_map[3]["skipped"] and not meta_map[10]["skipped"]


def test_m12_tau_none_never_skips():
    """M12: tau None (empty result) — nothing may be skipped; every
    threshold re-enumerates via the fallback."""
    a = _bare_analyzer()
    a._run_multi_threshold_replay = (
        lambda ds, thresholds, verbose_mode="silent": None)
    ran = []
    a.run_path_analysis = (
        lambda ds, t, verbose_mode="silent": ran.append(t) or pd.DataFrame(
            [{"type_pre": "A", "type_post": "B"}]))
    meta_map = {}
    a._execute_replay_batch(
        FAFB, [3, 10, 20], 3, meta_map,
        lambda t, meta: None, prev_meta_init=None, prev_t_init=None)
    assert ran == [3, 10, 20]  # per-threshold fallback: no skip possible


def test_banner_includes_aliased_canonical_folders(tmp_path):
    """Display fix: the effective-thresholds banner must list aliased
    canonical folders (e.g. 3->9) instead of omitting them, and carry
    the alias mapping in the text."""
    import comparison.comparison_analyzer as ca

    a = object.__new__(ca.ComparisonAnalyzer)
    a.verbose = False
    a._log = lambda msg, *arg, **kw: None
    a.parameters = type("P", (), {})()
    a.parameters.path_mode = "all"
    a.parameters.get_dataset_names = lambda: ["banc_v888"]
    a.parameters.get_thresholds_for_dataset = lambda ds: [3, 10, 20, 30]
    a.parameters.full_output_path = str(tmp_path)
    a._path_run_meta = {
        ("banc_v888", 3): {"tau": 9.0, "tau_canonical": 9,
                           "budget_bitten": True, "paths_complete": False,
                           "skipped": True, "duplicate_of": 9,
                           "applied_folder": 9, "edge_weight_floor": None},
        ("banc_v888", 10): {"tau": 10.0, "tau_canonical": 10,
                            "budget_bitten": False, "paths_complete": True,
                            "skipped": False, "duplicate_of": None,
                            "applied_folder": 10, "edge_weight_floor": None},
        ("banc_v888", 20): {"tau": None, "tau_canonical": None,
                            "budget_bitten": False, "paths_complete": True,
                            "skipped": False, "duplicate_of": None,
                            "applied_folder": 20, "edge_weight_floor": None},
        ("banc_v888", 30): {"tau": None, "tau_canonical": None,
                            "budget_bitten": False, "paths_complete": True,
                            "skipped": False, "duplicate_of": None,
                            "applied_folder": 30, "edge_weight_floor": None},
    }
    a._export_effective_threshold_banner()
    payload = json.load(
        open(tmp_path / "effective_thresholds.json", encoding="utf-8"))
    ds = payload["datasets"]["banc_v888"]
    # minsyn_9 is a REAL applied point on disk: it joins the applied list.
    assert ds["effective"] == [9, 10, 20, 30]
    assert ds["applied_folder"] == {"3": 9}
    assert "aliased: 3\u21929" in payload["banner"]


def test_applied_state_skipped_row_without_canonical_uses_folder():
    """The 24-vs-25 class: a collapsed row whose canonical bookkeeping is
    missing must report the applied FOLDER threshold, never the bare
    landing tau."""
    import comparison.comparison_analyzer as ca

    a = object.__new__(ca.ComparisonAnalyzer)
    a.parameters = type("P", (), {})()
    a._path_run_meta = {("banc_v888", 20): {
        "tau": 25.0, "tau_canonical": None, "budget_bitten": True,
        "paths_complete": False, "skipped": True, "duplicate_of": 10,
        "applied_folder": 24, "edge_weight_floor": None,
    }}
    applied, _pruned, _floor, _source = a._applied_state_for("banc_v888", 20)
    assert applied == 24


def test_slice_note_refresh_contract():
    """Source contract: _materialize_threshold rewrites the [path budget]
    note per slice so a materialized folder's notes describe IT."""
    src = (PROJECT_ROOT / "src" / "coana.py").read_text(encoding="utf-8")
    assert "[path budget] StrongestFirst slice for threshold" in src
