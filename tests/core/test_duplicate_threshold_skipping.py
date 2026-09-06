"""Feature G tests — duplicate-threshold skipping (τ collapse)
(plan-cross-dataset-threshold-alignment.md §13).

User framing: "if threshold = 1,3,5,10 in the input, but we get a
tau = 4, the program should skip the 3, and finally show thresholds of
4,5,10." With the unified rule, complete runs carry their natural τ too:
a run whose weakest emitted path has bottleneck τ yields the IDENTICAL
set for every input threshold <= τ — later such thresholds are skipped,
aliased, and marked in the exports.
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _make_analyzer(tmp_path, thresholds=(1, 3, 5, 10),
                   datasets=("male-cns:v1.0", "flywire_FAFB_v783")):
    from comparison.comparison_analyzer import ComparisonAnalyzer
    from comparison.comparison_parameters import ComparisonParameters

    params = ComparisonParameters(
        datasets=list(datasets),
        source_neurons=["L2"],
        target_neurons=["l-LNv"],
        max_interlayer=2,
        thresholds=list(thresholds),
        output_folder=str(tmp_path),
        auto_type_mapping=False,
        verbose=False,
    )
    an = ComparisonAnalyzer(params, verbose=False)
    an.parameters.replay_paths = False  # Feature G in isolation
    return an


def _frame(seed):
    return pd.DataFrame([(f"a{i}", f"b{i}", 9)
                         for i in range(seed)],
                        columns=["type_pre", "type_post", "weight"])


def test_g_skip_allowed_rule():
    from comparison.comparison_analyzer import ComparisonAnalyzer  # noqa: F401
    an = _make_analyzer(Path("/tmp"))  # loop rules don't touch the folder

    bitten = {"tau": 4, "budget_bitten": True, "paths_complete": False,
              "pathfinding": "StrongestFirst", "_threshold": 1,
              "skipped": False, "duplicate_of": None}
    assert an._g_skip_allowed("ds", 3, bitten) is True    # 3 <= 4
    assert an._g_skip_allowed("ds", 4, bitten) is True    # boundary: <= tau
    assert an._g_skip_allowed("ds", 5, bitten) is False   # new material

    complete = {"tau": 6, "budget_bitten": False, "paths_complete": True,
                "pathfinding": "StrongestFirst", "_threshold": 1,
                "skipped": False, "duplicate_of": None}
    assert an._g_skip_allowed("ds", 3, complete) is True  # natural tau bites too
    assert an._g_skip_allowed("ds", 6, complete) is True
    assert an._g_skip_allowed("ds", 7, complete) is False

    changed_algo = dict(bitten, pathfinding="MemoizedDFS")
    assert an._g_skip_allowed("ds", 3, changed_algo) is False  # paranoia clause

    no_tau = dict(bitten, tau=None)
    assert an._g_skip_allowed("ds", 3, no_tau) is False
    assert an._g_skip_allowed("ds", 3, None) is False


def test_budget_bitten_run_skips_later_thresholds(tmp_path, monkeypatch):
    """input [1,3,5,10]; t=1 bites with tau=4 -> skip 3; run 5 and 10."""
    an = _make_analyzer(tmp_path)
    ds = "male-cns:v1.0"
    calls = []

    def fake_run(dataset_name, threshold, verbose_mode="simple"):
        if dataset_name == ds:
            calls.append(threshold)
        # emulate run_path_analysis meta recording: t=1 budget-bitten @ 4;
        # later runs complete with natural tau 2 (min bn of their set)
        if threshold == 1:
            tau, bitten = 4, True
        else:
            tau, bitten = 2, False
        an._path_taus[(dataset_name, threshold)] = tau
        an._path_run_meta[(dataset_name, threshold)] = {
            "tau": tau, "budget_bitten": bitten,
            "paths_complete": not bitten, "skipped": False,
            "duplicate_of": None, "pathfinding": "StrongestFirst",
        }
        return _frame(5)

    monkeypatch.setattr(an, "run_path_analysis", fake_run)
    an._run_all_path_analyses(skip_existing=True)

    assert calls == [1, 5, 10]  # 3 skipped (3 <= tau 4)
    aliased = an.raw_results[ds][3]
    # Directive 2: the alias is an independent COPY (no shared-reference
    # hazard) with identical content — and no re-enumeration happened
    # (calls already asserts only thresholds 1, 5, 10 ran).
    assert aliased.equals(an.raw_results[ds][1])
    assert aliased is not an.raw_results[ds][1]

    meta = an._path_run_meta[(ds, 3)]
    assert meta["skipped"] is True and meta["duplicate_of"] == 1
    assert meta["tau"] == 4

    # sensitivity export carries the markers
    out = tmp_path / "cr"
    out.mkdir()
    an._export_intra_dataset_comparisons(str(out))
    sens = pd.read_csv(out / "threshold_sensitivity.csv")
    row = sens[(sens.dataset == ds) & (sens.threshold == 3)].iloc[0]
    assert bool(row["skipped"]) is True
    assert int(row["duplicate_of"]) == 1
    assert float(row["tau"]) == 4
    assert bool(row["paths_complete"]) is False  # aliased output is tau-bounded


def test_complete_run_natural_tau_also_collapses(tmp_path, monkeypatch):
    """A complete run whose weakest path has bn=6 collapses 3 and 5."""
    an = _make_analyzer(tmp_path, thresholds=(1, 3, 5, 8, 10))
    ds = "male-cns:v1.0"
    calls = []

    def fake_run(dataset_name, threshold, verbose_mode="simple"):
        if dataset_name == ds:
            calls.append(threshold)
        if threshold == 1:
            tau, bitten, complete = 6, False, True  # complete, natural tau 6
        else:
            tau, bitten, complete = 2, False, True
        an._path_taus[(dataset_name, threshold)] = tau
        an._path_run_meta[(dataset_name, threshold)] = {
            "tau": tau, "budget_bitten": bitten,
            "paths_complete": complete, "skipped": False,
            "duplicate_of": None, "pathfinding": "StrongestFirst",
        }
        return _frame(4)

    monkeypatch.setattr(an, "run_path_analysis", fake_run)
    an._run_all_path_analyses(skip_existing=True)

    assert calls == [1, 8, 10]  # 3 and 5 skipped (<= natural tau 6)
    meta = an._path_run_meta[(ds, 5)]
    assert meta["skipped"] and meta["duplicate_of"] == 1
    assert meta["paths_complete"] is True  # aliased set IS the complete set


def test_per_dataset_independence(tmp_path, monkeypatch):
    """τ collapse is per dataset: one collapsed dataset must not skip
    another dataset's thresholds (§13.3/§13.5 item 2)."""
    an = _make_analyzer(tmp_path, thresholds=(1, 3))
    ds_a, ds_b = "male-cns:v1.0", "flywire_FAFB_v783"
    calls = []

    def fake_run(dataset_name, threshold, verbose_mode="simple"):
        calls.append((dataset_name, threshold))
        tau, bitten = (4, True) if dataset_name == ds_a else (2, False)
        an._path_taus[(dataset_name, threshold)] = tau
        an._path_run_meta[(dataset_name, threshold)] = {
            "tau": tau, "budget_bitten": bitten,
            "paths_complete": not bitten, "skipped": False,
            "duplicate_of": None, "pathfinding": "StrongestFirst",
        }
        return _frame(3)

    monkeypatch.setattr(an, "run_path_analysis", fake_run)
    an._run_all_path_analyses(skip_existing=True)

    # ds_a: t=3 skipped (tau 4); ds_b: natural tau 2 < 3 -> runs normally
    assert calls == [(ds_a, 1), (ds_b, 1), (ds_b, 3)]
    assert an._path_run_meta[(ds_a, 3)]["skipped"] is True
    assert an._path_run_meta[(ds_b, 3)]["skipped"] is False


def test_shortest_mode_never_skips(tmp_path, monkeypatch):
    """Shortest sets are NOT nested across thresholds — no skip rule."""
    an = _make_analyzer(tmp_path, thresholds=(1, 3))
    an.parameters.path_mode = "shortest"
    ds = "male-cns:v1.0"
    calls = []

    def fake_run(dataset_name, threshold, verbose_mode="simple"):
        if dataset_name == ds:
            calls.append(threshold)
        an._path_taus[(dataset_name, threshold)] = 4
        an._path_run_meta[(dataset_name, threshold)] = {
            "tau": 4, "budget_bitten": True, "paths_complete": False,
            "skipped": False, "duplicate_of": None,
            "pathfinding": "StrongestFirst",
        }
        return _frame(2)

    monkeypatch.setattr(an, "run_path_analysis", fake_run)
    an._run_all_path_analyses(skip_existing=True)
    assert calls == [1, 3]  # no skipping in shortest mode


def test_threshold_meta_persistence_round_trip(tmp_path, monkeypatch):
    """Skip state is persisted next to the raw results so a resumed run
    reconstructs it (G §13.5 item 6)."""
    an = _make_analyzer(tmp_path, thresholds=(1, 3))
    ds = "male-cns:v1.0"

    meta_map = {1: {"tau": 4, "budget_bitten": True,
                    "paths_complete": False, "skipped": False,
                    "duplicate_of": None, "pathfinding": "StrongestFirst",
                    "_threshold": 1},
                3: {"tau": 4, "budget_bitten": True,
                    "paths_complete": False, "skipped": True,
                    "duplicate_of": 1, "pathfinding": "StrongestFirst",
                    "_threshold": 3}}
    an._store_threshold_meta(ds, meta_map)
    meta_file = (Path(an.parameters.full_output_path) / "dataset_data"
                 / "male-cns_v1_0" / "threshold_meta.json")
    assert meta_file.exists()

    loaded = an._load_threshold_meta(ds)
    assert loaded[3]["skipped"] is True
    assert loaded[3]["duplicate_of"] == 1
    assert loaded[1]["tau"] == 4
