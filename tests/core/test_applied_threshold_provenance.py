"""Plan 2026-09-07 Phase B: canonical applied-threshold/bottleneck
provenance.

Covers the shared ``applied_threshold_provenance`` formula (complete,
StrongestFirst-bitten, Edge-Budget-floored, and combined runs), the
instance-level finalization (all_attributes.json fields +
data_details/parameters.csv rows), the parameters.txt provenance block with
its backward-compatible aliases, and the shortest-mode contract (StrongestFirst
budget metadata exported, Edge Budget never applied).
"""

import json
import os

import pandas as pd
import polars as pl
import pytest

import coana
from coana import applied_threshold_provenance

from tests.core.test_pathfinding import _make_pipeline_fc


# ---------------------------------------------------------------------------
# Shared formula
# ---------------------------------------------------------------------------

def test_complete_run_applied_is_requested_tau_reported_separately():
    prov = applied_threshold_provenance(
        requested_threshold=3,
        strongest_first_tau=9,
        strongest_first_budget_bitten=False,
        tau_canonical=9,
    )
    assert prov['applied_threshold'] == 3
    assert prov['applied_threshold_source'] == 'requested'
    assert prov['paths_complete'] is True
    assert prov['strongest_first_tau'] == 9
    assert prov['tau_canonical'] == 9


def test_sf_bitten_run_applied_is_w2_plus_one():
    prov = applied_threshold_provenance(
        requested_threshold=3,
        strongest_first_tau=12,
        strongest_first_budget_bitten=True,
        strongest_dropped_bottleneck=7,
        tau_canonical=8,
    )
    assert prov['applied_threshold'] == 8
    assert prov['applied_threshold_source'] == 'strongest_first_budget'
    assert prov['paths_complete'] is False
    assert prov['strongest_dropped_bottleneck'] == 7


def test_edge_budget_only_run():
    prov = applied_threshold_provenance(
        requested_threshold=3,
        strongest_first_tau=10,
        strongest_first_budget_bitten=False,
        tau_canonical=10,
        edge_weight_floor=6,
        edge_budget_landing=5,
        edge_budget=1000,
    )
    assert prov['applied_threshold'] == 10
    assert prov['applied_threshold_source'] == 'edge_budget'
    assert prov['edge_budget_applied'] is True
    assert prov['paths_complete'] is False


def test_combined_run_names_both_sources():
    prov = applied_threshold_provenance(
        requested_threshold=3,
        strongest_first_tau=14,
        strongest_first_budget_bitten=True,
        strongest_dropped_bottleneck=11,
        tau_canonical=12,
        edge_weight_floor=6,
        edge_budget_landing=5,
        edge_budget=1000,
    )
    assert prov['applied_threshold'] == 12
    assert prov['applied_threshold_source'] == \
        'strongest_first_budget+edge_budget'
    assert prov['paths_complete'] is False


def test_floor_guard_against_degenerate_canonical():
    # A canonical below the floor is impossible; the formula must not
    # report a contradiction.
    prov = applied_threshold_provenance(
        requested_threshold=3,
        strongest_first_tau=9,
        strongest_first_budget_bitten=True,
        tau_canonical=4,
        edge_weight_floor=6,
        edge_budget=1000,
    )
    assert prov['applied_threshold'] == 6


# ---------------------------------------------------------------------------
# Instance-level finalization
# ---------------------------------------------------------------------------

def _state_fc(tmp_path):
    fc = object.__new__(coana.FindNeuronConnection)
    fc.allpath_folder = str(tmp_path)
    fc.source_fname = 'src'
    fc.target_fname = 'tgt'
    fc.dataset = 'test:v1'
    fc.min_synapse_num = 3
    fc.max_paths_bodyid = 0
    fc.graph_edge_limit_bodyid = 0
    fc.drop_untyped = True
    fc.parameter_dict = {'min synapse number': '3'}
    fc.parameter_df = pd.DataFrame(
        [('min synapse number', '3')], columns=['parameter', 'value'])
    fc.strongest_first_cutoff = None
    fc.strongest_first_budget_bitten = False
    fc.strongest_dropped_bottleneck = None
    fc.tau_canonical = None
    fc.edge_weight_floor = None
    fc.edge_budget_landing = None
    fc.strongest_retained_bottleneck = None
    fc._vprint = lambda *a, **k: None
    return fc


def test_finalize_records_full_contract(tmp_path):
    fc = _state_fc(tmp_path)
    fc.strongest_first_cutoff = 12
    fc.strongest_first_budget_bitten = True
    fc.strongest_dropped_bottleneck = 7
    fc.tau_canonical = 8
    fc.strongest_retained_bottleneck = 15

    prov = fc._finalize_threshold_provenance(path_mode='all')

    assert prov['requested_threshold'] == 3
    assert prov['applied_threshold'] == 8
    assert prov['applied_threshold_source'] == 'strongest_first_budget'
    assert prov['strongest_first_budget'] == 1000000  # auto budget
    assert prov['paths_complete'] is False
    # public instance attributes -> all_attributes.json export
    for attr in ('requested_threshold', 'applied_threshold',
                 'applied_threshold_source', 'strongest_first_budget',
                 'strongest_first_budget_bitten', 'strongest_first_tau',
                 'tau_canonical', 'strongest_dropped_bottleneck',
                 'edge_budget', 'edge_budget_applied', 'edge_budget_landing',
                 'edge_weight_floor', 'strongest_retained_bottleneck',
                 'paths_complete'):
        assert hasattr(fc, attr), attr
    # structured parameter export carries the same contract names
    assert fc.parameter_dict['applied_threshold'] == '8'
    assert fc.parameter_dict['applied_threshold_source'] == \
        'strongest_first_budget'
    assert 'applied_threshold' in set(fc.parameter_df['parameter'])


def test_finalize_shortest_mode_never_floored(tmp_path):
    fc = _state_fc(tmp_path)
    fc.graph_edge_limit_bodyid = 1000000  # even with a (ignored) budget
    fc.strongest_first_cutoff = 5
    fc.strongest_first_budget_bitten = True
    fc.strongest_dropped_bottleneck = 3
    fc.tau_canonical = 4

    prov = fc._finalize_threshold_provenance(path_mode='shortest')

    assert prov['edge_budget'] is None
    assert prov['edge_budget_applied'] is False
    assert prov['edge_weight_floor'] is None
    assert prov['applied_threshold'] == 4
    assert prov['applied_threshold_source'] == 'strongest_first_budget'


def test_write_run_metadata_writes_provenance_block(tmp_path):
    fc = _state_fc(tmp_path)
    fc.strongest_first_cutoff = 12
    fc.strongest_first_budget_bitten = True
    fc.strongest_dropped_bottleneck = 7
    fc.tau_canonical = 8
    fc._finalize_threshold_provenance(path_mode='all')
    fc._write_run_metadata('all')

    text = (tmp_path / 'parameters.txt').read_text(encoding='utf-8')
    for key in ('requested_threshold', 'applied_threshold',
                'applied_threshold_source', 'strongest_first_budget',
                'strongest_first_budget_bitten', 'strongest_first_tau',
                'tau_canonical', 'strongest_dropped_bottleneck',
                'edge_budget', 'edge_budget_applied', 'edge_budget_landing',
                'edge_weight_floor', 'strongest_retained_bottleneck',
                'paths_complete', 'path_mode'):
        assert f'{key}:' in text, key
    # backward-compatible aliases
    assert 'applied_tau (min path bottleneck):' in text
    assert 'edge_weight_floor:' in text

    attrs = json.loads((tmp_path / 'all_attributes.json').read_text())
    assert attrs['applied_threshold'] == 8
    assert attrs['applied_threshold_source'] == 'strongest_first_budget'
    assert attrs['paths_complete'] is False


# ---------------------------------------------------------------------------
# Pipeline-level provenance (offline harness)
# ---------------------------------------------------------------------------

# Three parallel 2-hop routes with distinct bottlenecks 9 / 7 / 3.
_BUDGET_EDGES = [
    ('S', 'A', 10), ('A', 'T', 9),
    ('S', 'B', 7), ('B', 'T', 8),
    ('S', 'C', 3), ('C', 'T', 9),
]


def test_complete_run_bitten_provenance_flows_to_files(
        monkeypatch, tmp_path):
    coana._FINDALLPATH_GRAPH_CACHE.clear()
    fc, fetch_calls, logs = _make_pipeline_fc(
        monkeypatch, tmp_path, _BUDGET_EDGES, max_interlayer=2)
    fc.pathfinding = 'StrongestFirst'
    fc.max_paths_bodyid = 1  # bite: only the bottleneck-9 path is kept
    fc.FindAllPath()

    assert fc.paths_complete is False
    assert fc.applied_threshold_source == 'strongest_first_budget'
    assert fc.applied_threshold == fc.tau_canonical
    assert fc.strongest_dropped_bottleneck == 7
    assert fc.strongest_first_tau == 9
    assert fc.strongest_retained_bottleneck == 9  # W* from lossless pruning

    text = open(os.path.join(fc.allpath_folder, 'parameters.txt'),
                encoding='utf-8').read()
    assert 'applied_threshold:' in text
    assert 'applied_threshold_source:' in text
    assert 'strongest_first_budget+edge_budget' not in text
    attrs = json.loads(open(os.path.join(
        fc.allpath_folder, 'all_attributes.json'),
        encoding='utf-8').read())
    assert attrs['applied_threshold'] == fc.applied_threshold
    assert attrs['paths_complete'] is False
    # data_details/parameters.csv carries the same block
    params_csv = pl.read_csv(os.path.join(
        fc.allpath_folder, 'data_details', 'parameters.csv'))
    keys = set(params_csv['parameter'].to_list())
    assert {'applied_threshold', 'applied_threshold_source',
            'paths_complete', 'strongest_retained_bottleneck'} <= keys


def test_complete_run_unbitten_provenance_is_requested(
        monkeypatch, tmp_path):
    coana._FINDALLPATH_GRAPH_CACHE.clear()
    fc, fetch_calls, logs = _make_pipeline_fc(
        monkeypatch, tmp_path, _BUDGET_EDGES, max_interlayer=2)
    fc.pathfinding = 'StrongestFirst'
    fc.FindAllPath()  # auto budget 1M — no bite

    assert fc.paths_complete is True
    assert fc.applied_threshold == 1  # the requested threshold
    assert fc.applied_threshold_source == 'requested'
    assert fc.strongest_first_tau == 3  # natural weakest emitted bottleneck
    assert fc.tau_canonical == 3
    text = open(os.path.join(fc.allpath_folder, 'parameters.txt'),
                encoding='utf-8').read()
    assert 'requested' in text


def test_shortest_run_exports_budget_metadata_and_no_floor(
        monkeypatch, tmp_path):
    coana._FINDALLPATH_GRAPH_CACHE.clear()
    fc, fetch_calls, logs = _make_pipeline_fc(
        monkeypatch, tmp_path, _BUDGET_EDGES, max_interlayer=2,
        graph_edge_limit=500000)
    fc.max_paths_bodyid = 1  # bite inside the shortest enumeration
    fc.FindShortestPath()

    assert fc.paths_complete is False
    assert fc.applied_threshold_source == 'strongest_first_budget'
    assert fc.edge_budget_applied is False
    assert fc.edge_weight_floor is None  # shortest mode is NEVER floored
    assert fc.edge_budget is None
    text = open(os.path.join(fc.allpath_folder, 'parameters.txt'),
                encoding='utf-8').read()
    assert 'edge_weight_floor:' in text
    assert 'n/a' in text  # floor rendered as n/a for shortest mode
