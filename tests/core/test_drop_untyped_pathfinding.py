"""Plan 2026-09-07 Phase A: shared untyped-neuron predicate and the
FindNeuronConnection ``drop_untyped`` pathfinding filter.

Covers the shared predicate (string and numeric values), the pandas/Polars
layer filters, toggle-off retention, the dropped-record schema, warning-note
gating, the graph-cache key, and end-to-end Complete/Shortest behavior via
the offline pipeline harness from test_pathfinding.
"""

import json
import os

import pandas as pd
import polars as pl
import pytest

import coana
from coana import FindNeuronConnection, _findallpath_cache_key
from utils.label_utils import is_untyped_type_label

from tests.core.test_pathfinding import (
    _CHAIN_EDGES,
    _make_pipeline_fc,
)

# ---------------------------------------------------------------------------
# Shared predicate
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value,expected", [
    ('', True),
    ('   ', True),
    ('Unknown', True),
    ('UNKNOWN', True),
    ('none', True),
    ('NaN', True),
    (None, True),                      # str(None) -> 'none'
    (float('nan'), True),              # str(nan) -> 'nan'
    ('123', True),                     # numeric bodyId-fallback label
    ('0', True),
    ('123.5', False),                  # not an all-digit fallback
    ('Tm3', False),
    ('MBON021_b', False),
    ('123a', False),
])
def test_shared_predicate_string_and_numeric_values(value, expected):
    assert is_untyped_type_label(value) is expected


def test_comparison_analyzer_delegates_to_shared_predicate():
    """The comparison analyzer's predicate must agree with the shared one
    (same interpretation of string and numeric fallback labels)."""
    from comparison.comparison_analyzer import ComparisonAnalyzer

    samples = ['', 'Unknown', 'nan', '123', '123.5', 'Tm3', None, 123]
    for sample in samples:
        assert (ComparisonAnalyzer._is_untyped_type_value(object(), sample)
                is is_untyped_type_label(sample))


# ---------------------------------------------------------------------------
# Unit-level layer filters
# ---------------------------------------------------------------------------

def _bare_fc(tmp_path, drop_untyped=True):
    fc = object.__new__(FindNeuronConnection)
    fc.dataset = 'test:v1'
    fc.min_synapse_num = 3
    fc.drop_untyped = drop_untyped
    fc.skip_bodyId = False
    fc._vprint = lambda *a, **k: None
    fc._reset_untyped_drop_tracking()
    return fc


def _polars_layer():
    return pl.DataFrame({
        'bodyId_pre': ['1', '2', '3', '4'],
        'bodyId_post': ['5', '6', '7', '8'],
        'weight': [10, 20, 30, 40],
        'type_pre': ['Tm3', '', 'Unknown', 'Tm3'],
        'type_post': ['Tm4', 'Tm4', '55', '123'],
    })


def test_polars_filter_drops_untyped_and_records(tmp_path):
    fc = _bare_fc(tmp_path)
    kept = fc._filter_untyped_polars(_polars_layer(), layer_label='0->1')

    # rows 2 ('' pre), 3 ('Unknown' pre + bodyId-fallback post) and
    # 4 (numeric bodyId-fallback post) must go; row 1 stays.
    assert kept.height == 1
    assert kept.row(0, named=True)['bodyId_pre'] == '1'

    stats = fc._untyped_drop_stats
    assert stats['rows'] == 3
    assert stats['untyped_pre'] == 2      # '' and 'Unknown'
    assert stats['untyped_post'] == 2     # '55' and '123'
    assert stats['neurons'] == {'2', '6', '3', '7', '4', '8'}
    assert len(fc._untyped_dropped_frames) == 1


def test_pandas_filter_matches_polars(tmp_path):
    fc = _bare_fc(tmp_path)
    pandas_layer = _polars_layer().to_pandas()
    kept = fc._filter_untyped_pandas(pandas_layer, layer_label='0->1')

    assert len(kept) == 1
    assert kept.iloc[0]['bodyId_pre'] == '1'
    assert fc._untyped_drop_stats['rows'] == 3


def test_toggle_off_retains_all_rows(tmp_path):
    fc = _bare_fc(tmp_path, drop_untyped=False)
    for frame in (_polars_layer(), _polars_layer().to_pandas()):
        filtered = (
            fc._filter_untyped_polars(frame)
            if isinstance(frame, pl.DataFrame)
            else fc._filter_untyped_pandas(frame))
        assert len(filtered) == 4
    assert fc._untyped_drop_stats['rows'] == 0
    assert fc._untyped_dropped_frames == []


def test_dropped_record_schema(tmp_path):
    fc = _bare_fc(tmp_path)
    fc._filter_untyped_polars(_polars_layer(), layer_label='0->1')
    run = tmp_path / 'find-paths-complete_x'
    run.mkdir()

    fc._export_untyped_drop_records(str(run))

    rec_path = run / 'data_details' / 'untyped_dropped_records.csv'
    assert rec_path.exists()
    record = pl.read_csv(rec_path)
    cols = record.columns
    # run/threshold context first, side flag last (stable schema)
    assert cols[:3] == ['dataset', 'threshold', 'conn_layer']
    assert cols[-1] == 'untyped_side'
    for core_col in ('bodyId_pre', 'bodyId_post', 'type_pre', 'type_post',
                     'weight'):
        assert core_col in cols
    sides = set(record['untyped_side'].to_list())
    assert sides <= {'pre', 'post', 'pre+post'}
    assert 'pre' in sides and 'post' in sides

    # nothing dropped -> no file
    fc2 = _bare_fc(tmp_path)
    run2 = tmp_path / 'find-paths-complete_y'
    run2.mkdir()
    fc2._export_untyped_drop_records(str(run2))
    assert not (run2 / 'data_details' / 'untyped_dropped_records.csv').exists()


def test_warning_note_gated_on_actual_drops(tmp_path):
    fc = _bare_fc(tmp_path)
    fc._warn_notes = []
    fc._write_user_warning_notes(str(tmp_path))
    assert not (tmp_path / 'user_warning_notes.txt').exists()

    fc2 = _bare_fc(tmp_path)
    fc2._warn_notes = []
    fc2._filter_untyped_polars(_polars_layer(), layer_label='0->1')
    fc2._write_user_warning_notes(str(tmp_path))
    note = (tmp_path / 'user_warning_notes.txt').read_text(encoding='utf-8')
    assert '[untyped dropped]' in note
    assert 'data_details/untyped_dropped_records.csv' in note
    assert 'drop_untyped=True' in note


def test_cache_key_distinguishes_drop_untyped():
    common = dict(
        dataset_safe='test_v1', source_ID=['1'], target_ID=['2'],
        max_interlayer=2, separate_hemispheres=False, filter_by='bodyId',
        min_ratio=0.0, min_traversal_probability=0.0,
        exclude_intra_type_connections=False,
    )
    assert (_findallpath_cache_key(drop_untyped=True, **common)
            != _findallpath_cache_key(drop_untyped=False, **common))
    # default is on
    assert (_findallpath_cache_key(**common)
            == _findallpath_cache_key(drop_untyped=True, **common))


# ---------------------------------------------------------------------------
# End-to-end pipeline behavior (offline harness)
# ---------------------------------------------------------------------------

# 'U' carries a NULL type (the harness ``untyped`` mechanism): the Polars
# layer conversion turns it into the empty label, which the shared
# predicate treats as untyped. 'U' itself resolves as its own type when
# the filter is off.
_UNTYPED_EDGES = [
    ('S', 'U', 10),
    ('U', 'T', 10),
    ('S', 'T', 2),  # weak direct edge keeps a typed result alive
]


def test_complete_paths_drop_untyped_removes_intermediate(
        monkeypatch, tmp_path):
    coana._FINDALLPATH_GRAPH_CACHE.clear()
    fc, fetch_calls, logs = _make_pipeline_fc(
        monkeypatch, tmp_path, _UNTYPED_EDGES, max_interlayer=2,
        untyped=('U',))
    fc.FindAllPath()

    # The edge touching the null-labeled neuron was dropped before the
    # graph was built: the S->U row is removed at the layer-0 fetch, so U
    # never enters the discovery frontier and U->T is never even fetched.
    assert fc._untyped_drop_stats['rows'] == 1
    assert fc._untyped_drop_stats['untyped_post'] == 1   # S->U
    assert fc._untyped_drop_stats['untyped_pre'] == 0
    assert fc._untyped_drop_stats['neurons'] == {'S', 'U'}

    rec = pl.read_csv(os.path.join(
        fc.allpath_folder, 'data_details', 'untyped_dropped_records.csv'))
    assert rec.height == 1
    assert set(rec['conn_layer'].to_list()) == {'0->1'}
    assert set(rec['untyped_side'].to_list()) == {'post'}

    note = os.path.join(fc.allpath_folder, 'user_warning_notes.txt')
    assert os.path.exists(note)
    note_text = open(note, encoding='utf-8').read()
    assert '[untyped dropped]' in note_text
    assert 'data_details/untyped_dropped_records.csv' in note_text

    # the intermediate untyped neuron is gone from the result set
    type_csv = os.path.join(
        fc.allpath_folder, 'src_to_tgt_allpaths_type.csv')
    df = pl.read_csv(type_csv)
    assert all('->U->' not in str(row) for row in df.iter_rows())


def test_complete_paths_toggle_off_keeps_untyped_neuron(
        monkeypatch, tmp_path):
    coana._FINDALLPATH_GRAPH_CACHE.clear()
    fc, fetch_calls, logs = _make_pipeline_fc(
        monkeypatch, tmp_path, _UNTYPED_EDGES, max_interlayer=2,
        untyped=('U',))
    fc.drop_untyped = False
    fc.FindAllPath()

    assert fc._untyped_drop_stats['rows'] == 0
    assert not os.path.exists(os.path.join(
        fc.allpath_folder, 'data_details', 'untyped_dropped_records.csv'))
    # The 2-hop path through the untyped neuron is back.
    type_csv = os.path.join(
        fc.allpath_folder, 'src_to_tgt_allpaths_type.csv')
    df = pl.read_csv(type_csv)
    assert any('->U->' in str(row) for row in df.iter_rows())
    # no untyped-drop warning for a run with the filter off (other notes,
    # e.g. skip_bodyId, may still apply)
    note_path = os.path.join(fc.allpath_folder, 'user_warning_notes.txt')
    if os.path.exists(note_path):
        assert '[untyped dropped]' not in open(
            note_path, encoding='utf-8').read()


def test_shortest_paths_drop_untyped_in_backward_discovery(
        monkeypatch, tmp_path):
    coana._FINDALLPATH_GRAPH_CACHE.clear()
    fc, fetch_calls, logs = _make_pipeline_fc(
        monkeypatch, tmp_path, _UNTYPED_EDGES, max_interlayer=99,
        untyped=('U',))
    fc.FindShortestPath()

    # The incoming edge U->T is dropped during backward discovery, so the
    # 2-hop route never exists and the direct edge is the only shortest
    # path.
    assert fc._untyped_drop_stats['rows'] >= 1
    assert fc._untyped_drop_stats['untyped_pre'] >= 1
    type_csv = os.path.join(
        fc.allpath_folder, 'src_to_tgt_allpaths_type.csv')
    df = pl.read_csv(type_csv)
    assert all('->U->' not in str(row) for row in df.iter_rows())


def test_typed_intermediate_is_never_dropped(monkeypatch, tmp_path):
    coana._FINDALLPATH_GRAPH_CACHE.clear()
    fc, fetch_calls, logs = _make_pipeline_fc(
        monkeypatch, tmp_path, _CHAIN_EDGES, max_interlayer=99)
    fc.FindShortestPath()

    assert fc._untyped_drop_stats['rows'] == 0
    assert not os.path.exists(os.path.join(
        fc.allpath_folder, 'data_details', 'untyped_dropped_records.csv'))
    type_csv = os.path.join(
        fc.allpath_folder, 'src_to_tgt_allpaths_type.csv')
    df = pl.read_csv(type_csv)
    assert 'TS->TA->TB->TC->TT' in str(df.row(0))
