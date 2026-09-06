"""Unit tests for the report-fixes plan (plan-cross-dataset-report-fixes.md).

Covers the export-level fixes that were verified against the
cross-dataset_L2_to_l-LNv_MF_20260904_191643 evidence run:

- N1: neuron_counts_summary reads source/target_neurons.csv from the
  minsyn folder ROOT (with data_details/ fallback);
- N2: edge_weight_comparison source/target split (rsplit, duplicate-index
  safe) and per-key dedupe;
- N3: presence columns are consistent 1/0 ints (not mixed True/0);
- N6: unified_summary + path_count_comparison count unique pairs.
"""

import os
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import coana  # noqa: E402


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_analyzer(tmp_path, thresholds=(3, 5, 10), datasets=None):
    """Minimal ComparisonAnalyzer wired to a tmp output folder."""
    from comparison.comparison_analyzer import ComparisonAnalyzer
    from comparison.comparison_parameters import ComparisonParameters

    params = ComparisonParameters(
        datasets=datasets or ['male-cns:v1.0', 'flywire_FAFB_v783'],
        source_neurons=['L2'],
        target_neurons=['l-LNv'],
        max_interlayer=2,
        thresholds=list(thresholds),
        output_folder=str(tmp_path),
        auto_type_mapping=False,
        verbose=False,
    )
    return ComparisonAnalyzer(params, verbose=False)


def _conn_df(rows):
    """Raw-results-style frame: (type_pre, type_post, weight) rows, one row
    per (pair, layer) occurrence — duplicates intentional."""
    return pd.DataFrame(
        rows, columns=['type_pre', 'type_post', 'weight'])


# ---------------------------------------------------------------------------
# Fix B / N6: unique-pair counting
# ---------------------------------------------------------------------------

def test_unified_summary_counts_unique_pairs(tmp_path):
    an = _make_analyzer(tmp_path)
    # 3 rows but only 2 unique pairs; layer dup must not inflate the count
    an.raw_results = {'male-cns:v1.0': {3: _conn_df([
        ('A', 'B', 10), ('A', 'B', 4), ('B', 'C', 2),
    ])}}
    out = tmp_path / 'cr'
    out.mkdir()
    an._export_unified_summary(str(out))

    summary = pd.read_csv(out / 'unified_summary.csv')
    row = summary[(summary.dataset == 'male-cns_v1_0') & (summary.threshold == 3)]
    assert len(row) == 1
    assert int(row['total_edges'].iloc[0]) == 2          # unique pairs
    assert int(row['total_layer_rows'].iloc[0]) == 3     # raw rows kept visible


def test_path_count_comparison_counts_unique_pairs(tmp_path):
    an = _make_analyzer(tmp_path)
    an.raw_results = {'male-cns:v1.0': {3: _conn_df([
        ('A', 'B', 10), ('A', 'B', 4), ('B', 'C', 2),
    ])}}
    out = tmp_path / 'cr'
    out.mkdir()
    an._export_cross_dataset_comparisons(str(out))

    pc = pd.read_csv(out / 'path_count_comparison.csv')
    row = pc[(pc.dataset == 'male-cns:v1.0') & (pc.threshold == 3)]
    assert int(row['connection_count'].iloc[0]) == 2


# ---------------------------------------------------------------------------
# N1: neuron counts read the minsyn-root file
# ---------------------------------------------------------------------------

def test_neuron_counts_reads_minsyn_root_file(tmp_path):
    an = _make_analyzer(tmp_path)
    # full_output_path = output_folder + a timestamped run folder
    ds_dir = Path(an.parameters.full_output_path) / 'dataset_data' / \
        'male-cns_v1_0' / 'minsyn_3'
    ds_dir.mkdir(parents=True)
    # FNC writes source/target_neurons.csv at the minsyn ROOT (N1 fix)
    pd.DataFrame({'bodyId': [1, 2, 3], 'type': ['A', 'B', 'C']}).to_csv(
        ds_dir / 'source_neurons.csv', index=False)
    pd.DataFrame({'bodyId': [9], 'type': ['T']}).to_csv(
        ds_dir / 'target_neurons.csv', index=False)

    cr = tmp_path / 'cr'
    cr.mkdir(parents=True, exist_ok=True)
    an._export_neuron_counts_comparison(str(cr))
    summary = pd.read_csv(cr / 'neuron_counts_summary.csv')
    row = summary[summary.dataset == 'male-cns_v1_0']
    assert int(row['source_count'].iloc[0]) == 3
    assert int(row['target_count'].iloc[0]) == 1


# ---------------------------------------------------------------------------
# N2: edge weight comparison source/target split + dedupe
# ---------------------------------------------------------------------------

def test_edge_weight_export_splits_keys_and_dedupes(tmp_path):
    from comparison.comparison_analyzer import ComparisonAnalyzer

    an = _make_analyzer(tmp_path, thresholds=[3])
    # aligned data: duplicate index labels (per-layer rows) that used to
    # crash the fillna-with-Series split and fall back to whole keys
    aligned = pd.DataFrame(
        {'male-cns:v1.0': [17, 4], 'flywire_FAFB_v783': [0, 6]},
        index=pd.Index(['184653 -> L3', 'L2 -> l-LNv'], name='edge'))
    an.aligned_results = {3: aligned}
    an.raw_results = {}

    out = tmp_path / 'cr'
    out.mkdir()
    # the edge-weight table is exported inside the cross-dataset pass
    an._export_cross_dataset_comparisons(str(out))

    df = pd.read_csv(out / 'edge_weight_comparison.csv', keep_default_na=False)
    assert (df['source'] == df['edge_key']).sum() == 0, 'whole-key source leaked'
    assert (df['target'] == '').sum() == 0, 'empty target leaked'
    row = df[df.edge_key == '184653 -> L3'].iloc[0]
    # N5: digit-only endpoints are labelled as untyped bodyIds
    assert row['source'] == 'bodyId:184653 (untyped)'
    assert row['target'] == 'L3'
    assert len(df) == 2  # deduped to one row per edge key


# ---------------------------------------------------------------------------
# N3: presence columns parse as ints
# ---------------------------------------------------------------------------

def test_unified_edge_presence_is_int(tmp_path):
    an = _make_analyzer(tmp_path, thresholds=[3])
    an.raw_results = {
        'male-cns:v1.0': {3: _conn_df([('A', 'B', 10)])},
        'flywire_FAFB_v783': {3: _conn_df([])},
    }
    out = tmp_path / 'cr'
    out.mkdir()
    an._export_unified_summary(str(out))

    df = pd.read_csv(out / 'unified_edge_comparison.csv')
    present_cols = [c for c in df.columns if c.endswith('_present')]
    assert present_cols, 'no presence columns exported'
    for c in present_cols:
        assert df[c].isin([0, 1]).all(), f'{c} has non-0/1 values'


# ---------------------------------------------------------------------------
# §5: export_conflicts dataset scoping
# ---------------------------------------------------------------------------

def test_export_conflicts_datasets_filter(tmp_path, monkeypatch):
    from comparison.cross_dataset_type_mapper import (
        CrossDatasetTypeMapper, TypeMappingConflict)

    mapper = CrossDatasetTypeMapper.__new__(CrossDatasetTypeMapper)
    mapper.verbose = False
    conflicts = [
        # (source_dataset, target_dataset, source_type, target_types, rel)
        TypeMappingConflict('male-cns:v1.0', 'flywire_FAFB_v783',
                            'PVLP123', {'PVLP123a'}, '1-to-N'),
        TypeMappingConflict('male-cns:v1.0', 'banc_v626',
                            'Xyz1', {'Xyz1a'}, '1-to-N'),
        TypeMappingConflict('male-cns:v1.0', 'hemibrain:v1.2.1',
                            'Abc2', {'Abc2a'}, '1-to-N'),
    ]
    monkeypatch.setattr(mapper, '_conflicts', conflicts, raising=False)

    out = tmp_path / 'conflicts.csv'
    mapper.export_conflicts(str(out), datasets=['male-cns:v1.0', 'flywire_FAFB_v783'])
    df = pd.read_csv(out)
    assert (df.target_dataset == 'flywire_FAFB_v783').all()
    assert 'banc_v626' not in df.target_dataset.values
    assert 'hemibrain:v1.2.1' not in df.target_dataset.values
