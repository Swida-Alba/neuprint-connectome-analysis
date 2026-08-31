"""Real-dataset validation for the cross-dataset type mapping.

The hermetic tests in ``test_type_mapper_coverage.py`` validate the mapping
logic on synthetic tables.  This module loads the *actual* local datasets
(male-cns v1.0, FlyWire FAFB v783, FlyWire BANC v626) and validates the
additional Type(S) rename resolution end to end:

* male-cns ``SLP249`` (flywireType ``SLP249``) must resolve to FAFB ``APDN3``
  because FAFB v783 renamed those neurons and keeps ``SLP249`` only in its
  ``additional_type(s)`` column;
* every resolved mapping target must be a *current* primary type of the
  target dataset (or a name genuinely absent from it) - never a stale
  additional-only name;
* ambiguous renames (one old name listed under several primaries) and
  comma-separated crosswalk cells must produce conflicts, not guesses;
* FAFB and BANC rename independently, so their namespaces are checked
  separately.

All tests skip when the local datasets are not present.
"""

from pathlib import Path

import pandas as pd
import pytest

from comparison.cross_dataset_type_mapper import (
    CrossDatasetTypeMapper,
    get_type_mapper,
)

MCNS = 'male-cns:v1.0'
FW = 'flywire_FAFB_v783'
BANC = 'flywire_BANC_v626'
HB = 'hemibrain:v1.2.1'

REPO_ROOT = Path(__file__).resolve().parents[2]
MCNS_CSV = REPO_ROOT / 'datasets' / 'male-cns_v1_0' / 'male-cns_v1_0_allneurons_neuron_df.csv'
FAFB_CSV = REPO_ROOT / 'datasets' / 'flywire_FAFB_v783' / 'flywire_FAFB_v783_allneurons_neuron_df.csv'
BANC_CSV = REPO_ROOT / 'datasets' / 'flywire_BANC_v626' / 'flywire_BANC_v626_allneurons_neuron_df.csv'

pytestmark = pytest.mark.skipif(
    not (MCNS_CSV.exists() and FAFB_CSV.exists()),
    reason='real male-cns v1.0 / FAFB v783 neuron tables not available locally',
)

requires_banc = pytest.mark.skipif(
    not BANC_CSV.exists(),
    reason='real BANC v626 neuron table not available locally',
)


@pytest.fixture(scope='module')
def mapper():
    m = CrossDatasetTypeMapper(verbose=False, workspace_path=str(REPO_ROOT))
    assert m.load() is True
    return m


@pytest.fixture(scope='module')
def fafb_names():
    """(primary, additional-only) type name sets recomputed from FAFB v783."""
    table = pd.read_csv(FAFB_CSV, usecols=['type', 'additional_type(s)'], low_memory=False)
    primaries = set(table['type'].dropna().astype(str).str.strip()) - {''}
    additional = set()
    for cell in table['additional_type(s)'].dropna().astype(str):
        additional.update(name.strip() for name in cell.split(',') if name.strip())
    return primaries, additional - primaries


@pytest.fixture(scope='module')
def banc_names():
    """(primary, additional-only) type name sets recomputed from BANC v626."""
    table = pd.read_csv(
        BANC_CSV, usecols=['type', 'Alternative Cell Type(s)'], low_memory=False)
    primaries = set(table['type'].dropna().astype(str).str.strip()) - {''}
    additional = set()
    for cell in table['Alternative Cell Type(s)'].dropna().astype(str):
        additional.update(name.strip() for name in cell.split(',') if name.strip())
    return primaries, additional - primaries


# ---------------------------------------------------------------------------
# Load integrity
# ---------------------------------------------------------------------------

def test_flywire_alt_tables_are_indexed(mapper):
    # SLP249 is additional-only in FAFB; APDN3 (its current primary) must
    # NOT be indexed as an additional-only name.
    assert 'SLP249' in mapper._flywire_alt_to_primary.get(FW, {})
    assert 'APDN3' not in mapper._flywire_alt_to_primary[FW]
    # observed: 1,208 FAFB / 642 BANC additional-only names; keep floors
    assert len(mapper._flywire_alt_to_primary[FW]) >= 700
    assert len(mapper._flywire_alt_to_primary[BANC]) >= 500


# ---------------------------------------------------------------------------
# The reported issue: male-cns SLP249 vs FAFB APDN3
# ---------------------------------------------------------------------------

def test_reported_issue_slp249_resolves_to_apdn3(mapper):
    # SLP249 is not a FAFB v783 primary type anymore; the male-cns crosswalk
    # name must resolve to the current FAFB primary name.
    assert mapper.get_mapped_type('SLP249', MCNS, FW) == 'APDN3'

    resolved = mapper.resolve_type_across_datasets(
        'SLP249', [MCNS, FW], source_dataset=MCNS)
    assert resolved == {MCNS: 'SLP249', FW: 'APDN3'}

    assert mapper.get_display_name('SLP249', [MCNS, FW]) == 'SLP249(APDN3)'
    _, hover = mapper.get_display_name_with_dataset_info(
        'SLP249', [MCNS, FW], source_dataset=MCNS)
    assert hover == {'M': 'SLP249', 'F': 'APDN3'}


def test_known_fafb_renames(mapper):
    expected = {
        'SLP249': 'APDN3',
        'CL125': 'APDN3',        # flywireType LMTe01
        'MDN': 'DNp50',
        'IPC': 'm_NSC_DILP',
        'mAL4A': 'mAL',          # flywireType mAL4
        'FS4B': 'FS4A',
        'FS4C': 'FS4A',
        'DNES1': 'l_NSC_unknown',
        'Hugin-RG': 'SEZ_NSC_Hugin',
        'CAPA': 'SEZ_NSC_CAPA',
        'ITP': 'l_NSC_ITP',
        'aSP22': 'DNa12',
        'TuBu04': 'TuBu03',
    }
    for mcns_type, fafb_type in expected.items():
        assert mapper.get_mapped_type(mcns_type, MCNS, FW) == fafb_type, mcns_type


def test_ambiguous_renames_become_conflicts(mapper):
    # Each old name is listed under several FAFB primaries (a split), so no
    # single target can be chosen: no mapping, exact conflict recorded.
    expected_splits = {
        'SLP295': {'SLP295a', 'SLP295b'},
        'AOTU008': {'AOTU008a', 'AOTU008b', 'AOTU008c', 'AOTU008d'},
        'SMP206': {'SLP327a', 'SLP327b'},
        'SMP426': {'SLP402a', 'SLP402b', 'SLP402c'},
    }
    conflicts = mapper.get_1_to_n_conflicts()
    for mcns_type, targets in expected_splits.items():
        assert mapper.get_mapped_type(mcns_type, MCNS, FW) is None, mcns_type
        assert any(
            c.source_type == mcns_type
            and c.target_dataset == FW
            and c.target_types == targets
            for c in conflicts
        ), mcns_type


def test_comma_separated_crosswalk_cell_expands_to_conflict(mapper):
    # male-cns 'VS' lists eight flywire types in one cell -> 1-to-N conflict
    vs_targets = {f'VS{i}' for i in range(1, 9)}
    assert mapper.get_mapped_type('VS', MCNS, FW) is None
    for fw_key in (FW, BANC):
        assert any(
            c.source_type == 'VS'
            and c.target_dataset == fw_key
            and c.target_types == vs_targets
            for c in mapper.get_1_to_n_conflicts()
        ), fw_key


def test_comma_separated_hemibrain_cell_expands_to_conflict(mapper):
    # male-cns IPC: hemibrainType 'PI1,PI2,PI3' -> no single hemibrain target
    assert mapper.get_mapped_type('IPC', MCNS, HB) is None
    assert any(
        c.source_type == 'IPC'
        and c.target_dataset == HB
        and c.target_types == {'PI1', 'PI2', 'PI3'}
        for c in mapper.get_1_to_n_conflicts()
    )


# ---------------------------------------------------------------------------
# Global invariants over every mapping
# ---------------------------------------------------------------------------

def _iter_forward_targets(mapper, fw_key):
    for targets in mapper._type_mappings[MCNS].values():
        value = targets.get(fw_key)
        if value:
            yield value


def test_no_fafb_target_is_a_stale_name(mapper, fafb_names):
    # A mapping to an additional-only name is exactly the bug this fix
    # addresses: targets must be current primaries (or unknown names).
    _, additional_only = fafb_names
    stale = {v for v in _iter_forward_targets(mapper, FW) if v in additional_only}
    assert not stale


@requires_banc
def test_no_banc_target_is_a_stale_name(mapper, banc_names):
    _, additional_only = banc_names
    stale = {v for v in _iter_forward_targets(mapper, BANC) if v in additional_only}
    assert not stale


def test_rename_scale(mapper, fafb_names):
    # Observed: 89 male-cns types resolve to a different FAFB primary name.
    primaries, _ = fafb_names
    raw_by_type = (
        mapper._neuron_df.groupby('type')['flywireType']
        .apply(lambda s: {
            name.strip()
            for cell in s.dropna().astype(str)
            for name in cell.split(',') if name.strip()
        })
        .to_dict()
    )
    renamed = {
        mcns_type
        for mcns_type, targets in mapper._type_mappings[MCNS].items()
        if (v := targets.get(FW))
        and v in primaries
        and v not in raw_by_type.get(mcns_type, set())
    }
    assert len(renamed) >= 50
    assert 'SLP249' in renamed


# ---------------------------------------------------------------------------
# FAFB / BANC namespace independence
# ---------------------------------------------------------------------------

@requires_banc
def test_fafb_and_banc_namespaces_resolve_independently(mapper):
    # FAFB renamed MDN -> DNp50; BANC still uses MDN as its primary type.
    assert mapper.get_mapped_type('MDN', MCNS, FW) == 'DNp50'
    assert mapper.get_mapped_type('MDN', MCNS, BANC) == 'MDN'
    # ...and the reverse: BANC renamed DNge036 -> DNfl042 while FAFB keeps it.
    assert mapper.get_mapped_type('DNge036', MCNS, FW) == 'DNge036'
    assert mapper.get_mapped_type('DNge036', MCNS, BANC) == 'DNfl042'
    # The namespaces stay connected through the shared male-cns crosswalk.
    assert mapper.get_mapped_type('DNp50', FW, BANC) == 'MDN'
    assert mapper.get_mapped_type('MDN', BANC, FW) == 'DNp50'


# ---------------------------------------------------------------------------
# Reverse direction / canonicalization of renamed types
# ---------------------------------------------------------------------------

def test_reverse_of_merged_renames_is_flagged_n_to_1(mapper):
    # FAFB APDN3 receives SLP249, CL125 (LMTe01), SLP250 (LTe71) and PLP080;
    # aggregation is refused and flagged instead.
    assert mapper.get_mapped_type('APDN3', FW, MCNS) is None
    assert mapper.is_n_to_1_type('APDN3', FW)
    conflict = next(
        c for c in mapper.get_n_to_1_conflicts()
        if c.source_dataset == FW and c.source_type == 'APDN3'
    )
    assert conflict.target_types == {'SLP249', 'CL125', 'SLP250', 'PLP080'}

    # FS4A collects FS4B and FS4C (plus FS4A itself).
    assert mapper.is_n_to_1_type('FS4A', FW)
    fs4a = next(
        c for c in mapper.get_n_to_1_conflicts()
        if c.source_dataset == FW and c.source_type == 'FS4A'
    )
    assert {'FS4B', 'FS4C'} <= fs4a.target_types


def test_unique_reverse_rename_canonicalizes_partners(mapper):
    # DNp50 (FAFB) is only fed by male-cns MDN, so partner standardization
    # can safely canonicalize it.
    assert mapper.get_canonical_type('DNp50', FW) == 'MDN'
    assert not mapper.is_n_to_1_type('DNp50', FW)
    standardized = mapper.standardize_partner_types({'DNp50': 2.0}, FW)
    assert standardized == {'MDN': pytest.approx(2.0)}


def test_source_detection_uses_resolved_names(mapper):
    # APDN3 is registered as a FAFB-namespace type; SLP249 only as male-cns.
    assert mapper._detect_type_source('APDN3') == FW
    assert mapper._detect_type_source('SLP249') == MCNS
    assert 'APDN3' in mapper._dataset_types[FW]
    assert 'SLP249' not in mapper._dataset_types[FW]
    assert 'SLP249' in mapper._dataset_types[MCNS]


def test_rename_round_trip_consistency(mapper):
    # For every mapped male-cns type, canonicalizing its FAFB target either
    # returns the target itself (merged/N-to-1) or the source type (unique).
    for mcns_type, targets in mapper._type_mappings[MCNS].items():
        value = targets.get(FW)
        if not value:
            continue
        canonical = mapper.get_canonical_type(value, FW)
        assert canonical in (value, mcns_type), (mcns_type, value, canonical)


# ---------------------------------------------------------------------------
# Exports and global wiring
# ---------------------------------------------------------------------------

def test_export_mapping_includes_renames(mapper, tmp_path):
    out = tmp_path / 'mapping.csv'
    mapper.export_mapping(str(out), filter_types={'SLP249'})
    df = pd.read_csv(out)
    row = df[df[MCNS] == 'SLP249']
    assert not row.empty
    assert row.iloc[0][FW] == 'APDN3'


def test_global_type_mapper_singleton_resolves_renames(mapper, monkeypatch):
    from comparison import cross_dataset_type_mapper as mapper_module

    monkeypatch.setattr(mapper_module, '_global_type_mapper', mapper)
    assert get_type_mapper() is mapper
    assert get_type_mapper().get_mapped_type('SLP249', MCNS, FW) == 'APDN3'


# ---------------------------------------------------------------------------
# User warning notes (expanded / N-to-1 / 1-to-N)
# ---------------------------------------------------------------------------

def test_build_user_warning_notes_slp249_rename(mapper):
    notes = mapper.build_user_warning_notes(['SLP249'], [MCNS, FW])
    # expanded to the current FAFB primary name...
    assert any(
        'expanded' in n and "'SLP249'" in n and "'APDN3'" in n for n in notes)
    # ...and flagged N-to-1: FAFB APDN3 also collects CL125/SLP250/PLP080,
    # so the reverse (FAFB -> male-cns) aggregation is refused.
    n_to_1 = [n for n in notes if 'N-to-1' in n]
    assert n_to_1 and all(t in n_to_1[0] for t in ('CL125', 'PLP080', 'SLP250'))
    assert sum('double check' in n.lower() for n in notes) == 1


def test_build_user_warning_notes_apdn3_n_to_1(mapper):
    # FAFB APDN3 aggregates four male-cns types: flagged, never merged
    notes = mapper.build_user_warning_notes(['APDN3'], [FW, MCNS])
    n_to_1 = [n for n in notes if 'N-to-1' in n]
    assert n_to_1, notes
    assert all(t in n_to_1[0] for t in ('CL125', 'PLP080', 'SLP249', 'SLP250'))
    assert any('double check' in n.lower() for n in notes)


def test_build_user_warning_notes_known_rename_batch(mapper):
    # A queried batch mixes renames (expanded) and clean types; every
    # expected rename is reported and the advice line is present once.
    notes = mapper.build_user_warning_notes(
        ['SLP249', 'MDN', 'IPC', 'aMe12'], [MCNS, FW])
    joined = '\n'.join(notes)
    for old, new in (('SLP249', 'APDN3'), ('MDN', 'DNp50'), ('IPC', 'm_NSC_DILP')):
        assert f"'{old}'" in joined and f"'{new}'" in joined
    assert sum('double check' in n.lower() for n in notes) == 1


# ---------------------------------------------------------------------------
# Alias candidates (expanded viewer search) on real data
# ---------------------------------------------------------------------------

def _candidate(res, dataset, name):
    for cand in res[dataset]['candidates']:
        if cand['name'] == name:
            return cand
    return None


def test_get_alias_candidates_slp249_real(mapper):
    res = mapper.get_alias_candidates('SLP249', [MCNS, FW])
    assert res[MCNS]['outcome'] == 'matched'
    assert _candidate(res, MCNS, 'SLP249')['kind'] == 'same name'
    assert res[FW]['outcome'] == 'matched'
    renamed = _candidate(res, FW, 'APDN3')
    assert renamed['kind'] == 'renamed'
    # orthogonal aggregation annotation: FAFB APDN3 also covers the other
    # three male-cns types that were renamed into it.
    assert renamed['aggregates'] == ['CL125', 'PLP080', 'SLP249', 'SLP250']


def test_get_alias_candidates_apdn3_real(mapper):
    res = mapper.get_alias_candidates(
        'APDN3', [MCNS, FW, BANC])
    # native in both FlyWire datasets, with the aggregation annotation on
    # the FAFB candidate (BANC keeps its own 1:1 annotation-free identity).
    fafb = _candidate(res, FW, 'APDN3')
    assert fafb['kind'] == 'same name'
    assert fafb['aggregates'] == ['CL125', 'PLP080', 'SLP249', 'SLP250']
    banc = _candidate(res, BANC, 'APDN3')
    assert banc['kind'] == 'same name'
    assert banc['aggregates'] is None
    # male-cns side: the refused reverse aggregation exposes the group.
    mcns = res[MCNS]
    assert mcns['outcome'] == 'matched'
    assert [c['name'] for c in mcns['candidates']] == [
        'CL125', 'PLP080', 'SLP249', 'SLP250']
    assert all(c['kind'] == 'one of N' for c in mcns['candidates'])


def test_get_alias_candidates_namespace_independence_real(mapper):
    res = mapper.get_alias_candidates('MDN', [MCNS, FW, BANC])
    assert _candidate(res, MCNS, 'MDN')['kind'] == 'same name'
    renamed = _candidate(res, FW, 'DNp50')
    assert renamed['kind'] == 'renamed'
    assert renamed['aggregates'] is None  # unique reverse (only male-cns MDN)
    assert _candidate(res, BANC, 'MDN')['kind'] == 'same name'

    # BANC renamed DNge036 -> DNfl042 while FAFB keeps it natively.
    res_d = mapper.get_alias_candidates('DNge036', [MCNS, FW, BANC])
    assert _candidate(res_d, FW, 'DNge036')['kind'] == 'same name'
    assert _candidate(res_d, BANC, 'DNfl042')['kind'] == 'renamed'


def test_get_alias_candidates_vs_splits_real(mapper):
    # male-cns 'VS' lists eight FAFB types in one crosswalk cell: the
    # candidates are the split targets, never a joined name.
    res = mapper.get_alias_candidates('VS', [MCNS, FW])
    fw_names = [c['name'] for c in res[FW]['candidates']]
    assert fw_names == [f'VS{i}' for i in range(1, 9)]
    assert all(c['kind'] == 'splits into' for c in res[FW]['candidates'])
    assert res[MCNS]['candidates'] == [
        {'name': 'VS', 'kind': 'same name', 'aggregates': None},
    ]


def test_get_alias_candidates_unknown_real(mapper):
    res = mapper.get_alias_candidates('NoRealType123', [MCNS, FW, BANC])
    assert all(
        res[ds]['outcome'] == 'no counterpart known'
        for ds in (MCNS, FW, BANC)
    )


def test_get_alias_candidates_kinds_exclusive_real(mapper):
    # property over a broad real-type batch: one valid kind per candidate,
    # unique names per dataset, aggregates never present on 'one of N' /
    # 'splits into' candidates (it is a candidate-side annotation).
    queries = [
        'SLP249', 'APDN3', 'MDN', 'DNp50', 'IPC', 'm_NSC_DILP', 'LPN',
        'MTe07', 'MeVPLo2', 'aMe12', 'VS', 'DNge036', 'DNfl042', 'TuBu03',
        'TuBu04', 'FS4A', 'FS4B', 'CL125', 'SLP250', 'PLP080', 'aMe12_L',
    ]
    for query in queries:
        res = mapper.get_alias_candidates(query, [MCNS, FW, BANC])
        for dataset, outcome in res.items():
            names = [c['name'] for c in outcome['candidates']]
            assert len(names) == len(set(names)), (query, dataset, names)
            for cand in outcome['candidates']:
                assert cand['kind'] in CrossDatasetTypeMapper.ALIAS_KINDS, (
                    query, dataset, cand)
                if cand['kind'] in ('one of N', 'splits into'):
                    assert cand['aggregates'] is None, (query, dataset, cand)


def test_alias_counts_match_expected_real(mapper):
    # the annotation lists real sibling types: SLP249 itself stays findable
    # in male-cns while FAFB APDN3 counts its four neurons.
    res = mapper.get_alias_candidates('SLP249', [MCNS, FW])
    assert 'SLP249' in res[MCNS]['candidates'][0]['name']
    assert res[FW]['candidates'][0]['name'] == 'APDN3'


def test_get_alias_candidates_combined_cell_name_real(mapper):
    # male-cns 'vDeltaB' is native here; in FAFB the old name 'vDeltaB' was
    # split into vDelta/vDeltaA (it now only survives inside combined
    # additional_type(s) cells) -> 'splits into' with both targets.
    res = mapper.get_alias_candidates('vDeltaB', [MCNS, FW])
    assert _candidate(res, MCNS, 'vDeltaB')['kind'] == 'same name'
    split_names = [c['name'] for c in res[FW]['candidates']]
    assert split_names == ['vDelta', 'vDeltaA']
    assert all(c['kind'] == 'splits into' for c in res[FW]['candidates'])
    assert all(c['aggregates'] is None for c in res[FW]['candidates'])

    # a literal comma-joined query is not a name: honest outcome everywhere.
    joined = mapper.get_alias_candidates(
        'vDeltaB, vDeltaC', [MCNS, FW])
    assert all(
        joined[ds]['outcome'] == 'no counterpart known'
        for ds in (MCNS, FW)
    )
