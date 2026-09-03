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
    preferred_bridge_chain,
    standardize_bridge,
)
from ui.neuron_index import (
    collect_native_type_matches,
    enrich_native_type_matches,
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


# =============================================================================
# Round 1 of the bodyId-level bridge plan (_plan/plan-type-mapping-bodyid-bridge.md):
# standardized linker extraction, the at-most-two-linker theorem, and the
# BANC type-name routing through FAFB annotations into the male-cns crosswalk.
# =============================================================================

def test_standardize_bridge_real_anchors(mapper):
    """CL125 / PLP080 / SLP250 chains standardize to the registry linkers."""
    from comparison.cross_dataset_type_mapper import standardize_bridge

    # CL125: the full two-linker standard (flywireType -> additional_type(s))
    chains = mapper.get_type_bridges('CL125', MCNS, FW)
    two_linker = next(
        chain for chain in chains
        if [('flywireType', 'LMTe01'), ('additional_type(s)', 'LMTe01')]
        == [(l['column'], l['value'])
            for l in standardize_bridge(chain, MCNS, FW)
            if l['kind'] == 'linker'])
    linkers = standardize_bridge(two_linker, MCNS, FW)
    linker_only = [l for l in linkers if l['kind'] == 'linker']
    assert sorted((l['column'], l['value']) for l in linker_only) == [
        ('additional_type(s)', 'LMTe01'), ('flywireType', 'LMTe01')]
    assert not any(l['indirect'] for l in linkers)

    # PLP080: same two-linker shape with its own value — the literal
    # PLP080 -- PLP080 (additional_type(s)) -- APDN3 path (located by
    # signature: chain order varies with the pruned walk)
    chains_plp = mapper.get_type_bridges('PLP080', MCNS, FW)
    chain = next(
        c for c in chains_plp
        if [('flywireType', 'PLP080'), ('additional_type(s)', 'PLP080')]
        == [(l['column'], l['value'])
            for l in standardize_bridge(c, MCNS, FW)
            if l['kind'] == 'linker'])
    linker_only = [l for l in standardize_bridge(chain, MCNS, FW)
                   if l['kind'] == 'linker']
    assert [(l['column'], l['value']) for l in linker_only] == [
        ('flywireType', 'PLP080'), ('additional_type(s)', 'PLP080')]

    # SLP250: the preferred chain is the registry two-linker standard
    chain = preferred_bridge_chain(
        mapper.get_type_bridges('SLP250', MCNS, FW), MCNS, FW)
    linker_only = [l for l in standardize_bridge(chain, MCNS, FW)
                   if l['kind'] == 'linker']
    # the annotation linker's value is the CELL entry on APDN3 rows:
    # 'LTe71' (SLP250's own crosswalk name echoed in the annotation)
    assert [(l['column'], l['value']) for l in linker_only] == [
        ('flywireType', 'LTe71'), ('additional_type(s)', 'LTe71')]

    # MDN -> BANC same-name chain: its crosswalk verification linker
    # (flywireType 'MDN' — the male-cns cell names the BANC type itself)
    chain = mapper.get_type_bridges('MDN', MCNS, BANC)[0]
    linkers = [l for l in standardize_bridge(chain, MCNS, BANC)
               if l['kind'] == 'linker']
    assert [(l['column'], l['value']) for l in linkers] == [
        ('flywireType', 'MDN')]
    # registry-less (BANC) pairs flag every linker indirect — that is the
    # designed honesty about the weaker evidence, not a defect
    assert all(l['indirect'] for l in linkers)

    # endpoints are always the two datasets' type identities
    assert chain[0]['column'] == 'type' and chain[0]['dataset'] == MCNS
    assert chain[-1]['dataset'] == BANC


def test_banc_type_names_route_through_annotations(mapper):
    """BANC type names map into male-cns through FAFB/BANC annotations.

    DNp50 is a BANC v626 primary whose male-cns counterpart is MDN (a
    rename): the chains must route through the additional Type(S)
    annotations, and the ambiguity (several chains) is reported.
    """
    chains = mapper.get_type_bridges('DNp50', BANC, MCNS)
    assert chains, 'DNp50 (BANC) must route into male-cns'
    # every chain ends at a male-cns type identity
    for chain in chains:
        assert chain[0]['dataset'] == BANC and chain[0]['column'] == 'type'
        assert chain[-1]['dataset'] == MCNS
        assert chain[-1]['column'] == 'type'
    # at least one chain routes through an annotation linker naming MDN
    assert any(
        any(hop['column'] in ('additional_type(s)', 'Alternative Cell Type(s)')
            and hop.get('via') == 'DNp50' for hop in chain)
        for chain in chains)
    # DNp50 has no male-cns same-name (it is a rename), so EVERY chain must
    # carry an annotation linker — the ambiguity is the chain count.
    assert len(chains) >= 2
    assert all(
        any(hop['column'] in ('additional_type(s)',
                              'Alternative Cell Type(s)')
            for hop in chain)
        for chain in chains)


def test_two_linker_cap_over_sample_matrix(mapper):
    """Every chain standardizes to at most two linker nodes (the theorem)."""
    pairs = [
        # registry pairs: the two-linker standard holds strictly
        (MCNS, FW), (MCNS, HB),
        # hub pairs (BANC routes through FAFB annotations): a third
        # annotation hop is legitimate and every linker is flagged indirect
        (MCNS, BANC), (BANC, MCNS), (FW, MCNS), (BANC, FW), (FW, BANC),
    ]
    types = ['MDN', 'CL125', 'APDN3', 'SLP249', 'aMe12', 'TmY9q', 'Dn3',
             'vDeltaB', 'LC10', 'DNp50']
    checked = 0
    for src, tgt in pairs:
        for type_name in types:
            for chain in mapper.get_type_bridges(type_name, src, tgt):
                linkers = standardize_bridge(chain, src, tgt)
                # The two-linker standard binds the DIRECT (registry) linkers;
                # deeper FAFB rename chains (e.g. vDeltaB -> vDeltaA ->
                # vDeltaL into hemibrain) are legitimate but every extra
                # linker is flagged indirect.
                direct = [l for l in linkers
                          if l['kind'] == 'linker' and not l['indirect']]
                assert len(direct) <= 2, (src, tgt, type_name, chain)
                checked += 1
    assert checked > 20, f'sample too small: {checked} chains'


def test_plp080_renamed_resolution_regression(mapper):
    """The FAFB-side search for PLP080 (additional-only there) resolves to
    APDN3 with the N-to-1 aggregates — preserved by the v2 changes."""
    res = mapper.get_alias_candidates('PLP080', [FW])
    info = res[FW]
    assert info['outcome'] == 'matched'
    assert [c['name'] for c in info['candidates']] == ['APDN3']
    assert info['candidates'][0]['kind'] == 'renamed'
    assert info['candidates'][0]['aggregates'] == [
        'CL125', 'PLP080', 'SLP249', 'SLP250']


def _entry(matches, dataset):
    for entry in matches:
        if entry['dataset'] == dataset:
            return entry
    return None


def test_bridge_linker_text_values_and_hub_note(mapper):
    """bridge_linker_text: values included, deduped, indirect (hub) linkers
    last with the hub note."""
    from comparison.cross_dataset_type_mapper import bridge_linker_text

    chains = mapper.get_type_bridges('CL125', MCNS, FW)
    info = bridge_linker_text(chains, MCNS, FW, 'APDN3')
    # the FAFB APDN3 rows carry 'CL125' directly, so the deduplicated
    # linker text is: direct annotation evidence first, then the
    # two-linker crosswalk standard
    assert info['text'] == (
        "additional_type(s) 'CL125' + flywireType 'LMTe01' "
        "+ additional_type(s) 'LMTe01'")
    assert [e['value'] for e in info['entries']] == [
        'CL125', 'LMTe01', 'LMTe01']
    assert not any(e['indirect'] for e in info['entries'])

    # the CL125 -> LTe71 (BANC) "hub chain" was pure transitivity noise
    # (BANC LTe71's own Alternative cell names only itself) — pruned and
    # stays pruned; indirect (hub-routed) linkers still occur on real pairs
    hub_chains = mapper.get_type_bridges('CL125', MCNS, BANC)
    assert not [c for c in hub_chains if c and c[-1]['value'] == 'LTe71']
    indirect_found = any(
        any(l['indirect'] for l in standardize_bridge(
            chain, MCNS, BANC))
        for type_name in ('l-LNv', 'DN1pA', 'CB3508')
        for chain in mapper.get_type_bridges(type_name, MCNS, BANC))
    assert indirect_found


def test_mapping_sankey_figure_restored(mapper):
    """The restored native sankey renders the standardized linker bands
    with pooled-count ribbons and the CSV-notice title."""
    import plotly.graph_objects as go

    from comparison.mapping_visualization import (
        build_mapping_flows,
        build_mapping_sankey_figure,
    )
    from ui.neuron_index import (
        load_cached_neuron_index,
        pool_bridge_body_ids,
    )

    native = collect_native_type_matches(MCNS, 'APDN3', uncapped=True)
    enrich_native_type_matches(native, MCNS)
    entry = _entry(native, FW)
    assert entry
    index = load_cached_neuron_index(MCNS)
    foreign_index = load_cached_neuron_index(FW)
    flows = build_mapping_flows(
        [entry], MCNS,
        source_counts=count_types_in_index_map(index, entry))
    assert flows
    pools = {}
    for flow in flows:
        chain = preferred_bridge_chain(flow['bridges'], MCNS, FW)
        if chain is None:
            continue
        pool = pool_bridge_body_ids(
            MCNS, FW, standardize_bridge(chain, MCNS, FW),
            flow['source_type'], flow['foreign_type'],
            indexes={MCNS: index, FW: foreign_index})
        pools[(flow['source_type'], flow['foreign_type'])] = pool

    fig = build_mapping_sankey_figure(flows, pools=pools)
    assert fig is not None and fig.data[0].type == 'sankey'
    labels = list(fig.data[0].node['label'])
    values = list(fig.data[0].link['value'])
    # linker bands colored per column: amber flywireType + violet annotations
    colors = list(fig.data[0].node['color'])
    assert '#f59e0b' in colors and '#a855f7' in colors
    # pooled ribbons: each drawn chain contributes its pooled count per
    # hop — 4-bodyId bridges (CL125, SLP249) and 2-bodyId bridges
    # (SLP250, PLP080); two derivation chains per flow survive (hops
    # aggregate into 20 ribbons)
    assert sorted(values) == [2] * 10 + [4] * 10
    # no cap hit at 4 flows -> no notice; capped -> notice points to CSV
    assert 'full mapping' not in fig.layout.title.text
    capped = build_mapping_sankey_figure(flows, pools=pools, max_flows=2)
    assert 'full mapping is in the CSV export' in capped.layout.title.text


def count_types_in_index_map(index, entry):
    from ui.neuron_index import count_types_in_index
    return count_types_in_index(index, entry.get('mapped_type_names', []))


def test_analyzer_mapping_export_imports():
    """The Cross-Dataset run's mapping_sankey export imports cleanly
    (regression: the removed plotly builder broke it silently)."""
    import importlib

    module = importlib.import_module('comparison.mapping_visualization')
    assert hasattr(module, 'build_mapping_sankey_figure')
    source = Path(  # the analyzer's lazy import names must all resolve
        Path(__file__).resolve().parents[2]
        / 'src' / 'comparison' / 'comparison_analyzer.py').read_text(
        encoding='utf-8')
    segment = source[source.index('build_mapping_sankey_figure'):]
    assert 'write_mapping_network_html' in segment


def test_mapping_type_sankey_and_in_memory_renderers(mapper):
    """Type-level sankey bands (§9C.6) + the in-memory vispath renderers
    (no repository writes — temp-dir render, HTML string back)."""
    from comparison.mapping_visualization import (
        build_mapping_flows,
        build_mapping_type_sankey_figure,
        render_bridge_linker_html,
        render_mapping_network_html,
    )
    from ui.neuron_index import (
        count_types_in_index,
        load_cached_neuron_index,
        pool_bridge_body_ids,
    )

    native = collect_native_type_matches(MCNS, 'APDN3', uncapped=True)
    enrich_native_type_matches(native, MCNS)
    entry = _entry(native, FW)
    assert entry
    index = load_cached_neuron_index(MCNS)
    foreign_index = load_cached_neuron_index(FW)
    flows = build_mapping_flows(
        [entry], MCNS, source_counts=count_types_in_index(
            index, entry.get('mapped_type_names', [])))
    assert flows
    pools = {}
    for flow in flows:
        chain = preferred_bridge_chain(flow['bridges'], MCNS, FW)
        if chain is None:
            continue
        pools[(flow['source_type'], flow['foreign_type'])] = (
            pool_bridge_body_ids(
                MCNS, FW, standardize_bridge(chain, MCNS, FW),
                flow['source_type'], flow['foreign_type'],
                indexes={MCNS: index, FW: foreign_index}))

    fig = build_mapping_type_sankey_figure(flows, pools=pools)
    assert fig is not None and fig.data[0].type == 'sankey'
    labels = list(fig.data[0].node['label'])
    values = list(fig.data[0].link['value'])
    # two bands only: the 4 mapped sources -> the single FAFB target,
    # one aggregated ribbon per pair with the pooled granularity
    assert len(labels) == len(flows) + 1
    assert sorted(values) == sorted(
        min(len(p['source_body_ids']), len(p['target_body_ids']))
        or 1 for p in pools.values())
    colors = set(list(fig.data[0].node['color']))
    assert colors == {'#5b8cff', '#22c55e'}  # source blue / target green
    # cap notice only when the cap trims flows
    assert 'full mapping' not in fig.layout.title.text
    capped = build_mapping_type_sankey_figure(flows, pools=pools,
                                              max_flows=1)
    assert 'full mapping is in the CSV export' in capped.layout.title.text

    # in-memory renderers: strings, correct renderer, no repo writes
    net_html = render_mapping_network_html(flows)
    assert net_html and 'cytoscape' in net_html.lower()
    linker_html = render_bridge_linker_html(
        flows, source_dataset=MCNS, target_dataset=FW, pools=pools)
    assert linker_html and 'cytoscape' in linker_html.lower()
    assert 'flywireType' in linker_html


def test_bridge_linker_text_warns_on_unverified_same_name():
    """A same-name pair whose only chain is bare name equality states
    'no metadata verification (please double check)'; a verified chain
    overrides the warning (§9F)."""
    from comparison.cross_dataset_type_mapper import bridge_linker_text

    chains = [[{"dataset": MCNS, "column": "type", "value": "X"},
               {"dataset": FW, "column": "type", "value": "X"}]]
    info = bridge_linker_text(chains, MCNS, FW, "X")
    assert info["text"] == ("same name — no metadata verification "
                            "(please double check)")

    # with a crosswalk-verification chain the warning is replaced by the
    # linker facts (same name + the verification)
    chains.append([{"dataset": MCNS, "column": "type", "value": "X"},
                   {"dataset": FW, "column": "flywireType", "value": "X"}])
    info = bridge_linker_text(chains, MCNS, FW, "X")
    assert info["text"] == ("same name + flywireType 'X'")


def test_dataset_abbreviations_and_version_suffixes():
    """4-char dataset names: FAFB (never FLYW), and family collisions
    (two male-cns or two BANC versions) get version suffixes."""
    from utils.naming_utils import dataset_abbrev, make_unique_dataset_labels

    assert dataset_abbrev('flywire_FAFB_v783') == 'FAFB'
    assert dataset_abbrev('flywire_BANC_v888') == 'BANC'
    assert dataset_abbrev('flywire_BANC_v626') == 'BANC'
    assert dataset_abbrev('male-cns:v1.0') == 'MCNS'
    assert dataset_abbrev('flywire') == 'FAFB'  # never the FLYW fallback

    labels = make_unique_dataset_labels([
        'male-cns:v1.0', 'male-cns:v0.9',
        'flywire_BANC_v888', 'flywire_BANC_v626',
        'flywire_FAFB_v783'])
    # BOTH colliding labels get versions — a bare MCNS would stay ambiguous
    assert labels == ['MCNS_v1_0', 'MCNS_v0_9', 'BANC_v888', 'BANC_v626',
                      'FAFB']
