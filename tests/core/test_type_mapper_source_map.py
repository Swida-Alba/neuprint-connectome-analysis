"""Source-map validation for the cross-dataset type mapper (§9I).

``BRIDGE_SOURCE_MAP`` is the declarative single source of truth for the
valid (dataset, column) derivation bridges.  This module grounds the
map in the *actual* local datasets and verifies that every derivation
chain the mapper emits obeys its licensing rules:

* grounding — every declared (home, column) exists in the dataset
  tables, and the crosswalk columns really hit their target namespaces
  at the expected rates;
* licensing sweep — across every directed namespace pair, chains end
  at target-namespace ``type`` nodes, crosswalk hops only appear when
  the bridge's endpoints include the column's family, no ping-pong
  (consecutive identical standardized linkers), and the ≤2-direct-
  linker standard holds;
* known-pair regression matrix — the screenshot regression
  (``DN1pA[MCNS·hemibrainType] → DN1pA[BANC·type]``) stays dead, the
  sanctioned hemibrain↔flywire route through male-cns survives, and
  the FAFB-annotation / BANC ``Alternative Cell Type(s)`` routing
  evidence is intact.

All tests skip when the local datasets are not present.
"""

from pathlib import Path

import pandas as pd
import pytest

from comparison.cross_dataset_type_mapper import (
    CROSSWALK_COLUMNS,
    BRIDGE_SOURCE_MAP,
    CrossDatasetTypeMapper,
    bridge_is_valid,
    crosswalk_route_licensed,
    standardize_bridge,
)

MCNS = 'male-cns:v1.0'
FAFB = 'flywire_FAFB_v783'
BANC = 'flywire_BANC_v626'
HB = 'hemibrain:v1.2.1'
MANC = 'manc:v1.0'

REPO_ROOT = Path(__file__).resolve().parents[2]
TABLES = {
    MCNS: REPO_ROOT / 'datasets' / 'male-cns_v1_0' / 'male-cns_v1_0_allneurons_neuron_df.csv',
    FAFB: REPO_ROOT / 'datasets' / 'flywire_FAFB_v783' / 'flywire_FAFB_v783_allneurons_neuron_df.csv',
    BANC: REPO_ROOT / 'datasets' / 'flywire_BANC_v626' / 'flywire_BANC_v626_allneurons_neuron_df.csv',
    MANC: REPO_ROOT / 'datasets' / 'manc_v1_0' / 'manc_v1_0_allneurons_neuron_df.csv',
    HB: REPO_ROOT / 'datasets' / 'hemibrain_v1_2_1' / 'hemibrain_v1_2_1_allneurons_neuron_df.csv',
}

pytestmark = pytest.mark.skipif(
    not (TABLES[MCNS].exists() and TABLES[FAFB].exists()),
    reason='real male-cns v1.0 / FAFB v783 neuron tables not available locally',
)


@pytest.fixture(scope='module')
def mapper():
    m = CrossDatasetTypeMapper(verbose=False, workspace_path=str(REPO_ROOT))
    assert m.load() is True
    return m


def _split_cell(value):
    if not isinstance(value, str):
        return set()
    return {n.strip() for n in value.split(',') if n.strip()}


def _column_values(path, column):
    table = pd.read_csv(path, usecols=lambda c: c in ('type', column),
                        low_memory=False)
    return table


def _namespace_names(m, key):
    names = set(m._dataset_types.get(key, {}))
    names |= set(m._flywire_primaries.get(key, ()))
    return names


# ---------------------------------------------------------------------------
# 1. grounding: the map describes the data that actually exists
# ---------------------------------------------------------------------------

def test_source_map_columns_exist_in_dataset_tables():
    for (home, column), _targets in BRIDGE_SOURCE_MAP.items():
        if home == '*' or column == 'type':
            continue
        path = TABLES.get(home)
        if path is None or not path.exists():
            continue  # dataset not cached locally; loader check covers it
        header = pd.read_csv(path, nrows=0).columns
        assert column in header, (home, column)


def test_crosswalk_values_hit_their_target_namespaces(mapper):
    """flywireType reaches both flywire namespaces, hemibrainType is
    hemibrain-only.  Asserting the hit RATES requires those namespaces'
    tables to be cached locally — the module-level skip only guarantees
    male-cns + FAFB."""
    if not (TABLES[BANC].exists() and TABLES[HB].exists()):
        pytest.skip('BANC / hemibrain tables not available locally')

    def values_of(column):
        table = _column_values(TABLES[MCNS], column)
        values = set()
        for cell in table[column].dropna().astype(str):
            values |= _split_cell(cell)
        return values

    flywire_vals = values_of('flywireType')
    hemi_vals = values_of('hemibrainType')

    fafb_types = _namespace_names(mapper, FAFB)
    banc_types = _namespace_names(mapper, BANC)
    hemi_types = _namespace_names(mapper, HB)

    # flywireType reaches BOTH flywire datasets (family licensing)
    assert len(flywire_vals & fafb_types) / len(flywire_vals) >= 0.90
    assert len(flywire_vals & banc_types) / len(flywire_vals) >= 0.90
    # hemibrainType is hemibrain-only (and essentially complete)
    assert len(hemi_vals & hemi_types) / len(hemi_vals) >= 0.99


# ---------------------------------------------------------------------------
# 2. licensing sweep across every directed namespace pair
# ---------------------------------------------------------------------------

PAIRS = [(s, t) for s in (MCNS, FAFB, BANC, HB, MANC)
         for t in (MCNS, FAFB, BANC, HB, MANC) if s != t]


def _sample_types(m, key, total=12):
    primaries = sorted(_namespace_names(m, key))
    if not primaries:
        return []
    # deterministic: alphabetical head + spread tail (long-tail names)
    head = primaries[:8]
    rest = primaries[8:]
    tail = rest[::max(1, len(rest) // 4)][:4] if rest else []
    return (head + tail)[:total]


@pytest.mark.parametrize('source,target', PAIRS)
def test_chains_obey_the_source_map(mapper, source, target):
    source_key = mapper._get_type_mapping_key(source)
    target_key = mapper._get_type_mapping_key(target)
    endpoints = {source_key, target_key}
    checked = 0
    for type_name in _sample_types(mapper, source_key):
        chains = mapper.get_type_bridges(type_name, source, target)
        for chain in chains:
            where = f'{source}->{target} {type_name}'
            # every chain ENDS at a type of the target namespace: the
            # landed value is a primary name there (the final hop's column
            # may be the licensing edge itself — the crosswalk-arrival
            # shape, e.g. [type, hemibrainType])
            end = chain[-1]
            assert end['dataset'] == target_key, (where, chain)
            assert end['value'] in _namespace_names(mapper, target_key), (
                where, chain)
            # every non-type hop is map-licensed (landing namespace)
            for hop in chain[1:]:
                column = hop['column']
                if column == 'type':
                    continue
                licensed = [
                    targets
                    for (home, col), targets in BRIDGE_SOURCE_MAP.items()
                    if col == column and home != '*'
                    and hop['dataset'] in targets
                ]
                assert licensed, (where, hop)
                if column in CROSSWALK_COLUMNS:
                    # endpoint-family licensing (the screenshot rule)
                    assert crosswalk_route_licensed(
                        column, source_key, target_key), (where, hop)
                    assert endpoints & set().union(*licensed), (where, hop)
                assert column not in (
                    'Alternative Cell Type(s)',) or hop['dataset'] == BANC
                assert column != 'additional_type(s)' or hop['dataset'] == FAFB
            # the map-derived validator agrees (belt and braces)
            assert bridge_is_valid(
                chain, source, target,
                key_of=mapper._get_type_mapping_key), (where, chain)
            # the ≤2-direct-linker standard and the ping-pong suppression
            linkers = [l for l in standardize_bridge(
                chain, source, target) if l['kind'] == 'linker']
            assert sum(1 for l in linkers if not l['indirect']) <= 2, (
                where, chain)
            for a, b in zip(linkers, linkers[1:]):
                assert (a['column'], a['value']) != (
                    b['column'], b['value']), (where, chain)
            assert len(chain) <= 6, (where, chain)
            checked += 1


# ---------------------------------------------------------------------------
# 3. known-pair regression matrix
# ---------------------------------------------------------------------------

def test_dn1pa_mcns_banc_has_no_third_family_crosswalk(mapper):
    """The screenshot regression: hemibrainType must never appear on a
    MCNS↔BANC bridge (no hemibrain endpoint); the flywireType-verified
    routes stay."""
    chains = mapper.get_type_bridges('DN1pA', MCNS, BANC)
    assert chains
    for chain in chains:
        for hop in chain[1:]:
            if hop['column'] in CROSSWALK_COLUMNS:
                assert hop['column'] == 'flywireType', (chain,)
    assert any(
        any(h['column'] == 'flywireType' and h['dataset'] == BANC
            for h in chain[1:])
        for chain in chains)


def test_mdn_mcns_banc_routes_only_through_flywire(mapper):
    chains = mapper.get_type_bridges('MDN', MCNS, BANC)
    assert chains
    for chain in chains:
        assert all(h['column'] != 'hemibrainType' for h in chain[1:])
        assert all(h['column'] != 'mancType' for h in chain[1:])


def test_dnp50_banc_annotation_routing_intact(mapper):
    """DNp50 → BANC MDN routes through the FAFB additional_type(s)
    evidence and the BANC Alternative Cell Type(s) cells — both keep
    working under the source map."""
    chains = mapper.get_type_bridges('DNp50', MCNS, BANC)
    ends = {chain[-1]['value'] for chain in chains}
    assert 'MDN' in ends
    assert any(
        any(h['column'] == 'additional_type(s)' and h['dataset'] == FAFB
            for h in chain[1:])
        for chain in chains)
    assert any(
        any(h['column'] == 'Alternative Cell Type(s)'
            for h in chain[1:])
        for chain in chains)


def test_cl125_apdn3_standard_chain_survives(mapper):
    """The MCNS→FAFB CL125→APDN3 rename keeps its flywireType 'LMTe01'
    verification under the source map."""
    chains = mapper.get_type_bridges('CL125', MCNS, FAFB)
    assert chains
    end_apdn3 = [c for c in chains if c[-1]['value'] == 'APDN3']
    assert end_apdn3
    assert any(
        any(h['column'] == 'flywireType' and h['value'] == 'LMTe01'
            for h in chain[1:])
        for chain in end_apdn3)


def test_hemibrain_fafb_sanctioned_route_through_mcns(mapper):
    """hemibrain↔flywire mapping goes through male-cns: hemibrain type →
    male-cns type → flywireType → flywire type."""
    chains = mapper.get_type_bridges('DN1pA', HB, FAFB)
    assert chains
    assert any(
        any(h['column'] == 'flywireType' for h in chain[1:])
        for chain in chains)
    for chain in chains:  # no mancType noise on a hemibrain↔flywire pair
        assert all(h['column'] != 'mancType' for h in chain[1:])


def test_mctype_licensing_manc_only(mapper):
    """mancType hops exist only on bridges with a manc endpoint."""
    for type_name in _sample_types(mapper, MCNS):
        for target in (FAFB, BANC, HB):
            for chain in mapper.get_type_bridges(type_name, MCNS, target):
                assert all(h['column'] != 'mancType' for h in chain[1:])


def test_fifth_lnv_no_pingpong_chains(mapper):
    """The 5th-LNv self-loop class (consecutive identical
    additional_type(s) linkers) stays pruned on real data."""
    for chain in mapper.get_type_bridges('5th-LNv', MCNS, FAFB):
        linkers = [l for l in standardize_bridge(
            chain, MCNS, FAFB) if l['kind'] == 'linker']
        for a, b in zip(linkers, linkers[1:]):
            assert (a['column'], a['value']) != (b['column'], b['value']), (
                chain)


def test_annotation_columns_land_in_their_own_namespace(mapper):
    """additional_type(s) only maps to FAFB; Alternative Cell Type(s)
    only works for the BANC — across a broad sweep."""
    for type_name in _sample_types(mapper, MCNS, total=10):
        for target in (FAFB, BANC):
            for chain in mapper.get_type_bridges(type_name, MCNS, target):
                for hop in chain[1:]:
                    if hop['column'] == 'additional_type(s)':
                        assert hop['dataset'] == FAFB, (type_name, chain)
                    if hop['column'] == 'Alternative Cell Type(s)':
                        assert hop['dataset'] == BANC, (type_name, chain)
