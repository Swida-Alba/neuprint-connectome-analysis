"""Real-data tests for the viewer's expanded cross-dataset alias search.

``collect_alias_matches`` powers the "See available neurons" viewer's
zero-hit panel: the search name is expanded through the auto type mapping
and every alias is counted in the other datasets' *cached* neuron indexes.
These tests need the local indexes (male-cns v1.0, FAFB v783, BANC v626 at
minimum) and are skipped when they are absent.
"""

from pathlib import Path

import pytest

from ui.neuron_index import collect_alias_matches

REPO_ROOT = Path(__file__).resolve().parents[2]

MCNS = 'male-cns:v1.0'
FW = 'flywire_FAFB_v783'
BANC = 'flywire_BANC_v626'

REQUIRED_INDEXES = [
    REPO_ROOT / 'neuron_indexes' / dataset_to_folder / 'neuron_index.parquet'
    for dataset_to_folder in (
        'male-cns_v1_0', 'flywire_FAFB_v783', 'flywire_BANC_v626',
    )
]

pytestmark = pytest.mark.skipif(
    not all(p.exists() for p in REQUIRED_INDEXES),
    reason='cached neuron indexes (male-cns v1.0 / FAFB v783 / BANC v626) '
           'not available locally',
)


def _entry(matches, dataset):
    for entry in matches:
        if entry['dataset'] == dataset:
            return entry
    return None


def _candidate(entry, name):
    for cand in entry['candidates']:
        if cand['name'] == name:
            return cand
    return None


def test_search_slp249_in_fafb_finds_rename_with_count():
    # The user-facing case: SLP249 does not exist in FAFB v783 (renamed to
    # APDN3); the expanded search must surface the alias with its count.
    matches = collect_alias_matches(FW, 'SLP249')
    fafb = _entry(matches, FW)
    assert fafb is not None and fafb['is_selected'] is True
    renamed = _candidate(fafb, 'APDN3')
    assert renamed is not None
    assert renamed['kind'] == 'renamed'
    # FAFB counts every APDN3 neuron (12), of which 4 carry the SLP249
    # additional-type alias.
    assert renamed['count'] == 12
    assert renamed['aggregates'] == ['CL125', 'PLP080', 'SLP249', 'SLP250']

    # the local alias still resolves in male-cns v1.0
    mcns = _entry(matches, MCNS)
    assert mcns is not None
    local = _candidate(mcns, 'SLP249')
    assert local is not None and local['kind'] == 'same name'
    assert local['count'] == 4


def test_search_apdn3_in_malecns_offers_group_members():
    # APDN3 aggregates four male-cns types; the selected dataset's entry
    # must offer each member as a searchable group (never merge silently).
    matches = collect_alias_matches(MCNS, 'APDN3')
    mcns = _entry(matches, MCNS)
    assert mcns is not None and mcns['is_selected'] is True
    counts = {c['name']: c['count'] for c in mcns['candidates']}
    assert all(c['kind'] == 'one of N' for c in mcns['candidates'])
    assert counts == {'SLP249': 4, 'CL125': 4, 'SLP250': 2, 'PLP080': 2}

    # both FlyWire datasets know the name natively
    fafb = _entry(matches, FW)
    native = _candidate(fafb, 'APDN3')
    assert native['kind'] == 'same name'
    assert native['count'] == 12
    banc = _entry(matches, BANC)
    assert _candidate(banc, 'APDN3')['count'] == 8


def test_search_mdn_across_namespaces():
    # FAFB renamed MDN -> DNp50; BANC still uses MDN natively.
    matches = collect_alias_matches(MCNS, 'MDN')
    fafb = _entry(matches, FW)
    renamed = _candidate(fafb, 'DNp50')
    assert renamed['kind'] == 'renamed'
    assert renamed['count'] == 4
    banc = _entry(matches, BANC)
    same = _candidate(banc, 'MDN')
    assert same['kind'] == 'same name'
    assert same['count'] == 4


def test_selected_dataset_entry_is_sorted_first():
    matches = collect_alias_matches(MCNS, 'APDN3')
    assert matches[0]['dataset'] == MCNS
    assert matches[0]['is_selected'] is True


def test_guards_return_no_matches():
    # numeric (bodyId-style), single char, pattern, empty: the expansion
    # never fires for non-plain-name queries.
    for query in ('123', '7', 'a', 'SLP*', ''):
        assert collect_alias_matches(MCNS, query) == []


# ---------------------------------------------------------------------------
# Comma-joined additional_type(s) cells
# ---------------------------------------------------------------------------

def test_suggestion_pools_split_combined_additional_type_cells():
    from ui.type_suggestions import (
        dataset_aware_suggestions,
        get_dataset_pools,
    )

    pools = get_dataset_pools(FW)
    additional = [v for v, _ in pools['additional_type(s)']]
    # every part of 'vDeltaB, vDeltaC, ...' is offered individually ...
    assert 'vDeltaB' in additional
    assert 'vDeltaG' in additional
    # ... and no combined cell survives as a suggestion value.
    assert not any(',' in v for v in additional)

    suggestions = dataset_aware_suggestions('vDeltaB', [FW], 'auto')
    values = [str(e[0]) for e in suggestions]
    assert values and all(',' not in v for v in values)
    assert 'vDeltaB' in values


def test_viewer_match_groups_split_combined_cells():
    from ui.neuron_index import load_cached_neuron_index, query_neuron_index

    index = load_cached_neuron_index(FW)
    page = query_neuron_index(index, search='vDeltaB', page=1, page_size=10)
    # the neurons are still all found (substring over the joined cells)
    assert page.total == 315
    group_values = [str(g.get('match_value')) for g in page.match_groups]
    assert 'vDeltaB' in group_values
    # a matched value is always one real name, never a joined cell
    assert all(',' not in v for v in group_values)


# ---------------------------------------------------------------------------
# Native (mapper-free) type-name expansion
# ---------------------------------------------------------------------------

def test_native_expansion_finds_dn3_relatives_in_other_datasets():
    # DN3 does not exist anywhere as a standalone type; the expansion finds
    # the name-similar types natively — no auto type mapping involved.
    matches = collect_alias_matches  # noqa: F841  (sibling API sanity)
    from ui.neuron_index import collect_native_type_matches

    native = collect_native_type_matches(MCNS, 'DN3')
    by_ds = {e['dataset']: e for e in native}

    fafb = by_ds[FW]
    fafb_types = {c['name']: c['count'] for c in fafb['types']}
    assert fafb_types['APDN3'] == 12
    assert fafb_types['s-CPDN3A'] == 38
    assert fafb_types['l-CPDN3'] == 2
    assert not any(c['exact'] for c in fafb['types'])
    # sorted by neuron count (descending) within the non-exact tier
    counts = [c['count'] for c in fafb['types']]
    assert counts == sorted(counts, reverse=True)

    banc = by_ds[BANC]
    assert {c['name']: c['count'] for c in banc['types']}['APDN3'] == 8

    # datasets with nothing related are reported as absent (not in the list)
    assert MCNS not in by_ds          # selected dataset is not scanned
    assert 'hemibrain:v1.2.1' not in by_ds


def test_native_expansion_enriches_with_mapped_current_dataset_names():
    from ui.neuron_index import collect_native_type_matches, enrich_native_type_matches

    native = collect_native_type_matches(MCNS, 'DN3')
    enrich_native_type_matches(native, MCNS)

    by_ds = {e['dataset']: e for e in native}
    apdn3 = next(c for c in by_ds[FW]['types'] if c['name'] == 'APDN3')
    assert apdn3['mapped']['kind'] == 'one of N'
    assert apdn3['mapped']['targets'] == ['CL125', 'PLP080', 'SLP249', 'SLP250']
    # l-CPDN3 maps uniquely to male-cns aMe13
    l_cpdn3 = next(c for c in by_ds[FW]['types'] if c['name'] == 'l-CPDN3')
    assert l_cpdn3['mapped'] == {'kind': 'renamed', 'targets': ['aMe13']}


def test_native_expansion_taxonomy_labels_map_covered_types():
    from ui.neuron_index import collect_native_type_matches, enrich_native_type_matches

    native = collect_native_type_matches(MCNS, 'circadian')
    enrich_native_type_matches(native, MCNS)
    by_ds = {e['dataset']: e for e in native}

    fafb_labels = {l['label']: l for l in by_ds[FW]['labels']}
    clock = fafb_labels['circadian_clock']
    assert clock['column'] == 'cell_type'
    covered = {t['name']: t for t in clock['types']}
    s_cpdn3a = covered['s-CPDN3A']
    assert s_cpdn3a['mapped']['kind'] == 'one of N'
    # taxonomy labels themselves get no annotation; their covered types do
    assert all(not l.get('mapped') for l in by_ds[FW]['labels'])

    banc_labels = {l['label']: l for l in by_ds[BANC]['labels']}
    neuron = banc_labels['circadian_neuron']
    assert neuron['column'] == 'Class'
    # covered types carry their mapping relation when one exists ...
    kinds = {t.get('mapped', {}).get('kind') for t in neuron['types']}
    assert {'renamed', 'same name', 'one of N'} <= kinds
    # ... and BANC v888 covers an unmapped type, still shown so the user is
    # led to inspect it in the other dataset.
    v888 = {e['dataset']: e for e in native}['flywire_BANC_v888']
    v888_labels = {l['label']: l for l in v888['labels']}
    assert any(
        t.get('mapped') is None for t in v888_labels['circadian_neuron']['types']
    )


def test_zero_hit_matches_merges_native_and_mapped_tiers():
    from ui.neuron_index import collect_zero_hit_matches

    result = collect_zero_hit_matches(MCNS, 'DN3')
    assert set(result.keys()) == {'native', 'mapped'}
    assert any(e['types'] for e in result['native'])
    # the mapper knows nothing about DN3: the mapped tier stays empty while
    # the native tier still delivers.
    assert not any(
        e['outcome'] == 'matched' and e['candidates'] for e in result['mapped']
    )


def test_native_expansion_is_mapper_free():
    """The native tier is a pure neuron-index search: its code path must not
    reference the auto type mapping at all."""
    import inspect

    import ui.neuron_index as ni

    for func in (ni.collect_native_type_matches, ni._native_type_matches,
                 ni._native_label_matches, ni._native_label_columns):
        source = inspect.getsource(func)
        # strip the docstring so prose mentions don't count as code usage
        if source.count('"""') >= 2:
            source = '"""'.join(source.split('"""')[2:])
        assert 'mapper' not in source.lower(), func.__name__
        assert 'comparison' not in source, func.__name__
