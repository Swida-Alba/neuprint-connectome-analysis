"""UI tests for the Round 2 Type Mapping entrance (spec §3/§10).

Drives the real entrance button + dialog for male-cns v1.0 + FAFB v783
with NiceGUI's test client: the button is enabled when both datasets have
cached indexes, the global search composes the mapping across the
selection, and the composed view + per-pair cards + CSV actions appear.
Needs the local cached indexes; skipped when they are absent.
"""

from pathlib import Path

import pytest

import ui.history_store as hs
import ui.type_mapping_history as tmh
from ui.neuron_index import clear_neuron_index_cache

REPO_ROOT = Path(__file__).resolve().parents[2]
MCNS_INDEX = REPO_ROOT / 'neuron_indexes' / 'male-cns_v1_0' / 'neuron_index.parquet'
FAFB_INDEX = REPO_ROOT / 'neuron_indexes' / 'flywire_FAFB_v783' / 'neuron_index.parquet'
MCNS = 'male-cns:v1.0'
FAFB = 'flywire_FAFB_v783'

pytestmark = pytest.mark.skipif(
    not (MCNS_INDEX.exists() and FAFB_INDEX.exists()),
    reason='cached male-cns v1.0 / FAFB v783 neuron indexes not available locally',
)


@pytest.fixture
def panel_client(tmp_path, monkeypatch):
    from nicegui import Client, ui
    from nicegui.page import page

    # the panel's query box keeps its own history store; isolate both it
    # and the shared neuron history so tests never touch the real files
    monkeypatch.setattr(tmh, "_HISTORY_PATH",
                        tmp_path / "type_mapping_history.json")
    monkeypatch.setattr(hs, "_HISTORY_PATH", tmp_path / "neuron_history.json")

    clear_neuron_index_cache()
    # Warm the type mapper up front: its first load takes a minute and
    # would otherwise race the panel assertions below.
    from comparison.cross_dataset_type_mapper import get_type_mapper

    assert get_type_mapper().load() is True

    client = Client(page('/type-mapping-panel-test'))
    with client:
        selection = {'value': [MCNS, FAFB]}
        from ui.components.type_mapping_panel import create_type_mapping_entry
        button = create_type_mapping_entry(lambda: list(selection['value']))
    try:
        yield client, button, selection
    finally:
        clear_neuron_index_cache()


def _buttons(client):
    return [e for e in client.elements.values()
            if type(e).__name__ == 'Button']


def _invoke(handler, client, element):
    """Run a click handler, driving any returned coroutine to completion.

    The loading-notice search handler is async (run.io_bound off the
    event loop) — the results only exist after it finishes, so the
    assertions below need the coroutine fully awaited.
    """
    import asyncio
    import inspect

    args = {'sender': element.id, 'client': client, 'args': None}
    # NiceGUI wraps the user handler (lambda e: handle_event(user, e));
    # unwrap it so the click actually EXECUTES here instead of being
    # deferred without a running app loop.
    target = handler
    for cell in (getattr(handler, '__closure__', None) or ()): 
        candidate = cell.cell_contents
        if (callable(candidate) and candidate is not handler
                and 'nicegui' not in getattr(
                    candidate, '__module__', 'nicegui')):
            target = candidate
            break
    try:
        result = target(args)
    except TypeError:
        result = target()
    if inspect.isawaitable(result):
        # drive the coroutine inside the sender's parent slot: NiceGUI
        # resolves the client for ui.notify/UI creation from the current
        # TASK's slot stack, which a fresh loop's task does not carry
        slot = getattr(element, 'parent_slot', None)

        async def _drive():
            if slot is not None:
                with slot:
                    await result
            else:
                await result

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_drive())
        finally:
            loop.close()


def _click_button(client, text_part: str) -> bool:
    for element in client.elements.values():
        if type(element).__name__ != 'Button':
            continue
        if text_part not in str(getattr(element, 'text', '')):
            continue
        for listener in element._event_listeners.values():
            if listener.type == 'click' and listener.handler:
                _invoke(listener.handler, client, element)
                return True
    return False


def _labels(client):
    return [
        str(element.text)
        for element in client.elements.values()
        if isinstance(getattr(element, 'text', None), str)
        and getattr(element, 'text', '')
    ]


def test_button_enabled_with_two_cached_datasets(panel_client):
    client, button, selection = panel_client
    mapping_buttons = [b for b in _buttons(client)
                       if 'Type Mapping' in str(getattr(b, 'text', ''))]
    assert mapping_buttons, 'Type Mapping entrance button not found'
    assert mapping_buttons[0].enabled


def test_button_disabled_below_two_datasets():
    """No cached-index requirement met: the entrance stays disabled."""
    from nicegui import Client, ui
    from nicegui.page import page

    clear_neuron_index_cache()
    client = Client(page('/type-mapping-panel-test-empty'))
    with client:
        from ui.components.type_mapping_panel import create_type_mapping_entry
        create_type_mapping_entry(lambda: [])
    buttons = [b for b in _buttons(client)
               if 'Type Mapping' in str(getattr(b, 'text', ''))]
    assert buttons
    assert not buttons[0].enabled


def test_global_search_composes_the_selection(panel_client):
    client, button, _selection = panel_client
    search = button.search_container
    # the standard filter-mode control rides on the chip input (§12)
    assert getattr(search, "filter_mode", None) is not None
    search.add_values(['APDN3'])
    assert _click_button(client, 'Search mappings')
    labels = _labels(client)
    # the composed view + per-pair card for the selection appear; the
    # exact chip resolves in FAFB, so the origin-seeded pair runs
    # FAFB -> male-cns (§12 direction rule: the query lives where it
    # matched)
    assert any('Composed view (HTML)' in label for label in labels) or \
        any('Composed view' in b_text for b_text in
            [str(getattr(b, 'text', '')) for b in _buttons(client)])
    assert any('flywire_FAFB_v783 → male-cns:v1.0' in label
               for label in labels)
    # bidirectional type-coverage presentation (user 2026-09-07): a
    # TOP-LEVEL panel per dataset pair, with the pair named in its title
    assert any(
        'Type coverage — flywire_FAFB_v783 → male-cns:v1.0' in label
        and 'bidirectional' in label
        for label in labels)
    # 2026-09-09: the second view is "Backward" (not "Reverse"), and both
    # views name their coverage columns by DATASET so nothing reads as
    # flipped
    assert any(label.startswith('Backward —') for label in labels)
    assert not any(label.startswith('Reverse —') for label in labels)
    coverage_tables = [
        e for e in client.elements.values()
        if type(e).__name__ == 'Table'
        and any('side (bodyIds)' in c.get('label', '')
                for c in e._props.get('columns', []))]
    assert coverage_tables, 'dataset-named coverage columns missing'
    for table in coverage_tables:
        column_labels = {c['label'] for c in table._props['columns']}
        assert 'flywire_FAFB_v783 side (bodyIds)' in column_labels
        assert 'male-cns:v1.0 side (bodyIds)' in column_labels
    # artifact + CSV actions of the per-pair card are present
    for action in ('Sankey (type-level)', 'Sankey (linker)',
                   'Network (type-level)', 'Network (linker)',
                   'Export mapping'):
        assert any(action in str(getattr(b, 'text', ''))
                   for b in _buttons(client)), action
    # the confirmed search is recorded in the panel's OWN history store —
    # never in the shared neuron-query history of the analysis tabs
    assert tmh.recent() == ['APDN3']
    assert tmh.datasets_of('APDN3') == sorted([MCNS, FAFB])
    assert hs.recent() == []


def test_failed_search_never_records_history(panel_client):
    """A zero-hit query matches no type, so nothing lands in the store."""
    client, button, _selection = panel_client
    search = button.search_container
    search.add_values(['zzz_no_such_type_zzz'])
    assert _click_button(client, 'Search mappings')
    assert tmh.recent() == []
    assert hs.recent() == []


def test_history_rows_show_dataset_and_column_hints(panel_client):
    """The panel's Recent list annotates rows like its suggestions do."""
    client, button, _selection = panel_client
    search = button.search_container
    search.add_values(['APDN3'])
    assert _click_button(client, 'Search mappings')

    # refocus the empty editor: the panel's own Recent list carries the
    # confirmed query with the suggestion-style gray hint
    focus = next(
        listener for listener in search.chip_input._event_listeners.values()
        if listener.type == 'focus'
    )
    search.chip_input._handle_event({'listener_id': focus.id, 'args': None})
    labels = _labels(client)
    assert 'APDN3' in labels
    # APDN3 lives in FAFB's type column only, so the hint is exactly the
    # matched column · dataset pair its suggestion row would show
    assert 'type · flywire_FAFB_v783' in labels


def test_empty_search_produces_no_results(panel_client):
    client, button, _selection = panel_client
    assert _click_button(client, 'Search mappings')
    # no composed view and no per-pair cards without a search
    assert not any('Composed view (HTML)' in str(getattr(b, 'text', ''))
                   for b in _buttons(client))
    assert not any('mapped pairs' in label for label in _labels(client))
