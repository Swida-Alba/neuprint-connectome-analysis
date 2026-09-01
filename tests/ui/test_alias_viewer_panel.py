"""UI-level tests for the viewer's expanded cross-dataset search panel.

Drives the real neuron-index viewer for male-cns v1.0 with NiceGUI's test
client: a zero-hit query (APDN3) must reveal the alias panel with explicit
notice, group members, counts, and the local search action, while a query
with hits must keep it hidden.  Needs the local cached indexes; skipped
when they are absent.
"""

from pathlib import Path

import pytest

from ui.components import neuron_index_viewer as viewer_mod
from ui.neuron_index import clear_neuron_index_cache

REPO_ROOT = Path(__file__).resolve().parents[2]
MCNS_INDEX = REPO_ROOT / 'neuron_indexes' / 'male-cns_v1_0' / 'neuron_index.parquet'

pytestmark = pytest.mark.skipif(
    not MCNS_INDEX.exists(),
    reason='cached male-cns v1.0 neuron index not available locally',
)


def _alias_section(client):
    for element in client.elements.values():
        if 'drocat-neuron-alias-panel' in str(getattr(element, '_classes', [])):
            return element
    return None


def _labels(client):
    return [
        str(element.text)
        for element in client.elements.values()
        if isinstance(getattr(element, 'text', None), str)
        and getattr(element, 'text', '')
    ]


def _wait_for_labels(client, predicate, timeout_seconds: float = 40.0):
    """Poll the rendered labels until predicate(texts) holds.

    The first alias expansion loads the type mapper in the background
    (seconds), so the panel update can land after set_value returns.
    """
    import time as _time

    deadline = _time.monotonic() + timeout_seconds
    texts = _labels(client)
    while not predicate(texts) and _time.monotonic() < deadline:
        _time.sleep(0.25)
        texts = _labels(client)
    return texts


@pytest.fixture
def viewer_client():
    from nicegui import Client, ui
    from nicegui.page import page

    clear_neuron_index_cache()
    # Warm the type mapper up front: its first load takes a minute and
    # would otherwise race the panel assertions below.
    from comparison.cross_dataset_type_mapper import get_type_mapper

    assert get_type_mapper().load() is True

    client = Client(page('/alias-panel-test'))
    with client:
        content = ui.element('div')
        viewer_mod._render_index(content, 'male-cns:v1.0')
    inputs = [e for e in client.elements.values() if isinstance(e, ui.input)]
    assert inputs, 'viewer search input not found'
    try:
        yield client, inputs[0]
    finally:
        clear_neuron_index_cache()


def test_zero_hit_query_reveals_alias_panel(viewer_client):
    client, search = viewer_client
    search.set_value('APDN3')  # no male-cns neuron carries this name

    texts = _wait_for_labels(
        client,
        lambda t: any('Cross-dataset matches — informational only' in x
                      for x in t),
    )
    assert any('Cross-dataset matches — informational only' in t for t in texts)
    assert any('please double check' in t for t in texts)

    # mapped tier: local group members offered as same-dataset search actions
    assert any("'SLP249' — one of N (4 neurons)" in t for t in texts)
    assert any("Search 'SLP249' here" in t for t in texts)
    # native tier: FAFB entry carries the aggregation annotation and count
    assert any(
        "'APDN3' (12 neurons) — here: one of CL125, PLP080, SLP249, SLP250"
        in t
        for t in texts
    )
    # native tier header states the name-similar caution
    assert any('name-similar — not necessarily the same type' in t
               for t in texts)
    # checked-but-unknown datasets are stated explicitly
    assert any(t.startswith('No known counterpart in:') for t in texts)

    section = _alias_section(client)
    assert section is not None
    assert 'hidden' not in section.classes


def test_hit_query_keeps_alias_panel_hidden(viewer_client):
    client, search = viewer_client
    search.set_value('aMe12')  # male-cns has plenty of aMe12 rows

    texts = _labels(client)
    assert not any('Cross-dataset matches — informational only' in t
                   for t in texts)
    section = _alias_section(client)
    assert section is not None
    assert 'hidden' in section.classes


def test_query_changes_refresh_the_panel(viewer_client):
    """Changing the query re-evaluates the expansion every time and returns
    to the normal (hit-based) workflow whenever the new query finds rows."""
    client, search = viewer_client

    # 1) zero-hit FAFB-native name -> panel with the local group member
    search.set_value('APDN3')
    texts = _wait_for_labels(
        client, lambda t: any("'SLP249' — one of N (4 neurons)" in x for x in t))
    assert any("'SLP249' — one of N (4 neurons)" in t for t in texts)

    # 2) a query with hits -> panel hidden, normal workflow
    search.set_value('aMe12')
    texts = _wait_for_labels(
        client,
        lambda t: any('aMe12' in x for x in t)
        and not any('Cross-dataset matches' in x for x in t),
    )
    section = _alias_section(client)
    assert 'hidden' in section.classes

    # 3) a different zero-hit name -> a fresh panel with its own alias
    search.set_value('DNp50')  # FAFB primary; male-cns calls it MDN
    texts = _wait_for_labels(
        client, lambda t: any("'MDN' — renamed (4 neurons)" in x for x in t))
    assert any("Cross-dataset matches — informational only" in t for t in texts)
    assert any("'MDN' — renamed (4 neurons)" in t for t in texts)

    # 4) back to a query with hits -> hidden again
    search.set_value('MDN')  # male-cns native, 4 neurons
    texts = _wait_for_labels(
        client,
        lambda t: any('MDN' in x for x in t)
        and not any('Cross-dataset matches' in x for x in t),
    )
    assert 'hidden' in _alias_section(client).classes
