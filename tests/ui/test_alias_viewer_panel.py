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


def _click_button(client, text_part: str) -> bool:
    """Invoke the click handler of the first button whose text matches."""
    for element in client.elements.values():
        if type(element).__name__ != 'Button':
            continue
        if text_part not in str(getattr(element, 'text', '')):
            continue
        for listener in element._event_listeners.values():
            if listener.type == 'click' and listener.handler:
                try:
                    listener.handler({
                        'sender': element.id, 'client': client, 'args': None,
                    })
                except TypeError:
                    listener.handler()
                return True
    return False


def _table(client):
    """The main neuron table (identified by its bodyId column)."""
    for element in client.elements.values():
        if type(element).__name__ != 'Table':
            continue
        if any(c.get('name') == 'bodyId' for c in getattr(element, 'columns', [])):
            return element
    return None


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


def test_mapped_type_view_button_warning_and_exit(viewer_client):
    """The panel button runs the equivalent search for the mapped types in
    the current dataset: warning banner, provenance columns, current-dataset
    rows only, and a clean way back to the normal workflow."""
    client, search = viewer_client
    search.set_value('circadian')  # zero local hits; taxonomy labels match

    texts = _wait_for_labels(
        client, lambda t: any('Show mapped types here' in x for x in t))
    assert any('Show mapped types here' in t for t in texts)

    # each button follows its own dataset badge (block-local, not detached)
    ordered = []
    for element in client.elements.values():
        text = str(getattr(element, 'text', '') or '')
        if type(element).__name__ == 'Button' and 'Show mapped types here' in text:
            ordered.append('BUTTON ' + text)
        elif type(element).__name__ == 'Badge' and any(
                ds in text for ds in ('flywire_', 'banc_', 'manc', 'hemibrain')):
            ordered.append('BADGE ' + text)
    assert len(ordered) == 6, ordered
    for i in range(3):
        assert ordered[2 * i].startswith('BADGE '), ordered
        assert ordered[2 * i + 1].startswith('BUTTON '), ordered
    assert 'flywire_FAFB_v783' in ordered[0]
    assert 'banc_v888' in ordered[2]
    assert 'banc_v626' in ordered[4]

    # enter the mapped view from the first dataset block (FAFB: it is the
    # first cached dataset with native matches in the scan order)
    assert _click_button(client, 'Show mapped types here')

    # warning banner with the foreign dataset and the double-check advice
    texts = _labels(client)
    assert any('Mapped-type view' in t and 'FAFB' in t
               for t in texts)
    assert any('please double check' in t for t in texts)
    assert any('Selection adds these neurons to the query' in t for t in texts)

    # provenance columns are appended to the main table: the two floating
    # columns only (Map source / Mapped entry moved into the cell hover)
    table = _table(client)
    column_names = [c['name'] for c in table.columns]
    assert '__map_foreign' in column_names
    assert '__map_origin' in column_names
    assert '__map_source' not in column_names
    assert '__map_entry' not in column_names

    # rows are current-dataset neurons only: no FlyWire root ids may leak in
    body_ids = [str(r.get('bodyId', '')) for r in table.rows]
    assert body_ids and all(
        b and not b.startswith('7205759') for b in body_ids
    )
    assert any('Mapped view' in t for t in texts)

    # explicit exit returns to the normal workflow (search is still the
    # zero-hit 'circadian', so the expansion panel re-appears)
    assert _click_button(client, 'Back to normal search')
    texts = _wait_for_labels(
        client, lambda t: any('Show mapped types here' in x for x in t))
    column_names = [c['name'] for c in _table(client).columns]
    assert not any(c.startswith('__map_') or c.startswith('__match_') for c in column_names)
    section = _alias_section(client)
    assert 'hidden' not in section.classes


def test_search_change_exits_mapped_view(viewer_client):
    client, search = viewer_client
    search.set_value('circadian')
    _wait_for_labels(
        client, lambda t: any('Show mapped types here' in x for x in t))
    assert _click_button(client, 'Show mapped types here')
    section = _alias_section(client)
    assert 'hidden' in section.classes  # mapped view active

    # a new query leaves the mapped view entirely
    search.set_value('aMe12')
    texts = _wait_for_labels(
        client,
        lambda t: any('aMe12' in x for x in t)
        and not any('Mapped-type view' in x for x in t),
    )
    column_names = [c['name'] for c in _table(client).columns]
    assert not any(c.startswith('__map_') or c.startswith('__match_') for c in column_names)
    assert 'hidden' in _alias_section(client).classes


def test_mapped_view_pinned_columns_and_bridge_hover(viewer_client):
    """The two floating columns are the table's last columns pinned to the
    RIGHT edge with matching header geometry, and their cells hover-name
    the bridge derivation."""
    import re

    client, search = viewer_client
    search.set_value('circadian')
    _wait_for_labels(
        client, lambda t: any('Show mapped types here' in x for x in t))
    assert _click_button(client, 'Show mapped types here')

    table = _table(client)
    column_names = [str(c.get('name', '')) for c in table.columns]
    # the floating provenance columns are the table's LAST columns
    assert column_names[-2:] == ['__map_foreign', '__map_origin']
    assert '__map_source' not in column_names
    assert '__map_entry' not in column_names

    map_columns = [c for c in table.columns
                   if str(c.get('name', '')).startswith('__map_')]
    foreign_style = map_columns[0]['style']
    origin_style = map_columns[1]['style']
    origin_right = int(re.search(r'right:(\d+)px', origin_style).group(1))
    foreign_right = int(re.search(r'right:(\d+)px', foreign_style).group(1))
    origin_w = int(re.search(r'width:(\d+)px', origin_style).group(1))
    # Matched column pins to the right edge; Foreign type(s) stacks left
    # of it, offset by exactly its neighbour's width
    assert origin_right == 0
    assert foreign_right == origin_w
    # body cells opaque, headers carry geometry but no inline background
    # (the distinct CSS head tint wins)
    assert 'background' in foreign_style and 'background' in origin_style
    for column in map_columns:
        header_style = column['headerStyle']
        assert 'position:sticky' in header_style
        assert 'z-index:9' in header_style
        assert 'background' not in header_style
    assert re.search(
        r'right:(\d+)px', map_columns[0]['headerStyle']
    ).group(1) == str(foreign_right)
    assert re.search(
        r'right:(\d+)px', map_columns[1]['headerStyle']
    ).group(1) == '0'

    # every stamped row hover-names its bridge derivation (one line per
    # mapped entry)
    stamped = [r for r in table.rows if r.get('__map_foreign')]
    assert stamped
    assert all(str(r.get('__map_bridge', '')).strip() for r in stamped)


def test_mapping_visualization_icons(viewer_client):
    """Sankey and Network buttons are visually distinguishable icons."""
    client, search = viewer_client
    search.set_value('circadian')
    _wait_for_labels(
        client, lambda t: any('Show mapped types here' in x for x in t))

    icons = {}
    for element in client.elements.values():
        if type(element).__name__ != 'Button':
            continue
        text = str(getattr(element, 'text', '') or '')
        if text in ('Sankey', 'Network'):
            icons[text] = getattr(element, '_props', {}).get('icon', '')
    assert icons.get('Sankey') == 'multiple_stop'
    assert icons.get('Network') == 'account_tree'


def test_mapped_view_survives_display_controls(viewer_client):
    """Sort / Order / Rows are display controls: they must not exit the
    mapped-type view (only query changes do)."""
    client, search = viewer_client
    search.set_value('circadian')
    _wait_for_labels(
        client, lambda t: any('Show mapped types here' in x for x in t))
    assert _click_button(client, 'Show mapped types here')

    def _mapped_active():
        table = _table(client)
        names = [str(c.get('name', '')) for c in table.columns]
        return any(n.startswith('__map_') for n in names)

    assert _mapped_active()
    # change ONLY the display controls (Sort by / Order / Rows) — query
    # controls (search, target column, match mode) legitimately exit
    display_labels = {'Sort by', 'Order', 'Rows'}
    changed = []
    for element in client.elements.values():
        if type(element).__name__ != 'Select':
            continue
        label = (getattr(element, '_props', {}) or {}).get('label', '')
        if label not in display_labels:
            continue
        options = list(getattr(element, 'options', {}) or {})
        alternative = next((o for o in options if o != element.value), None)
        if alternative is None:
            continue
        element.set_value(alternative)
        changed.append(label)
    assert set(changed) == display_labels, changed
    # the mapped view survived the display changes
    assert _mapped_active()
    texts = _labels(client)
    assert any('Mapped-type view' in t for t in texts)


def test_mapping_visualization_variants_download_not_saved(
        viewer_client, tmp_path, monkeypatch):
    """All four mapping artifacts (sankey/network x type-level/linker) are
    delivered as browser downloads with the persistent banner, and NOTHING
    is written under PROJECT_ROOT/outputs (§9B.4, §9C.6)."""
    import csv as csv_mod
    import io as io_mod

    from nicegui import ui as nicegui_ui

    client, search = viewer_client
    monkeypatch.setattr(viewer_mod, 'PROJECT_ROOT', tmp_path)
    downloads: list = []

    def _record(content, filename=None, media_type=''):
        downloads.append((str(content), str(filename or '')))

    monkeypatch.setattr(nicegui_ui.download, 'content', _record)

    search.set_value('circadian')
    _wait_for_labels(
        client, lambda t: any('Show mapped types here' in x for x in t))
    assert _click_button(client, 'Show mapped types here')

    # every 'Type-level' / 'Linker view' menu item belongs to a mapping
    # artifact button (mapped banner + expansion blocks); invoking them
    # covers all four (kind, variant) combinations.  MenuItem keeps its
    # label in a child section, so match via descendant label text.
    def _invoke_menu_items(text_exact):
        count = 0
        for element in client.elements.values():
            if type(element).__name__ != 'MenuItem':
                continue
            if not any(str(getattr(d, 'text', '')) == text_exact
                       for d in element.descendants()):
                continue
            for listener in element._event_listeners.values():
                if listener.type == 'click' and listener.handler:
                    try:
                        listener.handler({
                            'sender': element.id, 'client': client,
                            'args': None,
                        })
                    except TypeError:
                        listener.handler()
                    count += 1
                    break
        return count

    assert _invoke_menu_items('Type-level') >= 2
    assert _invoke_menu_items('Linker view') >= 2

    names = [name for _content, name in downloads if name.endswith('.html')]
    for artifact in ('mapping_sankey_type_', 'mapping_sankey_linker_',
                     'mapping_network_type_', 'mapping_network_linker_'):
        assert any(name.startswith(artifact) for name in names), names
    for content, name in downloads:
        if not name.endswith('.html'):
            continue
        if '_sankey_' in name:
            assert 'plotly' in content.lower(), name
        else:
            assert 'cytoscape' in content.lower(), name
    # nothing lands in the repository outputs dir
    out_dir = tmp_path / 'outputs'
    assert not out_dir.exists() or not list(out_dir.rglob('*.html'))

    # the mapped-view CSV carries the provenance columns (§9.3)
    downloads.clear()
    assert _click_button(client, 'Export matched rows (CSV)')
    csv_pairs = [(c, n) for c, n in downloads if n.endswith('.csv')]
    assert csv_pairs and 'matched_rows' in csv_pairs[0][1]
    csv_text = csv_pairs[0][0]
    rows = list(csv_mod.reader(io_mod.StringIO(csv_text)))
    header = rows[0]
    for column in ('foreign_dataset', 'foreign_type(s)',
                   'matched column(s)', 'bridge-flywireType',
                   'bridge-additional_type(s)'):
        assert column in header, header
    type_idx = header.index('type')
    bridge_idx = header.index('bridge-flywireType')
    foreign_idx = header.index('foreign_type(s)')
    by_type = {row[type_idx]: row for row in rows[1:] if row[type_idx]}
    cl125 = by_type.get('CL125')
    assert cl125, sorted(by_type)
    assert cl125[foreign_idx] == 'APDN3'
    assert cl125[bridge_idx] == 'LMTe01'
    # beyond-display-cap types (DN1a is not in the capped covered list)
    # still get filled provenance cells — the empty-row regression
    dn1a = by_type.get('DN1a')
    assert dn1a, sorted(by_type)
    assert dn1a[foreign_idx] == 'DN1a'
    assert dn1a[header.index('matched column(s)')] == "cell_type · circadian_clock"
