"""Tests for the collapsible type_tree legend panel injected into viewer HTML.

``legend_mode='type_tree'`` renders exactly like ``'type'`` (the native
legend used by static exports) and additionally tags traces with
``drocatLegend`` meta so the exported interactive HTML can embed a
collapsible type -> neuron legend panel built client-side.
"""

from pathlib import Path
import sys

import plotly.graph_objects as go


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import visualize_skeleton  # noqa: E402
from visualize_skeleton import LEGEND_MODES, VisualizeSkeleton  # noqa: E402


def _make_visualizer(legend_mode='type_tree'):
    visualizer = object.__new__(VisualizeSkeleton)
    visualizer.background_color = 'white'
    visualizer.legend_mode = legend_mode
    return visualizer


def _tagged_figure():
    """A figure shaped like a type_tree export: tagged neurons, a paired
    synapse legend dummy, and a mesh pinned last."""
    fig = go.Figure()
    for i, (group, item) in enumerate(
        [('Tm3', '101'), ('Tm3', '102'), ('Mi1', '201')]
    ):
        trace = go.Scatter3d(
            x=[0, i + 1], y=[0, 1], z=[0, 1], mode='lines',
            line=dict(color='#1f77b4', width=4),
            name=group, legendgroup=group, showlegend=(i != 1),
        )
        trace.legendrank = i * 100
        trace.meta = {'drocatLegend': {
            'kind': 'neuron', 'group': group, 'item': item,
        }}
        fig.add_trace(trace)

    dummy = go.Scatter3d(
        x=[None], y=[None], z=[None], mode='markers',
        name='synapses 0 -> 1', legendgroup='synapses 0 -> 1',
        showlegend=True, marker=dict(size=10, color='rgb(44,160,44)'),
    )
    fig.add_trace(dummy)

    mesh = go.Mesh3d(
        x=[0, 1, 0], y=[0, 0, 1], z=[0, 0, 0], i=[0], j=[1], k=[2],
        color='#c8e6f0', opacity=0.1, name='Brain mesh', showlegend=True,
    )
    mesh.legendrank = visualize_skeleton.BRAIN_MESH_LEGEND_RANK
    fig.add_trace(mesh)
    return fig


def test_legend_modes_include_type_tree():
    assert 'type_tree' in LEGEND_MODES
    assert 'type' in LEGEND_MODES


def test_legend_tree_html_contains_panel_and_markers():
    html = _make_visualizer()._legend_tree_html()
    assert 'drocat-legend-tree' in html
    assert 'drocatLegend' in html  # meta key read client-side
    assert '"meshRankBase": 100000000' in html
    # theme-aware via the body class the theme switch toggles
    assert 'drocat-theme-dark' in html
    # replaces the native legend on injected pages only
    assert '.js-plotly-plot .legend{display:none !important;}' in html
    # banner-aware positioning + webdriver guard
    assert 'drocat-warning-container' in html
    assert 'navigator.webdriver' in html
    # panel sits on the right, below the theme switch
    assert 'position:fixed;right:10px;top:60px;' in html
    # triangle caret that rotates when a group expands
    assert '\\u25B6' in html
    assert '.drocat-lt-expanded .drocat-lt-caret{transform:rotate(90deg);}' in html


def test_write_plotly_html_embeds_tree_only_when_requested(tmp_path):
    visualizer = _make_visualizer()

    on_path = tmp_path / 'on.html'
    visualizer._write_plotly_html(
        _tagged_figure(), str(on_path), legend_tree=True
    )
    on_html = on_path.read_text(encoding='utf-8')
    assert 'drocat-legend-tree' in on_html
    assert 'drocatLegend' in on_html  # meta serialized into the figure

    off_path = tmp_path / 'off.html'
    visualizer._write_plotly_html(_tagged_figure(), str(off_path))
    off_html = off_path.read_text(encoding='utf-8')
    assert 'drocat-legend-tree' not in off_html
    assert 'drocatLegend' in off_html  # meta is harmless without the panel


def test_tree_injection_is_idempotent(tmp_path):
    visualizer = _make_visualizer()
    page = tmp_path / 'idempotent.html'
    visualizer._write_plotly_html(
        _tagged_figure(), str(page), legend_tree=True
    )
    before = page.read_text(encoding='utf-8')
    visualizer._inject_page_extras(str(page), legend_tree=True)
    assert page.read_text(encoding='utf-8') == before
