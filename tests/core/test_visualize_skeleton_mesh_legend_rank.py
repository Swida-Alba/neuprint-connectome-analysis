"""Tests ensuring brain/VNC mesh legend entries always rank last.

Plotly's legend sorts by ``legendrank`` and coerces missing ranks to 0, so
mesh traces without an explicit rank used to cluster into the first neuron
legend group and appear between two neuron entries.
"""

from pathlib import Path
import sys

import plotly.graph_objects as go


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from visualize_skeleton import (  # noqa: E402
    BRAIN_MESH_LEGEND_RANK,
    ROI_MESH_LEGEND_RANK_BASE,
    VNC_MESH_LEGEND_RANK,
    _configure_roi_mesh_traces,
    VisualizeSkeleton,
)


def test_mesh_legend_ranks_order_meshes_after_neurons():
    # Neuron ranks grow by 100 per legend group; the mesh bands must stay
    # far above any realistic neuron count, with ROI meshes before the
    # brain mesh and the VNC mesh always last.
    assert BRAIN_MESH_LEGEND_RANK > ROI_MESH_LEGEND_RANK_BASE
    assert VNC_MESH_LEGEND_RANK > BRAIN_MESH_LEGEND_RANK
    assert ROI_MESH_LEGEND_RANK_BASE > 1_000_000


def test_configure_roi_mesh_traces_sets_legend_rank():
    traces = [go.Mesh3d(), go.Mesh3d()]
    result = _configure_roi_mesh_traces(traces, 'AME', legend_rank=1234)

    assert result is traces
    assert all(trace.legendrank == 1234 for trace in result)
    assert all(trace.name == 'brain region [AME]' for trace in result)
    assert result[0].showlegend is True
    assert result[1].showlegend is False


def test_configure_roi_mesh_traces_without_rank_keeps_default():
    traces = [go.Mesh3d()]
    _configure_roi_mesh_traces(traces, 'AME')
    assert traces[0].legendrank is None


def test_visualize_skeleton_constants_are_importable_from_class_module():
    # The plotting code references the constants at mesh creation time;
    # guard against accidental renames.
    for name in (
        'ROI_MESH_LEGEND_RANK_BASE',
        'BRAIN_MESH_LEGEND_RANK',
        'VNC_MESH_LEGEND_RANK',
    ):
        assert hasattr(VisualizeSkeleton.__init__.__globals__, name) or hasattr(
            sys.modules['visualize_skeleton'], name
        )
