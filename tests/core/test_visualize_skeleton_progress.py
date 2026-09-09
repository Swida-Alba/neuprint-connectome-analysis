"""Regression tests for compact skeleton transformation progress output."""

import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import navis  # noqa: E402
import visualize_skeleton as vs_mod  # noqa: E402
from visualize_skeleton import VisualizeSkeleton  # noqa: E402


class CaptureProgressBar:
    """Minimal outer-bar stub that records postfix refreshes."""

    def __init__(self):
        self.statuses = []

    def set_postfix_str(self, status, refresh=True):
        self.statuses.append((status, refresh))


def make_neuron(name):
    neuron = navis.TreeNeuron(pd.DataFrame({
        "node_id": np.array([0], dtype=np.int64),
        "parent_id": np.array([-1], dtype=np.int64),
        "x": np.array([0.0]),
        "y": np.array([0.0]),
        "z": np.array([0.0]),
        "radius": np.array([1.0]),
        "type": ["root"],
    }))
    neuron.name = name
    return neuron


def make_chain_neuron(n_nodes=2000, body_id="42"):
    """Straight chain TreeNeuron large enough for the node-reduction stage."""
    types = ["root"] + ["slab"] * max(0, n_nodes - 2) + ["end"]
    neuron = navis.TreeNeuron(pd.DataFrame({
        "node_id": np.arange(n_nodes, dtype=np.int64),
        "parent_id": np.array([-1] + list(range(n_nodes - 1)),
                               dtype=np.int64),
        "x": np.arange(n_nodes, dtype=float) * 1000.0,
        "y": np.zeros(n_nodes),
        "z": np.zeros(n_nodes),
        "radius": np.ones(n_nodes),
        "type": types[:n_nodes],
    }))
    neuron.soma = None
    neuron.id = body_id
    return neuron


def make_mesh(body_id=42):
    vertices = np.array([
        (0.0, 0.0, 0.0),
        (100.0, 0.0, 0.0),
        (0.0, 100.0, 0.0),
        (0.0, 0.0, 100.0),
    ])
    faces = np.array([
        (0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3),
    ], dtype=np.int64)
    mesh = navis.MeshNeuron({"vertices": vertices, "faces": faces})
    mesh.id = body_id
    return mesh


def test_transform_progress_reuses_outer_bar(monkeypatch, capsys):
    """Successful transforms refresh one bar instead of printing status lines."""
    monkeypatch.setitem(sys.modules, "flybrains", types.ModuleType("flybrains"))
    monkeypatch.setattr(
        vs_mod.navis.transforms.registry,
        "shortest_bridging_seq",
        lambda source, target: ([source, target], ["step"]),
    )
    monkeypatch.setattr(vs_mod.navis, "xform", lambda neuron, transform: neuron)

    visualizer = object.__new__(VisualizeSkeleton)
    visualizer.verbose = True
    progress = CaptureProgressBar()

    result = visualizer._xform_neurons_safe(
        navis.NeuronList([make_neuron("left"), make_neuron("right")]),
        source="raw",
        target="template",
        layer_label="Layer 1 (example)",
        progress_bar=progress,
    )

    assert len(result) == 2
    statuses = [status for status, _ in progress.statuses]
    assert statuses[0].endswith("2 pts | raw→template")
    assert any("current=left" in status for status in statuses)
    assert any("current=right" in status for status in statuses)
    assert statuses[-1].endswith("done 0.0s")

    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == ""


def test_fafb_visualizer_cave_fetch_skeletonizes_and_honours_cache_policy(
        tmp_path, monkeypatch):
    """The CAVE fetch wavefront-skeletonizes the raw mesh and passes the
    cache_neurons policy through as the fetch's use_cache policy."""
    import cave_data_fetcher as cave

    calls = []

    class FakeCaveFetcher:
        def __init__(self, *args, **kwargs):
            assert kwargs["project_root"] == str(tmp_path)

        def fetch_skeleton(self, body_id, use_cache=True):
            calls.append((body_id, use_cache))
            return make_chain_neuron(body_id=str(body_id))

    monkeypatch.setattr(cave, "CAVEDataFetcher", FakeCaveFetcher)

    visualizer = object.__new__(VisualizeSkeleton)
    visualizer.dataset = "flywire_FAFB_v783"
    visualizer.script_path = str(tmp_path)
    visualizer.verbose = False
    visualizer._flywire_skeleton_access = {"cave_token": "test-token"}

    visualizer.cache_neurons = False
    result = visualizer._fetch_fafb_skeletons_via_cave([42])
    assert calls == [(42, False)]
    assert isinstance(result["42"], navis.TreeNeuron)

    calls.clear()
    visualizer.cache_neurons = True
    result = visualizer._fetch_fafb_skeletons_via_cave([42])
    assert calls == [(42, True)]
    assert isinstance(result["42"], navis.TreeNeuron)


def _stub_fafb_layer_visualizer(mode):
    """Visualizer stub whose _vprint records the FAFB layer stage stream."""
    visualizer = object.__new__(VisualizeSkeleton)
    visualizer.skeleton_mode = mode
    visualizer.skeleton_mesh_simplification = 0.95
    visualizer.soma_mesh_simplification = 0.80
    visualizer.soma_region_radius = 20.0
    messages = []
    visualizer._vprint = lambda msg, *args, **kwargs: messages.append(msg)
    visualizer._simplify_mesh_fafb_fine = (
        lambda trimesh_obj, target_faces: trimesh_obj)
    return visualizer, messages


def test_fafb_fast_tube_stages_reduce_then_tube_then_decimate():
    """Fast tube emits node reduction strictly before tube meshing and
    surface decimation."""
    visualizer, messages = _stub_fafb_layer_visualizer("tube")
    neuron = make_chain_neuron(2000)

    out, done = visualizer._process_fafb_layer(
        navis.NeuronList([neuron]), "fast",
        render_mesh_cache={})

    assert done is True
    assert isinstance(out[0], navis.MeshNeuron)
    joined = "".join(messages)
    assert "🔽 reduce nodes 42 2000→" in joined
    assert "🧪 tube mesh 42" in joined
    assert "⚡ decimate 42" in joined
    assert (joined.index("🔽 reduce nodes") < joined.index("🧪 tube mesh")
            < joined.index("⚡ decimate"))


def test_fafb_line_mode_never_emits_tube_or_mesh_stages():
    """Line rendering emits node reduction only; tube/mesh stages are
    absent from the progress stream."""
    visualizer, messages = _stub_fafb_layer_visualizer("line")
    neuron = make_chain_neuron(2000)

    out, done = visualizer._process_fafb_layer(
        navis.NeuronList([neuron]), "fast",
        render_mesh_cache={})

    assert done is False
    assert isinstance(out[0], navis.TreeNeuron)
    joined = "".join(messages)
    assert "🔽 reduce nodes 42 2000→" in joined
    assert "🧪 tube mesh" not in joined
    assert "⚡ decimate" not in joined
    assert "🧱 mesh source" not in joined


def test_fafb_progress_callback_reports_stages_and_completion():
    """FAFB sources update the shared render bar without nesting tqdm bars."""
    visualizer, _ = _stub_fafb_layer_visualizer("tube")
    events = []

    visualizer._process_fafb_layer(
        navis.NeuronList([make_chain_neuron(2000)]),
        "fast",
        render_mesh_cache={},
        progress_callback=lambda neuron_id, stage, done: events.append(
            (neuron_id, stage, done)),
    )

    completed = [event for event in events if event[2]]
    assert {event[0] for event in completed} == {"42"}
    assert any(event[0] == "42" and event[1] == "reduce nodes"
               for event in events)
    assert any(event[0] == "42" and event[1] == "tube mesh"
               for event in events)
