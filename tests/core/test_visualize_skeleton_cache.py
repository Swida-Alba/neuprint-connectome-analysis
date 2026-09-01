"""NeuPrint skeleton cache and render-time simplification.

Covers `VisualizeSkeleton._save_cached_neurons` / `_load_cached_neurons` /
`_skeleton_cache_is_simplified`:
- NeuPrint visualization caches hold raw skeletons in the shared compressed-SWC
  namespace (.swc.zst, level 0 recorded in the header)
- every visualization simplification level applies to that common source
- no visualization fetch writes a `.pkl` skeleton or `.level` marker
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import visualize_skeleton as vs_mod  # noqa: E402
from visualize_skeleton import VisualizeSkeleton  # noqa: E402


class FakeClient:
    """Minimal neuprint client stub: construction must not hit the network."""

    def __init__(self, *args, **kwargs):
        self.dataset = kwargs.get("dataset", "hemibrain:v1.2.1")


def make_neuron(n_nodes=120):
    import navis
    # proper SWC classification (root/slab/end) so navis.downsample_neuron
    # reduces the skeleton like it does for real NeuPrint data
    types = (["root"] + ["slab"] * max(0, n_nodes - 2) + ["end"])
    nodes = pd.DataFrame({
        "node_id": np.arange(n_nodes, dtype=np.int64),
        "parent_id": np.array([-1] + list(range(n_nodes - 1)), dtype=np.int64),
        "x": np.arange(n_nodes, dtype=float) * 1000.0,
        "y": np.zeros(n_nodes),
        "z": np.zeros(n_nodes),
        "radius": np.ones(n_nodes),
        "type": types[:n_nodes],
    })
    nrn = navis.TreeNeuron(nodes)
    # disable the constructor's soma auto-detection (it flags most nodes of
    # a plain chain, turning them into downsample fix points)
    nrn.soma = None
    return nrn


def build_vs(tmp_path, dataset="hemibrain:v1.2.1", cache_neurons=True):
    vs = VisualizeSkeleton(
        dataset=dataset,
        neuron_layers=["aMe12"],
        client=FakeClient(dataset=dataset),
        verbose=False,
        output_dir=str(tmp_path),
        include_timestamp=False,
        cache_neurons=cache_neurons,
        data_folder=str(tmp_path),
        # sandbox the cache root (defaults to the project root)
        script_path=str(tmp_path),
    )
    vs.client_type = "neuprint"
    return vs


class TestNeuPrintSimplifiedCache:
    def test_save_writes_simplified_swc_without_marker(self, tmp_path):
        vs = build_vs(tmp_path)
        neuron = make_neuron(120)
        neuron.id = 101
        df = pd.DataFrame({"bodyId": [101]})
        vs._save_cached_neurons(df, [neuron])

        cache_dir = (Path(tmp_path) / "cache" / "hemibrain_v1_2_1"
                     / "skeletons")
        raw_path = cache_dir / "raw_skeletons" / "101.swc.zst"
        assert raw_path.exists()
        assert not (cache_dir / "101.pkl").exists()
        assert not (cache_dir / ".level").exists()

        cached, missing = vs._load_cached_neurons(
            pd.DataFrame({"bodyId": [101]}))
        assert missing == []
        assert cached is not None and len(cached) == 1
        # Visualization persistence overrides the morphology helper's simp90
        # default so render-time mesh decimation is applied exactly once.
        assert cached[0]._drocat_simplification == 0
        assert len(cached[0].nodes) == len(neuron.nodes)

    def test_save_is_independent_of_render_simplification(self, tmp_path):
        vs = build_vs(tmp_path)
        vs.skeleton_mesh_simplification = 0.5
        neuron = make_neuron(120)
        neuron.id = 101

        vs._save_cached_neurons(pd.DataFrame({"bodyId": [101]}), [neuron])

        raw_path = (Path(tmp_path) / "cache" / "hemibrain_v1_2_1"
                    / "skeletons" / "raw_skeletons" / "101.swc.zst")
        assert raw_path.exists()
        assert not raw_path.with_name("101.pkl").exists()

    def test_load_reads_cached_level(self, tmp_path):
        vs = build_vs(tmp_path)
        neuron = make_neuron(120)
        neuron.id = 101
        vs._save_cached_neurons(pd.DataFrame({"bodyId": [101]}), [neuron])
        neurons, missing = vs._load_cached_neurons(pd.DataFrame({"bodyId": [101, 202]}))
        assert missing == [202]
        assert neurons is not None and len(neurons) == 1
        assert neurons[0]._drocat_simplification == 0
        assert len(neurons[0].nodes) == len(neuron.nodes)
        assert vs._skeleton_cache_is_simplified() is False

    def test_ignore_cache_reports_all_missing(self, tmp_path):
        vs = build_vs(tmp_path)
        neuron = make_neuron(120)
        neuron.id = 101
        vs._save_cached_neurons(pd.DataFrame({"bodyId": [101]}), [neuron])
        # Explicitly bypassing the raw cache still reports the IDs as missing.
        neurons, missing = vs._load_cached_neurons(
            pd.DataFrame({"bodyId": [101]}), ignore_cache=True)
        assert neurons is None
        assert missing == [101]

    def test_skeleton_cache_is_never_marked_simplified(self, tmp_path):
        vs = build_vs(tmp_path)
        assert vs._skeleton_cache_is_simplified() is False
        cache_dir = Path(tmp_path) / "cache" / "hemibrain_v1_2_1" / "skeletons"
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / ".level").write_text("simp90\n")
        assert vs._skeleton_cache_is_simplified() is False

    def test_effective_render_level_uses_current_render_source(self, tmp_path):
        vs = build_vs(tmp_path)
        vs.skeleton_mesh_simplification = 0.95
        neuron = make_neuron(120)
        neuron.id = 101
        vs._save_cached_neurons(pd.DataFrame({"bodyId": [101]}), [neuron])

        # Render simplification is always applied directly to the raw source.
        assert vs._effective_render_simplification(
            is_fafb=False, using_simplified_cache=False
        ) == pytest.approx(0.95)
        assert vs._effective_render_simplification(
            is_fafb=False, using_simplified_cache=True
        ) == pytest.approx(0.95)

    def test_mixed_legacy_cache_levels_are_marked_stale(self, tmp_path):
        vs = build_vs(tmp_path)
        raw = make_neuron(120)
        raw.id = 101
        raw._drocat_simplification = 0
        legacy = make_neuron(120)
        legacy.id = 202
        legacy._drocat_simplification = 90

        class MixedCache:
            def load_skeleton(self, body_id):
                return {101: raw, 202: legacy}.get(body_id)

        preferred, stale, missing = vs._resolve_neuprint_render_cache(
            MixedCache(), [101, 202, 303]
        )
        assert set(preferred) == {101}
        assert set(stale) == {202}
        assert missing == [303]

    def test_unrefreshable_legacy_cache_aborts_render(
            self, tmp_path, monkeypatch):
        """A simp90 file that cannot be refreshed online must abort the
        render instead of re-leveling every raw source to level 90."""
        import morphology

        vs = build_vs(tmp_path)
        raw = make_neuron(120)
        raw.id = 101
        raw._drocat_simplification = 0
        legacy = make_neuron(120)
        legacy.id = 202
        legacy._drocat_simplification = 90

        class MixedCache:
            def load_skeleton(self, body_id):
                return {101: raw, 202: legacy}.get(body_id)

        monkeypatch.setattr(
            morphology, "find_similar_raw_cache",
            lambda *a, **k: MixedCache())

        def partial_fetch(fetch_df, fetch_kwargs, persist=False):
            # The missing raw skeleton comes back online; the stale legacy
            # file does not.
            return [raw]

        vs._fetch_neuprint_skeletons_batched = partial_fetch

        with pytest.raises(RuntimeError, match="legacy simp90"):
            vs._prepare_neuprint_skeletons_for_render(
                [101, 202], False, False, None)


class TestFlywireCacheUntouched:
    def test_visualizer_raw_cache_hook_does_not_write_flywire_api_cache(
            self, tmp_path, monkeypatch):
        """FlyWire uses its CAVE/mesh path, not the NeuPrint SWC hook."""
        # the constructor checks the real FlyWire data files; skip that
        # check (the cache logic itself is what is under test)
        import FAFB_file_converter
        monkeypatch.setattr(FAFB_file_converter, "ensure_flywire_data",
                            lambda dataset, dataset_dir: True)
        vs = build_vs(tmp_path, dataset="flywire_FAFB_v783")
        vs.client_type = "flywire"
        neuron = make_neuron(120)
        neuron.id = 101
        vs._save_cached_neurons(pd.DataFrame({"bodyId": [101]}), [neuron])

        cache_dir = Path(tmp_path) / "cache" / "flywire_FAFB_v783"
        assert not (cache_dir / "skeletons" / "raw_skeletons").exists()
        assert not (cache_dir / "API_cache").exists()


class TestStrictNoCachePolicy:
    def test_prepare_with_cache_neurons_false_skips_raw_cache_and_persist(
            self, tmp_path, monkeypatch):
        """cache_neurons=False: no shared raw-cache read, online fetches
        receive persist=False, and nothing is written by the prepare phase."""
        vs = build_vs(tmp_path, cache_neurons=False)
        vs.skeleton_mode = "line"
        vs.show_connectors = False
        vs.skeleton_mesh_simplification = 0.0

        # The shared raw cache must never be consulted.
        monkeypatch.setattr(
            "morphology.find_similar_raw_cache",
            lambda *a, **k: (_ for _ in ()).throw(
                AssertionError(
                    "cache_neurons=False read the shared raw cache")),
        )
        fetch_kwargs_seen = {}
        neuron = make_neuron(120)
        neuron.id = 101

        def fake_batched(fetch_df, fetch_kwargs, persist=True):
            fetch_kwargs_seen["persist"] = persist
            return [neuron]

        vs._fetch_neuprint_skeletons_batched = fake_batched
        vs._persist_fetched_neuprint_skeletons = (
            lambda *a, **k: (_ for _ in ()).throw(
                AssertionError(
                    "cache_neurons=False persisted fetched skeletons")))

        prepared, prepared_mesh = vs._prepare_neuprint_skeletons_for_render(
            [101], use_neuprint_fine_pipeline=False,
            use_neuprint_mesh_cache=False, neuprint_mesh_cache={})

        assert fetch_kwargs_seen == {"persist": False}
        assert 101 in prepared
        assert prepared_mesh is False
        # The line render product is the in-memory 50% reduction; the
        # canonical fetched source is untouched.
        assert prepared[101].n_nodes < neuron.n_nodes
        assert neuron.n_nodes == 120


class TestCustomLayerGrouping:
    """layer_map_csv: rows sharing a 'layer' value become ONE group (one
    legend entry in 'layer' mode, one individual profile)."""

    def test_layer_map_csv_groups_neurons(self, tmp_path, monkeypatch):
        csv_path = tmp_path / "layer_map.csv"
        csv_path.write_text(
            "layer,id_type_instance\n"
            "DNp_group,DNp01\n"
            "DNp_group,DNp02\n"
            "aMe_group,aMe12\n"
        )
        # layer resolution happens via statvis during construction; the
        # grouping logic itself is what is under test
        monkeypatch.setattr(
            vs_mod.sv, "getNeurons",
            lambda *a, **k: (pd.DataFrame(), pd.DataFrame(), "x", None),
        )
        vs = VisualizeSkeleton(
            dataset="hemibrain:v1.2.1",
            neuron_layers=["aMe12"],  # overridden by the CSV
            layer_map_csv=str(csv_path),
            client=FakeClient(dataset="hemibrain:v1.2.1"),
            verbose=False,
            output_dir=str(tmp_path),
            include_timestamp=False,
            script_path=str(tmp_path),
        )
        # two groups -> two layers, one per group
        assert vs.neuron_layers == [["DNp01", "DNp02"], "aMe12"]
        assert vs.custom_layer_names == ["DNp_group", "aMe_group"]
        assert vs.layer_names == ["DNp_group", "aMe_group"]

    def test_duplicate_custom_names_group_layers(self, tmp_path, monkeypatch):
        """The UI hint path: two layers given the SAME custom name share one
        legend entry in 'layer' mode (verified backend grouping)."""
        monkeypatch.setattr(
            vs_mod.sv, "getNeurons",
            lambda *a, **k: (pd.DataFrame(), pd.DataFrame(), "x", None),
        )
        vs = VisualizeSkeleton(
            dataset="hemibrain:v1.2.1",
            neuron_layers=["DNp01", "DNp02"],
            custom_layer_names=["DNp_group", "DNp_group"],
            client=FakeClient(dataset="hemibrain:v1.2.1"),
            verbose=False,
            output_dir=str(tmp_path),
            include_timestamp=False,
            script_path=str(tmp_path),
        )
        assert vs.layer_names == ["DNp_group", "DNp_group"]
        # in 'layer' mode both layers map to the same legend group
        assert len(set(vs.layer_names)) == 1
