"""Tests for `VisualizeSkeleton._resolve_fafb_sources`.

Covers the per-body render priority:
    tube mode:  prepared CAVE mesh cache -> raw SWC cache -> healed ZIP
                -> CAVE API
    line mode:  raw SWC cache -> healed ZIP -> CAVE API (SWC-first)
and the strict `use_cache=False` policy (cache sources skipped, extrusion
parquet check cache untouched).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import navis  # noqa: E402
from visualize_skeleton import VisualizeSkeleton  # noqa: E402


def make_tree(body_id):
    neuron = navis.TreeNeuron(pd.DataFrame({
        "node_id": np.array([0], dtype=np.int64),
        "parent_id": np.array([-1], dtype=np.int64),
        "x": np.array([0.0]),
        "y": np.array([0.0]),
        "z": np.array([0.0]),
        "radius": np.array([1.0]),
        "type": ["root"],
    }))
    neuron.soma = None
    neuron.id = body_id
    return neuron


def make_spiky_tree(body_id):
    neuron = navis.TreeNeuron(pd.DataFrame({
        "node_id": np.array([0, 1, 2, 3, 4], dtype=np.int64),
        "parent_id": np.array([-1, 0, 1, 1, 3], dtype=np.int64),
        "x": np.array([0.0, 1.0, 2.0, 100.0, 101.0]),
        "y": np.zeros(5),
        "z": np.zeros(5),
        "radius": np.ones(5),
    }))
    neuron.id = body_id
    return neuron


def make_mesh(body_id):
    vertices = np.array([
        (0.0, 0.0, 0.0), (100.0, 0.0, 0.0),
        (0.0, 100.0, 0.0), (0.0, 0.0, 100.0),
    ])
    faces = np.array([
        (0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3),
    ], dtype=np.int64)
    mesh = navis.MeshNeuron({"vertices": vertices, "faces": faces})
    mesh.id = body_id
    return mesh


class RecordingResolver:
    """Stubbed resolver recording every source-layer call."""

    def __init__(self, zip_hits=None, raw_hits=None, mesh_hits=None,
                 api_hits=None):
        self.visualizer = object.__new__(VisualizeSkeleton)
        self.visualizer.dataset = "flywire_FAFB_v783"
        self.visualizer.cache_neurons = True
        self.visualizer.auto_fix_extrusions = False
        self.visualizer._vprint = lambda *args, **kwargs: None

        self.zip_hits = dict(zip_hits or {})
        self.raw_hits = dict(raw_hits or {})
        self.mesh_hits = dict(mesh_hits or {})
        self.api_hits = dict(api_hits or {})

        self.calls = {"zip": [], "raw": [], "mesh": [], "api": [],
                      "extrusion": []}

        self.visualizer._preload_fafb_skeletons = self._fake_zip
        self.visualizer._load_api_cached_skeletons = self._fake_raw
        self.visualizer._load_cached_fafb_meshes = self._fake_mesh
        self.visualizer._fetch_fafb_skeletons_via_api = self._fake_api
        self.visualizer._detect_extrusions_in_skeletons = self._fake_extrusion

    # --- mock source layers --------------------------------------------
    def _fake_zip(self, body_ids_filter=None):
        ids = list(body_ids_filter or [])
        self.calls["zip"].append(ids)
        return {bid: self.zip_hits[bid] for bid in ids
                if bid in self.zip_hits}

    def _fake_raw(self, body_ids):
        self.calls["raw"].append(list(body_ids))
        found = {bid: self.raw_hits[bid] for bid in body_ids
                 if bid in self.raw_hits}
        missing = [bid for bid in body_ids if bid not in self.raw_hits]
        return found, missing

    def _fake_mesh(self, body_ids):
        self.calls["mesh"].append(list(body_ids))
        found = {bid: self.mesh_hits[bid] for bid in body_ids
                 if bid in self.mesh_hits}
        missing = [bid for bid in body_ids if bid not in self.mesh_hits]
        return found, missing

    def _fake_api(self, body_ids, **kwargs):
        self.calls["api"].append((list(body_ids), kwargs))
        return {bid: self.api_hits[bid] for bid in body_ids
                if bid in self.api_hits}

    def _fake_extrusion(self, skeletons, **kwargs):
        self.calls["extrusion"].append((list(skeletons), kwargs))
        return []

    def resolve(self, body_ids, **kwargs):
        return self.visualizer._resolve_fafb_sources(body_ids, **kwargs)


class TestSourcePriority:
    def test_mesh_then_raw_then_zip_then_cave(self):
        resolver = RecordingResolver(
            zip_hits={"1": make_tree("1")},
            raw_hits={"2": make_tree("2")},
            mesh_hits={"3": make_mesh("3")},
            api_hits={"4": make_mesh("4")},
        )
        sources, skeleton_cache, mesh_cache = resolver.resolve([1, 2, 3, 4])

        assert sources == {"1": "zip", "2": "raw_cache", "3": "mesh_cache",
                           "4": "cave"}
        assert set(skeleton_cache) == {"1", "2"}
        assert set(mesh_cache) == {"3", "4"}
        # Tube mode priority: the prepared mesh cache wins, then the raw
        # SWC cache, then the healed bundle; CAVE serves the rest.
        assert resolver.calls["mesh"] == [["1", "2", "3", "4"]]
        assert resolver.calls["raw"] == [["1", "2", "4"]]
        assert resolver.calls["zip"] == [["1", "4"]]
        assert resolver.calls["api"][0][0] == ["4"]
        assert resolver.calls["api"][0][1]["cache_prepared"] is True

    def test_raw_cache_before_bundle_when_mesh_absent(self):
        """Without a prepared mesh hit, the raw SWC cache precedes the
        healed bundle (pipeline priority), and both stay SWC sources."""
        resolver = RecordingResolver(
            zip_hits={"7": make_tree("7")},
            raw_hits={"7": make_tree("7")},
        )
        sources, skeleton_cache, _ = resolver.resolve([7])

        assert sources == {"7": "raw_cache"}
        assert set(skeleton_cache) == {"7"}
        assert resolver.calls["mesh"] == [["7"]]
        assert resolver.calls["raw"] == [["7"]]
        assert resolver.calls["zip"] == []

    def test_every_local_miss_falls_through_to_cave(self):
        resolver = RecordingResolver(api_hits={"9": make_mesh("9")})
        sources, skeleton_cache, mesh_cache = resolver.resolve([9])

        assert sources == {"9": "cave"}
        assert skeleton_cache == {}
        assert set(mesh_cache) == {"9"}

    def test_api_only_routes_straight_to_cave(self):
        resolver = RecordingResolver(
            zip_hits={"1": make_tree("1")},
            raw_hits={"1": make_tree("1")},
            api_hits={"1": make_mesh("1")},
        )
        sources, skeleton_cache, mesh_cache = resolver.resolve(
            [1], api_only=True)

        assert sources == {"1": "cave"}
        assert skeleton_cache == {}
        assert set(mesh_cache) == {"1"}
        assert resolver.calls["zip"] == []
        assert resolver.calls["raw"] == []
        assert resolver.calls["mesh"] == []
        assert resolver.calls["api"][0][1]["cache_prepared"] is True


class TestStrictUseCache:
    def test_cache_sources_skipped_when_caching_disabled(self):
        resolver = RecordingResolver(
            raw_hits={"2": make_tree("2")},
            mesh_hits={"2": make_mesh("2")},
            api_hits={"2": make_mesh("2")},
        )
        resolver.visualizer.cache_neurons = False

        sources, _, mesh_cache = resolver.resolve(
            [2], allow_mesh_cache=False)

        assert sources == {"2": "cave"}
        assert set(mesh_cache) == {"2"}
        assert resolver.calls["raw"] == []
        assert resolver.calls["mesh"] == []
        # CAVE was asked to bypass its prepared cache too.
        assert resolver.calls["api"][0][1]["cache_prepared"] is False

    def test_healed_zip_stays_eligible_without_cache(self):
        """The healed ZIP is the canonical raw source and stays eligible
        under the strict use_cache=False policy."""
        resolver = RecordingResolver(zip_hits={"5": make_tree("5")})
        resolver.visualizer.cache_neurons = False

        sources, skeleton_cache, _ = resolver.resolve(
            [5], allow_mesh_cache=False)

        assert sources == {"5": "zip"}
        assert set(skeleton_cache) == {"5"}
        assert resolver.calls["raw"] == []
        assert resolver.calls["mesh"] == []
        assert resolver.calls["api"] == []

    def test_extrusion_check_honors_use_cache_policy(self):
        resolver = RecordingResolver(zip_hits={"6": make_tree("6")})
        resolver.visualizer.auto_fix_extrusions = True

        # caching enabled -> extrusion check may use its parquet cache
        resolver.resolve([6], allow_mesh_cache=True)
        assert resolver.calls["extrusion"][0][1]["use_cache"] is True

        # caching disabled -> strict in-memory-only extrusion check
        resolver.calls["extrusion"].clear()
        resolver.visualizer.cache_neurons = False
        resolver.resolve([6], allow_mesh_cache=False)
        assert resolver.calls["extrusion"][0][1]["use_cache"] is False


class TestExtrusionRepair:
    def test_extrusion_affected_tree_source_moves_to_cave(self):
        resolver = RecordingResolver(
            zip_hits={"6": make_tree("6")},
            api_hits={"6": make_mesh("6")},
        )
        resolver.visualizer.auto_fix_extrusions = True
        resolver.visualizer._detect_extrusions_in_skeletons = (
            lambda skeletons, **kwargs: ["6"])

        sources, skeleton_cache, mesh_cache = resolver.resolve(
            [6], allow_mesh_cache=True)

        assert sources == {"6": "cave"}
        assert skeleton_cache == {}
        assert set(mesh_cache) == {"6"}
        # Repair requests force a fresh fetch.
        assert resolver.calls["api"][0][0] == ["6"]
        assert resolver.calls["api"][0][1]["force_refresh"] is True

    def test_failed_cave_repair_prunes_local_extrusion_branch(self):
        resolver = RecordingResolver(
            zip_hits={"7": make_spiky_tree("7")},
        )
        resolver.visualizer.auto_fix_extrusions = True
        resolver.visualizer._detect_extrusions_in_skeletons = (
            lambda skeletons, **kwargs: ["7"])

        sources, skeleton_cache, mesh_cache = resolver.resolve(
            [7], allow_mesh_cache=True)

        assert sources == {"7": "local_repaired"}
        assert mesh_cache == {}
        assert set(skeleton_cache["7"].nodes["node_id"]) == {0, 1, 2}
        assert resolver.calls["api"][0][0] == ["7"]
        assert resolver.calls["api"][0][1]["force_refresh"] is True
