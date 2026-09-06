"""Offline tests for the isolated BANC render path.

Covers the render-space mapping, the BANC template branch, the source
resolver, and the BANC node-reduction rules — all with stubbed fetchers
and geometry helpers (no network, no CloudVolume).
"""
from pathlib import Path
import sys

import numpy as np
import navis
import pytest
import trimesh

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import visualize_skeleton  # noqa: E402
from visualize_skeleton import (  # noqa: E402
    dataset_native_space,
    dataset_render_space,
    VisualizeSkeleton,
)

SWC = """1 0 0 0 0 2 -1
2 0 4 0 0 2 1
3 0 8 0 0 2 2
4 0 12 0 0 2 3
"""


def _make_visualizer(dataset='banc_v888', skeleton_mode='tube'):
    vs = object.__new__(VisualizeSkeleton)
    vs.dataset = dataset
    vs.skeleton_mode = skeleton_mode
    vs.skeleton_mesh_simplification = 0.95
    vs.soma_mesh_simplification = None
    vs.soma_region_radius = 20000
    vs.cache_neurons = True
    vs.banc_skeleton_resolution = 'l2'
    vs.script_path = PROJECT_ROOT
    vs.FAKE_CACHE = {}
    vs._vprint = lambda *a, **k: None
    # Geometry helpers become pass-through recorders.
    vs._simplify_mesh_fafb_fine = lambda mesh, target: mesh
    vs._simplify_mesh_vertex_clustering = lambda mesh, target: mesh
    vs._simplify_mesh_with_soma_awareness = lambda *a, **k: a[0]
    return vs


def _tree(resolution=None):
    neuron = navis.read_swc(__import__('io').StringIO(SWC))
    if resolution:
        neuron._drocat_banc_resolution = resolution
    return neuron


class TestRenderSpace:
    def test_banc_renders_in_native_space(self):
        assert dataset_render_space('banc_v888') == 'BANC'
        assert dataset_native_space('banc_v626') == 'BANC'

    def test_fafb_unchanged(self):
        assert dataset_render_space('flywire_FAFB_v783') == 'FLYWIRE'


class TestViewCameras:
    """Plan item C: the shared per-dataset camera table."""

    def test_banc_front_matches_plan_table(self):
        from visualize_skeleton import dataset_view_cameras

        cams = dataset_view_cameras('banc_v888')
        assert cams['Front']['eye'] == dict(x=0, y=-2.5, z=0)
        assert cams['Front']['up'] == dict(x=0, y=0, z=1)
        assert cams['Top']['eye'] == dict(x=0, y=0, z=2.5)
        assert cams['Top']['up'] == dict(x=0, y=-1, z=0)
        assert cams['Left']['eye'] == dict(x=2.5, y=0, z=0)
        assert cams['Right']['eye'] == dict(x=-2.5, y=0, z=0)

    def test_banc_lowercase_keys_for_png_export(self):
        from visualize_skeleton import dataset_view_cameras

        upper = dataset_view_cameras('banc_v626')
        lower = dataset_view_cameras('banc_v626', lowercase=True)
        assert set(lower) == {'front', 'back', 'top', 'bottom',
                              'left', 'right'}
        assert lower['front'] == upper['Front']

    def test_distance_scaling(self):
        from visualize_skeleton import dataset_view_cameras

        cams = dataset_view_cameras('banc_v888', distance=1.5)
        assert cams['Front']['eye'] == dict(x=0, y=-1.5, z=0)

    def test_other_families_unchanged(self):
        from visualize_skeleton import dataset_view_cameras

        assert (dataset_view_cameras('manc:v1.2.1')['Front']['eye']
                == dict(x=0, y=0, z=2.5))
        assert (dataset_view_cameras(
            'hemibrain:v1.2.1', brain_mesh='template')['Front']['eye']
            == dict(x=0, y=2.5, z=0))
        assert (dataset_view_cameras('male-cns:v1.0')['Front']['eye']
                == dict(x=0, y=0, z=-2.5))


class TestTemplateInfo:
    def test_banc_branch_skips_transform(self):
        vs = _make_visualizer()
        vs.brain_mesh = 'template'
        sentinel = object()
        vs._get_banc_template_volume = lambda: sentinel
        info = vs._get_template_info()
        assert info['source'] == 'BANC'
        assert info['target'] == 'BANC'
        assert info['skip_transform'] is True
        assert info['template_obj'] is sentinel

    def test_banc_whole_falls_back_to_template(self):
        vs = _make_visualizer()
        vs.brain_mesh = 'whole'
        vs._get_banc_template_volume = lambda: object()
        info = vs._get_template_info()
        # 'whole' (JRC2018F) is not wired for BANC: native outline instead.
        assert info['skip_transform'] is True

    def test_transform_not_needed_for_banc(self):
        vs = _make_visualizer()
        vs.brain_mesh = 'template'
        vs._get_banc_template_volume = lambda: object()
        assert vs._needs_skeleton_transform() is False


class TestBancProcessor:
    def _run_tube(self, monkeypatch, pipeline='fast', resolutions=('l2', 'full')):
        calls = []
        monkeypatch.setattr(
            visualize_skeleton, 'simplify_skeleton_nodes',
            lambda n, factor: calls.append((getattr(n, 'id'), factor))
            or (_tree(), {'raw_nodes': 4, 'achieved_nodes': 1}))
        # A tiny but valid mesh stands in for the tube product.
        box = trimesh.creation.box(extents=(1, 1, 1))
        monkeypatch.setattr(
            navis.conversion, 'tree2meshneuron',
            lambda work, tube_points=6: navis.MeshNeuron(box))

        vs = _make_visualizer(skeleton_mode='tube')
        vs._vprint = lambda *a, **k: None
        neurons = [_tree(r) for r in resolutions]
        for i, n in enumerate(neurons):
            n.id = 1000 + i
        out, already = vs._process_banc_layer(
            navis.NeuronList(neurons), pipeline)
        return out, already, calls

    def test_fast_reduces_only_full_resolution(self, monkeypatch):
        out, already, calls = self._run_tube(
            monkeypatch, pipeline='fast', resolutions=('l2', 'full'))
        reduced = [c[0] for c in calls]
        # The L2 source keeps every node; only the full-res one is reduced.
        assert reduced == [1001]
        assert already is True
        assert all(isinstance(n, navis.MeshNeuron) for n in out)

    def test_fine_never_reduces(self, monkeypatch):
        out, already, calls = self._run_tube(
            monkeypatch, pipeline='fine', resolutions=('l2', 'full'))
        assert calls == []

    def test_line_reduces_only_full_resolution(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            visualize_skeleton, 'simplify_skeleton_nodes',
            lambda n, factor: calls.append((getattr(n, 'id'), factor))
            or (n, {'raw_nodes': 4, 'achieved_nodes': 4}))
        vs = _make_visualizer(skeleton_mode='line')
        l2, full = _tree('l2'), _tree('full')
        l2.id, full.id = 1, 2
        out, already = vs._process_banc_layer(
            navis.NeuronList([l2, full]), 'fast')
        assert already is False
        # Reduction is attempted for the full-res source only; the L2
        # source keeps every node untouched.
        assert calls == [(2, visualize_skeleton.BANC_LINE_FULL_NODE_REDUCTION)]
        assert {n.id: len(n.nodes) for n in out} == {1: 4, 2: 4}

    def test_faces_l2_kept_full_decimated(self, monkeypatch):
        """Plan item B: L2 tubes skip the face-decimation stage entirely;
        full-resolution sources are decimated to the slider target with the
        4,000-face safety floor."""
        dense = trimesh.creation.icosphere(subdivisions=4)  # 5120 faces
        assert len(dense.faces) == 5120
        decimate_calls = []
        monkeypatch.setattr(
            visualize_skeleton, 'simplify_skeleton_nodes',
            lambda n, factor: (n, {'raw_nodes': 4, 'achieved_nodes': 4}))
        monkeypatch.setattr(
            navis.conversion, 'tree2meshneuron',
            lambda work, tube_points=6: navis.MeshNeuron(dense))

        vs = _make_visualizer(skeleton_mode='tube')
        vs._vprint = lambda *a, **k: None
        vs._simplify_mesh_fafb_fine = (
            lambda mesh, target: decimate_calls.append(target) or mesh)

        l2, full = _tree('l2'), _tree('full')
        l2.id, full.id = 1, 2
        out, already = vs._process_banc_layer(
            navis.NeuronList([l2, full]), 'fast')
        assert already is True
        # L2: untouched 5120-face tube. Full: one decimation call with the
        # floored target — max(max(100, int(5120 * 0.05)), 4000) = 4000.
        by_id = {n.id: n for n in out}
        assert len(by_id[1].faces) == 5120
        assert decimate_calls == [4000]
        assert len(by_id[2].faces) == 5120  # pass-through decimator


class TestBancResolver:
    def test_resolver_uses_bucket_and_raw_cache(self, tmp_path, monkeypatch):
        import banc_public_data

        fetched = []

        def fake_fetch(dataset, body_id, resolution='l2', project_root=None,
                       use_cache=True):
            fetched.append((dataset, str(body_id), resolution, use_cache))
            return _tree('l2')

        monkeypatch.setattr(banc_public_data, "fetch_banc_swc", fake_fetch)

        vs = _make_visualizer()
        vs.script_path = str(tmp_path)
        vs._load_api_cached_skeletons = (
            lambda ids: ({}, list(ids)))

        sources, skeleton_cache, mesh_cache = vs._resolve_banc_sources(
            [12345], use_cache=True)
        assert mesh_cache == {}
        assert sources == {'12345': 'banc_gcs'}
        assert '12345' in skeleton_cache
        assert fetched == [('banc_v888', '12345', 'l2', True)]

    def test_resolver_raw_cache_wins(self, tmp_path, monkeypatch):
        import banc_public_data

        def fail_fetch(*a, **k):
            raise AssertionError("bucket fetch should not run on cache hit")

        monkeypatch.setattr(banc_public_data, "fetch_banc_swc", fail_fetch)

        vs = _make_visualizer()
        vs.script_path = str(tmp_path)
        cached = _tree()
        vs._load_api_cached_skeletons = (
            lambda ids: ({'12345': cached}, []))

        sources, skeleton_cache, _ = vs._resolve_banc_sources([12345])
        assert sources == {'12345': 'raw_cache'}
        assert skeleton_cache['12345'] is cached

    def test_resolver_restores_resolution_provenance(self, tmp_path,
                                                     monkeypatch):
        """Plan item R2: cache reloads must recover the resolution.

        The shared raw cache strips in-memory attributes; the resolver
        restores them from the provenance header ('banc_gcs_full') so
        full-resolution neurons keep their pipeline stages.
        """
        import banc_public_data

        monkeypatch.setattr(
            banc_public_data, "fetch_banc_swc",
            lambda *a, **k: pytest.fail(
                "cache hits must not reach the bucket"))

        vs = _make_visualizer()
        vs.script_path = str(tmp_path)
        cached_full = _tree()   # no in-memory resolution attribute
        cached_full._drocat_source = 'banc_gcs_full'
        cached_l2 = _tree()
        cached_l2._drocat_source = 'banc_gcs_l2'
        cached_legacy = _tree()  # headerless legacy entry
        vs._load_api_cached_skeletons = lambda ids: (
            {'11': cached_full, '22': cached_l2, '33': cached_legacy}, [])

        _, skeleton_cache, _ = vs._resolve_banc_sources(
            [11, 22, 33], use_cache=True)
        assert skeleton_cache['11']._drocat_banc_resolution == 'full'
        assert skeleton_cache['22']._drocat_banc_resolution == 'l2'
        assert skeleton_cache['33']._drocat_banc_resolution == 'l2'

    def test_missing_skeleton_is_skipped(self, tmp_path, monkeypatch):
        import banc_public_data

        monkeypatch.setattr(
            banc_public_data, "fetch_banc_swc",
            lambda *a, **k: None)
        vs = _make_visualizer()
        vs.script_path = str(tmp_path)
        vs._load_api_cached_skeletons = lambda ids: ({}, list(ids))
        sources, skeleton_cache, _ = vs._resolve_banc_sources([999])
        assert sources == {} and skeleton_cache == {}


class TestBancOutlineProducts:
    """E4: the whole-CNS outline is cached as render (~100k faces) + full."""

    def _make_vs(self):
        vs = _make_visualizer()
        vs.script_path = str(PROJECT_ROOT)
        return vs

    def test_render_full_product_split(self, tmp_path, monkeypatch):
        import types

        vs = self._make_vs()
        # Synthetic 300k-face "outline" on a small vertex ring.
        n_faces = 300_000
        verts = np.array([[0, 0, 0], [1000, 0, 0], [0, 1000, 0]], dtype=float)
        rng = np.random.default_rng(0)
        faces = rng.integers(0, 3, size=(n_faces, 3))
        mesh = types.SimpleNamespace(vertices=verts, faces=faces)

        fake_cv_mod = types.ModuleType("cloudvolume")

        def _fake_get(segid, _mesh=mesh):
            return _mesh

        class FakeMesh:
            get = staticmethod(_fake_get)

        class FakeCV:
            def __init__(self, url, **kwargs):
                pass

            mesh = FakeMesh()

        fake_cv_mod.CloudVolume = FakeCV
        monkeypatch.setitem(sys.modules, "cloudvolume", fake_cv_mod)
        import trimesh
        small = trimesh.Trimesh(
            vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]],
            faces=[[0, 1, 2], [1, 3, 2], [0, 2, 1], [1, 2, 3]])
        monkeypatch.setattr(
            vs, "_simplify_mesh_open3d", lambda tm, target: small)
        # script_path must point at tmp so the cache writes stay isolated.
        vs.script_path = str(tmp_path)

        volume = vs._get_banc_region_volume("BANC_outline", segid=1)
        assert volume is not None
        assert len(volume.faces) == 4
        # Pair written: render product under the standard name, full next to
        # it — both under their PLAIN names (not the per-lowercase encoding).
        import json as _json
        import os
        mesh_dir = os.path.join(str(tmp_path), "cache", "banc_v888", "meshes")
        std = vs._banc_mesh_file_path(mesh_dir, "BANC_outline")
        full = vs._banc_mesh_file_path(mesh_dir, "BANC_outline_full")
        assert std.endswith("BANC_outline.json")
        assert full.endswith("BANC_outline_full.json")
        assert os.path.exists(std)
        assert os.path.exists(full)
        std_faces = len(_json.load(open(std))["faces"])
        full_faces = len(_json.load(open(full))["faces"])
        assert std_faces == 4 and full_faces == n_faces

    def test_self_heal_oversized_cached_outline(self, tmp_path, monkeypatch):
        import json as _json
        import os
        import types

        vs = self._make_vs()
        vs.script_path = str(tmp_path)
        # Pre-seed an oversized standard product (pre-E4 cache), under the
        # old per-lowercase-encoded file name.
        mesh_dir = vs._get_dataset_mesh_dir()
        os.makedirs(mesh_dir, exist_ok=True)
        rng = np.random.default_rng(1)
        verts = np.array([[0, 0, 0], [1000, 0, 0], [0, 1000, 0]], dtype=float)
        faces = rng.integers(0, 3, size=(200_000, 3))
        std = vs._banc_mesh_file_path(mesh_dir, "BANC_outline")
        with open(std, "w") as handle:
            _json.dump({"vertices": verts.tolist(), "faces": faces.tolist()}, handle)

        mesh = types.SimpleNamespace(vertices=verts, faces=faces)
        fake_cv_mod = types.ModuleType("cloudvolume")

        import trimesh
        small = trimesh.Trimesh(
            vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]],
            faces=[[0, 1, 2], [1, 3, 2], [0, 2, 1], [1, 2, 3]])

        def _fake_get(segid, _mesh=mesh):
            return _mesh

        class FakeMesh:
            get = staticmethod(_fake_get)

        class FakeCV:
            def __init__(self, url, **kwargs):
                pass

            mesh = FakeMesh()

        fake_cv_mod.CloudVolume = FakeCV
        monkeypatch.setitem(sys.modules, "cloudvolume", fake_cv_mod)
        monkeypatch.setattr(
            vs, "_simplify_mesh_open3d", lambda tm, target: small)

        volume = vs._get_banc_region_volume("BANC_outline", segid=1)
        # Self-healed: standard product decimated, full copy retained.
        assert len(volume.faces) == 4
        full = vs._banc_mesh_file_path(mesh_dir, "BANC_outline_full")
        assert os.path.exists(full)

    def test_clean_names_and_legacy_migration(self, tmp_path):
        """Aggregate products cache under their plain names (not the
        per-lowercase encoding), and cache files written with the old
        encoding are renamed in place on first load."""
        import json as _json
        import os

        vs = self._make_vs()
        vs.script_path = str(tmp_path)
        mesh_dir = vs._get_dataset_mesh_dir()
        os.makedirs(mesh_dir, exist_ok=True)
        payload = {"vertices": [[0, 0, 0]], "faces": [[0, 0, 0]]}

        legacy = os.path.join(
            mesh_dir, vs._roi_to_filename("BANC_neuropil"))
        with open(legacy, "w") as handle:
            _json.dump(payload, handle)
        clean = os.path.join(mesh_dir, "BANC_neuropil.json")
        assert not os.path.exists(clean)

        loaded = vs._load_banc_region_mesh_json("BANC_neuropil")
        assert loaded is not None
        assert os.path.exists(clean) and not os.path.exists(legacy)

        # Non-aggregate regions keep the generic case-safe encoding
        # (lowercase letters stay prefixed, e.g. for 'aL(L)' vs 'AL(L)').
        fine = vs._banc_mesh_file_path(mesh_dir, "aL(L)")
        assert os.path.basename(fine) == "_aL(L).json"


class TestSplitBancCns:
    def test_split_partitions_faces_at_neck(self, tmp_path):
        import navis
        import numpy as np
        from banc_public_data import BANC_BRAIN_VNC_Y_CUTOFF

        vs = _make_visualizer()
        vs._vprint = lambda *a, **k: None
        verts = np.array([
            [0.0, 0.0, 0.0],        # brain-side vertex
            [10.0, float(BANC_BRAIN_VNC_Y_CUTOFF - 1), 0.0],
            [20.0, float(BANC_BRAIN_VNC_Y_CUTOFF + 1), 0.0],
            [30.0, float(BANC_BRAIN_VNC_Y_CUTOFF + 2), 5.0],
        ])
        faces = np.array([
            [0, 1, 0],  # brain face (all y < cutoff)
            [2, 3, 2],  # VNC face (all y >= cutoff)
            [1, 2, 1],  # crossing face -> assigned by centroid
        ], dtype=np.int64)
        import trimesh
        volume = navis.Volume(trimesh.Trimesh(vertices=verts, faces=faces))
        brain, vnc = vs._split_banc_cns_volume(volume)
        # Centroid assignment: nothing is dropped (no crack), and the
        # crossing face lands on the brain side (its centroid is anterior).
        assert len(brain.faces) == 2 and len(vnc.faces) == 1
        assert len(brain.faces) + len(vnc.faces) == len(faces)

    def _two_sided_volume(self):
        """Small outline with faces strictly on both sides of the neck."""
        import navis
        import numpy as np
        from banc_public_data import BANC_BRAIN_VNC_Y_CUTOFF

        verts = np.array([
            [0.0, float(BANC_BRAIN_VNC_Y_CUTOFF - 10), 0.0],
            [5.0, float(BANC_BRAIN_VNC_Y_CUTOFF - 10), 5.0],
            [10.0, float(BANC_BRAIN_VNC_Y_CUTOFF + 10), 0.0],
            [15.0, float(BANC_BRAIN_VNC_Y_CUTOFF + 10), 5.0],
        ])
        faces = np.array([[0, 1, 0], [2, 3, 2]], dtype=np.int64)
        return navis.Volume(trimesh.Trimesh(vertices=verts, faces=faces))

    def test_template_info_names_brain_portion(self):
        """Plan item E2: template mode is labelled '(brain)'."""
        vs = _make_visualizer()
        vs.brain_mesh = 'template'
        vs._get_banc_template_volume = self._two_sided_volume
        info = vs._get_template_info()
        assert info['mesh_name'] == 'BANC (brain)'

    def test_whole_info_keeps_cns_outline_name(self):
        vs = _make_visualizer()
        vs._vprint = lambda *a, **k: None
        vs.brain_mesh = 'whole'
        vs._get_banc_template_volume = self._two_sided_volume
        info = vs._get_template_info()
        assert info['mesh_name'] == 'BANC (CNS outline)'

    def test_vnc_template_info_names_vnc_portion(self):
        """Plan item E3: the VNC branch is labelled '(VNC)' and carries
        only the posterior faces."""
        import numpy as np

        vs = _make_visualizer()
        vs._vprint = lambda *a, **k: None
        vs._get_banc_template_volume = self._two_sided_volume
        info = vs._get_vnc_template_info()
        assert info['mesh_name'] == 'BANC (VNC)'
        assert len(info['mesh'].faces) == 1


class TestOverlayDropWarning:
    """Plan item R6: overlays between spaces without a bridge are dropped
    with an explicit warning naming the dropped bodies."""

    def test_no_bridge_warning_lists_body_ids(self, capsys):
        from types import SimpleNamespace
        import flybrains  # noqa: F401 - the production transform registry

        flybrains.register_transforms()
        # 'NOT_A_TEMPLATE_SPACE' can never be registered, so the lookup
        # deterministically raises and takes the drop-with-warning path.
        neurons = [SimpleNamespace(id=111), SimpleNamespace(id=222)]
        out = visualize_skeleton.transform_neurons_to_space(
            neurons, 'FLYWIRE', 'NOT_A_TEMPLATE_SPACE')
        assert out == []
        output = capsys.readouterr().out
        assert '2 overlay neuron(s) dropped' in output
        assert '111' in output and '222' in output


class TestResolutionKnob:
    def test_resolver_uses_unified_chain(self, tmp_path, monkeypatch):
        """Source selection is removed: the resolver never forwards a
        resolution preference to the fetcher (unified L2->full->pcg)."""
        import banc_public_data

        seen = {}

        def fake_fetch(dataset, body_id, project_root=None,
                       use_cache=True, *args, **kwargs):
            seen['args'] = (dataset, str(body_id))
            seen['resolution_forwarded'] = (
                len(args) >= 1 or 'resolution' in kwargs)
            return _tree()

        monkeypatch.setattr(banc_public_data, "fetch_banc_swc", fake_fetch)
        vs = _make_visualizer()
        vs.banc_skeleton_resolution = 'full'  # deprecated/ignored
        vs.script_path = str(tmp_path)
        vs._load_api_cached_skeletons = lambda ids: ({}, list(ids))
        vs._resolve_banc_sources([1], resolution='full')
        assert seen['args'] == ('banc_v888', '1')
        assert seen['resolution_forwarded'] is False

    def test_fetch_rejects_invalid_resolution(self, tmp_path):
        import banc_public_data

        with pytest.raises(ValueError):
            banc_public_data.fetch_banc_swc(
                "banc_v888", "1", resolution="coarse", project_root=tmp_path)
