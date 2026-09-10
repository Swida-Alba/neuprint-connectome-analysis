"""FAFB skeleton denoising (twig pruning) in cave_data_fetcher.

`fetch_skeleton`/`fetch_skeletons` prune terminal twigs shorter than the
denoise threshold (default 10 µm) after skeletonization — and also on cache
hits, so cached skeletons return denoised consistently.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import cave_data_fetcher as cdf  # noqa: E402


def mesh_neuron():
    """A small surface mesh suitable for cache and type-contract tests."""
    import navis

    vertices = np.array([
        (0.0, 0.0, 0.0),
        (100.0, 0.0, 0.0),
        (0.0, 100.0, 0.0),
        (0.0, 0.0, 100.0),
    ])
    faces = np.array([
        (0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3),
    ], dtype=np.int64)
    return navis.MeshNeuron({"vertices": vertices, "faces": faces})


def test_cache_disabled_does_not_create_api_cache_directories(tmp_path):
    fetcher = cdf.CAVEDataFetcher(
        dataset="flywire_FAFB_v783",
        cave_token="test-token",
        cache_enabled=False,
        project_root=str(tmp_path),
        verbose=False,
    )

    assert not (tmp_path / "cache").exists()
    assert Path(fetcher.get_cache_path()).as_posix().endswith(
        "cache/flywire_FAFB_v783/API_cache")


def test_fafb_mesh_fetch_disables_unsupported_draco_deduplication(
        tmp_path, capsys):
    """FAFB Graphene fetches must not emit CloudVolume's no-op warning."""
    calls = []
    warning = (
        "Warning: deduplication not currently supported for this layer's "
        "variable layered draco meshes"
    )

    class FakeMeshSource:
        def get(self, body_id, **kwargs):
            calls.append((body_id, kwargs))
            if kwargs.get("deduplicate_chunk_boundaries", True):
                print(warning)
            return {body_id: mesh_neuron().trimesh}

    class FakeCloudVolume:
        mesh = FakeMeshSource()

    fetcher = cdf.CAVEDataFetcher(
        dataset="flywire_FAFB_v783",
        cave_token="test-token",
        cache_enabled=False,
        project_root=str(tmp_path),
        verbose=False,
        _cv=FakeCloudVolume(),
    )

    result = fetcher.fetch_mesh(42)

    assert result is not None
    assert calls == [(42, {"deduplicate_chunk_boundaries": False})]
    output = capsys.readouterr()
    assert warning not in output.out
    assert warning not in output.err


def test_banc_cache_namespace_keeps_requested_release(tmp_path):
    fetcher = cdf.CAVEDataFetcher(
        dataset="banc_v888",
        cave_token="test-token",
        project_root=str(tmp_path),
        verbose=False,
    )

    assert Path(fetcher.get_cache_path()).as_posix().endswith(
        "cache/banc_v888/API_cache"
    )
    assert Path(
        fetcher._get_skeleton_cache_path("72057594037927937")).as_posix().endswith(
        "cache/banc_v888/skeletons/cave_skeletons/"
        "72057594037927937.swc.zst"
    )


def test_use_cache_false_is_online_only_even_when_cache_enabled(
        tmp_path, monkeypatch):
    """Per-call ``use_cache=False`` must not read or write API skeletons."""
    fetcher = cdf.CAVEDataFetcher(
        dataset="flywire_FAFB_v783",
        cave_token="test-token",
        cache_enabled=True,
        project_root=str(tmp_path),
        verbose=False,
    )
    monkeypatch.setattr(
        fetcher, "_load_from_cache",
        lambda _path: pytest.fail("online-only fetch inspected the cache"),
    )
    monkeypatch.setattr(
        fetcher, "fetch_mesh",
        lambda _body_id, use_cache=False: object(),
    )
    monkeypatch.setattr(
        cdf.navis, "skeletonize",
        lambda *_args, **_kwargs: twiggy_neuron(),
    )

    skeleton = fetcher.fetch_skeleton(
        42, use_cache=False, simplify_mesh=0.0, denoise_twigs=None
    )

    assert skeleton is not None
    assert not (tmp_path / "cache").exists()


def test_skeleton_cache_roundtrip_is_written_only_for_cache_enabled_call(
        tmp_path, monkeypatch):
    """An opted-in fetch writes raw compressed SWC for reuse."""
    fetcher = cdf.CAVEDataFetcher(
        dataset="flywire_FAFB_v783",
        cave_token="test-token",
        cache_enabled=True,
        project_root=str(tmp_path),
        verbose=False,
    )
    monkeypatch.setattr(
        fetcher, "fetch_mesh",
        lambda _body_id, use_cache=False: object(),
    )
    monkeypatch.setattr(
        cdf.navis, "skeletonize",
        lambda *_args, **_kwargs: twiggy_neuron(),
    )

    first = fetcher.fetch_skeleton(
        42, use_cache=True, simplify_mesh=0.0, denoise_twigs=None
    )
    cache_path = tmp_path / "cache" / "flywire_FAFB_v783" \
        / "skeletons" / "cave_skeletons" / "42.swc.zst"
    assert first is not None and cache_path.exists()
    # Replacements carry provenance: they are CAVE-derived trees, never
    # healed-bundle mirrors.
    import zstandard as zstd
    with open(cache_path, "rb") as handle:
        with zstd.ZstdDecompressor().stream_reader(handle) as reader:
            head = reader.read(512).decode("utf-8", "replace").splitlines()
    assert "# DROCAT simpl: 0" in head
    assert "# DROCAT source: cave_mesh_wavefront" in head

    monkeypatch.setattr(
        fetcher, "fetch_mesh",
        lambda *_args, **_kwargs: pytest.fail("cache hit fetched the mesh"),
    )
    second = fetcher.fetch_skeleton(
        42, use_cache=True, simplify_mesh=0.0, denoise_twigs=None
    )
    assert second is not None
    assert len(second.nodes) == len(first.nodes)


def test_fafb_mesh_cache_roundtrip_is_mesh_native_and_never_skeletonizes(
        tmp_path, monkeypatch):
    """CAVE FAFB preparation writes/reads a dedicated MeshNeuron cache."""
    fetcher = cdf.CAVEDataFetcher(
        dataset="flywire_FAFB_v783",
        cave_token="test-token",
        cache_enabled=True,
        project_root=str(tmp_path),
        verbose=False,
    )
    raw_mesh = mesh_neuron()
    monkeypatch.setattr(
        fetcher, "fetch_mesh",
        lambda _body_id, use_cache=False: raw_mesh,
    )
    monkeypatch.setattr(
        cdf.navis,
        "skeletonize",
        lambda *_args, **_kwargs: pytest.fail(
            "FAFB mesh preparation must not skeletonize"),
    )

    first = fetcher.fetch_fafb_mesh(
        42, use_cache=True, soma_pos=[0.0, 0.0, 0.0])
    import navis
    assert isinstance(first, navis.MeshNeuron)
    cache_path = (tmp_path / "cache" / "flywire_FAFB_v783" / "meshes"
                  / "FLYWIRE_simp95_soma80_r20" / "42.pkl.zst")
    assert cache_path.exists()
    assert not cache_path.with_suffix("").with_suffix(".pkl").exists()
    assert not (tmp_path / "cache" / "flywire_FAFB_v783" / "skeletons"
                / "raw_skeletons" / "42.swc.gz").exists()

    monkeypatch.setattr(
        fetcher, "fetch_mesh",
        lambda *_args, **_kwargs: pytest.fail("cache hit fetched the mesh"),
    )
    second = fetcher.fetch_fafb_mesh(
        42, use_cache=True, soma_pos=[0.0, 0.0, 0.0])
    assert isinstance(second, navis.MeshNeuron)
    assert str(second.id) == "42"


def test_fafb_mesh_use_cache_false_is_online_only(tmp_path, monkeypatch):
    """An online-only FAFB mesh fetch does not touch the local mesh cache."""
    fetcher = cdf.CAVEDataFetcher(
        dataset="flywire_FAFB_v783",
        cave_token="test-token",
        cache_enabled=True,
        project_root=str(tmp_path),
        verbose=False,
    )
    monkeypatch.setattr(
        fetcher, "fetch_mesh",
        lambda _body_id, use_cache=False: mesh_neuron(),
    )

    result = fetcher.fetch_fafb_mesh(42, use_cache=False)

    import navis
    assert isinstance(result, navis.MeshNeuron)
    assert not (tmp_path / "cache").exists()


def test_fafb_mesh_force_refresh_bypasses_cache_and_rewrites_prepared_mesh(
        tmp_path, monkeypatch):
    """Extrusion repair must replace, not reuse, a prepared cache entry."""
    fetcher = cdf.CAVEDataFetcher(
        dataset="flywire_FAFB_v783",
        cave_token="test-token",
        cache_enabled=True,
        project_root=str(tmp_path),
        verbose=False,
    )
    calls = []

    def fetch_raw(body_id, use_cache=False):
        calls.append((body_id, use_cache))
        refreshed = mesh_neuron()
        refreshed.id = body_id
        return refreshed

    monkeypatch.setattr(fetcher, "fetch_mesh", fetch_raw)

    first = fetcher.fetch_fafb_mesh(
        42, use_cache=True, soma_pos=[0.0, 0.0, 0.0])
    assert first is not None
    calls.clear()

    saved = []
    original_save = cdf.FlyWireMeshCache.save

    def record_save(cache, meshes):
        saved.append(meshes)
        return original_save(cache, meshes)

    monkeypatch.setattr(cdf.FlyWireMeshCache, "save", record_save)
    monkeypatch.setattr(
        cdf.FlyWireMeshCache,
        "load",
        lambda *_args, **_kwargs: pytest.fail(
            "force_refresh must bypass the prepared mesh cache read"),
    )

    refreshed = fetcher.fetch_fafb_mesh(
        42,
        use_cache=True,
        force_refresh=True,
        soma_pos=[0.0, 0.0, 0.0],
    )

    assert refreshed is not None
    assert calls == [(42, False)]
    assert saved and 42 in saved[0]


def twiggy_neuron():
    """A long neurite (200 µm) with a short 2 µm terminal twig."""
    import navis
    pts = [(i * 1000.0, 0.0, 0.0) for i in range(201)]
    parents = [-1] + list(range(200))
    # twig: 2 extra nodes at 1 µm spacing off node 50
    pts += [(50.0 * 1000.0, 1000.0, 0.0), (50.0 * 1000.0, 2000.0, 0.0)]
    parents += [50, 201]
    n = len(pts)
    nodes = pd.DataFrame({
        "node_id": np.arange(n, dtype=np.int64),
        "parent_id": np.array(parents, dtype=np.int64),
        "x": [p[0] for p in pts],
        "y": [p[1] for p in pts],
        "z": [p[2] for p in pts],
        "radius": np.ones(n),
        "type": ["slab"] * n,
    })
    nodes.loc[0, "type"] = "root"
    nodes.loc[200, "type"] = "end"
    nodes.loc[201, "type"] = "end"
    nodes.loc[202, "type"] = "end"
    nrn = navis.TreeNeuron(nodes)
    nrn.soma = None
    nrn.units = "nm"
    return nrn


class TestDenoiseSkeleton:
    def test_prune_twigs_removes_short_twig(self):
        nrn = twiggy_neuron()
        assert len(nrn.nodes) == 203
        out = cdf.CAVEDataFetcher._denoise_skeleton(nrn, 10000.0)  # 10 µm
        # the 2 µm twig is pruned; the 200 µm neurite survives
        assert len(out.nodes) < len(nrn.nodes)
        assert len(out.nodes) <= 201
        # the original is untouched (pruning is not in-place)
        assert len(nrn.nodes) == 203

    def test_fetch_skeleton_denoises_cache_hits(self, monkeypatch):
        """Denoising is transient and does not alter the raw cache source."""
        fetcher = object.__new__(cdf.CAVEDataFetcher)
        fetcher.verbose = False
        monkeypatch.setattr(fetcher, "_get_skeleton_cache_path",
                            lambda bid: "/tmp/x.pkl")
        monkeypatch.setattr(fetcher, "_load_from_cache",
                            lambda path: twiggy_neuron())
        # denoise_twigs=None keeps the cached skeleton as-is
        raw = fetcher.fetch_skeleton(42, use_cache=True, denoise_twigs=None)
        assert len(raw.nodes) == 203
        # An explicit 10 µm request prunes the twig on a cache hit.
        den = fetcher.fetch_skeleton(
            42, use_cache=True, denoise_twigs=10000.0)
        assert len(den.nodes) < len(raw.nodes)

    def test_fetch_skeleton_caches_denoised(self, monkeypatch):
        """A fresh fetch caches raw data before transient denoising."""
        import pickle
        import navis
        fetcher = object.__new__(cdf.CAVEDataFetcher)
        fetcher.verbose = False
        saved = {}

        mesh_pkl = (PROJECT_ROOT / "cache" / "flywire_FAFB_v783" / "skeletons"
                    / "FLYWIRE_simp95_soma80_r20" / "720575940597856265.pkl")
        if not mesh_pkl.exists():
            pytest.skip("FAFB mesh cache missing")
        with open(mesh_pkl, "rb") as f:
            real_mesh = pickle.load(f)

        def fake_fetch_mesh(bid, use_cache=False):
            return navis.MeshNeuron(real_mesh.trimesh,
                                    id=bid, name=str(bid), units="nm")

        monkeypatch.setattr(fetcher, "fetch_mesh", fake_fetch_mesh)
        monkeypatch.setattr(fetcher, "_get_skeleton_cache_path",
                            lambda bid: f"/tmp/{bid}.pkl")
        monkeypatch.setattr(fetcher, "_load_from_cache", lambda path: None)

        def fake_save(neuron, path):
            saved[path] = len(neuron.nodes)

        monkeypatch.setattr(fetcher, "_save_to_cache", fake_save)
        skel = fetcher.fetch_skeleton(42, use_cache=True, denoise_twigs=10000.0)
        assert skel is not None
        assert "/tmp/42.pkl" in saved
        assert saved["/tmp/42.pkl"] >= len(skel.nodes)


def test_extrusion_fix_store_roundtrip_with_provenance(tmp_path):
    """Locally pruned fixes persist to ``extrusion_fixes`` with an explicit
    ``local_extrusion_fix`` provenance header and reload network-free."""
    import zstandard as zstd

    fetcher = cdf.CAVEDataFetcher(
        dataset="flywire_FAFB_v783",
        cave_token="",
        cache_enabled=True,
        project_root=str(tmp_path),
        verbose=False,
    )
    fix = twiggy_neuron()
    assert fetcher.save_extrusion_fix_skeleton(42, fix) is True

    fix_path = tmp_path / "cache" / "flywire_FAFB_v783" / "skeletons" \
        / "extrusion_fixes" / "42.swc.zst"
    assert fix_path.exists()
    with open(fix_path, "rb") as handle:
        with zstd.ZstdDecompressor().stream_reader(handle) as reader:
            head = reader.read(512).decode("utf-8", "replace").splitlines()
    assert "# DROCAT simpl: 0" in head
    assert "# DROCAT source: local_extrusion_fix" in head

    loaded = fetcher.load_cached_fix_skeleton("42")
    assert loaded is not None
    assert len(loaded.nodes) == len(fix.nodes)
    assert loaded._drocat_source == "local_extrusion_fix"
    # the CAVE replacement store stays untouched by the fix store
    assert fetcher.load_cached_skeleton(42) is None
    assert not (tmp_path / "cache" / "flywire_FAFB_v783" / "skeletons"
                / "cave_skeletons").exists()


def test_load_repaired_skeletons_prefers_cave_then_fix_store(tmp_path):
    """``load_repaired_skeletons`` serves ``api_repaired`` bodies from the
    ``cave_skeletons`` store and ``local_fallback`` bodies from
    ``cave_skeletons`` first, then ``extrusion_fixes``; unknown and failed
    statuses stay out so callers run the repair chain for them."""
    from fafb_utils import (
        EXTRUSION_REPAIR_API_FAILED,
        EXTRUSION_REPAIR_API_REPAIRED,
        EXTRUSION_REPAIR_LOCAL_FALLBACK,
        set_extrusion_repair_status,
    )

    fetcher = cdf.CAVEDataFetcher(
        dataset="flywire_FAFB_v783",
        cave_token="",
        cache_enabled=True,
        project_root=str(tmp_path),
        verbose=False,
    )
    assert fetcher.save_extrusion_fix_skeleton(7, twiggy_neuron())

    set_extrusion_repair_status(str(tmp_path), "flywire_FAFB_v783", {
        "5": EXTRUSION_REPAIR_API_REPAIRED,     # store miss -> unresolved
        "6": EXTRUSION_REPAIR_API_FAILED,       # retryable -> unresolved
        "7": EXTRUSION_REPAIR_LOCAL_FALLBACK,   # fix store hit
        "8": EXTRUSION_REPAIR_LOCAL_FALLBACK,   # no store entry -> unresolved
    })

    out = cdf.load_repaired_skeletons(
        "flywire_FAFB_v783", [5, 6, 7, 8, 9],
        project_root=str(tmp_path))
    assert set(out) == {7}
    assert out[7]._drocat_source == "local_extrusion_fix"

    # a cave_skeletons entry wins over the persisted fix for the same body
    fetcher._save_cave_skeleton(
        twiggy_neuron(),
        fetcher._get_skeleton_cache_path(cdf.body_id_to_api_int(7)))
    served = cdf.load_repaired_skeletons(
        "flywire_FAFB_v783", [7], project_root=str(tmp_path))
    assert set(served) == {7}
    assert served[7]._drocat_source == "cave_mesh_wavefront"

    # BANC never reaches the repair caches
    assert cdf.load_repaired_skeletons(
        "banc_public_v888", [7], project_root=str(tmp_path)) == {}
