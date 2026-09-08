"""Coverage-focused tests for fafb_utils, flywire_mesh_cache, flywire_ids
and connection_map.

Targets the specific uncovered branches from coverage_missing.txt.  All
filesystem access is confined to pytest tmp_path; optional heavy backends
(open3d, concurrent.futures process pools) are faked through sys.modules
injection so no real multiprocessing or network access occurs.
"""

import gzip
import io
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import navis  # noqa: E402
import trimesh  # noqa: E402

import connection_map as cmap  # noqa: E402
import fafb_utils as fau  # noqa: E402
import flywire_ids as fid  # noqa: E402
import flywire_mesh_cache as fmc  # noqa: E402


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

def make_chain_neuron(n=10, spike_from=None, spike_length=250000.0,
                      spike_child=False):
    """A simple chain TreeNeuron; optionally one far-away spike subtree."""
    rows = []
    for i in range(1, n + 1):
        rows.append({'node_id': i, 'parent_id': -1 if i == 1 else i - 1,
                     'x': float(i * 1000), 'y': 0.0, 'z': 0.0,
                     'radius': 5.0, 'type': 3})
    if spike_from is not None:
        spike_x = float(spike_from * 1000 + spike_length)
        rows.append({'node_id': n + 1, 'parent_id': spike_from,
                     'x': spike_x, 'y': 0.0, 'z': 0.0,
                     'radius': 5.0, 'type': 3})
        if spike_child:
            rows.append({'node_id': n + 2, 'parent_id': n + 1,
                         'x': spike_x + 1000.0, 'y': 0.0, 'z': 0.0,
                         'radius': 5.0, 'type': 3})
    return navis.TreeNeuron(pd.DataFrame(rows))


def box_trimesh(size=1000.0):
    return trimesh.creation.box(extents=(size, size, size))


def mesh_neuron(tm=None, body_id=1):
    return navis.MeshNeuron(tm if tm is not None else box_trimesh(),
                            id=body_id)


# ===========================================================================
# fafb_utils
# ===========================================================================

class TestExtractGz:
    def test_roundtrip(self, tmp_path):
        src = tmp_path / 'data.csv.gz'
        src.write_bytes(gzip.compress(b'col\n1\n'))
        out = tmp_path / 'data.csv'
        fau.extract_gz(str(src), str(out))
        assert out.read_text() == 'col\n1\n'


class TestPrepareFlywireData:
    def test_merged_files_short_circuit(self, tmp_path):
        data = tmp_path / 'flywire_FAFB_v783'
        data.mkdir()
        conn = data / 'flywire_FAFB_v783_merged_connections.parquet'
        neurons = data / 'flywire_FAFB_v783_allneurons_neuron_df.parquet'
        conn.write_bytes(b'x')
        neurons.write_bytes(b'x')
        assert fau.prepare_flywire_data(str(data)) == (str(neurons),
                                                       str(conn))

    def test_banc_missing_raises(self, tmp_path):
        data = tmp_path / 'banc_v888'
        data.mkdir()
        with pytest.raises(FileNotFoundError, match="BANC"):
            fau.prepare_flywire_data(str(data))

    def test_missing_everything_raises(self, tmp_path):
        data = tmp_path / 'flywire_FAFB_v783'
        data.mkdir()
        with pytest.raises(FileNotFoundError, match="Required data files"):
            fau.prepare_flywire_data(str(data))

    def test_legacy_gz_generation(self, tmp_path):
        data = tmp_path / 'flywire_FAFB_v783'
        downloads = data / 'downloads'
        downloads.mkdir(parents=True)
        conn_csv = ("pre_root_id,post_root_id,syn_count\n"
                    "100,200,3\n100,300,2\n200,100,1\n")
        types_csv = "root_id,primary_type\n100,T1\n200,T2\n300,T3\n"
        (downloads / 'connections_princeton_no_threshold.csv.gz').write_bytes(
            gzip.compress(conn_csv.encode()))
        (downloads / 'consolidated_cell_types.csv.gz').write_bytes(
            gzip.compress(types_csv.encode()))

        neuron_path, conn_path = fau.prepare_flywire_data(str(data))
        assert Path(conn_path).exists()
        df = pd.read_csv(neuron_path, dtype={'bodyId': str})
        row = df[df['bodyId'] == '100'].iloc[0]
        assert row['pre'] == 5 and row['post'] == 1
        assert row['type'] == 'T1'

        # second call hits the "outputs already exist" fast path
        again = fau.prepare_flywire_data(str(data))
        assert again == (neuron_path, conn_path)

    def test_uncompressed_files_present(self, tmp_path):
        data = tmp_path / 'flywire_FAFB_v783'
        downloads = data / 'downloads'
        downloads.mkdir(parents=True)
        conn_csv = ("pre_root_id,post_root_id,syn_count\n"
                    "100,200,3\n")
        types_csv = "root_id,primary_type\n100,T1\n200,T2\n"
        (downloads / 'connections_princeton_no_threshold.csv').write_text(
            conn_csv)
        (downloads / 'consolidated_cell_types.csv').write_text(types_csv)
        neuron_path, conn_path = fau.prepare_flywire_data(str(data))
        assert Path(neuron_path).exists()
        assert Path(conn_path).name == 'connections_princeton_no_threshold.csv'


class TestSkeletonPaths:
    def test_zip_resolution(self, tmp_path, capsys):
        data = tmp_path / 'flywire_FAFB_v783'
        data.mkdir()
        assert fau.get_fafb_skeleton_zip(str(data)) is None
        assert 'Warning' in capsys.readouterr().out

        banc = tmp_path / 'banc_v888'
        banc.mkdir()
        assert fau.get_fafb_skeleton_zip(str(banc)) is None

        std = data / 'flywire_FAFB_v783_skeletons.zip'
        std.write_bytes(b'z')
        assert fau.get_fafb_skeleton_zip(str(data)) == str(std)

        std.unlink()
        original = data / 'sk_lod1_783_healed.zip'
        original.write_bytes(b'z')
        assert fau.get_fafb_skeleton_zip(str(data)) == str(original)

        original.unlink()
        downloads = data / 'downloads'
        downloads.mkdir()
        in_downloads = downloads / 'sk_lod1_783_healed.zip'
        in_downloads.write_bytes(b'z')
        assert fau.get_fafb_skeleton_zip(str(data)) == str(in_downloads)

    def test_parquet_resolution(self, tmp_path):
        data = tmp_path / 'flywire_FAFB_v783'
        data.mkdir()
        assert fau.get_fafb_skeleton_parquet(str(data)) is None
        pq = data / 'flywire_FAFB_v783_skeletons.parquet'
        pq.write_bytes(b'p')
        assert fau.get_fafb_skeleton_parquet(str(data)) == str(pq)


class TestExtrusionCache:
    FOLDER = 'flywire_FAFB_v783'

    def test_load_missing_and_corrupt(self, tmp_path):
        assert fau.load_extrusion_check_cache(str(tmp_path), self.FOLDER) == {}
        assert fau.load_extrusion_repair_status(str(tmp_path),
                                                self.FOLDER) == {}
        path = fau.extrusion_check_cache_path(str(tmp_path), self.FOLDER)
        path.parent.mkdir(parents=True)
        path.write_bytes(b'not a parquet file')
        assert fau.load_extrusion_check_cache(str(tmp_path), self.FOLDER) == {}
        assert fau.load_extrusion_repair_status(str(tmp_path),
                                                self.FOLDER) == {}

    def test_repair_status_without_status_column(self, tmp_path):
        path = fau.extrusion_check_cache_path(str(tmp_path), self.FOLDER)
        path.parent.mkdir(parents=True)
        pd.DataFrame({'bodyId': ['1', '2'],
                      'has_extrusion': [True, False]}).to_parquet(path)
        statuses = fau.load_extrusion_repair_status(str(tmp_path),
                                                    self.FOLDER)
        assert statuses == {'1': fau.EXTRUSION_REPAIR_PENDING,
                            '2': fau.EXTRUSION_REPAIR_CLEAN}

    def test_detector_metadata_invalidates_old_results_but_keeps_statuses(
            self, tmp_path):
        fau.save_extrusion_check_cache(str(tmp_path), self.FOLDER, {1: True})
        path = fau.extrusion_check_cache_path(str(tmp_path), self.FOLDER)
        frame = pd.read_parquet(path)
        frame.loc[0, 'detector_version'] = 'old-detector'
        frame.to_parquet(path, index=False)

        assert fau.load_extrusion_check_cache(
            str(tmp_path), self.FOLDER) == {}
        assert fau.load_extrusion_repair_status(
            str(tmp_path), self.FOLDER)['1'] == fau.EXTRUSION_REPAIR_PENDING

    def test_repair_status_with_na_values_and_wrong_schema(self, tmp_path):
        path = fau.extrusion_check_cache_path(str(tmp_path), self.FOLDER)
        path.parent.mkdir(parents=True)
        pd.DataFrame({
            'bodyId': ['1', '2', '3'],
            'has_extrusion': [True, False, True],
            'repair_status': [None, '', 'api_repaired'],
        }).to_parquet(path)
        statuses = fau.load_extrusion_repair_status(str(tmp_path),
                                                    self.FOLDER)
        assert statuses['1'] == fau.EXTRUSION_REPAIR_PENDING
        assert statuses['2'] == fau.EXTRUSION_REPAIR_CLEAN
        assert statuses['3'] == 'api_repaired'

        # schema without the required columns -> empty dict
        pd.DataFrame({'other': [1]}).to_parquet(path)
        assert fau.load_extrusion_repair_status(str(tmp_path),
                                                self.FOLDER) == {}

    def test_save_and_set_roundtrip(self, tmp_path):
        # empty inputs are no-ops
        fau.save_extrusion_check_cache(str(tmp_path), self.FOLDER, {})
        fau.set_extrusion_repair_status(str(tmp_path), self.FOLDER, {})
        assert not fau.extrusion_check_cache_path(
            str(tmp_path), self.FOLDER).exists()

        fau.save_extrusion_check_cache(str(tmp_path), self.FOLDER, {
            1: True,
            2: False,
            3: {'has_extrusion': True,
                'repair_status': fau.EXTRUSION_REPAIR_API_REPAIRED},
        })
        cache = fau.load_extrusion_check_cache(str(tmp_path), self.FOLDER)
        statuses = fau.load_extrusion_repair_status(str(tmp_path),
                                                    self.FOLDER)
        assert cache == {'1': True, '2': False, '3': True}
        assert statuses['1'] == fau.EXTRUSION_REPAIR_PENDING
        assert statuses['2'] == fau.EXTRUSION_REPAIR_CLEAN
        assert statuses['3'] == fau.EXTRUSION_REPAIR_API_REPAIRED

        # repair-status-only update keeps detection results and creates
        # conservative flagged rows for unknown ids
        fau.set_extrusion_repair_status(str(tmp_path), self.FOLDER, {
            '1': fau.EXTRUSION_REPAIR_API_FAILED,
            '9': fau.EXTRUSION_REPAIR_LOCAL_FALLBACK,
        })
        statuses = fau.load_extrusion_repair_status(str(tmp_path),
                                                    self.FOLDER)
        cache = fau.load_extrusion_check_cache(str(tmp_path), self.FOLDER)
        assert statuses['1'] == fau.EXTRUSION_REPAIR_API_FAILED
        assert statuses['9'] == fau.EXTRUSION_REPAIR_LOCAL_FALLBACK
        assert cache['9'] is True


class TestDiagnoseExtrusion:
    def test_missing_nodes_or_columns(self):
        result = fau.diagnose_extrusion_nodes(object())
        assert result['detected'] is False
        assert result['median_edge_length'] == 0.0

        class NoColumns:
            nodes = pd.DataFrame({'x': [0.0]})

        assert fau.diagnose_extrusion_nodes(NoColumns())['detected'] is False

    def test_no_edges(self):
        neuron = make_chain_neuron(n=1)  # single root node, no edges
        result = fau.diagnose_extrusion_nodes(neuron)
        assert result['detected'] is False
        assert result['candidate_child_ids'] == []

    def test_uniform_chain_not_detected(self):
        neuron = make_chain_neuron(n=10)
        result = fau.diagnose_extrusion_nodes(neuron)
        assert result['detected'] is False
        assert result['candidate_child_ids'] == []
        assert result['median_edge_length'] == pytest.approx(1000.0)

    def test_spike_detected(self):
        neuron = make_chain_neuron(n=10, spike_from=5)
        result = fau.diagnose_extrusion_nodes(neuron)
        assert result['detected'] is True
        assert result['candidate_child_ids'] == [11]
        assert result['candidate_parent_ids'] == [5]


class TestRepairExtrudedSkeleton:
    def test_no_candidates_returns_original(self):
        neuron = make_chain_neuron(n=10)
        repaired, stats = fau.repair_extruded_skeleton(neuron)
        assert repaired is neuron
        assert stats['repaired'] is False

    def test_successful_repair(self):
        neuron = make_chain_neuron(n=10, spike_from=5)
        repaired, stats = fau.repair_extruded_skeleton(neuron)
        assert stats['repaired'] is True
        assert stats['removed_node_ids'] == [11]
        assert len(repaired.nodes) == 10

    def test_repair_too_large_is_skipped(self):
        neuron = make_chain_neuron(n=10, spike_from=5, spike_child=True)
        repaired, stats = fau.repair_extruded_skeleton(
            neuron, max_removed_fraction=0.05)
        assert repaired is neuron
        assert stats['repaired'] is False

    def test_repair_leaving_too_few_nodes_is_skipped(self):
        neuron = make_chain_neuron(n=10, spike_from=5)
        repaired, stats = fau.repair_extruded_skeleton(
            neuron, min_remaining_nodes=11)
        assert repaired is neuron
        assert stats['repaired'] is False

    def test_remove_nodes_failure_returns_original(self, monkeypatch):
        neuron = make_chain_neuron(n=10, spike_from=5)

        def boom(neuron, ids, inplace=False):
            raise RuntimeError("remove failed")

        monkeypatch.setattr(navis, 'remove_nodes', boom)
        repaired, stats = fau.repair_extruded_skeleton(neuron)
        assert repaired is neuron
        assert stats['repaired'] is False

    def test_repaired_too_small_returns_original(self, monkeypatch):
        neuron = make_chain_neuron(n=10, spike_from=5)

        class Tiny:
            nodes = pd.DataFrame({'node_id': [1]})

        monkeypatch.setattr(navis, 'remove_nodes',
                            lambda neuron, ids, inplace=False: Tiny())
        repaired, stats = fau.repair_extruded_skeleton(neuron)
        assert repaired is neuron
        assert stats['repaired'] is False


class TestDetectExtrusion:
    def test_radius_repair_and_no_trimesh(self, monkeypatch):
        df = pd.DataFrame({'node_id': [1, 2], 'parent_id': [-1, 1],
                           'x': [0.0, 1000.0], 'y': [0.0, 0.0],
                           'z': [0.0, 0.0]})
        neuron = types.SimpleNamespace(nodes=df)  # no radius column
        monkeypatch.setattr(navis.conversion, 'tree2meshneuron',
                            lambda neuron, **kw: object())
        assert fau.detect_extrusion(neuron) is False
        assert (neuron.nodes['radius'] == 1).all()

    def test_invalid_radii_repaired(self, monkeypatch):
        neuron = make_chain_neuron(n=3)
        neuron.nodes.loc[neuron.nodes.index[0], 'radius'] = 0
        monkeypatch.setattr(navis.conversion, 'tree2meshneuron',
                            lambda neuron, **kw: object())
        assert fau.detect_extrusion(neuron) is False
        assert (neuron.nodes['radius'] > 0).all()

    def test_conversion_failure_returns_false(self, monkeypatch):
        neuron = make_chain_neuron(n=3)

        def boom(neuron, **kwargs):
            raise RuntimeError("conversion failed")

        monkeypatch.setattr(navis.conversion, 'tree2meshneuron', boom)
        assert fau.detect_extrusion(neuron) is False

    def test_empty_edges_returns_false(self, monkeypatch):
        neuron = make_chain_neuron(n=3)
        fake_mesh = types.SimpleNamespace(
            faces=np.zeros((50, 3), dtype=int),
            vertices=np.zeros((0, 3)),
            edges_unique=np.zeros((0, 2), dtype=int),
        )
        monkeypatch.setattr(
            navis.conversion, 'tree2meshneuron',
            lambda neuron, **kw: types.SimpleNamespace(trimesh=fake_mesh))
        assert fau.detect_extrusion(neuron) is False

    def test_small_uniform_mesh_not_flagged(self, monkeypatch):
        neuron = make_chain_neuron(n=3)
        monkeypatch.setattr(
            navis.conversion, 'tree2meshneuron',
            lambda neuron, **kw: types.SimpleNamespace(
                trimesh=box_trimesh(1000.0)))
        assert fau.detect_extrusion(neuron) is False

    def test_large_mesh_uses_fine_simplification(self, monkeypatch):
        neuron = make_chain_neuron(n=3)
        big = trimesh.creation.icosphere(subdivisions=3)  # 1280 faces
        calls = []

        def fake_fine(mesh, target):
            calls.append(target)
            return box_trimesh(1000.0)

        monkeypatch.setattr(fmc, 'simplify_mesh_fine', fake_fine)
        monkeypatch.setattr(
            navis.conversion, 'tree2meshneuron',
            lambda neuron, **kw: types.SimpleNamespace(trimesh=big))
        assert fau.detect_extrusion(neuron) is False
        assert calls == [100]


class TestFlagExtrusions:
    FOLDER = 'flywire_FAFB_v783'

    def test_all_from_cache(self, tmp_path, monkeypatch, capsys):
        fau.save_extrusion_check_cache(str(tmp_path), self.FOLDER,
                                       {1: True, 2: False})

        def boom(*args, **kwargs):
            raise AssertionError("cached neurons must not be analyzed")

        monkeypatch.setattr(fau, 'detect_extrusion', boom)
        flagged = fau.flag_extrusions(str(tmp_path), self.FOLDER,
                                      {1: object(), 2: object()},
                                      verbose=True)
        assert flagged == [1]
        assert 'known issue(s) from cache' in capsys.readouterr().out

    def test_serial_check_and_cache_write(self, tmp_path, monkeypatch):
        outcomes = {11: True, 12: False}
        monkeypatch.setattr(
            fau, 'detect_extrusion',
            lambda neuron, simplification=0.95: outcomes[int(neuron)])
        flagged = fau.flag_extrusions(
            str(tmp_path), self.FOLDER, {11: 11, '12': 12}, n_workers=1)
        assert flagged == [11]
        assert fau.load_extrusion_check_cache(
            str(tmp_path), self.FOLDER) == {'11': True, '12': False}

    def test_serial_check_without_cache_write(self, tmp_path, monkeypatch):
        monkeypatch.setattr(fau, 'detect_extrusion',
                            lambda neuron, simplification=0.95: False)
        flagged = fau.flag_extrusions(str(tmp_path), self.FOLDER,
                                      {11: 11}, n_workers=1, use_cache=False)
        assert flagged == []
        assert not fau.extrusion_check_cache_path(
            str(tmp_path), self.FOLDER).exists()

    def test_pool_failure_falls_back_to_serial(self, tmp_path, monkeypatch):
        """A broken ProcessPoolExecutor must trigger the serial fallback."""

        class BrokenExecutor:
            def __init__(self, *args, **kwargs):
                raise RuntimeError("pool cannot start")

        fake_cf = types.ModuleType('concurrent.futures')
        fake_cf.ProcessPoolExecutor = BrokenExecutor
        monkeypatch.setitem(sys.modules, 'concurrent.futures', fake_cf)

        outcomes = {21: True, 22: False}
        messages = []
        monkeypatch.setattr(
            fau, 'detect_extrusion',
            lambda neuron, simplification=0.95: outcomes[int(neuron)])
        flagged = fau.flag_extrusions(
            str(tmp_path), self.FOLDER, {21: 21, 22: 22},
            log=messages.append)
        assert flagged == [21]
        assert any('flagged for' in msg for msg in messages)
        assert fau.load_extrusion_check_cache(
            str(tmp_path), self.FOLDER)['21'] is True


# ===========================================================================
# flywire_mesh_cache
# ===========================================================================

class TestMeshCacheCore:
    def test_save_load_roundtrip(self, tmp_path):
        cache = fmc.FlyWireMeshCache('flywire_FAFB_v783',
                                     project_root=str(tmp_path))
        mesh = mesh_neuron(body_id=42)
        assert cache.save({42: mesh}) == 1
        assert cache.path(42).exists()
        loaded = cache.load(42)
        assert isinstance(loaded, navis.MeshNeuron)
        assert loaded.id == '42'

    def test_save_empty_and_invalid_entries(self, tmp_path):
        cache = fmc.FlyWireMeshCache('flywire_FAFB_v783',
                                     project_root=str(tmp_path))
        assert cache.save({}) == 0
        assert cache.save({'7': object()}) == 0  # invalid mesh skipped
        # an unnormalizable body id is rejected at the path boundary
        with pytest.raises(fid.FlyWireBodyIdError):
            cache.save({-5: mesh_neuron()})

    def test_save_write_failure_is_cleaned_up(self, tmp_path, monkeypatch):
        cache = fmc.FlyWireMeshCache('flywire_FAFB_v783',
                                     project_root=str(tmp_path))

        def boom(obj, writer, protocol=None):
            raise RuntimeError("pickle stream broken")

        monkeypatch.setattr(fmc.pickle, 'dump', boom)
        assert cache.save({'8': mesh_neuron()}) == 0
        # no stray temp files remain in the cache dir
        assert not list(cache.cache_dir.glob('.*'))

    def test_save_requires_zstandard(self, tmp_path, monkeypatch):
        cache = fmc.FlyWireMeshCache('flywire_FAFB_v783',
                                     project_root=str(tmp_path))
        monkeypatch.setattr(fmc, 'zstd', None)
        with pytest.raises(ImportError, match="zstandard"):
            cache.save({1: mesh_neuron()})

    def test_load_requires_zstandard_for_zst(self, tmp_path, monkeypatch):
        cache = fmc.FlyWireMeshCache('flywire_FAFB_v783',
                                     project_root=str(tmp_path))
        cache.save({1: mesh_neuron()})
        monkeypatch.setattr(fmc, 'zstd', None)
        with pytest.raises(ImportError, match="zstandard"):
            cache._load_pickle(cache.path(1))

    def test_load_invalid_content_and_corrupt_files(self, tmp_path):
        import pickle as _pickle
        cache = fmc.FlyWireMeshCache('flywire_FAFB_v783',
                                     project_root=str(tmp_path))
        cache.cache_dir.mkdir(parents=True, exist_ok=True)
        # valid zst pickle but not a MeshNeuron -> skipped
        mesh = mesh_neuron()
        cache.save({'5': mesh})
        junk = cache.cache_dir / '6.pkl.zst'
        import zstandard as _zstd
        junk.write_bytes(_zstd.ZstdCompressor().compress(
            _pickle.dumps("not a mesh")))
        corrupt = cache.cache_dir / '7.pkl.zst'
        corrupt.write_bytes(b'garbage')
        assert cache.load(6) is None
        assert cache.load(7) is None

    def test_legacy_pickle_migration(self, tmp_path):
        import pickle as _pickle
        cache = fmc.FlyWireMeshCache('flywire_FAFB_v783',
                                     project_root=str(tmp_path))
        cache.cache_dir.mkdir(parents=True, exist_ok=True)
        legacy = cache.cache_dir / '9.pkl'
        legacy.write_bytes(_pickle.dumps(mesh_neuron()))
        loaded = cache.load(9)
        assert isinstance(loaded, navis.MeshNeuron)
        assert cache.path(9).exists()  # migrated to .pkl.zst

    def test_legacy_dir_read(self, tmp_path):
        import pickle as _pickle
        cache = fmc.FlyWireMeshCache('flywire_FAFB_v783',
                                     project_root=str(tmp_path))
        cache.legacy_cache_dir.mkdir(parents=True, exist_ok=True)
        (cache.legacy_cache_dir / '8.pkl').write_bytes(
            _pickle.dumps(mesh_neuron()))
        loaded = cache.load(8, migrate=False)
        assert isinstance(loaded, navis.MeshNeuron)

    def test_existing_ids(self, tmp_path):
        import pickle as _pickle
        cache = fmc.FlyWireMeshCache('flywire_FAFB_v783',
                                     project_root=str(tmp_path))
        cache.save({'10': mesh_neuron()})
        cache.cache_dir.mkdir(parents=True, exist_ok=True)
        (cache.cache_dir / '11.pkl').write_bytes(
            _pickle.dumps(mesh_neuron()))
        (cache.cache_dir / 'notanumber.pkl').write_bytes(b'x')
        assert cache.existing_ids() == {'10', '11'}


class _FakeO3DMesh:
    """Minimal open3d.geometry.TriangleMesh stand-in."""

    shrink_on_cleanup = False
    cluster_result = None

    def __init__(self):
        self.vertices = np.zeros((0, 3))
        self.triangles = np.zeros((0, 3), dtype=int)

    def remove_duplicated_vertices(self):
        pass

    def remove_duplicated_triangles(self):
        pass

    def remove_degenerate_triangles(self):
        if _FakeO3DMesh.shrink_on_cleanup:
            self.triangles = np.zeros((0, 3), dtype=int)

    def remove_unreferenced_vertices(self):
        pass

    def get_max_bound(self):
        return [1000.0, 1000.0, 1000.0]

    def get_min_bound(self):
        return [0.0, 0.0, 0.0]

    def simplify_vertex_clustering(self, voxel_size, contraction=None):
        result = _FakeO3DMesh()
        if _FakeO3DMesh.cluster_result == 'empty':
            result.vertices = np.zeros((0, 3))
            result.triangles = np.zeros((0, 3), dtype=int)
        else:
            result.vertices = np.array(
                [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
            result.triangles = np.array(
                [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
        return result


def _install_fake_open3d(monkeypatch):
    _FakeO3DMesh.shrink_on_cleanup = False
    _FakeO3DMesh.cluster_result = None
    fake = types.ModuleType('open3d')
    geometry = types.ModuleType('open3d.geometry')
    utility = types.ModuleType('open3d.utility')
    geometry.TriangleMesh = _FakeO3DMesh
    geometry.Vector3dVector = np.asarray
    geometry.Vector3iVector = np.asarray

    class _Contraction:
        Average = 'average'

    geometry.SimplificationContraction = _Contraction
    utility.Vector3dVector = np.asarray
    utility.Vector3iVector = np.asarray
    fake.geometry = geometry
    fake.utility = utility
    monkeypatch.setitem(sys.modules, 'open3d', fake)
    monkeypatch.setitem(sys.modules, 'open3d.geometry', geometry)
    monkeypatch.setitem(sys.modules, 'open3d.utility', utility)


class _FakeLargeMesh:
    """trimesh-like object that only provides arrays/counts."""

    def __init__(self, n_faces=150000, area=1e9, extent_ok=True):
        self._n_faces = n_faces
        self.area = area
        self.vertices = np.zeros((8, 3))
        self.faces = np.zeros((n_faces, 3), dtype=int)
        self._extent_ok = extent_ok

    # used by trimesh.Trimesh construction of the prefilter output only


class TestVertexClusterPrefilter:
    def test_small_mesh_returns_none(self):
        assert fmc._vertex_cluster_prefilter(box_trimesh(), 10) is None

    def test_open3d_missing_returns_none(self, monkeypatch):
        monkeypatch.setitem(sys.modules, 'open3d', None)
        assert fmc._vertex_cluster_prefilter(_FakeLargeMesh(), 100) is None

    def test_target_not_smaller(self, monkeypatch):
        _install_fake_open3d(monkeypatch)
        assert fmc._vertex_cluster_prefilter(
            _FakeLargeMesh(), 200000) is None

    def test_empty_geometry(self, monkeypatch):
        _install_fake_open3d(monkeypatch)
        fake = _FakeLargeMesh()
        fake.vertices = np.zeros((0, 3))
        assert fmc._vertex_cluster_prefilter(fake, 100) is None

    def test_cleanup_shrinks_below_minimum(self, monkeypatch):
        _install_fake_open3d(monkeypatch)
        _FakeO3DMesh.shrink_on_cleanup = True
        assert fmc._vertex_cluster_prefilter(_FakeLargeMesh(), 100) is None

    def test_cluster_empty_result(self, monkeypatch):
        _install_fake_open3d(monkeypatch)
        _FakeO3DMesh.cluster_result = 'empty'
        assert fmc._vertex_cluster_prefilter(_FakeLargeMesh(), 100) is None

    def test_bad_area_falls_back(self, monkeypatch):
        _install_fake_open3d(monkeypatch)

        class NoArea:
            vertices = np.zeros((8, 3))
            faces = np.zeros((150000, 3), dtype=int)

            @property
            def area(self):
                raise RuntimeError("no area")

        result = fmc._vertex_cluster_prefilter(NoArea(), 100)
        assert isinstance(result, trimesh.Trimesh)

    def test_success_path(self, monkeypatch):
        _install_fake_open3d(monkeypatch)
        result = fmc._vertex_cluster_prefilter(_FakeLargeMesh(), 100)
        assert isinstance(result, trimesh.Trimesh)
        assert len(result.faces) == 4


class TestSimplifyMeshOpen3d:
    def test_open3d_missing_fallback(self, monkeypatch):
        monkeypatch.setitem(sys.modules, 'open3d', None)
        result = fmc._simplify_mesh_open3d(box_trimesh(), 2)
        assert isinstance(result, trimesh.Trimesh)

    def test_open3d_failure_returns_input(self, monkeypatch):
        fake = types.ModuleType('open3d')

        class BoomGeometry:
            def __init__(self):
                raise RuntimeError("open3d broken")

        fake.geometry = types.SimpleNamespace(TriangleMesh=BoomGeometry)
        fake.utility = types.SimpleNamespace(Vector3dVector=np.asarray,
                                             Vector3iVector=np.asarray)
        monkeypatch.setitem(sys.modules, 'open3d', fake)
        box = box_trimesh()
        result = fmc._simplify_mesh_open3d(box, 2)
        assert result is box


class TestSimplifyMeshFine:
    def test_noop_cases(self):
        box = box_trimesh()
        assert fmc.simplify_mesh_fine(box, 10**9) is box
        empty = types.SimpleNamespace(faces=np.zeros((0, 3)))
        assert fmc.simplify_mesh_fine(empty, 10) is empty

    def test_candidate_below_target_retries_source(self, monkeypatch):
        source = trimesh.creation.icosphere(subdivisions=3)  # 1280 faces
        candidate = box_trimesh(1000.0)  # 12 faces <= target 100
        monkeypatch.setattr(fmc, '_vertex_cluster_prefilter',
                            lambda mesh, target, **kw: candidate)
        seen = []

        def decimator(mesh, target):
            seen.append(mesh)
            return mesh

        monkeypatch.setattr(fmc, '_simplify_mesh_open3d', decimator)
        result = fmc.simplify_mesh_fine(source, 100)
        assert result is source  # retried QEM on the source
        assert seen == [source]

    def test_candidate_kept_when_qem_returns_input(self, monkeypatch):
        source = trimesh.creation.icosphere(subdivisions=3)
        candidate = trimesh.creation.icosphere(subdivisions=2)  # 320 faces
        monkeypatch.setattr(fmc, '_vertex_cluster_prefilter',
                            lambda mesh, target, **kw: candidate)
        monkeypatch.setattr(fmc, '_simplify_mesh_open3d',
                            lambda mesh, target: mesh)  # QEM "fails"
        result = fmc.simplify_mesh_fine(source, 100)
        assert result is candidate

    def test_qem_result_used(self, monkeypatch):
        source = trimesh.creation.icosphere(subdivisions=3)
        candidate = trimesh.creation.icosphere(subdivisions=2)
        final = box_trimesh(500.0)
        monkeypatch.setattr(fmc, '_vertex_cluster_prefilter',
                            lambda mesh, target, **kw: candidate)
        monkeypatch.setattr(fmc, '_simplify_mesh_open3d',
                            lambda mesh, target: final)
        assert fmc.simplify_mesh_fine(source, 100) is final

    def test_no_prefilter_direct_qem(self, monkeypatch):
        source = trimesh.creation.icosphere(subdivisions=3)
        final = box_trimesh(500.0)
        monkeypatch.setattr(fmc, '_vertex_cluster_prefilter',
                            lambda mesh, target, **kw: None)
        monkeypatch.setattr(fmc, '_simplify_mesh_open3d',
                            lambda mesh, target: final)
        assert fmc.simplify_mesh_fine(source, 100) is final


class TestSomaAwareSimplification:
    @staticmethod
    def recording_decimator():
        seen = []

        def decimator(mesh, target):
            seen.append(target)
            return mesh

        decimator.seen = seen
        return decimator

    def test_no_soma_uniform(self):
        dec = self.recording_decimator()
        mesh = trimesh.creation.icosphere(subdivisions=3)
        fmc.simplify_mesh_with_soma_awareness(mesh, soma_pos=None,
                                              decimator=dec)
        assert dec.seen == [max(100, int(len(mesh.faces) * 0.05))]

    def test_invalid_soma_uniform(self):
        dec = self.recording_decimator()
        mesh = trimesh.creation.icosphere(subdivisions=3)
        fmc.simplify_mesh_with_soma_awareness(mesh, soma_pos=[1.0, 2.0],
                                              decimator=dec)
        assert len(dec.seen) == 1
        dec2 = self.recording_decimator()
        fmc.simplify_mesh_with_soma_awareness(
            mesh, soma_pos=[float('nan'), 0.0, 0.0], decimator=dec2)
        assert len(dec2.seen) == 1

    def test_soma_far_away_all_skeleton(self):
        dec = self.recording_decimator()
        mesh = box_trimesh(1000.0)
        fmc.simplify_mesh_with_soma_awareness(
            mesh, soma_pos=[10**9, 10**9, 10**9], decimator=dec)
        assert dec.seen == [max(100, int(12 * 0.05))]

    def test_soma_envelopes_mesh_all_soma(self):
        dec = self.recording_decimator()
        mesh = box_trimesh(1000.0)
        fmc.simplify_mesh_with_soma_awareness(
            mesh, soma_pos=[0.0, 0.0, 0.0], soma_radius=10**9,
            decimator=dec)
        assert dec.seen == [max(100, int(12 * 0.2))]

    def test_mixed_regions_concatenate(self):
        mesh = box_trimesh(100000.0)
        seen = []

        def decimator(part, target):
            seen.append(len(part.faces))
            return part

        result = fmc.simplify_mesh_with_soma_awareness(
            mesh, soma_pos=[50000.0, 0.0, 0.0], soma_radius=60000.0,
            decimator=decimator)
        assert len(seen) == 2  # soma part + skeleton part
        assert isinstance(result, trimesh.Trimesh)

    def test_mixed_failure_falls_back(self):
        mesh = box_trimesh(100000.0)
        calls = []

        def decimator(part, target):
            calls.append(target)
            if len(calls) == 1:
                raise RuntimeError("decimation failed")
            return part

        result = fmc.simplify_mesh_with_soma_awareness(
            mesh, soma_pos=[50000.0, 0.0, 0.0], soma_radius=60000.0,
            decimator=decimator)
        assert result is mesh  # uniform fallback returned the source
        assert len(calls) == 2


class OneShotArray:
    """Converts to a numpy array once, then raises on later conversions."""

    def __init__(self):
        self._used = False

    def __len__(self):
        return 3

    def __array__(self, dtype=None):
        if self._used:
            raise ValueError("second conversion fails")
        self._used = True
        return np.array([1.0, 2.0, 3.0])


class TestPrepareFlywireMesh:
    def test_non_meshneuron_raises(self):
        with pytest.raises(TypeError, match="MeshNeuron"):
            fmc.prepare_flywire_mesh(object(), 1)

    def test_uniform_preparation(self):
        mesh = mesh_neuron(trimesh.creation.icosphere(subdivisions=3),
                           body_id=5)
        prepared = fmc.prepare_flywire_mesh(
            mesh, 5, decimator=lambda part, target: part)
        assert isinstance(prepared, navis.MeshNeuron)
        assert prepared.id == '5'

    def test_soma_pos_from_mesh_attribute(self):
        mesh = mesh_neuron(box_trimesh(100.0))
        mesh.soma_pos = [10.0, 10.0, 10.0]
        prepared = fmc.prepare_flywire_mesh(
            mesh, 6, soma_radius=10**9,
            decimator=lambda part, target: part)
        assert np.allclose(np.asarray(prepared.soma_pos), [10, 10, 10])

    def test_soma_pos_assignment_failure_is_tolerated(self):
        mesh = mesh_neuron(box_trimesh(100.0))
        prepared = fmc.prepare_flywire_mesh(
            mesh, 7, soma_pos=OneShotArray(), soma_radius=10**9,
            decimator=lambda part, target: part)
        assert isinstance(prepared, navis.MeshNeuron)


class TestParseSomaPosition:
    def test_variants(self):
        assert fmc.parse_soma_position(None) is None
        assert fmc.parse_soma_position(float('nan')) is None
        assert np.allclose(fmc.parse_soma_position([1, 2, 3, 4]), [1, 2, 3])
        parsed = fmc.parse_soma_position("[710944 262716 205680]")
        assert np.allclose(parsed, [710944, 262716, 205680])
        assert fmc.parse_soma_position("1 2") is None
        assert fmc.parse_soma_position("nan nan nan") is None


# ===========================================================================
# flywire_ids
# ===========================================================================

class TestFlyWireIds:
    def test_string_forms(self):
        assert fid.normalize_flywire_body_id('007') == '7'
        assert fid.normalize_flywire_body_id('123.0') == '123'
        with pytest.raises(fid.FlyWireBodyIdError, match="empty"):
            fid.normalize_flywire_body_id('   ')
        with pytest.raises(fid.FlyWireBodyIdError, match="exact decimal"):
            fid.normalize_flywire_body_id('12a')
        with pytest.raises(fid.FlyWireBodyIdError, match="rounded float"):
            fid.normalize_flywire_body_id(f'{2**53}.0')

    def test_none_and_bool_rejected(self):
        with pytest.raises(fid.FlyWireBodyIdError):
            fid.normalize_flywire_body_id(None)
        with pytest.raises(fid.FlyWireBodyIdError):
            fid.normalize_flywire_body_id(True)

    def test_negative_rejected(self):
        with pytest.raises(fid.FlyWireBodyIdError, match="non-negative"):
            fid.normalize_flywire_body_id(-5)

    def test_decimal_forms(self):
        from decimal import Decimal
        assert fid.normalize_flywire_body_id(Decimal('5')) == '5'
        with pytest.raises(fid.FlyWireBodyIdError, match="not integral"):
            fid.normalize_flywire_body_id(Decimal('5.5'))
        with pytest.raises(fid.FlyWireBodyIdError, match="not integral"):
            fid.normalize_flywire_body_id(Decimal('NaN'))

    def test_float_forms(self):
        assert fid.normalize_flywire_body_id(5.0) == '5'
        with pytest.raises(fid.FlyWireBodyIdError, match="not a finite"):
            fid.normalize_flywire_body_id(float('nan'))
        with pytest.raises(fid.FlyWireBodyIdError, match="not a finite"):
            fid.normalize_flywire_body_id(5.5)
        with pytest.raises(fid.FlyWireBodyIdError, match="unsafe float"):
            fid.normalize_flywire_body_id(float(2**53 + 2))

    def test_str_protocol_fallback(self):
        class StrId:
            def __str__(self):
                return '0042'

        assert fid.normalize_flywire_body_id(StrId()) == '42'

        class BadStr:
            def __str__(self):
                return 'x42'

        with pytest.raises(fid.FlyWireBodyIdError, match="invalid"):
            fid.normalize_flywire_body_id(BadStr())

    def test_api_int_bounds(self):
        assert fid.body_id_to_api_int('7') == 7
        with pytest.raises(fid.FlyWireBodyIdError, match="64-bit"):
            fid.body_id_to_api_int(str(2**63))

    def test_frame_normalization_skips_missing_columns(self):
        df = pd.DataFrame({'bodyId': ['007', 5]})
        fid.normalize_flywire_id_columns(df, ['bodyId', 'absent'])
        assert list(df['bodyId']) == ['7', '5']

    def test_dataset_helpers(self):
        assert fid.is_banc_dataset('banc_v888')
        assert not fid.is_banc_dataset('flywire_FAFB_v783')
        assert fid.is_flywire_dataset('flywire_FAFB_v783')
        assert not fid.is_flywire_dataset('banc_v888')
        assert fid.is_local_connectome_dataset('banc_v888')
        assert not fid.is_flywire_dataset('hemibrain')
        assert fid.dataset_folder('a:b.c') == 'a_b_c'
        assert fid.resolve_flywire_dataset_dir('/nonexistent/root',
                                               'x') is None


# ===========================================================================
# connection_map
# ===========================================================================

def _conn_frame():
    return pl.DataFrame({
        'bodyId_pre': ['1', '1', '2'],
        'bodyId_post': ['2', '3', '3'],
        'weight': [4, 2, 1],
    })


class TestConnectionMap:
    def test_min_weight_property_and_filter(self, tmp_path):
        index_path = tmp_path / 'index.parquet'
        pl.DataFrame({'bodyId': ['2', '3'],
                      'type': ['A', 'B']}).write_parquet(index_path)
        m = cmap.ThresholdedConnectionMap(
            db_path='', neuron_index_path=str(index_path),
            min_weight=2, conn_frame=_conn_frame())
        assert m.min_weight == 2
        by_id = m.total_incoming_by_bodyid()
        assert dict(zip(by_id['bodyId_post'].to_list(),
                        by_id['total_incoming_weight'].to_list())) == {
            '2': 4, '3': 2}
        by_type = m.total_incoming_by_type()
        assert dict(zip(by_type['type_post'].to_list(),
                        by_type['total_incoming_weight'].to_list())) == {
            'A': 4, 'B': 2}

    def test_pandas_frame_input(self, tmp_path):
        index_path = tmp_path / 'index.parquet'
        pl.DataFrame({'bodyId': ['2'],
                      'type': ['A']}).write_parquet(index_path)
        pdf = pd.DataFrame({'bodyId_pre': ['1'], 'bodyId_post': ['2'],
                            'weight': [7]})
        m = cmap.ThresholdedConnectionMap(
            db_path='', neuron_index_path=str(index_path),
            conn_frame=pdf)
        assert m.total_incoming_by_bodyid().height == 1

    def test_no_source_files_empty(self, tmp_path):
        m = cmap.ThresholdedConnectionMap(
            db_path=str(tmp_path / 'missing.parquet'),
            neuron_index_path=str(tmp_path / 'missing_index.parquet'))
        assert m.total_incoming_by_bodyid().height == 0

    def test_none_db_path_empty(self):
        m = cmap.ThresholdedConnectionMap(db_path=None,
                                          neuron_index_path='')
        assert m.total_incoming_by_bodyid().height == 0

    def test_batch_files_with_schema_tolerance(self, tmp_path):
        db_path = tmp_path / 'connections.parquet'
        batch_dir = tmp_path / '_batch_files'
        batch_dir.mkdir()
        # a legacy batch missing the weight column must be skipped
        pl.DataFrame({'bodyId_pre': ['x'], 'bodyId_post': ['y'],
                      'other': [1]}).write_parquet(batch_dir /
                                                    'batch_0000.parquet')
        pl.DataFrame({'bodyId_pre': ['1'], 'bodyId_post': ['2'],
                      'weight': [3]}).write_parquet(batch_dir /
                                                     'batch_0001.parquet')
        index_path = tmp_path / 'index.parquet'
        pl.DataFrame({'bodyId': ['2'],
                      'type': ['A']}).write_parquet(index_path)
        m = cmap.ThresholdedConnectionMap(db_path=str(db_path),
                                          neuron_index_path=str(index_path))
        by_id = m.total_incoming_by_bodyid()
        assert by_id['total_incoming_weight'].to_list() == [3]

    def test_batch_files_all_invalid(self, tmp_path):
        db_path = tmp_path / 'connections.parquet'
        batch_dir = tmp_path / '_batch_files'
        batch_dir.mkdir()
        pl.DataFrame({'junk': [1]}).write_parquet(batch_dir /
                                                  'batch_0000.parquet')
        m = cmap.ThresholdedConnectionMap(db_path=str(db_path),
                                          neuron_index_path='')
        assert m.total_incoming_by_bodyid().height == 0

    def test_double_check_lock_cache(self, tmp_path):
        class PreloadingLock:
            """A lock whose acquisition pre-fills the aggregate cache."""

            def __init__(self, cache, key, value):
                self.cache, self.key, self.value = cache, key, value

            def __enter__(self):
                self.cache[self.key] = self.value
                return self

            def __exit__(self, *exc):
                return False

        sentinel = pl.DataFrame({'sentinel': [1]})
        m = cmap.ThresholdedConnectionMap(db_path='',
                                          neuron_index_path='',
                                          conn_frame=_conn_frame())
        m._lock = PreloadingLock(m._cache, 'by_bodyid', sentinel)
        assert m.total_incoming_by_bodyid() is sentinel

        m2 = cmap.ThresholdedConnectionMap(db_path='',
                                           neuron_index_path='',
                                           conn_frame=_conn_frame())
        m2._lock = PreloadingLock(m2._cache, 'by_type', sentinel)
        assert m2.total_incoming_by_type() is sentinel
