"""FlyWire mesh preparation and cache helpers.

FlyWire/FAFB CAVE fetches are surface meshes, not SWC trees.  This module
keeps that representation separate from the NeuPrint raw-skeleton cache and
provides the same soma-aware decimation used by the visualizer.
"""

from __future__ import annotations

import os
import pickle
import re
import tempfile
from pathlib import Path
from typing import Dict, Mapping, Optional, Union

import numpy as np
import navis

try:
    import zstandard as zstd
except ImportError:  # pragma: no cover - installation is covered by requirements
    zstd = None

try:
    from .flywire_ids import normalize_flywire_body_id
except ImportError:
    from flywire_ids import normalize_flywire_body_id

try:
    from .utils.naming_utils import canonical_dataset_name
except ImportError:  # pragma: no cover - src laid bare on sys.path
    from utils.naming_utils import canonical_dataset_name


FLYWIRE_MESH_CACHE_SIMPLIFICATION = 0.95
FLYWIRE_MESH_CACHE_SOMA_SIMPLIFICATION = 0.8
FLYWIRE_MESH_CACHE_SOMA_RADIUS = 20000
FLYWIRE_MESH_CACHE_ZSTD_LEVEL = 1
FLYWIRE_MESH_CACHE_SUFFIX = ".pkl.zst"

# Large FAFB tube meshes are expensive inputs for global QEM decimation.  A
# small, one-pass vertex-clustering prepass reduces the QEM input while the
# final QEM pass still controls the requested face target and preserves the
# fine-render quality.  These are deliberately conservative: small meshes
# continue to use the direct QEM path.
FAFB_FINE_PREFILTER_MIN_FACES = 100_000
FAFB_FINE_PREFILTER_RATIO = 4.0
FAFB_FINE_CLUSTER_VOXEL_FACTOR = 0.30


def _dataset_folder(dataset: str) -> str:
    return canonical_dataset_name(str(dataset)).replace(":", "_").replace(".", "_")


def flywire_mesh_cache_key(
        simplification: float = FLYWIRE_MESH_CACHE_SIMPLIFICATION,
        soma_simplification: float = FLYWIRE_MESH_CACHE_SOMA_SIMPLIFICATION,
        soma_radius: float = FLYWIRE_MESH_CACHE_SOMA_RADIUS,
        ) -> str:
    """Return the representation/configuration key used on disk."""
    return (
        f"FLYWIRE_simp{int(float(simplification) * 100)}"
        f"_soma{int(float(soma_simplification) * 100)}"
        f"_r{int(float(soma_radius) / 1000)}"
    )


def _mesh_cache_dirs(
        dataset: str,
        project_root: Optional[Union[str, Path]] = None,
        simplification: float = FLYWIRE_MESH_CACHE_SIMPLIFICATION,
        soma_simplification: float = FLYWIRE_MESH_CACHE_SOMA_SIMPLIFICATION,
        soma_radius: float = FLYWIRE_MESH_CACHE_SOMA_RADIUS,
        ) -> tuple[Path, Path]:
    root = Path(project_root) if project_root else Path(__file__).parent.parent
    key = flywire_mesh_cache_key(
        simplification, soma_simplification, soma_radius)
    base = root / "cache" / _dataset_folder(dataset)
    # New writes are explicitly mesh-owned.  The old skeletons/{key} folder
    # is retained as a read-only migration source for existing users.
    return base / "meshes" / key, base / "skeletons" / key


class FlyWireMeshCache:
    """Cache of prepared FlyWire ``MeshNeuron`` objects.

    New files are written below ``cache/{dataset}/meshes``.  Existing
    visualization caches under ``cache/{dataset}/skeletons`` are read and
    migrated non-destructively when encountered.  The canonical file format
    is a Zstandard-compressed pickle (``{bodyId}.pkl.zst``); older
    uncompressed ``.pkl`` files remain readable and are migrated on access.
    """

    def __init__(
            self,
            dataset: str,
            project_root: Optional[Union[str, Path]] = None,
            simplification: float = FLYWIRE_MESH_CACHE_SIMPLIFICATION,
            soma_simplification: float = FLYWIRE_MESH_CACHE_SOMA_SIMPLIFICATION,
            soma_radius: float = FLYWIRE_MESH_CACHE_SOMA_RADIUS,
            ):
        self.dataset = dataset
        self.project_root = (
            Path(project_root) if project_root
            else Path(__file__).parent.parent
        )
        self.simplification = float(simplification)
        self.soma_simplification = float(soma_simplification)
        self.soma_radius = float(soma_radius)
        self.cache_dir, self.legacy_cache_dir = _mesh_cache_dirs(
            dataset,
            project_root=self.project_root,
            simplification=self.simplification,
            soma_simplification=self.soma_simplification,
            soma_radius=self.soma_radius,
        )

    @staticmethod
    def _body_id(body_id) -> str:
        return normalize_flywire_body_id(body_id)

    def path(self, body_id) -> Path:
        return self.cache_dir / (
            f"{self._body_id(body_id)}{FLYWIRE_MESH_CACHE_SUFFIX}")

    def _candidate_paths(self, body_id) -> list[Path]:
        key = self._body_id(body_id)
        return [
            self.cache_dir / f"{key}{FLYWIRE_MESH_CACHE_SUFFIX}",
            self.cache_dir / f"{key}.pkl",
            self.legacy_cache_dir / f"{key}{FLYWIRE_MESH_CACHE_SUFFIX}",
            self.legacy_cache_dir / f"{key}.pkl",
        ]

    @staticmethod
    def _valid_mesh(obj) -> bool:
        return isinstance(obj, navis.MeshNeuron)

    @staticmethod
    def _load_pickle(path: Path):
        """Load either a legacy pickle or a Zstandard-compressed pickle."""
        if str(path).endswith(FLYWIRE_MESH_CACHE_SUFFIX):
            if zstd is None:
                raise ImportError(
                    "zstandard is required to read FlyWire .pkl.zst caches")
            with path.open("rb") as handle:
                with zstd.ZstdDecompressor().stream_reader(handle) as reader:
                    return pickle.load(reader)
        with path.open("rb") as handle:
            return pickle.load(handle)

    @staticmethod
    def _path_body_id(path: Path) -> str:
        """Extract a body ID from either supported mesh-cache suffix."""
        name = Path(path).name
        if name.endswith(FLYWIRE_MESH_CACHE_SUFFIX):
            name = name[:-len(FLYWIRE_MESH_CACHE_SUFFIX)]
        elif name.endswith(".pkl"):
            name = name[:-len(".pkl")]
        return name

    def load(self, body_id, migrate: bool = True):
        """Load a cached mesh, optionally migrating the old path."""
        for path in self._candidate_paths(body_id):
            if not path.exists():
                continue
            try:
                mesh = self._load_pickle(path)
                if not self._valid_mesh(mesh):
                    continue
                mesh.id = self._body_id(body_id)
                if migrate and path != self.path(body_id):
                    self.save({body_id: mesh})
                return mesh
            except Exception:
                continue
        return None

    def existing_ids(self) -> set[str]:
        """Return IDs available in either the new or legacy mesh folder."""
        ids: set[str] = set()
        for directory in (self.cache_dir, self.legacy_cache_dir):
            if not directory.exists():
                continue
            paths = list(directory.glob("*.pkl.zst"))
            paths.extend(directory.glob("*.pkl"))
            for path in paths:
                try:
                    ids.add(self._body_id(self._path_body_id(path)))
                except (TypeError, ValueError):
                    continue
        return ids

    def save(self, meshes: Mapping[Union[int, str], object]) -> int:
        """Atomically persist only ``MeshNeuron`` values as ``.pkl.zst``."""
        if not meshes:
            return 0
        if zstd is None:
            raise ImportError(
                "zstandard is required to write FlyWire .pkl.zst caches")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        written = 0
        compressor = zstd.ZstdCompressor(level=FLYWIRE_MESH_CACHE_ZSTD_LEVEL)
        for body_id, mesh in meshes.items():
            if not self._valid_mesh(mesh):
                continue
            path = self.path(body_id)
            temp_path = None
            try:
                mesh.id = self._body_id(body_id)
                with tempfile.NamedTemporaryFile(
                        mode="wb",
                        prefix=f".{path.name}.",
                        suffix=".tmp",
                        dir=str(path.parent),
                        delete=False) as handle:
                    temp_path = Path(handle.name)
                    # Keep the file handle open while closing the zstd writer
                    # so the frame footer is finalized before the atomic
                    # replace. Pickle still streams directly into zstd rather
                    # than materializing a second uncompressed byte buffer.
                    with compressor.stream_writer(
                            handle, closefd=False) as writer:
                        pickle.dump(
                            mesh, writer, protocol=pickle.HIGHEST_PROTOCOL)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temp_path, path)
                written += 1
            except Exception:
                if temp_path is not None:
                    temp_path.unlink(missing_ok=True)
        return written


def _simplify_mesh_open3d(trimesh_obj, target_faces: int):
    """Decimate a mesh using the visualizer's Open3D/QEM fallback policy."""
    try:
        import open3d as o3d
        import trimesh

        target_faces = max(1, int(target_faces))
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(
            np.asarray(trimesh_obj.vertices))
        o3d_mesh.triangles = o3d.utility.Vector3iVector(
            np.asarray(trimesh_obj.faces))
        simplified = o3d_mesh.simplify_quadric_decimation(target_faces)
        return trimesh.Trimesh(
            vertices=np.asarray(simplified.vertices),
            faces=np.asarray(simplified.triangles),
        )
    except ImportError:
        try:
            return trimesh_obj.simplify_quadric_decimation(
                face_count=max(1, int(target_faces)))
        except Exception:
            return trimesh_obj
    except Exception:
        return trimesh_obj


def _vertex_cluster_prefilter(
        trimesh_obj,
        target_faces: int,
        prefilter_ratio: float = FAFB_FINE_PREFILTER_RATIO,
        voxel_factor: float = FAFB_FINE_CLUSTER_VOXEL_FACTOR,
        ):
    """Reduce a large mesh cheaply before the quality QEM pass.

    Open3D's vertex clustering is used only as a bounded prefilter.  The
    result is intentionally allowed to be approximate; a subsequent QEM
    decimation produces the final requested target.  Returning ``None``
    means that the prefilter was unavailable or not useful.
    """
    try:
        import open3d as o3d
        import trimesh

        source_faces = len(getattr(trimesh_obj, "faces", ()))
        target_faces = max(1, int(target_faces))
        if source_faces < FAFB_FINE_PREFILTER_MIN_FACES:
            return None
        if target_faces >= source_faces:
            return None

        prefilter_ratio = max(1.0, float(prefilter_ratio))
        prefilter_target = min(
            source_faces - 1,
            max(target_faces + 1, int(target_faces * prefilter_ratio)),
        )
        if prefilter_target >= source_faces:
            return None

        vertices = np.asarray(trimesh_obj.vertices, dtype=np.float64)
        faces = np.asarray(trimesh_obj.faces, dtype=np.int32)
        if len(vertices) == 0 or len(faces) == 0:
            return None

        base = o3d.geometry.TriangleMesh()
        base.vertices = o3d.utility.Vector3dVector(vertices)
        base.triangles = o3d.utility.Vector3iVector(faces)
        base.remove_duplicated_vertices()
        base.remove_duplicated_triangles()
        base.remove_degenerate_triangles()
        base.remove_unreferenced_vertices()
        source_faces = len(base.triangles)
        if source_faces < FAFB_FINE_PREFILTER_MIN_FACES:
            return None

        bounds = np.asarray(base.get_max_bound()) - np.asarray(
            base.get_min_bound())
        max_extent = float(np.max(bounds)) if len(bounds) else 0.0
        if not np.isfinite(max_extent) or max_extent <= 0:
            return None

        try:
            surface_area = float(trimesh_obj.area)
        except Exception:
            surface_area = 0.0
        if not np.isfinite(surface_area) or surface_area <= 0:
            surface_area = max_extent * max_extent

        voxel_size = np.sqrt(surface_area / prefilter_target) * float(
            voxel_factor)
        voxel_size = max(voxel_size, max_extent * 1e-5, 1e-6)
        clustered = base.simplify_vertex_clustering(
            float(voxel_size),
            contraction=o3d.geometry.SimplificationContraction.Average,
        )
        clustered.remove_duplicated_triangles()
        clustered.remove_degenerate_triangles()
        clustered.remove_unreferenced_vertices()
        if len(clustered.triangles) <= 0 or len(clustered.triangles) >= source_faces:
            return None

        return trimesh.Trimesh(
            vertices=np.asarray(clustered.vertices),
            faces=np.asarray(clustered.triangles),
            process=False,
        )
    except Exception:
        return None


def simplify_mesh_fine(
        trimesh_obj,
        target_faces: int,
        prefilter_ratio: float = FAFB_FINE_PREFILTER_RATIO,
        ):
    """Fast fine-quality mesh decimation for large FAFB/CAVE surfaces.

    Large meshes first pass through one inexpensive vertex-clustering stage,
    usually reducing the QEM input to about four times the final target.  QEM
    then performs the final decimation, so callers retain an explicit face
    target instead of accepting the approximate output of clustering alone.
    Smaller meshes use direct QEM because the prepass would not amortize its
    setup cost.  If Open3D or the prepass is unavailable, the direct fallback
    remains fully compatible with the previous behavior.
    """
    source_faces = len(getattr(trimesh_obj, "faces", ()))
    target_faces = max(1, int(target_faces))
    if source_faces == 0 or target_faces >= source_faces:
        return trimesh_obj

    candidate = _vertex_cluster_prefilter(
        trimesh_obj,
        target_faces,
        prefilter_ratio=prefilter_ratio,
    )
    if candidate is not None:
        if len(candidate.faces) <= target_faces:
            # Do not silently oversimplify when the approximate clustering
            # pass crossed the requested target; retry QEM on the source so
            # the caller's face-level contract remains authoritative.
            return _simplify_mesh_open3d(trimesh_obj, target_faces)
        simplified = _simplify_mesh_open3d(candidate, target_faces)
        if simplified is not candidate or len(candidate.faces) <= target_faces:
            return simplified
        # Preserve the useful prefilter if the final QEM pass fails.
        return candidate
    return _simplify_mesh_open3d(trimesh_obj, target_faces)


def simplify_mesh_with_soma_awareness(
        trimesh_obj,
        skeleton_simp: float = FLYWIRE_MESH_CACHE_SIMPLIFICATION,
        soma_simp: float = FLYWIRE_MESH_CACHE_SOMA_SIMPLIFICATION,
        soma_pos=None,
        soma_radius: float = FLYWIRE_MESH_CACHE_SOMA_RADIUS,
        decimator=None,
        ):
    """Apply branch/soma decimation matching the FAFB visualizer workflow.

    ``skeleton_simp=0.95`` removes 95% of branch-region faces and
    ``soma_simp=0.8`` removes 80% of faces in the soma region.  Without a
    soma coordinate, the established workflow falls back to uniform branch
    simplification.
    """
    import trimesh

    decimator = decimator or simplify_mesh_fine
    if soma_pos is None or len(soma_pos) == 0:
        target = max(100, int(len(trimesh_obj.faces) * (1 - skeleton_simp)))
        return decimator(trimesh_obj, target)

    soma_pos = np.asarray(soma_pos, dtype=float).flatten()[:3]
    if soma_pos.size != 3 or not np.all(np.isfinite(soma_pos)):
        target = max(100, int(len(trimesh_obj.faces) * (1 - skeleton_simp)))
        return decimator(trimesh_obj, target)

    vertices = np.asarray(trimesh_obj.vertices)
    faces = np.asarray(trimesh_obj.faces)
    centroids = vertices[faces].mean(axis=1)
    distances = np.linalg.norm(centroids - soma_pos, axis=1)
    soma_faces = faces[distances <= float(soma_radius)]
    skeleton_faces = faces[distances > float(soma_radius)]

    if len(soma_faces) == 0:
        target = max(100, int(len(faces) * (1 - skeleton_simp)))
        return decimator(trimesh_obj, target)
    if len(skeleton_faces) == 0:
        target = max(100, int(len(faces) * (1 - soma_simp)))
        return decimator(trimesh_obj, target)

    try:
        soma_mesh = trimesh.Trimesh(
            vertices=vertices, faces=soma_faces, process=False)
        skeleton_mesh = trimesh.Trimesh(
            vertices=vertices, faces=skeleton_faces, process=False)
        soma_mesh.remove_unreferenced_vertices()
        skeleton_mesh.remove_unreferenced_vertices()
        soma_target = max(50, int(len(soma_mesh.faces) * (1 - soma_simp)))
        skeleton_target = max(
            50, int(len(skeleton_mesh.faces) * (1 - skeleton_simp)))
        return trimesh.util.concatenate([
            decimator(soma_mesh, soma_target),
            decimator(skeleton_mesh, skeleton_target),
        ])
    except Exception:
        target = max(100, int(len(faces) * (1 - skeleton_simp)))
        return decimator(trimesh_obj, target)


def prepare_flywire_mesh(
        mesh,
        body_id,
        soma_pos=None,
        simplification: float = FLYWIRE_MESH_CACHE_SIMPLIFICATION,
        soma_simplification: float = FLYWIRE_MESH_CACHE_SOMA_SIMPLIFICATION,
        soma_radius: float = FLYWIRE_MESH_CACHE_SOMA_RADIUS,
        decimator=None,
        ):
    """Return a prepared ``MeshNeuron`` without skeletonizing it."""
    if not isinstance(mesh, navis.MeshNeuron):
        raise TypeError(
            f"FlyWire mesh preparation expected MeshNeuron, got "
            f"{type(mesh).__name__}")
    source_soma_pos = soma_pos
    if source_soma_pos is None:
        source_soma_pos = getattr(mesh, "soma_pos", None)
    simplified = simplify_mesh_with_soma_awareness(
        mesh.trimesh,
        skeleton_simp=simplification,
        soma_simp=soma_simplification,
        soma_pos=source_soma_pos,
        soma_radius=soma_radius,
        decimator=decimator,
    )
    prepared = navis.MeshNeuron(
        simplified,
        id=normalize_flywire_body_id(body_id),
        name=str(body_id),
        units="nm",
    )
    if source_soma_pos is not None:
        try:
            prepared.soma_pos = np.asarray(source_soma_pos, dtype=float)
        except Exception:
            pass
    return prepared


def parse_soma_position(value):
    """Parse FAFB ``position`` metadata such as ``"[710944 262716 205680]"``."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(value, dtype=float).flatten()
    else:
        numbers = re.findall(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?",
                             str(value))
        arr = np.asarray([float(n) for n in numbers], dtype=float)
    if arr.size < 3 or not np.all(np.isfinite(arr[:3])):
        return None
    return arr[:3]
