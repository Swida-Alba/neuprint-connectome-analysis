"""
Dataset Service for DROCAT UI
Fetches available datasets from NeuPrint server dynamically.
"""

import json
import os
import threading
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from .config import PROJECT_ROOT

try:
    # Shared token resolution: the server probe that lets a rejected
    # candidate fall through to the next location in the chain.
    from src.utils.token_manager import token_manager as _shared_token_manager
except ImportError:  # src not on sys.path; fall back to first-found tokens
    _shared_token_manager = None


@dataclass
class DatasetInfo:
    """Information about a dataset."""
    name: str
    source: str  # 'neuprint', 'flywire', 'local'
    available: bool = False
    neuron_count: int = 0
    typed_count: int = 0
    local_cache: bool = False
    local_prepared: bool = False  # Has local data files ready to use
    display_name: str = ""  # Human-readable name from Codex/server
    metadata: Dict = field(default_factory=dict)
    error: Optional[str] = None


# Dataset name normalization: folder name <-> dataset name
# e.g. 'hemibrain_v1_2_1' <-> 'hemibrain:v1.2.1'
def folder_to_dataset(folder_name: str) -> str:
    """Convert a folder name to a dataset identifier."""
    # FlyWire datasets keep their names as-is
    if "flywire" in folder_name.lower():
        return folder_name
    # NeuPrint: first _ becomes :, remaining _ become .
    # e.g. hemibrain_v1_2_1 -> hemibrain:v1.2.1
    #      male-cns_v0_9 -> male-cns:v0.9
    parts = folder_name.split("_", 1)
    if len(parts) == 2:
        prefix, version = parts
        version = version.replace("_", ".")
        return f"{prefix}:{version}"
    return folder_name


def dataset_to_folder(dataset: str) -> str:
    """Convert a dataset identifier to a folder name."""
    # FlyWire datasets keep their names as-is
    if "flywire" in dataset.lower():
        return dataset
    # NeuPrint: : becomes _, . becomes _
    return dataset.replace(":", "_").replace(".", "_")


def is_flywire_dataset(dataset: str) -> bool:
    """Return whether *dataset* is a FlyWire identifier.

    The NeuPrint server metadata lists a hidden dataset named ``banc:v888``
    that is not queryable through the API (BANC is served via FlyWire/
    Codex).  Do not classify that identifier as the local FlyWire BANC
    release merely because it contains the word ``banc``.
    """
    normalized = dataset.strip().lower()
    return normalized.startswith("flywire_") or "fafb" in normalized


def is_banc_dataset(dataset: str) -> bool:
    """Return whether *dataset* identifies a BANC release.

    BANC remains listed in the general dataset catalog for supported
    connectivity tools, but morphology/skeleton tabs use this predicate to
    show the option as unavailable.
    """

    return "banc" in str(dataset or "").strip().lower()


class DatasetService:
    """Service for fetching and managing dataset availability."""

    AVAILABILITY_CACHE_FORMAT = "drocat_dataset_availability/v1"
    AVAILABILITY_CACHE_FILENAME = "dataset_availability.json"

    # Known NeuPrint dataset candidates (fallback if /api/dbmeta/datasets fails)
    NEUPRINT_CANDIDATES = [
        "male-cns:v1.0",
        "male-cns:v0.9",
        "hemibrain:v1.2.1",
        "hemibrain:v1.1",
        "optic-lobe:v1.1",
        "optic-lobe:v1.0.1",
        "manc:v1.2.3",
        "manc:v1.2.1",
        "manc:v1.0",
        "fib19:v1.0",
        "mushroombody",
    ]

    # FlyWire/Codex datasets (from https://codex.flywire.ai/)
    # These require local files + CAVE token for API access
    FLYWIRE_DATASETS = [
        "flywire_FAFB_v783",
        "flywire_BANC_v888",
        "flywire_BANC_v626",
    ]

    # Codex display info (fetched from codex.flywire.ai rendered page)
    CODEX_DATASETS = {
        "flywire_FAFB_v783": {"display": "FAFB v783 (CB)", "desc": "Female Adult Fly Brain", "neurons": 139255},
        "flywire_BANC_v888": {"display": "BANC v888 (CNS)", "desc": "Brain and Nerve Cord", "neurons": 158262},
        "flywire_BANC_v626": {"display": "BANC v626 (CNS)", "desc": "Brain and Nerve Cord (older)", "neurons": 115151},
    }

    NEUPRINT_SERVER = "https://neuprint.janelia.org"
    CODEX_URL = "https://codex.flywire.ai/"

    def __init__(self):
        self._token: Optional[str] = None
        self._cave_token: Optional[str] = None
        # Ordered neuprint candidates from the config files:
        # [('config.json', '...'), ('config_local.json', '...')].
        self._token_chain: List[Tuple[str, str]] = []
        self._cache: Dict[str, DatasetInfo] = {}
        self._lock = threading.Lock()
        self._datasets_dir = PROJECT_ROOT / "datasets"
        self._cache_dir = PROJECT_ROOT / "cache"
        self._index_dir = PROJECT_ROOT / "neuron_indexes"
        self._available_neuprint: Optional[List[str]] = None
        self._server_datasets: Dict[str, dict] = {}  # Full server response from /api/dbmeta/datasets
        self._last_fetch_time: float = 0
        self._availability_snapshot: Dict[str, DatasetInfo] = {}
        self._availability_updated_at: Optional[str] = None
        self._availability_loaded = False

    @property
    def availability_cache_path(self) -> Path:
        """Persistent file containing the last complete availability refresh."""
        return self._cache_dir / self.AVAILABILITY_CACHE_FILENAME

    @staticmethod
    def _serialize_info(info: DatasetInfo) -> dict:
        """Convert a DatasetInfo into the JSON-safe snapshot representation."""
        return {
            "name": info.name,
            "source": info.source,
            "available": bool(info.available),
            "neuron_count": int(info.neuron_count or 0),
            "typed_count": int(info.typed_count or 0),
            "local_cache": bool(info.local_cache),
            "local_prepared": bool(info.local_prepared),
            "display_name": info.display_name or "",
            "metadata": info.metadata or {},
            "error": info.error,
        }

    @staticmethod
    def _deserialize_info(name: str, data: dict) -> Optional[DatasetInfo]:
        """Rebuild one DatasetInfo from a persisted snapshot row."""
        if not isinstance(data, dict):
            return None
        try:
            return DatasetInfo(
                name=str(data.get("name") or name),
                source=str(data.get("source") or "unknown"),
                available=bool(data.get("available", False)),
                neuron_count=int(data.get("neuron_count", 0) or 0),
                typed_count=int(data.get("typed_count", 0) or 0),
                local_cache=bool(data.get("local_cache", False)),
                local_prepared=bool(data.get("local_prepared", False)),
                display_name=str(data.get("display_name") or ""),
                metadata=data.get("metadata") if isinstance(data.get("metadata"), dict) else {},
                error=data.get("error"),
            )
        except (TypeError, ValueError):
            return None

    def _load_persisted_availability(self) -> None:
        """Load the last refresh into the runtime cache, if it exists."""
        if self._availability_loaded:
            return

        snapshot: Dict[str, DatasetInfo] = {}
        updated_at: Optional[str] = None
        path = self.availability_cache_path
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                updated_at = str(payload.get("updated_at") or "") or None
                rows = payload.get("datasets") or {}
                if isinstance(rows, dict):
                    for name, raw in rows.items():
                        info = self._deserialize_info(str(name), raw)
                        if info is not None:
                            snapshot[info.name] = info
        except (OSError, ValueError, TypeError):
            # A corrupt or partially-written snapshot must never prevent the
            # Settings page from loading; the next successful refresh replaces
            # it atomically.
            snapshot = {}
            updated_at = None

        with self._lock:
            self._availability_snapshot = snapshot
            self._availability_updated_at = updated_at
            self._availability_loaded = True
            # Keep the normal in-memory lookup path useful to selectors and
            # tools that are opened before the Settings tab.
            self._cache.update(snapshot)

    def get_cached_availability(self) -> Tuple[Dict[str, DatasetInfo], Optional[str]]:
        """Return the persisted availability rows and their refresh time."""
        self._load_persisted_availability()
        with self._lock:
            # Settings can clear the transient service cache after a token
            # change; restore the persisted rows when the next tab asks for
            # the snapshot so dataset selectors keep the same status labels.
            self._cache.update(self._availability_snapshot)
            return dict(self._availability_snapshot), self._availability_updated_at

    def get_availability_updated_at(self) -> Optional[str]:
        """Return the timestamp of the last successful availability refresh."""
        _results, updated_at = self.get_cached_availability()
        return updated_at

    def _persist_availability(
        self,
        results: Dict[str, DatasetInfo],
        replace: bool = True,
    ) -> str:
        """Persist a completed refresh and replace the active snapshot."""
        self._load_persisted_availability()
        with self._lock:
            snapshot = dict(results) if replace else {
                **self._availability_snapshot,
                **results,
            }
        updated_at = datetime.now().astimezone().isoformat(timespec="seconds")
        payload = {
            "format": self.AVAILABILITY_CACHE_FORMAT,
            "updated_at": updated_at,
            "datasets": {
                name: self._serialize_info(info)
                for name, info in snapshot.items()
            },
        }

        path = self.availability_cache_path
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_name(f".{path.name}.tmp")
        try:
            temp_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
            temp_path.replace(path)
        except OSError:
            try:
                temp_path.unlink()
            except OSError:
                pass
            raise

        with self._lock:
            self._availability_snapshot = snapshot
            self._availability_updated_at = updated_at
            self._availability_loaded = True
            self._cache.clear()
            self._cache.update(snapshot)
        return updated_at

    def _load_tokens(self):
        """Load tokens from config.json (primary) then config_local.json."""
        if self._token is not None and self._cave_token is not None:
            return

        loaded = {}
        chain = []
        # config.json comes first per key (the file a GitHub-pulled copy
        # edits directly); the gitignored config_local.json follows in the
        # fallback chain.
        for filename in ("config.json", "config_local.json"):
            config_path = PROJECT_ROOT / filename
            if not config_path.exists():
                continue
            try:
                # utf-8-sig tolerates the UTF-8 BOM that Windows editors
                # prepend to saved JSON files.
                with open(config_path, "r", encoding="utf-8-sig") as f:
                    cfg = json.load(f)
                cfg_tokens = cfg.get("tokens") or {}
                if isinstance(cfg_tokens, dict):
                    for key in ("neuprint", "cave"):
                        value = cfg_tokens.get(key)
                        if isinstance(value, str) and value.strip() and not value.startswith("YOUR_"):
                            # First non-empty value wins: config.json is read first.
                            loaded.setdefault(key, value.strip())
                            if key == "neuprint":
                                chain.append((filename, value.strip()))
            except (OSError, ValueError):
                pass

        if self._token is None:
            self._token = loaded.get("neuprint")
        if self._cave_token is None:
            self._cave_token = loaded.get("cave")
        self._token_chain = chain

    def get_token(self) -> Optional[str]:
        """Get NeuPrint token (config.json -> config_local.json -> env).

        Config wins per the standard chain, so a config update overrides a
        shell-exported NEUPRINT_APPLICATION_CREDENTIALS/NEUPRINT_TOKEN.
        Candidates the NeuPrint server refuses (401/403) are skipped and the
        next location is checked instead of failing the request; a probe
        that cannot run (e.g. offline) never disqualifies a candidate. With
        every candidate refused, the first is returned so callers surface
        the rejected-token error.
        """
        self._load_tokens()
        candidates = []
        if self._token:
            candidates.append(("config", self._token))
            # Deeper config candidates (config_local.json when config.json
            # also supplied a value) stay in the chain behind the winner.
            for source, value in self._token_chain:
                if value != self._token:
                    candidates.append((source, value))
        env_token = (os.environ.get("NEUPRINT_APPLICATION_CREDENTIALS")
                     or os.environ.get("NEUPRINT_TOKEN"))
        if env_token and env_token.strip() \
                and not env_token.strip().startswith("YOUR_"):
            candidates.append(("environment", env_token.strip()))
        if not candidates:
            return None
        if _shared_token_manager is None or len(candidates) == 1:
            return candidates[0][1]
        first = candidates[0][1]
        probed = set()
        for _source, value in candidates:
            if value in probed:
                continue
            probed.add(value)
            if not _shared_token_manager.neuprint_token_rejected(value):
                return value
        return first

    def get_cave_token(self) -> Optional[str]:
        """Get CAVE token (config.json -> config_local.json -> env)."""
        self._load_tokens()
        if self._cave_token:
            return self._cave_token
        return os.environ.get("CAVE_TOKEN")

    def fetch_neuprint_datasets(self) -> List[str]:
        """
        Fetch the FULL list of available NeuPrint datasets from the server API.
        Uses /api/dbmeta/datasets endpoint with Bearer token auth.
        Falls back to probing known candidates if API call fails.
        Returns a list of available dataset names.
        """
        self._load_tokens()

        token = self.get_token()
        if not token:
            return []

        # Try the proper API endpoint first
        try:
            import requests
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-type": "application/json",
            }
            r = requests.get(
                f"{self.NEUPRINT_SERVER}/api/dbmeta/datasets",
                headers=headers,
                timeout=15,
            )
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, dict):
                    # Store full server metadata, excluding datasets the
                    # server marks as hidden (e.g. banc:v888): they are
                    # listed but not queryable through the API.
                    self._server_datasets = data
                    available = sorted(
                        name for name, meta in data.items()
                        if not (isinstance(meta, dict)
                                and str(meta.get("hidden", "")).lower() == "true")
                    )
                    self._available_neuprint = available
                    self._last_fetch_time = time.time()
                    return available
        except Exception:
            pass

        # Fallback: probe known candidates individually
        available = []
        for ds in self.NEUPRINT_CANDIDATES:
            info = self._probe_neuprint_dataset(ds)
            if info.available:
                available.append(ds)

        self._available_neuprint = available
        self._last_fetch_time = time.time()
        return available

    def get_all_datasets(self) -> List[str]:
        """
        Get list of all available datasets (NeuPrint + FlyWire).
        If we have fetched from the server, includes ALL server datasets
        (including older versions like hemibrain:v1.1, fib19:v1.0, etc.).
        """
        if self._available_neuprint is not None:
            return self._available_neuprint + self.FLYWIRE_DATASETS
        return self.NEUPRINT_CANDIDATES + self.FLYWIRE_DATASETS

    def get_neuprint_datasets(self) -> List[str]:
        """Get list of available NeuPrint datasets."""
        if self._available_neuprint is not None:
            return self._available_neuprint.copy()
        return self.NEUPRINT_CANDIDATES.copy()

    def get_flywire_datasets(self) -> List[str]:
        """Get list of FlyWire datasets."""
        return self.FLYWIRE_DATASETS.copy()

    def check_dataset_availability(self, dataset: str) -> DatasetInfo:
        """
        Check if a dataset is available.
        Uses cached results if available.
        """
        with self._lock:
            if dataset in self._cache:
                return self._cache[dataset]

        self._load_tokens()

        # Determine source
        if is_flywire_dataset(dataset):
            info = self._check_flywire_dataset(dataset)
        else:
            info = self._probe_neuprint_dataset(dataset)

        # Check local cache/prepared status
        info.local_cache = self._check_local_cache(dataset)
        info.local_prepared = self._check_local_prepared(dataset)
        # Backfill the neuron count from local files when the server path
        # could not provide one (datasets without metadata files).
        if info.neuron_count == 0:
            total, typed = self._load_local_neuron_counts(dataset)
            info.neuron_count = total
            info.typed_count = typed
        if info.source == "flywire":
            # FlyWire has no server-backed dataset status in this UI.  A
            # directory (or a lone neuron table) is not enough to call it
            # available; both converter outputs must be present.
            info.available = info.local_prepared
            if not info.local_prepared:
                info.error = "Local FlyWire neuron and connection tables are not both prepared."

        # Last resorts so every listed dataset still reports a count:
        # known FlyWire release sizes, a live NeuPrint count query, and
        # finally the local pull-cache index.
        if info.neuron_count == 0 and info.source == "flywire":
            codex = self.CODEX_DATASETS.get(dataset) or {}
            if codex.get("neurons"):
                info.neuron_count = int(codex["neurons"])
        if info.neuron_count == 0 and info.available and info.source == "neuprint":
            total, typed = self._fetch_neuprint_counts(dataset)
            info.neuron_count = total
            info.typed_count = typed
        if info.neuron_count == 0 and info.local_cache:
            total, typed = self._load_cache_neuron_counts(dataset)
            info.neuron_count = total
            info.typed_count = typed

        # Set display name
        if not info.display_name:
            if dataset in self.CODEX_DATASETS:
                info.display_name = self.CODEX_DATASETS[dataset]["display"]
            else:
                info.display_name = dataset

        # Cache result
        with self._lock:
            self._cache[dataset] = info

        return info

    def fetch_codex_datasets(self) -> Dict[str, dict]:
        """
        Fetch available FlyWire datasets from Codex (codex.flywire.ai).
        Returns dict of {dataset_name: {display, desc, neurons}}.
        Falls back to hardcoded CODEX_DATASETS if fetch fails.
        """
        try:
            import requests
            from bs4 import BeautifulSoup
            r = requests.get(self.CODEX_URL, timeout=10)
            if r.status_code == 200:
                soup = BeautifulSoup(r.text, 'html.parser')
                # Look for dataset cards in the rendered HTML
                # The page structure has dataset names like "FAFB v783 (CB)"
                text = soup.get_text()
                # Parse known patterns
                import re
                patterns = [
                    (r'FAFB\s+v(\d+)', 'flywire_FAFB_v{}', 'FAFB v{} (CB)', 'Female Adult Fly Brain'),
                    (r'BANC\s+v(\d+)', 'flywire_BANC_v{}', 'BANC v{} (CNS)', 'Brain and Nerve Cord'),
                ]
                found = {}
                for pattern, key_fmt, display_fmt, desc in patterns:
                    matches = re.findall(pattern, text)
                    for ver in matches:
                        key = key_fmt.format(ver)
                        display = display_fmt.format(ver)
                        found[key] = {"display": display, "desc": desc, "neurons": 0}
                if found:
                    # Update our CODEX_DATASETS with fetched info
                    self.CODEX_DATASETS.update(found)
                    # Also update FLYWIRE_DATASETS list
                    for k in found:
                        if k not in self.FLYWIRE_DATASETS:
                            self.FLYWIRE_DATASETS.append(k)
                    return found
        except Exception:
            pass
        return self.CODEX_DATASETS

    def _probe_neuprint_dataset(self, dataset: str) -> DatasetInfo:
        """Check NeuPrint dataset availability.
        Uses server metadata from /api/dbmeta/datasets if available (fast).
        Falls back to individual query for neuron counts (slow).
        """
        info = DatasetInfo(name=dataset, source="neuprint")

        if not self.get_token():
            info.error = "No NeuPrint token configured"
            return info

        # Fast path: use server metadata if we have it
        if self._server_datasets and dataset in self._server_datasets:
            server_info = self._server_datasets[dataset]
            # Hidden datasets (e.g. banc:v888) are listed by the server but
            # not queryable through the API, so they stay unavailable here.
            if isinstance(server_info, dict) and str(
                    server_info.get("hidden", "")).lower() == "true":
                info.error = (
                    "Hidden on the NeuPrint server (not queryable through the API)"
                )
                return info
            info.available = True
            info.metadata = {
                "server": self.NEUPRINT_SERVER,
                "last_mod": server_info.get("last-mod", ""),
                "uuid": server_info.get("uuid", ""),
                "rois": server_info.get("ROIs", []),
            }
            # Set display name from server info
            info.display_name = dataset
            return info

        # Slow path: individual query (only for datasets not in server list)
        total, typed = self._fetch_neuprint_counts(dataset)
        if total:
            info.available = True
            info.neuron_count = total
            info.typed_count = typed
            info.metadata = {
                "server": self.NEUPRINT_SERVER,
                "checked_at": datetime.now().isoformat(),
            }
        else:
            info.error = "Empty response from server"

        return info

    def _fetch_neuprint_counts(self, dataset: str) -> tuple:
        """Query the NeuPrint server for a dataset's total/typed neuron count.

        Returns (0, 0) when the dataset cannot be queried (no token, network
        failure, or an empty response).  Used both by the slow availability
        probe and as a last resort for server datasets without local files.
        """
        token = self.get_token()
        if not token:
            return 0, 0
        try:
            from neuprint import Client

            client = Client(self.NEUPRINT_SERVER, dataset, token)
            result = client.fetch_custom(
                "MATCH (n:Neuron) RETURN count(n) as total, "
                "sum(CASE WHEN n.type IS NOT NULL AND n.type <> '' THEN 1 ELSE 0 END) as typed"
            )
            if not result.empty:
                return int(result["total"].iloc[0]), int(result["typed"].iloc[0])
        except Exception:
            pass
        return 0, 0

    def _check_flywire_dataset(self, dataset: str) -> DatasetInfo:
        """Check FlyWire dataset availability."""
        info = DatasetInfo(name=dataset, source="flywire")

        local_path = self._get_dataset_path(dataset)
        if local_path and local_path.exists():
            info.available = True
            info.local_cache = True

            # Counts: metadata file first, local neuron table fallback
            total, typed = self._load_local_neuron_counts(dataset)
            info.neuron_count = total
            info.typed_count = typed
            metadata_file = self._find_metadata_file(dataset)
            if metadata_file and metadata_file.exists():
                try:
                    with open(metadata_file, "r") as f:
                        info.metadata = json.load(f)
                except Exception:
                    pass
        else:
            info.error = (
                "Local FlyWire dataset is not prepared. Put the raw Codex files "
                "under datasets/<dataset>/downloads/ and run the converter."
            )

        return info

    def _check_local_cache(self, dataset: str) -> bool:
        """Check if dataset has local cache files."""
        dataset_path = self._get_dataset_path(dataset)
        if dataset_path and dataset_path.exists():
            for pattern in ["*_neuron_df.csv", "*_neuron_df.parquet", "*_allneurons*.csv"]:
                if list(dataset_path.glob(pattern)):
                    return True

        cache_path = self._cache_dir / dataset_to_folder(dataset)
        if cache_path.exists():
            if (cache_path / "connections.parquet").exists():
                return True
            # The neuron index is an app-owned "system file" now
            # (neuron_indexes/), not part of the cache.

        return False

    def _check_local_prepared(self, dataset: str) -> bool:
        """Check if dataset has local data files ready for analysis (not just cache)."""
        dataset_path = self._get_dataset_path(dataset)
        if dataset_path and dataset_path.exists():
            if not is_flywire_dataset(dataset):
                # NeuPrint can legitimately use a local neuron table while
                # connections remain server-backed; preserve that behavior.
                return any(
                    path
                    for pattern in ("*_neuron_df.csv", "*_neuron_df.parquet")
                    for path in dataset_path.glob(pattern)
                )

            # A FlyWire conversion is usable only when both generated tables
            # exist.  A neuron table by itself is not enough for pathfinding:
            # the converter also writes the merged connection table.
            neuron_ready = any(
                path
                for pattern in (
                    "*_allneurons_neuron_df.parquet",
                    "*_allneurons_neuron_df.csv",
                )
                for path in dataset_path.glob(pattern)
            )
            connections_ready = any(
                path
                for pattern in (
                    "*_merged_connections.parquet",
                    "*_merged_connections.csv",
                )
                for path in dataset_path.glob(pattern)
            )
            if neuron_ready and connections_ready:
                return True
        return False

    def _get_dataset_path(self, dataset: str) -> Optional[Path]:
        """Get the local path for a dataset."""
        safe_name = dataset_to_folder(dataset)
        return self._datasets_dir / safe_name

    def _find_metadata_file(self, dataset: str) -> Optional[Path]:
        """Find metadata file for a dataset."""
        dataset_path = self._get_dataset_path(dataset)
        if not dataset_path or not dataset_path.exists():
            return None

        safe_name = dataset_to_folder(dataset)
        metadata_file = dataset_path / f"{safe_name}_metadata.json"
        if metadata_file.exists():
            return metadata_file

        for f in dataset_path.glob("*_metadata.json"):
            return f

        return None

    # neuron-count memoization: (dataset, mtime_ns) -> (total, typed).
    # Counting a large neuron CSV on every page load is wasteful; the count is
    # re-read only when the local table changes.
    _neuron_counts_cache: Dict[tuple, tuple] = {}

    def _load_local_neuron_counts(self, dataset: str) -> tuple:
        """
        Total / typed neuron counts for a locally prepared dataset.

        Priority:
          1. ``*_metadata.json`` (``neuron_counts.total`` / ``typed``)
          2. local neuron table (``*_allneurons_neuron_df.parquet`` or
             ``.csv``, or a plain ``*_neuron_df.parquet``/``.csv`` from a
             NeuPrint conversion) counted via a streaming Polars scan - this
             covers datasets that were pulled/downloaded without a metadata
             file.

        The result is memoized per (dataset, table mtime); falls back to
        (0, 0) when neither source is available.
        """
        dataset_path = self._get_dataset_path(dataset)
        if not dataset_path or not dataset_path.exists():
            return 0, 0

        # 1. Metadata file wins when present (it is authoritative and cheap).
        metadata_file = self._find_metadata_file(dataset)
        if metadata_file and metadata_file.exists():
            try:
                with open(metadata_file, "r") as f:
                    meta = json.load(f)
                counts = meta.get("neuron_counts", {}) or {}
                if counts.get("total"):
                    return int(counts["total"]), int(counts.get("typed", 0) or 0)
            except Exception:
                pass

        # 2. Fall back to counting the local neuron table.  Prefer the full
        #    ``*_allneurons_neuron_df.*`` table; the plain ``*_neuron_df.*``
        #    names must match too, or prepared datasets would show no count.
        table = None
        for pattern in (
            "*_allneurons_neuron_df.parquet",
            "*_allneurons_neuron_df.csv",
            "*_neuron_df.parquet",
            "*_neuron_df.csv",
        ):
            matches = sorted(dataset_path.glob(pattern))
            if matches:
                table = matches[0]
                break
        if table is None:
            return 0, 0

        key = (dataset, table.stat().st_mtime_ns)
        if key in self._neuron_counts_cache:
            return self._neuron_counts_cache[key]

        total = typed = 0
        try:
            import polars as pl

            lazy = pl.scan_parquet(table) if table.suffix == ".parquet" else pl.scan_csv(
                table, infer_schema_length=0, ignore_errors=True
            )
            if "type" in lazy.collect_schema().names():
                # NOTE: do not apply a frame-level `.sum()` to `pl.len()` -
                # that corrupts the total (e.g. 158262 rows -> 3572024164).
                row = lazy.select(
                    pl.len().alias("total"),
                    (
                        pl.col("type").is_not_null()
                        & (pl.col("type") != "")
                        & (pl.col("type").cast(pl.Utf8) != "nan")
                    ).sum().alias("typed"),
                ).collect().row(0)
                total, typed = int(row[0]), int(row[1])
            else:
                total = int(lazy.select(pl.len()).collect().item())
        except Exception:
            # Last resort: plain pandas row count
            try:
                import pandas as pd

                if table.suffix == ".parquet":
                    df = pd.read_parquet(table)
                else:
                    df = pd.read_csv(table, index_col=0, low_memory=False)
                total = len(df)
                typed = int(df["type"].notna().sum()) if "type" in df.columns else 0
            except Exception:
                return 0, 0

        self._neuron_counts_cache[key] = (total, typed)
        return total, typed

    def _load_cache_neuron_counts(self, dataset: str) -> tuple:
        """
        Total / typed neuron counts from the app-owned neuron index.

        ``neuron_indexes/<dataset>/neuron_index.parquet`` is written by the
        DatasetPuller with one row per cached neuron (or shipped as a bundled
        seed).  The count may be partial (only what has been pulled so far),
        so it is used only as a last resort behind the dataset tables and the
        server query.
        """
        index = self._index_dir / dataset_to_folder(dataset) / "neuron_index.parquet"
        if not index.exists():
            return 0, 0

        key = ("cache", dataset, index.stat().st_mtime_ns)
        if key in self._neuron_counts_cache:
            return self._neuron_counts_cache[key]

        total = typed = 0
        try:
            import polars as pl

            lazy = pl.scan_parquet(index)
            if "type" in lazy.collect_schema().names():
                row = lazy.select(
                    pl.len().alias("total"),
                    (
                        pl.col("type").is_not_null()
                        & (pl.col("type") != "")
                        & (pl.col("type").cast(pl.Utf8) != "nan")
                    ).sum().alias("typed"),
                ).collect().row(0)
                total, typed = int(row[0]), int(row[1])
            else:
                total = int(lazy.select(pl.len()).collect().item())
        except Exception:
            return 0, 0

        self._neuron_counts_cache[key] = (total, typed)
        return total, typed

    def get_local_datasets(self) -> List[DatasetInfo]:
        """Get information about locally available datasets."""
        datasets = []

        if not self._datasets_dir.exists():
            return datasets

        for folder in self._datasets_dir.iterdir():
            if folder.is_dir() and not folder.name.startswith("."):
                name = folder_to_dataset(folder.name)

                info = DatasetInfo(
                    name=name,
                    source="flywire" if is_flywire_dataset(name) else "neuprint",
                    local_cache=True,
                )

                # Keep the local listing consistent with
                # check_dataset_availability().  In particular, a FlyWire
                # directory containing only the neuron table is not ready for
                # pathfinding until its merged connection table is present.
                info.local_prepared = self._check_local_prepared(name)
                info.available = info.local_prepared

                # Counts: metadata file first, local neuron table fallback
                # (datasets pulled without a metadata file still show counts).
                total, typed = self._load_local_neuron_counts(name)
                info.neuron_count = total
                info.typed_count = typed
                metadata_file = self._find_metadata_file(name)
                if metadata_file and metadata_file.exists():
                    try:
                        with open(metadata_file, "r") as f:
                            info.metadata = json.load(f)
                    except Exception:
                        pass

                datasets.append(info)

        return datasets

    def refresh_availability(self, datasets: Optional[List[str]] = None) -> Dict[str, DatasetInfo]:
        """
        Refresh availability for all or specific datasets.
        Also fetches FlyWire dataset list from Codex. A completed refresh is
        written to the persistent availability snapshot so the next Settings
        page load starts from this result instead of an older in-memory value.
        """
        refresh_all = datasets is None
        with self._lock:
            self._cache.clear()

        if datasets is None:
            # Fetch FlyWire datasets from Codex
            self.fetch_codex_datasets()
            # Fetch NeuPrint datasets from server
            neuprint_available = self.fetch_neuprint_datasets()
            datasets = neuprint_available + self.FLYWIRE_DATASETS

        results = {}
        for dataset in datasets:
            results[dataset] = self.check_dataset_availability(dataset)

        # A full refresh is authoritative and replaces the prior snapshot.
        # A targeted refresh updates only the requested rows and preserves the
        # rest of the last complete snapshot.
        self._persist_availability(results, replace=refresh_all)
        return results

    def is_cache_fresh(self, max_age_seconds: int = 300) -> bool:
        """Check if the cached availability data is still fresh."""
        if self._last_fetch_time == 0:
            return False
        return (time.time() - self._last_fetch_time) < max_age_seconds


# Global instance
_dataset_service: Optional[DatasetService] = None


def get_dataset_service() -> DatasetService:
    """Get the global DatasetService instance."""
    global _dataset_service
    if _dataset_service is None:
        _dataset_service = DatasetService()
    return _dataset_service
