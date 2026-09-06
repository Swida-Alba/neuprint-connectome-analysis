"""Regenerate the bundled neuron indexes shipped with the repository.

The app-owned ``neuron_indexes/`` directory doubles as the runtime index
store: the pull pipeline builds every dataset's index there, and a few
bundled datasets ship committed "seed" indexes so auto-suggestions and the
available-neurons viewer work immediately after a fresh install.  This script
rebuilds those seeds from the local ``datasets/`` metadata tables with
zero-filled cache-progress flags, and refreshes
``neuron_indexes/manifest.json``.

Usage:
    python src/build_seed_indexes.py
    python src/build_seed_indexes.py --datasets male-cns:v1.0,flywire_FAFB_v783
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Import the builder as a top-level module (src/ on sys.path) so this script
# does not trigger the heavy ``src/__init__`` package import tree.
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from neuron_index_builder import (  # noqa: E402
    build_search_cache_frame,
    dataset_folder,
    is_search_cache_compatible,
    metadata_path,
    ordered_projection_columns,
    read_metadata_projection,
    search_cache_path,
    system_neuron_index_path,
)

# Datasets whose indexes are committed to the repository.
SEED_DATASETS = (
    "male-cns:v1.0",
    "flywire_FAFB_v783",
    "banc_v888",
)

MANIFEST_FILENAME = "manifest.json"


def _atomic_write_parquet(frame, path: Path) -> None:
    """Write *frame* atomically so a reader never sees a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = f'{path}.tmp-{os.getpid()}'
    try:
        frame.write_parquet(temporary, compression='zstd')
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            try:
                os.remove(temporary)
            except OSError:
                pass


def _seed_frame(source: Path):
    """Materialize the seed projection with zeroed cache-progress flags.

    A fresh install must never claim connections are cached, so the shipped
    index carries the pull-built schema but with ``downstream_complete`` /
    ``last_fetched`` / ``connection_count`` all at their defaults.
    """
    import polars as pl

    frame = read_metadata_projection(source)
    for column, default in (
        ('downstream_complete', False),
        ('last_fetched', ''),
        ('connection_count', 0),
    ):
        if column in frame.columns:
            frame = frame.with_columns(
                pl.lit(default).alias(column)
            )
        else:
            frame = frame.with_columns(pl.lit(default).alias(column))
    if 'post' not in frame.columns:
        frame = frame.with_columns(pl.lit(0).alias('post'))
    return frame.select(ordered_projection_columns(frame.columns))


def build_seed_index(dataset: str, index_dir: Path) -> Optional[dict]:
    """Rebuild one bundled seed index and return its manifest entry."""
    import polars as pl

    source = metadata_path(dataset, _PROJECT_ROOT / 'datasets')
    if source is None:
        print(f'  ! No local metadata table for {dataset}; seed left untouched')
        return None

    index_path = system_neuron_index_path(dataset, index_dir)
    search_path = search_cache_path(index_path)

    frame = _seed_frame(source)
    if frame.height == 0 or frame['bodyId'].is_null().all():
        raise ValueError(f'{dataset}: seed projection has no usable bodyIds')

    _atomic_write_parquet(frame, index_path)
    _atomic_write_parquet(build_search_cache_frame(frame), search_path)

    # A corrupt or stale seed would silently degrade suggestions and the
    # viewer for every fresh install; fail loudly instead.
    if not is_search_cache_compatible(
        pl.read_parquet(search_path), frame.columns
    ):
        raise ValueError(f'{dataset}: built search sidecar failed validation')
    if frame['bodyId'].n_unique() != frame.height:
        raise ValueError(f'{dataset}: seed projection has duplicate bodyIds')

    return {
        'dataset': dataset,
        'folder': dataset_folder(dataset),
        'source': source.name,
        'rows': frame.height,
        'index_bytes': index_path.stat().st_size,
        'search_bytes': search_path.stat().st_size,
    }


def _read_manifest(index_dir: Path) -> dict:
    manifest_path = index_dir / MANIFEST_FILENAME
    if manifest_path.is_file():
        try:
            return json.loads(manifest_path.read_text(encoding='utf-8'))
        except (ValueError, OSError):
            pass
    return {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--datasets',
        help='Comma-separated dataset ids to rebuild (default: all bundled)',
    )
    args = parser.parse_args()

    index_dir = _PROJECT_ROOT / 'neuron_indexes'
    index_dir.mkdir(parents=True, exist_ok=True)

    datasets: List[str] = [
        item.strip()
        for item in (args.datasets or '').split(',')
        if item.strip()
    ] or list(SEED_DATASETS)

    entries: Dict[str, dict] = {}
    print(f'Rebuilding bundled neuron indexes in {index_dir}:')
    for dataset in datasets:
        print(f'- {dataset}')
        try:
            entry = build_seed_index(dataset, index_dir)
        except Exception as exc:
            print(f'  ✗ Failed: {exc}')
            return 1
        if entry is None:
            continue
        entries[dataset] = entry
        print(
            f'  ✓ {entry["rows"]:,} rows · index {entry["index_bytes"] / 1048576:.1f} MB · '
            f'search {entry["search_bytes"] / 1048576:.1f} MB'
        )

    if not entries:
        print('Nothing was rebuilt.')
        return 1

    manifest = {
        'generated_utc': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'index_dir': 'neuron_indexes',
        'datasets': entries,
    }
    manifest_path = index_dir / MANIFEST_FILENAME
    temporary = f'{manifest_path}.tmp-{os.getpid()}'
    with open(temporary, 'w', encoding='utf-8') as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write('\n')
    os.replace(temporary, manifest_path)
    print(f'✓ Manifest written: {manifest_path.name}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
