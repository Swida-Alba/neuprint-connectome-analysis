"""Offline tests for the BANC connectivity cache, metadata regeneration,
and v626/v888 namespace discrimination (audit fix round 2026-09-05).

Covers:
- Fix 2: the stale per-neuron connection cache is rebuilt from the same
  version's merged table, with a strictly per-version state sidecar;
- Fix 3: the metadata sidecar is regenerated table-true (counts, banc
  coverage note, real type coverage excluding 'Unknown');
- Fix 6: the pathfinding cache-only skip must not apply to BANC.
"""
import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import BANC_file_converter as banc  # noqa: E402
from utils.naming_utils import canonical_dataset_name  # noqa: E402


def _make_tables(dataset_dir: Path, rows=3):
    dataset_dir.mkdir(parents=True, exist_ok=True)
    neurons = pd.DataFrame({
        "bodyId": [str(1000 + i) for i in range(rows)],
        "type": ["l-LNv", "Unknown", "aMe12"],
    })
    neurons.to_parquet(
        dataset_dir / "ds_test_allneurons_neuron_df.parquet", index=False)
    conn = pd.DataFrame({
        "bodyId_pre": ["1000", "1000", "1002"],
        "bodyId_post": ["1001", "1002", "1001"],
        "weight": [10, 7, 3],
        "roi": ["AL", "AL", "AL"],
    })
    conn.to_parquet(
        dataset_dir / "ds_test_merged_connections.parquet", index=False)
    return neurons, conn


class TestConnectionCacheRebuild:
    def test_stale_cache_is_rebuilt_from_tables(self, tmp_path):
        dataset_dir = tmp_path / "datasets" / "ds_test"
        cache_dir = tmp_path / "cache" / "ds_test"
        _make_tables(dataset_dir)
        cache_dir.mkdir(parents=True)
        # A stale generation: older than the merged table, wrong rows.
        stale = cache_dir / "connections.parquet"
        stale.write_bytes(b"not-a-parquet")

        assert banc.build_connection_cache_from_tables(
            str(dataset_dir), str(cache_dir)) is True

        cache = pd.read_parquet(stale)
        assert list(cache.columns) == [
            "bodyId_pre", "bodyId_post", "weight", "roi", "cached_date"]
        assert len(cache) == 3
        # id '0' placeholder rows are filtered during the rebuild
        assert set(cache["bodyId_pre"]) == {"1000", "1002"}

    def test_fresh_cache_is_not_rebuilt(self, tmp_path):
        dataset_dir = tmp_path / "datasets" / "ds_test"
        cache_dir = tmp_path / "cache" / "ds_test"
        _make_tables(dataset_dir)
        assert banc.build_connection_cache_from_tables(
            str(dataset_dir), str(cache_dir)) is True
        first_mtime = (cache_dir / "connections.parquet").stat().st_mtime_ns
        # Unchanged merged table (same generation marker): no rebuild.
        assert banc.build_connection_cache_from_tables(
            str(dataset_dir), str(cache_dir)) is True
        assert (cache_dir / "connections.parquet").stat().st_mtime_ns \
            == first_mtime

    def test_state_sidecar_is_strictly_per_version(self, tmp_path):
        """Fix 6: state rows must be a subset of the release's neuron
        table — stale/foreign ids from earlier eras are dropped."""
        dataset_dir = tmp_path / "datasets" / "ds_test"
        cache_dir = tmp_path / "cache" / "ds_test"
        _make_tables(dataset_dir)
        # Pre-existing state carrying a foreign (other-release) id.
        cache_dir.mkdir(parents=True)
        pd.DataFrame({
            "bodyId": ["999999999", "1000"],
            "downstream_complete": [True, False],
            "last_fetched": ["2020-01-01", "2020-01-01"],
            "connection_count": [5, 0],
        }).to_parquet(cache_dir / "neuron_index_state.parquet", index=False)

        banc.build_connection_cache_from_tables(
            str(dataset_dir), str(cache_dir))

        state = pd.read_parquet(cache_dir / "neuron_index_state.parquet")
        assert set(state["bodyId"].astype(str)) == {"1000", "1001", "1002"}
        assert set(state["downstream_complete"]) == {True}
        counts = state.set_index("bodyId")["connection_count"].to_dict()
        # BANC-05: partner-row COUNT (aligned with the NeuPrint pull and
        # the FlyWire import), not the weight sum — neuron 1000 has two
        # outgoing partner rows (weights 10 + 7).
        assert counts["1000"] == 2
        assert counts["1001"] == 0


class TestBancMetadataRegeneration:
    def test_stats_are_table_true(self, tmp_path):
        dataset_dir = tmp_path / "datasets" / "ds_test"
        _make_tables(dataset_dir)
        # Stale sidecar from an earlier era (wrong note + wrong counts).
        (dataset_dir / "ds_test_metadata.json").write_text(json.dumps({
            "coverage_notes": "Full adult female brain (FAFB).",
            "neuron_counts": {"total": 99, "typed": 99,
                              "type_coverage": 1.0},
        }))

        assert banc._regenerate_banc_metadata("ds_test",
                                              str(dataset_dir)) is True
        meta = json.loads(
            (dataset_dir / "ds_test_metadata.json").read_text())
        assert meta["neuron_counts"]["total"] == 3
        # 'Unknown' is the untyped placeholder: only 2 of 3 are typed.
        assert meta["neuron_counts"]["typed"] == 2
        assert meta["neuron_counts"]["type_coverage"] == pytest.approx(2 / 3)
        assert meta["coverage_notes"] == "Full brain and VNC connectome."
        assert meta["synapse_counts"]["total"] == 20
        assert meta["roi_coverage"]["status"] == "not_available_for_banc"

    def test_missing_neuron_table_is_a_noop(self, tmp_path):
        assert banc._regenerate_banc_metadata(
            "ds_test", str(tmp_path)) is False


class TestPathfindingCacheOnlyGate:
    def test_cache_only_skip_does_not_apply_to_banc(self):
        """Fix 1 regression: the outgoing-expansion cache-only skip is
        gated off for BANC, whose connectivity IS the local table."""
        import coana

        assert coana.cache_only_skips_local_fetch(
            "banc_v888", cache_only=True) is False
        assert coana.cache_only_skips_local_fetch(
            "banc_v888", cache_only=False) is False
        # NeuPrint datasets keep the historical skip.
        assert coana.cache_only_skips_local_fetch(
            "hemibrain:v1.2.1", cache_only=True) is True
        assert coana.cache_only_skips_local_fetch(
            "hemibrain:v1.2.1", cache_only=False) is False
