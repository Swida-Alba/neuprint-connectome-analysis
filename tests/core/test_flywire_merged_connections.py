"""Tests for the FlyWire merged-connections parquet switch.

The FAFB/BANC converters used to export ``*_merged_connections.csv`` next to
the parquet; the CSV was ~5x larger and became redundant once every reader
prefers the parquet. These tests pin the new behavior: converters write only
the parquet, and all readers resolve the parquet first with CSV fallback.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import fafb_utils  # noqa: E402
import FAFB_file_converter  # noqa: E402
import BANC_file_converter  # noqa: E402
import coana  # noqa: E402


DATASET = "flywire_FAFB_v783"

RAW_ROWS = pd.DataFrame({
    # FlyWire root ids are large uint64s; kept as strings everywhere.
    # Rows 0+3 share (pre, post) so the converter must merge their ROIs.
    "pre_root_id": [7205759406123451, 7205759406123451,
                    7205759406129999, 7205759406123451],
    "post_root_id": [7205759406198765, 7205759406188888,
                     7205759406198765, 7205759406198765],
    "neuropil": ["AL(R)", "MB(R)", "AL(R)", "MB(R)"],
    "syn_count": [3, 5, 2, 5],
})


def _make_dataset_dir(tmp_path, dataset=DATASET):
    base = tmp_path / "datasets" / dataset
    (base / "downloads").mkdir(parents=True)
    # prepare_fafb_data requires the merged neuron table next to the conn file
    pd.DataFrame({"bodyId": [1], "type": ["T"]}).to_csv(
        base / f"{dataset}_allneurons_neuron_df.csv", index=False)
    return base


class TestPrepareFafbData:
    def test_prefers_merged_parquet_over_csv(self, tmp_path):
        base = _make_dataset_dir(tmp_path)
        (base / f"{DATASET}_merged_connections.parquet").touch()
        (base / f"{DATASET}_merged_connections.csv").touch()
        _, conn_file = fafb_utils.prepare_fafb_data(str(base))
        assert conn_file.endswith("_merged_connections.parquet")

    def test_falls_back_to_legacy_csv(self, tmp_path):
        base = _make_dataset_dir(tmp_path)
        (base / f"{DATASET}_merged_connections.csv").touch()
        _, conn_file = fafb_utils.prepare_fafb_data(str(base))
        assert conn_file.endswith("_merged_connections.csv")


class TestConvertersWriteOnlyParquet:
    def _run(self, converter, tmp_path):
        src = tmp_path / "connections_princeton.csv.gz"
        RAW_ROWS.to_csv(src, index=False, compression="gzip")
        out = tmp_path / "ds_merged_connections.parquet"
        assert converter.process_connections_to_parquet(str(src), str(out))
        assert out.exists()
        # The CSV export is gone: only the parquet is produced.
        assert list(tmp_path.glob("ds_merged_connections.*")) == [out]
        df = pd.read_parquet(out)
        # Same normalization the CSV carried: string ids, renamed columns.
        assert list(df.columns) == ["bodyId_pre", "bodyId_post", "weight", "roi"]
        assert len(df) == 3  # rows sharing (pre, post) merged across ROIs
        return df

    def test_fafb_converter(self, tmp_path):
        df = self._run(FAFB_file_converter, tmp_path)
        merged = df[df["bodyId_pre"] == "7205759406123451"].set_index("bodyId_post")
        assert merged.loc["7205759406198765", "weight"] == 8  # 3 + 5
        assert merged.loc["7205759406198765", "roi"] == "AL(R)|MB(R)"
        assert merged.loc["7205759406188888", "weight"] == 5

    def test_banc_converter(self, tmp_path):
        df = self._run(BANC_file_converter, tmp_path)
        merged = df[df["bodyId_pre"] == "7205759406123451"].set_index("bodyId_post")
        assert merged.loc["7205759406198765", "weight"] == 8
        assert merged.loc["7205759406198765", "roi"] == "AL(R)|MB(R)"

    def test_existing_parquet_short_circuits(self, tmp_path):
        src = tmp_path / "connections_princeton.csv.gz"
        RAW_ROWS.to_csv(src, index=False, compression="gzip")
        out = tmp_path / "ds_merged_connections.parquet"
        # A complete existing table is reused untouched.
        pd.DataFrame({"bodyId_pre": ["1"], "bodyId_post": ["2"],
                      "weight": [1], "roi": ["AL"]}).to_parquet(out,
                                                                index=False)
        before = out.read_bytes()
        assert FAFB_file_converter.process_connections_to_parquet(
            str(src), str(out))
        assert out.read_bytes() == before  # untouched

    def test_truncated_parquet_is_rebuilt(self, tmp_path):
        """An unreadable leftover must not be trusted: it is discarded and
        converted again from the raw CSV."""
        src = tmp_path / "connections_princeton.csv.gz"
        RAW_ROWS.to_csv(src, index=False, compression="gzip")
        out = tmp_path / "ds_merged_connections.parquet"
        out.write_bytes(b"existing")
        assert FAFB_file_converter.process_connections_to_parquet(
            str(src), str(out))
        assert out.read_bytes() != b"existing"
        assert "bodyId_pre" in pd.read_parquet(out).columns


class TestLoadFlywireMergedConnections:
    def test_parquet_yields_string_ids_and_engine_columns(self, tmp_path):
        pq = tmp_path / "ds_merged_connections.parquet"
        # Mimic converter output: ids stored as strings, engine column names.
        df = pd.DataFrame({
            "bodyId_pre": ["7205759406123451"],
            "bodyId_post": ["7205759406198765"],
            "weight": [3],
            "roi": ["AL(R)"],
        })
        df.to_parquet(pq, index=False)
        loaded = coana.load_flywire_merged_connections(str(pq))
        assert loaded["bodyId_pre"].tolist() == ["7205759406123451"]
        assert loaded["bodyId_pre"].map(type).eq(str).all()
        assert "weight" in loaded.columns

    def test_csv_with_raw_column_names_is_normalized(self, tmp_path):
        csv = tmp_path / "ds_merged_connections.csv"
        RAW_ROWS.to_csv(csv, index=False)
        loaded = coana.load_flywire_merged_connections(str(csv))
        assert "bodyId_pre" in loaded.columns
        assert loaded["bodyId_pre"].map(type).eq(str).all()
