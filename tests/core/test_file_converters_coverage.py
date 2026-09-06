"""Hermetic coverage tests for ``src/BANC_file_converter.py`` and
``src/FAFB_file_converter.py``.

All converters operate on explicit paths, so every test builds tiny synthetic
raw inputs (CSV / gzip CSV / ZIP of SWC) inside ``tmp_path`` and checks the
converted parquet outputs.  The FAFB skeleton pipeline's process pool is
replaced by an in-process fake executor — no multiprocessing is spawned.
"""

import gzip
import sys
import zipfile
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import BANC_file_converter as banc  # noqa: E402
import FAFB_file_converter as fafb  # noqa: E402


def _write_gz(path, text):
    Path(path).write_bytes(gzip.compress(text.encode("utf-8")))


# ===========================================================================
# BANC: process_neurons_to_parquet
# ===========================================================================


BANC_NEURON_HEADER = (
    "Root ID,Primary Cell Type,Super Class,Flow,Nerve,Hemilineage,"
    "Predicted NT type,Class,Sub Class,Soma side"
)


def _banc_neuron_rows():
    return "\n".join(
        [
            BANC_NEURON_HEADER,
            "720575940000000002,Mi1,visual,input,,dorsal,ACh,PN,medulla,right",
            "720575940000000001,,,,,,,,right",
            "720575940000000002,Mi1,visual,input,,dorsal,ACh,PN,medulla,right",
        ]
    )


def test_banc_neurons_conversion_full(tmp_path):
    read_path = tmp_path / "neurons.csv"
    read_path.write_text(_banc_neuron_rows(), encoding="utf-8")
    save_path = tmp_path / "neuron_df.parquet"
    save_csv = tmp_path / "neuron_df.csv"

    assert banc.process_neurons_to_parquet(str(read_path), str(save_path),
                                            save_csv_path=str(save_csv)) is True
    assert save_csv.exists()

    df = pd.read_parquet(save_path)
    # Deduplicated and sorted by bodyId.
    assert df["bodyId"].tolist() == [
        "720575940000000001", "720575940000000002",
    ]
    # Renamed columns present, standard ordering first.
    assert df.columns.tolist()[:4] == ["bodyId", "type", "instance", "post"]
    assert {"super_class", "flow", "nerve", "hemilineage", "nt_type"} <= set(
        df.columns
    )
    # Missing type filled with 'Unknown'; instance mirrors type; post zeroed.
    assert df.loc[df["bodyId"] == "720575940000000001", "type"].iloc[0] == "Unknown"
    assert df.loc[df["bodyId"] == "720575940000000002", "type"].iloc[0] == "Mi1"
    assert (df["instance"] == df["type"]).all()
    assert (df["post"] == 0).all()


def test_banc_neurons_cable_length_unit_relabel(tmp_path):
    """The Codex CSV mislabels micrometre cable lengths as 'Cable length
    (nm)'; the conversion must emit the honest unit so the Codex fallback
    path and the bucket-prepared path agree."""
    read_path = tmp_path / "neurons.csv"
    read_path.write_text(
        "\n".join([
            BANC_NEURON_HEADER + ",Cable length (nm)",
            "720575940000000001,Unknown,visual,input,,dorsal,ACh,PN,medulla,"
            "right,315.2",
        ]),
        encoding="utf-8",
    )
    save_path = tmp_path / "neuron_df.parquet"

    assert banc.process_neurons_to_parquet(
        str(read_path), str(save_path)) is True

    df = pd.read_parquet(save_path)
    assert "Cable length (µm)" in df.columns
    assert "Cable length (nm)" not in df.columns
    # Values pass through unchanged (they were always micrometres).
    assert df["Cable length (µm)"].iloc[0] == 315.2


def test_banc_neurons_gzip_input(tmp_path):
    read_path = tmp_path / "neurons.csv.gz"
    _write_gz(read_path, _banc_neuron_rows())
    save_path = tmp_path / "neuron_df.parquet"
    assert banc.process_neurons_to_parquet(str(read_path), str(save_path)) is True
    df = pd.read_parquet(save_path)
    assert len(df) == 2


def test_banc_neurons_missing_type_column(tmp_path):
    read_path = tmp_path / "neurons.csv"
    read_path.write_text(
        "Root ID,Super Class\n720575940000000001,visual\n", encoding="utf-8"
    )
    save_path = tmp_path / "neuron_df.parquet"
    assert banc.process_neurons_to_parquet(str(read_path), str(save_path)) is True
    df = pd.read_parquet(save_path)
    assert df["type"].tolist() == ["Unknown"]
    assert df["instance"].tolist() == ["Unknown"]


def test_banc_neurons_existing_output_short_circuits(tmp_path, capsys):
    save_path = tmp_path / "neuron_df.parquet"
    save_path.write_bytes(b"already")
    assert banc.process_neurons_to_parquet(str(tmp_path / "missing.csv"),
                                           str(save_path)) is True
    assert "Found existing converted file" in capsys.readouterr().out


def test_banc_neurons_missing_input(tmp_path):
    assert banc.process_neurons_to_parquet(
        str(tmp_path / "missing.csv"), str(tmp_path / "out.parquet")
    ) is False


def test_banc_neurons_invalid_input(tmp_path):
    read_path = tmp_path / "bad.csv"
    read_path.write_text("foo\nbar\n", encoding="utf-8")
    assert banc.process_neurons_to_parquet(
        str(read_path), str(tmp_path / "out.parquet")
    ) is False


# ===========================================================================
# BANC: process_connections_to_parquet
# ===========================================================================


def _banc_conn_rows():
    return "\n".join(
        [
            "pre_root_id,post_root_id,neuropil,syn_count,nt_type",
            "2,1,B,4,ACh",
            "2,1,A,3,ACh",
            "1,3,C,2,GABA",
        ]
    )


def test_banc_connections_aggregates_rois(tmp_path):
    read_path = tmp_path / "connections.csv"
    read_path.write_text(_banc_conn_rows(), encoding="utf-8")
    save_path = tmp_path / "connections.parquet"

    assert banc.process_connections_to_parquet(str(read_path),
                                               str(save_path)) is True
    df = pd.read_parquet(save_path)
    assert df.columns.tolist()[:4] == ["bodyId_pre", "bodyId_post", "weight", "roi"]
    pair = df[(df["bodyId_pre"] == "2") & (df["bodyId_post"] == "1")]
    assert pair["weight"].iloc[0] == 7
    assert pair["roi"].iloc[0] == "A|B"
    assert df["bodyId_pre"].tolist() == ["1", "2"]


def test_banc_connections_without_roi_column(tmp_path):
    read_path = tmp_path / "connections.csv"
    read_path.write_text(
        "pre_root_id,post_root_id,syn_count\n1,2,3\n1,2,4\n", encoding="utf-8"
    )
    save_path = tmp_path / "connections.parquet"
    assert banc.process_connections_to_parquet(str(read_path),
                                               str(save_path)) is True
    df = pd.read_parquet(save_path)
    assert df["weight"].tolist() == [7]
    assert df["roi"].tolist() == ["WholeBrain"]


def test_banc_connections_existing_output_short_circuits(tmp_path):
    save_path = tmp_path / "connections.parquet"
    save_path.write_bytes(b"already")
    assert banc.process_connections_to_parquet(
        str(tmp_path / "missing.csv"), str(save_path)
    ) is True


def test_banc_connections_missing_input(tmp_path):
    assert banc.process_connections_to_parquet(
        str(tmp_path / "missing.csv"), str(tmp_path / "out.parquet")
    ) is False


def test_banc_connections_invalid_input(tmp_path):
    read_path = tmp_path / "bad.csv"
    read_path.write_text("foo,bar\n1,2\n", encoding="utf-8")
    assert banc.process_connections_to_parquet(
        str(read_path), str(tmp_path / "out.parquet")
    ) is False


# ===========================================================================
# BANC: update_neuron_post_counts
# ===========================================================================


def test_banc_update_post_counts_parquet(tmp_path):
    neuron_path = tmp_path / "neurons.parquet"
    pd.DataFrame(
        {"bodyId": ["1", "2", "3"], "type": ["A", "B", "C"], "post": [0, 0, 0]}
    ).to_parquet(neuron_path, index=False)
    conn_path = tmp_path / "connections.parquet"
    pd.DataFrame(
        {
            "bodyId_pre": ["9", "9", "2"],
            "bodyId_post": ["1", "1", "2"],
            "weight": [5, 2, 3],
        }
    ).to_parquet(conn_path, index=False)

    save_csv = tmp_path / "neurons_updated.csv"
    assert banc.update_neuron_post_counts(
        str(neuron_path), str(conn_path), save_csv_path=str(save_csv)
    ) is True

    df = pd.read_parquet(neuron_path)
    assert df.set_index("bodyId")["post"].to_dict() == {"1": 7, "2": 3, "3": 0}
    assert save_csv.exists()


def test_banc_update_post_counts_csv(tmp_path):
    neuron_path = tmp_path / "neurons.csv"
    pd.DataFrame({"bodyId": ["1", "2"], "type": ["A", "B"], "post": [0, 0]}).to_csv(
        neuron_path, index=False
    )
    conn_path = tmp_path / "connections.csv"
    pd.DataFrame(
        {"bodyId_post": ["2", "2"], "weight": [1, 4]}
    ).to_csv(conn_path, index=False)

    assert banc.update_neuron_post_counts(str(neuron_path), str(conn_path)) is True
    df = pd.read_csv(neuron_path, dtype={"bodyId": str})
    assert df.set_index("bodyId")["post"].to_dict() == {"1": 0, "2": 5}


def test_banc_update_post_counts_error(tmp_path):
    assert banc.update_neuron_post_counts(
        str(tmp_path / "missing.parquet"), str(tmp_path / "also.parquet")
    ) is False


# ===========================================================================
# BANC: ensure_banc_data
# ===========================================================================


def test_banc_ensure_data_full_pipeline(tmp_path, capsys):
    dataset_dir = tmp_path / "datasets" / "banc_v999"
    downloads = dataset_dir / "downloads"
    downloads.mkdir(parents=True)
    _write_gz(downloads / "neurons.csv.gz", _banc_neuron_rows())
    # Use full FlyWire ids consistent with _banc_neuron_rows(): the post-count
    # backfill merges neuron bodyId with connection bodyId_post after canonical
    # normalization, which keeps full ids intact but does not complete a bare
    # short id to its 720575940… form, so bare ids would never match.
    _write_gz(downloads / "connections_princeton.csv.gz", "\n".join([
        "pre_root_id,post_root_id,neuropil,syn_count,nt_type",
        "720575940000000002,720575940000000001,B,4,ACh",
        "720575940000000002,720575940000000001,A,3,ACh",
        "720575940000000001,720575940000000002,C,2,GABA",
    ]))

    assert banc.ensure_banc_data("banc_v999", str(dataset_dir)) is True

    neuron_pq = dataset_dir / "banc_v999_allneurons_neuron_df.parquet"
    conn_pq = dataset_dir / "banc_v999_merged_connections.parquet"
    assert neuron_pq.exists() and conn_pq.exists()
    assert (dataset_dir / "banc_v999_allneurons_neuron_df.csv").exists()

    # Post counts were back-filled from the connection weights.
    df = pd.read_parquet(neuron_pq)
    assert df.set_index("bodyId")["post"].to_dict() == {
        "720575940000000001": 7,
        "720575940000000002": 2,
    }

    # Second run: everything already present, post counts already populated.
    capsys.readouterr()
    assert banc.ensure_banc_data("banc_v999", str(dataset_dir)) is True
    output = capsys.readouterr().out
    assert "Found existing neurons" in output
    assert "Found existing connections" in output
    assert "Post counts already populated" in output


def test_banc_ensure_data_missing_files(tmp_path, monkeypatch):
    # Bucket preparation unavailable (offline): falls back to the manual
    # Codex instructions and reports failure.  Downloads are stubbed out so
    # the test stays hermetic.
    import banc_public_data

    monkeypatch.setattr(banc_public_data, "download_meta_feather",
                        lambda *a, **k: None)
    monkeypatch.setattr(banc_public_data, "download_connections_product",
                        lambda *a, **k: None)
    dataset_dir = tmp_path / "datasets" / "banc_v999"
    assert banc.ensure_banc_data("banc_v999", str(dataset_dir)) is False
    assert (dataset_dir / "downloads").is_dir()


def test_banc_ensure_data_bucket_path_fills_post_counts(tmp_path, monkeypatch,
                                                        capsys):
    """Bucket-prepared tables carry post = 0; the early-return bucket path
    must still back-fill the column from the merged connections."""
    import banc_public_data

    dataset_dir = tmp_path / "datasets" / "banc_v999"
    dataset_dir.mkdir(parents=True)

    def fake_prepare(dataset_name, dir_, project_root=None):
        dir_ = Path(dir_)
        pd.DataFrame({
            "bodyId": ["720575940000000001", "720575940000000002"],
            "type": ["A", "B"],
            "post": [0, 0],
        }).to_parquet(
            dir_ / f"{dataset_name}_allneurons_neuron_df.parquet",
            index=False)
        pd.DataFrame({
            "bodyId_pre": ["720575940000000002",
                           "720575940000000002",
                           "720575940000000001"],
            "bodyId_post": ["720575940000000001",
                            "720575940000000001",
                            "720575940000000002"],
            "weight": [4, 3, 2],
        }).to_parquet(
            dir_ / f"{dataset_name}_merged_connections.parquet",
            index=False)
        return True

    monkeypatch.setattr(banc_public_data, "prepare_dataset_tables",
                        fake_prepare)

    assert banc.ensure_banc_data("banc_v999", str(dataset_dir)) is True
    df = pd.read_parquet(
        dataset_dir / "banc_v999_allneurons_neuron_df.parquet")
    assert df.set_index("bodyId")["post"].to_dict() == {
        "720575940000000001": 7,
        "720575940000000002": 2,
    }


# ===========================================================================
# FAFB: _parse_swc_batch
# ===========================================================================


def _build_swc_zip(path, entries):
    with zipfile.ZipFile(path, "w") as handle:
        for name, payload in entries.items():
            handle.writestr(name, payload)


def test_fafb_parse_swc_batch(tmp_path, capsys):
    zip_path = tmp_path / "skeletons.zip"
    swc_ok = "# comment\n\n1 0 10.4 20.6 30.2 2.0 -1\n2 3 11 21 31 1.5 1\nshort\n"
    swc_nested = "1 2 5 6 7 1 0\n"
    _build_swc_zip(
        zip_path,
        {
            "111.swc": swc_ok,
            "subdir/222.swc": swc_nested,
            "333.swc": b"\xff\xfe\x00".decode("latin-1"),
            "notes.txt": "not a skeleton",
        },
    )

    data = fafb._parse_swc_batch(str(zip_path),
                                 ["111.swc", "subdir/222.swc", "333.swc",
                                  "missing.swc"])
    assert data["bodyId"] == ["111", "111", "222"]
    assert data["node_id"] == [1, 2, 1]
    assert data["x"] == [10.4, 11.0, 5.0]
    assert data["parent_id"] == [-1, 1, 0]


def test_fafb_parse_swc_batch_unreadable_zip(tmp_path, capsys):
    data = fafb._parse_swc_batch(str(tmp_path / "nope.zip"), ["1.swc"])
    assert data["bodyId"] == []
    assert "Error in batch" in capsys.readouterr().out


# ===========================================================================
# FAFB: process_skeletons_to_parquet (with in-process fake pool)
# ===========================================================================


class _FakeFuture:
    def __init__(self, func, *args):
        self._value = func(*args)

    def result(self):
        return self._value


class _FakePool:
    def __init__(self, max_workers=None):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def submit(self, func, *args):
        return _FakeFuture(func, *args)


@pytest.fixture
def fake_pool(monkeypatch):
    monkeypatch.setattr(fafb.concurrent.futures, "ProcessPoolExecutor", _FakePool)
    monkeypatch.setattr(
        fafb.concurrent.futures, "as_completed", lambda futures, **kwargs: list(futures)
    )


def test_fafb_skeletons_conversion(tmp_path, fake_pool):
    zip_path = tmp_path / "sk.zip"
    _build_swc_zip(
        zip_path,
        {
            "111.swc": "1 0 10.4 20.6 30.2 2.0 -1\n2 3 11 21 31 1.5 1\n",
            "subdir/222.swc": "1 2 5 6 7 1 0\n",
        },
    )
    save_path = tmp_path / "skeletons.parquet"

    assert fafb.process_skeletons_to_parquet(str(zip_path), str(save_path),
                                             batch_size=1) is True
    assert save_path.exists()
    assert not save_path.with_name(save_path.name + ".tmp").exists()

    df = pd.read_parquet(save_path)
    assert df["bodyId"].tolist() == ["111", "111", "222"]
    assert df["node_id"].tolist() == [1, 2, 1]
    # Coordinates/radius are rounded to int32.
    assert df.loc[df["bodyId"] == "111", "x"].tolist() == [10, 11]
    assert df.loc[(df["bodyId"] == "111") & (df["node_id"] == 2),
                  "radius"].iloc[0] == 2
    assert str(df["node_id"].dtype) == "int32"


def test_fafb_skeletons_no_valid_nodes(tmp_path, fake_pool):
    zip_path = tmp_path / "sk.zip"
    _build_swc_zip(zip_path, {"111.swc": "# only a comment\n"})
    save_path = tmp_path / "skeletons.parquet"
    assert fafb.process_skeletons_to_parquet(str(zip_path),
                                             str(save_path)) is False
    assert not save_path.exists()
    assert not (tmp_path / "skeletons.parquet.tmp").exists()


def test_fafb_skeletons_no_swc_files(tmp_path, fake_pool):
    zip_path = tmp_path / "sk.zip"
    _build_swc_zip(zip_path, {"readme.txt": "nothing"})
    assert fafb.process_skeletons_to_parquet(
        str(zip_path), str(tmp_path / "out.parquet")
    ) is False


def test_fafb_skeletons_corrupt_zip(tmp_path, fake_pool):
    zip_path = tmp_path / "sk.zip"
    zip_path.write_bytes(b"this is not a zip")
    assert fafb.process_skeletons_to_parquet(
        str(zip_path), str(tmp_path / "out.parquet")
    ) is False


def test_fafb_skeletons_existing_output_short_circuits(tmp_path):
    save_path = tmp_path / "skeletons.parquet"
    save_path.write_bytes(b"already")
    assert fafb.process_skeletons_to_parquet(
        str(tmp_path / "missing.zip"), str(save_path)
    ) is True


def test_fafb_skeletons_missing_zip(tmp_path):
    assert fafb.process_skeletons_to_parquet(
        str(tmp_path / "missing.zip"), str(tmp_path / "out.parquet")
    ) is False


# ===========================================================================
# FAFB: process_neurons_to_parquet
# ===========================================================================


FAFB_CLASSIFICATION = "\n".join(
    [
        "root_id,class,sub_class,side",
        "100,ITP,Lateral,left",
        "100,ITP,Lateral,left",
        "200,Local,,right",
        "300,,Medulla,left",
        "400,,,right",
    ]
)


def _write_fafb_enrichment(downloads):
    downloads.mkdir(parents=True, exist_ok=True)
    (downloads / "names.csv").write_text(
        "root_id,name,group\n100,MeMe10_L,g1\n", encoding="utf-8"
    )
    (downloads / "coordinates.csv").write_text(
        "root_id,x,y,z,supervoxel_id\n100,1.5,2.5,3.5,9\n", encoding="utf-8"
    )
    (downloads / "neurons.csv").write_text(
        "root_id,nt_type,group\n100,ACh,g1\n200,GABA,g2\n", encoding="utf-8"
    )
    (downloads / "cell_stats.csv").write_text(
        "root_id,cable_length\n100,1234.5\n", encoding="utf-8"
    )
    (downloads / "consolidated_cell_types.csv").write_text(
        "root_id,primary_type\n100,MeMe10\n", encoding="utf-8"
    )
    return {
        "names": str(downloads / "names.csv"),
        "coordinates": str(downloads / "coordinates.csv"),
        "neurons": str(downloads / "neurons.csv"),
        "cell_stats": str(downloads / "cell_stats.csv"),
        "cell_types": str(downloads / "consolidated_cell_types.csv"),
    }


def test_fafb_neurons_with_enrichment(tmp_path):
    class_path = tmp_path / "classification.csv.gz"
    _write_gz(class_path, FAFB_CLASSIFICATION)
    enrichment = _write_fafb_enrichment(tmp_path / "downloads")
    save_path = tmp_path / "neuron_df.parquet"
    save_csv = tmp_path / "neuron_df.csv"

    assert fafb.process_neurons_to_parquet(
        str(class_path), str(save_path), save_csv_path=str(save_csv),
        enrichment_files=enrichment,
    ) is True
    assert save_csv.exists()

    df = pd.read_parquet(save_path).set_index("bodyId")
    assert df.index.tolist() == ["100", "200", "300", "400"]
    assert df.columns.tolist()[:4] == ["type", "instance", "post", "super_class"]
    # Type resolution: consolidated primary_type > sub_class > class > Unknown.
    assert df.loc["100", "type"] == "MeMe10"
    assert df.loc["200", "type"] == "Local"
    assert df.loc["300", "type"] == "Medulla"
    assert df.loc["400", "type"] == "Unknown"
    # Instance falls back to type when the names merge missed the neuron.
    assert df.loc["100", "instance"] == "MeMe10_L"
    assert df.loc["200", "instance"] == "Local"
    # Enriched columns survived the merge.
    assert {"nt_type", "cable_length", "x", "hemisphere"} <= set(df.columns)
    assert df.loc["100", "nt_type"] == "ACh"
    assert (df["post"] == 0).all()


def test_fafb_neurons_without_enrichment(tmp_path):
    class_path = tmp_path / "classification.csv.gz"
    _write_gz(class_path, FAFB_CLASSIFICATION)
    save_path = tmp_path / "neuron_df.parquet"

    assert fafb.process_neurons_to_parquet(str(class_path), str(save_path)) is True
    df = pd.read_parquet(save_path).set_index("bodyId")
    assert df.loc["100", "type"] == "Lateral"
    assert df.loc["400", "type"] == "Unknown"
    assert df.loc["200", "instance"] == "Local"


def test_fafb_neurons_existing_output_short_circuits(tmp_path):
    save_path = tmp_path / "neuron_df.parquet"
    save_path.write_bytes(b"already")
    assert fafb.process_neurons_to_parquet(
        str(tmp_path / "missing.csv.gz"), str(save_path)
    ) is True


def test_fafb_neurons_missing_input(tmp_path):
    assert fafb.process_neurons_to_parquet(
        str(tmp_path / "missing.csv.gz"), str(tmp_path / "out.parquet")
    ) is False


def test_fafb_neurons_invalid_input(tmp_path):
    bad_path = tmp_path / "bad.csv"
    bad_path.write_text("foo,bar\n1,2\n", encoding="utf-8")
    assert fafb.process_neurons_to_parquet(
        str(bad_path), str(tmp_path / "out.parquet")
    ) is False


# ===========================================================================
# FAFB: process_connections_to_parquet
# ===========================================================================


def test_fafb_connections_aggregates_rois(tmp_path):
    read_path = tmp_path / "connections.csv.gz"
    _write_gz(
        read_path,
        "\n".join(
            [
                "pre_root_id,post_root_id,neuropil,syn_count",
                "2,1,A,3",
                "2,1,B,4",
                "1,3,C,2",
            ]
        ),
    )
    save_path = tmp_path / "connections.parquet"
    assert fafb.process_connections_to_parquet(str(read_path),
                                               str(save_path)) is True
    df = pd.read_parquet(save_path)
    pair = df[(df["bodyId_pre"] == "2") & (df["bodyId_post"] == "1")]
    assert pair["weight"].iloc[0] == 7
    assert pair["roi"].iloc[0] == "A|B"


def test_fafb_connections_without_roi(tmp_path):
    read_path = tmp_path / "connections.csv.gz"
    _write_gz(read_path, "pre_root_id,post_root_id,syn_count\n1,2,3\n")
    save_path = tmp_path / "connections.parquet"
    assert fafb.process_connections_to_parquet(str(read_path),
                                               str(save_path)) is True
    df = pd.read_parquet(save_path)
    assert df["roi"].tolist() == ["WholeBrain"]


def test_fafb_connections_existing_and_missing(tmp_path):
    save_path = tmp_path / "connections.parquet"
    save_path.write_bytes(b"already")
    assert fafb.process_connections_to_parquet(
        str(tmp_path / "missing.csv.gz"), str(save_path)
    ) is True
    assert fafb.process_connections_to_parquet(
        str(tmp_path / "missing.csv.gz"), str(tmp_path / "other.parquet")
    ) is False


def test_fafb_connections_invalid_input(tmp_path):
    read_path = tmp_path / "bad.csv.gz"
    _write_gz(read_path, "foo,bar\n1,2\n")
    assert fafb.process_connections_to_parquet(
        str(read_path), str(tmp_path / "out.parquet")
    ) is False


# ===========================================================================
# FAFB: process_synapse_table_to_parquet
# ===========================================================================


def test_fafb_synapse_table_short_ids_prefixed(tmp_path):
    read_path = tmp_path / "synapse.csv.gz"
    _write_gz(
        read_path,
        "\n".join(
            [
                "pre_root_id,post_root_id,x,y,z",
                "123456789,111111111,1,2,3",
                "222222222,333333333,4,5,6",
                "444444444,123456789,7,8,9",
            ]
        ),
    )
    save_path = tmp_path / "synapse.parquet"
    assert fafb.process_synapse_table_to_parquet(
        str(read_path), str(save_path), chunksize=2
    ) is True
    df = pd.read_parquet(save_path)
    assert all(value.startswith("720575940") for value in df["pre_root_id"])
    assert len(df) == 3
    # Sorted by the fixed ids.
    assert df["pre_root_id"].tolist() == sorted(df["pre_root_id"].tolist())


def test_fafb_synapse_table_full_length_ids_untouched(tmp_path):
    read_path = tmp_path / "synapse.csv.gz"
    _write_gz(
        read_path,
        "pre_root_id,post_root_id\n"
        "720575940368841570,720575940368841571\n",
    )
    save_path = tmp_path / "synapse.parquet"
    assert fafb.process_synapse_table_to_parquet(str(read_path),
                                                 str(save_path)) is True
    df = pd.read_parquet(save_path)
    assert df["pre_root_id"].tolist() == ["720575940368841570"]


def test_fafb_synapse_table_missing_root_columns(tmp_path):
    read_path = tmp_path / "synapse.csv.gz"
    _write_gz(read_path, "foo,bar\n1,2\n")
    assert fafb.process_synapse_table_to_parquet(
        str(read_path), str(tmp_path / "out.parquet")
    ) is False


def test_fafb_synapse_table_empty_body(tmp_path):
    read_path = tmp_path / "synapse.csv.gz"
    _write_gz(read_path, "pre_root_id,post_root_id\n")
    assert fafb.process_synapse_table_to_parquet(
        str(read_path), str(tmp_path / "out.parquet")
    ) is False


def test_fafb_synapse_table_existing_and_missing(tmp_path):
    save_path = tmp_path / "synapse.parquet"
    save_path.write_bytes(b"already")
    assert fafb.process_synapse_table_to_parquet(
        str(tmp_path / "missing.csv.gz"), str(save_path)
    ) is True
    assert fafb.process_synapse_table_to_parquet(
        str(tmp_path / "missing.csv.gz"), str(tmp_path / "other.parquet")
    ) is False


# ===========================================================================
# FAFB: update_neuron_post_counts
# ===========================================================================


def test_fafb_update_post_counts_parquet(tmp_path):
    neuron_path = tmp_path / "neurons.parquet"
    pd.DataFrame({"bodyId": ["1", "2"], "type": ["A", "B"], "post": [0, 0]}).to_parquet(
        neuron_path, index=False
    )
    conn_path = tmp_path / "connections.parquet"
    pd.DataFrame(
        {"bodyId_pre": ["9", "9"], "bodyId_post": ["2", "2"], "weight": [1, 4]}
    ).to_parquet(conn_path, index=False)

    save_csv = tmp_path / "neurons.csv"
    assert fafb.update_neuron_post_counts(
        str(neuron_path), str(conn_path), save_csv_path=str(save_csv)
    ) is True
    df = pd.read_parquet(neuron_path)
    assert df.set_index("bodyId")["post"].to_dict() == {"1": 0, "2": 5}
    assert save_csv.exists()


def test_fafb_update_post_counts_csv(tmp_path):
    neuron_path = tmp_path / "neurons.csv"
    pd.DataFrame({"bodyId": ["1"], "type": ["A"], "post": [0]}).to_csv(
        neuron_path, index=False
    )
    conn_path = tmp_path / "connections.csv"
    pd.DataFrame({"bodyId_post": ["1"], "weight": [9]}).to_csv(
        conn_path, index=False
    )
    assert fafb.update_neuron_post_counts(str(neuron_path), str(conn_path)) is True
    df = pd.read_csv(neuron_path, dtype={"bodyId": str})
    assert df["post"].tolist() == [9]


def test_fafb_update_post_counts_error(tmp_path):
    assert fafb.update_neuron_post_counts(
        str(tmp_path / "missing.parquet"), str(tmp_path / "also.parquet")
    ) is False


# ===========================================================================
# FAFB: print_download_instructions
# ===========================================================================


def test_fafb_download_instructions(tmp_path, capsys):
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    _write_gz(downloads / "classification.csv.gz", "root_id\n100\n")
    (downloads / "names.csv").write_text("root_id,name\n", encoding="utf-8")
    _write_gz(downloads / "connections_princeton.csv.gz", "pre_root_id\n")

    fafb.print_download_instructions(str(downloads))
    output = capsys.readouterr().out
    assert "classification.csv.gz" in output
    # All critical files are present (two via fallback spellings).
    assert "❌" not in output
    assert "existed" in output
    assert "[optional]" in output


# ===========================================================================
# FAFB: ensure_flywire_data
# ===========================================================================


def _fafb_connections_rows():
    return "\n".join(
        [
            "pre_root_id,post_root_id,neuropil,syn_count",
            "200,100,A,5",
            "300,400,B,1",
        ]
    )


def test_fafb_ensure_data_full_pipeline(tmp_path, capsys):
    dataset_dir = tmp_path / "datasets" / "flywire_FAFB_v999"
    downloads = dataset_dir / "downloads"
    downloads.mkdir(parents=True)

    _write_gz(downloads / "classification.csv.gz", FAFB_CLASSIFICATION)
    _write_gz(downloads / "names.csv.gz", "root_id,name,group\n100,MeMe10_L,g\n")
    _write_gz(downloads / "coordinates.csv.gz",
              "root_id,x,y,z,supervoxel_id\n100,1,2,3,9\n")
    _write_gz(downloads / "neurons.csv.gz", "root_id,nt_type,group\n100,ACh,g\n")
    _write_gz(downloads / "cell_stats.csv.gz", "root_id,cable_length\n100,9.5\n")
    _write_gz(downloads / "consolidated_cell_types.csv.gz",
              "root_id,primary_type\n100,MeMe10\n")
    _write_gz(downloads / "connections_princeton_no_threshold.csv.gz",
              _fafb_connections_rows())
    _write_gz(
        downloads / "fafb_v783_princeton_synapse_table.csv.gz",
        "pre_root_id,post_root_id,x\n720575940368841570,720575940368841571,1\n",
    )
    with zipfile.ZipFile(downloads / "sk_lod1_783_healed.zip", "w") as handle:
        handle.writestr("1.swc", "1 0 0 0 0 1 -1\n")

    assert fafb.ensure_flywire_data("flywire_FAFB_v999", str(dataset_dir)) is True

    neuron_pq = dataset_dir / "flywire_FAFB_v999_allneurons_neuron_df.parquet"
    conn_pq = dataset_dir / "flywire_FAFB_v999_merged_connections.parquet"
    syn_pq = dataset_dir / "flywire_FAFB_v999_synapse_table.parquet"
    assert neuron_pq.exists() and conn_pq.exists() and syn_pq.exists()
    assert (dataset_dir / "sk_lod1_783_healed.zip").exists()

    # Post counts back-filled from connection weights (only 100/400 are post).
    df = pd.read_parquet(neuron_pq)
    posts = df.set_index("bodyId")["post"].to_dict()
    assert posts["100"] == 5
    assert posts["400"] == 1
    assert posts["200"] == 0

    # Second run: everything already present.
    capsys.readouterr()
    assert fafb.ensure_flywire_data("flywire_FAFB_v999", str(dataset_dir)) is True
    output = capsys.readouterr().out
    assert "Found existing neurons" in output
    assert "Found existing connections" in output
    assert "Found existing synapse table" in output
    assert "Found existing skeletons" in output
    assert "Post counts already populated" in output


def test_fafb_ensure_data_minimal_with_fallbacks(tmp_path, capsys):
    dataset_dir = tmp_path / "datasets" / "flywire_FAFB_v999"
    downloads = dataset_dir / "downloads"
    downloads.mkdir(parents=True)

    # Only the two critical files; connections under the last fallback name,
    # and a synapse table under a non-standard name (discovery loop).
    _write_gz(downloads / "classification.csv.gz", FAFB_CLASSIFICATION)
    _write_gz(downloads / "connections.csv.gz", _fafb_connections_rows())
    _write_gz(downloads / "custom_synapse_export.csv.gz",
              "pre_root_id,post_root_id\n720575940368841570,720575940368841571\n")
    with zipfile.ZipFile(downloads / "my_skeletons.zip", "w") as handle:
        handle.writestr("1.swc", "1 0 0 0 0 1 -1\n")

    assert fafb.ensure_flywire_data("flywire_FAFB_v999", str(dataset_dir)) is True
    output = capsys.readouterr().out
    assert "Missing neuron metadata files" in output
    assert (dataset_dir / "flywire_FAFB_v999_merged_connections.parquet").exists()
    assert (dataset_dir / "flywire_FAFB_v999_synapse_table.parquet").exists()
    # The skeleton zip is moved to the canonical destination name.
    assert (dataset_dir / "sk_lod1_783_healed.zip").exists()
    assert not (downloads / "my_skeletons.zip").exists()


def test_fafb_ensure_data_missing_critical(tmp_path, capsys):
    dataset_dir = tmp_path / "datasets" / "flywire_FAFB_v999"
    assert fafb.ensure_flywire_data("flywire_FAFB_v999", str(dataset_dir)) is False
    output = capsys.readouterr().out
    assert "MISSING CRITICAL FILES" in output
    assert (dataset_dir / "downloads").is_dir()
