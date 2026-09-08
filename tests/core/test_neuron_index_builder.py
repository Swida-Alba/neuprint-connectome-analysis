"""Tests for the local projection of pulled neuron metadata."""

import polars as pl


def test_metadata_projection_keeps_metadata_except_large_roi_fields(tmp_path):
    from src.neuron_index_builder import read_metadata_projection

    source = tmp_path / "sample_allneurons_neuron_df.csv"
    pl.DataFrame(
        {
            "bodyId": [12345678901234567890, 42],
            "type": ["aMe12", "APL"],
            "instance": ["aMe12_L", "APL_1"],
            "post": [10, 3],
            "size": [100, 200],
            "confidence": [0.9, 0.8],
            "somaSide": ["L", "R"],
            "roiInfo": ["{\"AL\": 1}", "{\"MB\": 2}"],
            "inputRois": ["[AL]", "[MB]"],
            "outputRois": ["[AL]", "[MB]"],
            "notes": ["large note", "another note"],
            "matchingNotes": ["match one", "match two"],
            "synonyms": ["syn one", "syn two"],
            "last_fetched": ["old", "old"],
        }
    ).write_csv(source)

    projection = read_metadata_projection(source)

    assert projection.columns == [
        "bodyId", "type", "instance", "post", "size", "confidence",
        "somaSide", "notes", "matchingNotes",
        "synonyms",
    ]
    assert projection["bodyId"].to_list() == ["12345678901234567890", "42"]
    assert projection["type"].to_list() == ["aMe12", "APL"]
    assert projection["size"].to_list() == [100, 200]
    assert projection["confidence"].to_list() == [0.9, 0.8]
    assert "roiInfo" not in projection.columns
    assert "inputRois" not in projection.columns
    assert "outputRois" not in projection.columns
    assert "notes" in projection.columns
    assert "matchingNotes" in projection.columns
    assert "synonyms" in projection.columns
    assert "last_fetched" not in projection.columns


def test_metadata_projection_normalizes_flywire_body_id_alias(tmp_path):
    from src.neuron_index_builder import read_metadata_projection

    source = tmp_path / "flywire_neuron_df.csv"
    source.write_text(
        "root_id,type,instance,flywireType,super_class,size_nm\n"
        "720575940000000001,aMe12,aMe12_L,Me,central,123\n"
        "720575940000000002,APL,APL_1,APL,visual,456\n",
        encoding="utf-8",
    )

    projection = read_metadata_projection(source)

    assert projection.columns == [
        "bodyId", "type", "instance", "flywireType", "super_class",
        "size_nm",
    ]
    assert projection["bodyId"].to_list() == [
        "720575940000000001", "720575940000000002",
    ]
    assert projection["size_nm"].to_list() == [123, 456]


def test_metadata_path_prefers_the_pulled_csv_over_an_old_projection(tmp_path):
    from src.neuron_index_builder import metadata_path

    dataset = "fixture:v1.0"
    folder = tmp_path / "fixture_v1_0"
    folder.mkdir()
    parquet = folder / "fixture_v1_0_allneurons_neuron_df.parquet"
    csv = folder / "fixture_v1_0_allneurons_neuron_df.csv"
    pl.DataFrame({"bodyId": [1], "type": ["old"]}).write_parquet(parquet)
    pl.DataFrame({"bodyId": [1], "type": ["pulled"]}).write_csv(csv)

    assert metadata_path(dataset, tmp_path) == csv


def test_dataset_folder_discovery_handles_releases_and_future_names(tmp_path):
    from src.neuron_index_builder import (
        dataset_identifier_from_folder,
        discover_metadata_datasets,
    )

    assert dataset_identifier_from_folder("male-cns_v1_0") == "male-cns:v1.0"
    assert dataset_identifier_from_folder("optic-lobe_v1_1") == "optic-lobe:v1.1"
    assert dataset_identifier_from_folder("manc_v1_2_1") == "manc:v1.2.1"
    assert dataset_identifier_from_folder("flywire_FAFB_v783") == "flywire_FAFB_v783"
    assert dataset_identifier_from_folder("banc_v888") == "banc_v888"
    assert dataset_identifier_from_folder("future_release") == "future_release"

    for folder in (
        "male-cns_v1_0", "optic-lobe_v1_1", "flywire_FAFB_v783",
    ):
        path = tmp_path / folder
        path.mkdir()
        (path / f"{folder}_allneurons_neuron_df.csv").write_text(
            "bodyId,type\n1,A\n", encoding="utf-8"
        )
    (tmp_path / "not_a_dataset").mkdir()
    assert discover_metadata_datasets(tmp_path) == [
        "flywire_FAFB_v783", "male-cns:v1.0", "optic-lobe:v1.1",
    ]


def test_metadata_candidates_accepts_future_schema_validated_filename(tmp_path):
    from src.neuron_index_builder import metadata_candidates

    folder = tmp_path / "future_v2_0"
    folder.mkdir()
    future = folder / "neurons_export_2026.csv"
    future.write_text("root_id,cell_class\n720575940000000001,Kenyon\n", encoding="utf-8")
    # A similarly named derived table must not win the fallback.
    (folder / "synapse_summary.csv").write_text(
        "bodyId,weight\n1,2\n", encoding="utf-8"
    )

    assert metadata_candidates("future:v2.0", tmp_path) == [future]


def test_priority_columns_put_cross_dataset_taxonomy_after_instance():
    from src.neuron_index_builder import ordered_projection_columns

    assert ordered_projection_columns([
        "bodyId", "type", "instance", "notes", "class", "mancType",
        "flywireType", "hemibrainType", "super_class", "cell_type", "post",
    ]) == [
        "bodyId", "type", "instance", "flywireType", "hemibrainType",
        "mancType", "cell_type", "class", "super_class", "notes", "post",
    ]


def test_priority_columns_preserve_non_priority_source_order():
    from src.neuron_index_builder import ordered_projection_columns

    assert ordered_projection_columns([
        "bodyId", "instance", "type", "pre", "score", "hemibrainType",
        "comment", "class", "post", "connection_count",
    ]) == [
        "bodyId", "type", "instance", "hemibrainType", "class", "pre",
        "score", "comment", "post", "connection_count",
    ]


def test_priority_columns_leave_measurements_in_source_order():
    from src.neuron_index_builder import ordered_projection_columns

    assert ordered_projection_columns([
        "bodyId", "type", "instance", "size",
        "celltypePredictedNtConfidence", "celltypePredictedNt",
        "celltypeTotalNtPredictions", "receptorType", "notes",
    ]) == [
        "bodyId", "type", "instance", "celltypePredictedNt", "receptorType",
        "size", "celltypePredictedNtConfidence", "celltypeTotalNtPredictions",
        "notes",
    ]


def test_search_cache_is_distinct_and_presorted_by_viewer_priority():
    from src.neuron_index_builder import build_search_cache_frame

    frame = pl.DataFrame(
        {
            "bodyId": ["2", "1", "3"],
            "type": ["MeVPaMe2", "aMe01", "aMe01"],
            "instance": ["MeVPaMe2_L", "aMe01_L", ""],
            "flywireType": ["aMe19a", "", "aMe01"],
            "post": [1, 2, 3],
        }
    )

    cache = build_search_cache_frame(frame)

    assert cache.columns == [
        "search_column", "search_priority", "search_value",
        "search_value_folded", "__neuron_rows",
    ]
    columns_by_priority = (
        cache.group_by("search_column")
        .agg(pl.col("search_priority").first())
        .sort("search_priority")["search_column"]
        .to_list()
    )
    assert columns_by_priority == [
        "bodyId", "type", "instance", "flywireType",
    ]
    type_a_me = cache.filter(
        (pl.col("search_column") == "type")
        & (pl.col("search_value") == "aMe01")
    )
    assert type_a_me.height == 1
    assert type_a_me["__neuron_rows"].to_list() == [[1, 2]]
    ordered = cache.select(["search_priority", "search_value"]).to_dicts()
    assert ordered == sorted(
        ordered,
        key=lambda row: (row["search_priority"], row["search_value"]),
    )


def test_search_cache_compatibility_rejects_stale_priority_columns():
    from src.neuron_index_builder import (
        build_search_cache_frame,
        is_search_cache_compatible,
        viewer_search_columns,
    )

    frame = pl.DataFrame({
        "bodyId": ["1"],
        "type": ["aMe17a"],
        "instance": ["aMe17a_L"],
        "flywireType": ["aMe17a1"],
        "locationType": ["aMe17a-central"],
    })
    current_columns = viewer_search_columns(frame.columns)
    current = build_search_cache_frame(frame)
    stale = build_search_cache_frame(frame, current_columns[:-1])

    assert is_search_cache_compatible(current, frame.columns)
    assert not is_search_cache_compatible(stale, frame.columns)


def test_metadata_pull_materializes_index_and_search_cache(tmp_path):
    """The first cache-enabled run builds both searchable parquet files."""
    import pandas as pd

    from src.coana import FindNeuronConnection
    from src.neuron_index_builder import (
        is_search_cache_compatible,
        search_cache_path,
    )

    dataset = "fixture:v1.0"
    folder = "fixture_v1_0"
    metadata_dir = tmp_path / "datasets" / folder
    metadata_dir.mkdir(parents=True)
    metadata = metadata_dir / f"{folder}_allneurons_neuron_df.csv"
    pl.DataFrame({
        "bodyId": ["100", "200"],
        "type": ["aMe12", ""],
        "instance": ["aMe12_L", "Other_R"],
        "flywireType": ["aMe12", "Other"],
        "roiInfo": ["large", "large"],
    }).write_csv(metadata)

    index_path = tmp_path / "neuron_indexes" / folder / "neuron_index.parquet"
    search_path = search_cache_path(index_path)
    finder = object.__new__(FindNeuronConnection)
    finder.use_cache = True
    finder.cache_folder = str(index_path.parent)
    finder.script_path = str(tmp_path)
    finder.dataset = dataset
    finder._dataset_safe = folder
    finder._neuron_index_cache = None
    finder._neuron_index_dict = {}
    finder._vprint = lambda *args, **kwargs: None
    finder._get_neuron_index_path = lambda: str(index_path)
    finder._get_neuron_search_cache_path = lambda: str(search_path)
    finder._read_neuron_index_disk = lambda: pd.DataFrame()

    assert FindNeuronConnection._ensure_neuron_index_from_metadata(finder) is True
    assert index_path.is_file()
    assert search_path.is_file()

    index = pl.read_parquet(index_path)
    search = pl.read_parquet(search_path)
    assert index.columns[:4] == ["bodyId", "type", "instance", "flywireType"]
    assert "roiInfo" not in index.columns
    assert is_search_cache_compatible(search, index.columns)


def test_metadata_pull_stabilizes_missing_type_and_instance_schema(tmp_path):
    """A sparse future table is padded once, not rebuilt on every init."""
    import pandas as pd

    from src.coana import FindNeuronConnection

    dataset = "future:v2.0"
    folder = "future_v2_0"
    metadata_dir = tmp_path / "datasets" / folder
    metadata_dir.mkdir(parents=True)
    metadata = metadata_dir / f"{folder}_allneurons_neuron_df.csv"
    pl.DataFrame({
        "bodyId": ["100", "200"],
        "cell_class": ["Kenyon", "projection"],
    }).write_csv(metadata)

    index_path = tmp_path / "neuron_indexes" / folder / "neuron_index.parquet"
    finder = object.__new__(FindNeuronConnection)
    finder.use_cache = True
    finder.cache_folder = str(index_path.parent)
    finder.script_path = str(tmp_path)
    finder.dataset = dataset
    finder._dataset_safe = folder
    finder._neuron_index_cache = None
    finder._neuron_index_dict = {}
    finder._vprint = lambda *args, **kwargs: None
    finder._get_neuron_index_path = lambda: str(index_path)
    finder._get_neuron_search_cache_path = lambda: str(
        index_path.with_name("neuron_index_search.parquet")
    )
    finder._read_neuron_index_disk = lambda: pd.DataFrame()

    assert FindNeuronConnection._ensure_neuron_index_from_metadata(finder) is True
    assert pl.read_parquet(index_path).columns[:4] == [
        "bodyId", "type", "instance", "cell_class"
    ]
    # The synthesized columns are included in the expected schema, so a
    # second initialization only refreshes the sidecar if needed.
    assert FindNeuronConnection._ensure_neuron_index_from_metadata(finder) is False


def test_legacy_index_migration_moves_index_and_sidecar(tmp_path):
    """A legacy cache/ index + sidecar move into the app-owned directory once."""
    from src.neuron_index_builder import (
        build_search_cache_frame,
        migrate_legacy_neuron_index,
        search_cache_path,
        system_neuron_index_path,
    )

    dataset = "legacy:v1.0"
    cache_dir = tmp_path / "cache"
    index_dir = tmp_path / "neuron_indexes"
    legacy = cache_dir / "legacy_v1_0" / "neuron_index.parquet"
    legacy.parent.mkdir(parents=True)
    frame = pl.DataFrame({
        "bodyId": ["1", "2"],
        "type": ["aMe12", "APL"],
        "instance": ["aMe12_L", "APL_1"],
        "downstream_complete": [True, False],
        "last_fetched": ["2026-08-12", ""],
        "connection_count": [17, 0],
    })
    frame.write_parquet(legacy)
    build_search_cache_frame(frame).write_parquet(search_cache_path(legacy))

    assert migrate_legacy_neuron_index(dataset, cache_dir=cache_dir, index_dir=index_dir) is True
    target = system_neuron_index_path(dataset, index_dir)
    assert target.is_file()
    assert search_cache_path(target).is_file()
    # The legacy location is emptied and the move keeps the progress flags.
    assert not legacy.exists()
    assert pl.read_parquet(target)["connection_count"].to_list() == [17, 0]
    # Idempotent: a second call is a no-op.
    assert migrate_legacy_neuron_index(dataset, cache_dir=cache_dir, index_dir=index_dir) is False


def test_legacy_index_migration_never_overwrites_existing_target(tmp_path):
    """An existing app-owned index wins over a legacy cache/ index."""
    from src.neuron_index_builder import (
        migrate_legacy_neuron_index,
        system_neuron_index_path,
    )

    dataset = "existing:v1.0"
    cache_dir = tmp_path / "cache"
    index_dir = tmp_path / "neuron_indexes"
    target = system_neuron_index_path(dataset, index_dir)
    target.parent.mkdir(parents=True)
    pl.DataFrame({"bodyId": ["new"], "type": ["new-type"]}).write_parquet(target)
    legacy = cache_dir / "existing_v1_0" / "neuron_index.parquet"
    legacy.parent.mkdir(parents=True)
    pl.DataFrame({"bodyId": ["old"], "type": ["old-type"]}).write_parquet(legacy)

    assert migrate_legacy_neuron_index(dataset, cache_dir=cache_dir, index_dir=index_dir) is False
    assert pl.read_parquet(target)["type"].to_list() == ["new-type"]
    assert legacy.is_file()  # untouched for a later manual recovery


def test_reset_index_progress_zeroes_flags_after_cache_clear(tmp_path):
    """Force-rebuild keeps the index but resets the progress it described."""
    from src.coana import FindNeuronConnection

    index_path = tmp_path / "neuron_indexes" / "reset_v1_0" / "neuron_index.parquet"
    index_path.parent.mkdir(parents=True)
    pl.DataFrame({
        "bodyId": ["1", "2"],
        "type": ["aMe12", "APL"],
        "instance": ["aMe12_L", "APL_1"],
        "post": [10, 3],
        "downstream_complete": [True, False],
        "last_fetched": ["2026-08-12T16:00:00", ""],
        "connection_count": [17, 0],
    }).write_parquet(index_path)

    finder = object.__new__(FindNeuronConnection)
    finder._get_neuron_index_path = lambda: str(index_path)
    finder._vprint = lambda *args, **kwargs: None

    FindNeuronConnection._reset_index_progress(finder)

    reset = pl.read_parquet(index_path)
    assert reset["downstream_complete"].to_list() == [False, False]
    assert reset["last_fetched"].to_list() == ["", ""]
    assert reset["connection_count"].to_list() == [0, 0]
    # Metadata columns are untouched.
    assert reset["type"].to_list() == ["aMe12", "APL"]
