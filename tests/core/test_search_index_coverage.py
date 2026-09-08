"""Extra hermetic coverage for ``neuron_search`` and ``neuron_index_builder``.

Every test builds tiny synthetic parquet/CSV fixtures under ``tmp_path`` and
never touches the repository's ``cache/``, ``datasets/``, ``neuron_indexes/``
directories, the network, or any user configuration.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

import neuron_index_builder as nib
import neuron_search as ns


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _folder_for(dataset: str) -> str:
    return nib.dataset_folder(dataset)


def _write_index(tmp_path, dataset, frame, *, sidecar=True, sidecar_columns=None):
    """Write a synthetic neuron index (+ optional search sidecar) to tmp."""
    root = tmp_path / "neuron_indexes"
    folder = root / _folder_for(dataset)
    folder.mkdir(parents=True, exist_ok=True)
    index_path = folder / "neuron_index.parquet"
    frame.write_parquet(index_path)
    if sidecar:
        nib.build_search_cache_frame(frame, sidecar_columns).write_parquet(
            nib.search_cache_path(index_path)
        )
    return root, index_path


def _main_frame():
    return pl.DataFrame(
        {
            "bodyId": ["100", "200", "300", "400"],
            "type": ["aMe12", "APL", "", "123"],
            "instance": ["aMe12_L", "APL_1", "Other_R", "123_L"],
        }
    )


@pytest.fixture(autouse=True)
def _clear_search_cache():
    ns.clear_cached_neuron_search()
    yield
    ns.clear_cached_neuron_search()


class _PlainColumnFrame:
    """Frame-like object whose columns expose neither to_list nor tolist."""

    def __init__(self, data):
        self._data = data

    def __getitem__(self, column):
        return self._data[column]


class _BrokenFrame:
    """Frame-like object whose non-canonical column access always fails."""

    columns = ["bodyId", "weird"]

    def __getitem__(self, column):
        if column == "weird":
            raise RuntimeError("cannot read column")

        class _Series:
            dtype = object

        return _Series()


# ---------------------------------------------------------------------------
# neuron_index_builder: column helpers
# ---------------------------------------------------------------------------


def test_body_id_column_alias_and_none():
    assert nib.body_id_column(["type", "instance"]) is None
    assert nib.body_id_column(["rootId", "type"]) == "rootId"
    assert nib.body_id_column(["flywire_id", "type"]) == "flywire_id"
    # The canonical name wins over later aliases.
    assert nib.body_id_column(["root_id", "bodyId"]) == "bodyId"


def test_priority_metadata_columns_rank_seven_class_field():
    # A class-like name outside the taxonomy tuple gets the fallback rank 7,
    # i.e. it sorts after the explicit *Type promotions and the taxonomy.
    ordered = nib.priority_metadata_columns(["megaClass", "flywireType", "class"])
    assert ordered == ["flywireType", "class", "megaClass"]
    # Measurement-sounding tokens are rejected even when class-like.
    assert nib.priority_metadata_columns(["classCount"]) == []
    assert nib.is_priority_metadata_column("superclass") is True
    assert nib.is_priority_metadata_column("prediction") is False


def test_is_large_serialized_metadata_column():
    assert nib.is_large_serialized_metadata_column("roiInfo")
    assert nib.is_large_serialized_metadata_column("input ROIs")
    assert not nib.is_large_serialized_metadata_column("somaSide")


def test_searchable_columns_dedupes_and_drops_excluded():
    assert nib.searchable_columns(
        [
            "bodyId",
            "bodyId",
            "type",
            "Unnamed: 0",
            "roiInfo",
            "notes",
            "",
            "last_fetched",
        ]
    ) == ["bodyId", "type", "notes"]


def test_viewer_search_columns_order():
    assert nib.viewer_search_columns(
        ["notes", "instance", "bodyId", "hemibrainType", "type", "size"]
    ) == ["bodyId", "type", "instance", "hemibrainType"]
    assert nib.viewer_search_columns(["size"]) == []


def test_dataset_folder_normalization():
    assert nib.dataset_folder("fly:v1.0") == "fly_v1_0"
    assert nib.dataset_folder(None) == ""


def test_dataset_identifier_from_folder_preserves_release_names():
    assert nib.dataset_identifier_from_folder("manc_v1_2_1") == "manc:v1.2.1"
    assert nib.dataset_identifier_from_folder("banc_v888") == "banc_v888"
    assert nib.dataset_identifier_from_folder("flywire_FAFB_v783") == "flywire_FAFB_v783"


def test_ui_cross_dataset_index_discovery_includes_future_folder(
    tmp_path, monkeypatch
):
    from ui import neuron_index

    folder = tmp_path / "neuron_indexes" / "future_v2_0"
    folder.mkdir(parents=True)
    pl.DataFrame({"bodyId": ["1"], "type": ["FutureA"]}).write_parquet(
        folder / "neuron_index.parquet"
    )
    monkeypatch.setattr(neuron_index, "PROJECT_ROOT", tmp_path)

    assert neuron_index.datasets_with_cached_indexes() == ["future:v2.0"]


def test_search_cache_path_and_system_path(tmp_path):
    index = tmp_path / "x" / "neuron_index.parquet"
    assert nib.search_cache_path(index).name == "neuron_index_search.parquet"
    assert nib.system_neuron_index_path("d:v1", tmp_path) == (
        tmp_path / "d_v1" / "neuron_index.parquet"
    )


# ---------------------------------------------------------------------------
# neuron_index_builder: migration of legacy cache/ indexes
# ---------------------------------------------------------------------------


def test_migrate_legacy_neuron_indexes_all_branches(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    index_dir = tmp_path / "neuron_indexes"

    # A missing cache root is a no-op.
    assert nib.migrate_legacy_neuron_indexes(cache_dir, index_dir) == []

    # Folder with index + sidecar.
    a = cache_dir / "a_v1"
    a.mkdir(parents=True)
    pl.DataFrame({"bodyId": ["1"], "type": ["A"]}).write_parquet(
        a / "neuron_index.parquet"
    )
    pl.DataFrame({"search_column": ["bodyId"]}).write_parquet(
        a / "neuron_index_search.parquet"
    )
    # Folder with only the index.
    b = cache_dir / "b_v1"
    b.mkdir(parents=True)
    pl.DataFrame({"bodyId": ["2"]}).write_parquet(b / "neuron_index.parquet")
    # Folder without any index is skipped.
    (cache_dir / "c_v1").mkdir()
    # A stray file (not a directory) is ignored.
    (cache_dir / "stray.txt").write_text("x", encoding="utf-8")

    migrated = nib.migrate_legacy_neuron_indexes(cache_dir, index_dir)
    assert migrated == ["a_v1", "b_v1"]
    assert (index_dir / "a_v1" / "neuron_index.parquet").is_file()
    assert (index_dir / "a_v1" / "neuron_index_search.parquet").is_file()
    assert (index_dir / "b_v1" / "neuron_index.parquet").is_file()
    assert not (index_dir / "b_v1" / "neuron_index_search.parquet").exists()
    assert not (index_dir / "c_v1").exists()

    # A migration failure is swallowed and the loop continues.
    again_dir = tmp_path / "cache2"
    d = again_dir / "d_v1"
    d.mkdir(parents=True)
    pl.DataFrame({"bodyId": ["3"]}).write_parquet(d / "neuron_index.parquet")

    def boom(dataset, *, cache_dir, index_dir):
        raise OSError("disk gone")

    monkeypatch.setattr(nib, "migrate_legacy_neuron_index", boom)
    assert nib.migrate_legacy_neuron_indexes(again_dir, tmp_path / "idx2") == []


# ---------------------------------------------------------------------------
# neuron_index_builder: search-cache compatibility checks
# ---------------------------------------------------------------------------


def _sidecar_frame(columns, priorities):
    return pl.DataFrame(
        {
            "__neuron_rows": [[0] for _ in columns],
            "search_column": list(columns),
            "search_priority": pl.Series(list(priorities), dtype=pl.UInt16),
            "search_value": ["v" for _ in columns],
            "search_value_folded": ["v" for _ in columns],
        }
    )


def test_is_search_cache_compatible_none_and_schema():
    assert nib.is_search_cache_compatible(None, ["bodyId"]) is False
    missing = pl.DataFrame({"search_column": ["bodyId"]})
    assert nib.is_search_cache_compatible(missing, ["bodyId"]) is False


def test_is_search_cache_compatible_select_failure():
    class _ExplodingFrame:
        columns = [
            "__neuron_rows",
            "search_column",
            "search_priority",
            "search_value",
            "search_value_folded",
        ]

        def select(self, *_args, **_kwargs):
            raise RuntimeError("unreadable")

    assert nib.is_search_cache_compatible(_ExplodingFrame(), ["bodyId"]) is False


def test_is_search_cache_compatible_priority_problems():
    # An unexpected column fails the expected-priority lookup.
    unknown = _sidecar_frame(["bodyId", "mystery"], [0, 1])
    assert nib.is_search_cache_compatible(unknown, ["bodyId"]) is False

    # A wrong priority for a known column is rejected.
    swapped = _sidecar_frame(["type", "bodyId"], [0, 1])
    assert nib.is_search_cache_compatible(swapped, ["bodyId", "type"]) is False

    # A non-integer priority is rejected.
    bad_priority = pl.DataFrame(
        {
            "__neuron_rows": [[0]],
            "search_column": ["bodyId"],
            "search_priority": ["not-a-number"],
            "search_value": ["v"],
            "search_value_folded": ["v"],
        }
    )
    assert nib.is_search_cache_compatible(bad_priority, ["bodyId"]) is False

    # A correct sidecar is accepted.
    good = _sidecar_frame(["bodyId", "type"], [0, 1])
    assert nib.is_search_cache_compatible(good, ["bodyId", "type"]) is True


def test_build_search_cache_frame_empty_projection():
    frame = pl.DataFrame({"size": [1, 2]})
    cache = nib.build_search_cache_frame(frame)
    assert cache.height == 0
    assert set(cache.columns) == {
        "__neuron_rows",
        "search_column",
        "search_priority",
        "search_value",
        "search_value_folded",
    }


def test_build_search_cache_frame_explicit_columns_and_markers():
    frame = pl.DataFrame({"bodyId": ["1", "2"], "type": ["", None]})
    # "instance" is not a frame column, so it drops out of the projection;
    # the all-empty "type" column keeps a self-describing marker row.
    cache = nib.build_search_cache_frame(frame, ["bodyId", "type", "instance"])
    columns = set(cache.get_column("search_column").to_list())
    assert columns == {"bodyId", "type"}
    marker = cache.filter(pl.col("search_column") == "type")
    assert marker.height == 1
    assert marker["__neuron_rows"].to_list() == [[]]


# ---------------------------------------------------------------------------
# neuron_index_builder: metadata discovery and projection
# ---------------------------------------------------------------------------


def test_metadata_candidates_missing_folder(tmp_path):
    assert nib.metadata_candidates("nope:v1", tmp_path) == []
    assert nib.metadata_path("nope:v1", tmp_path) is None


def test_metadata_candidates_discovers_non_exact_names(tmp_path):
    folder = tmp_path / "d_v1"
    folder.mkdir(parents=True)
    exact = folder / "d_v1_neuron_df.csv"
    discovered = folder / "other_neuron_df.csv"
    exact.write_text("bodyId,type\n1,A\n", encoding="utf-8")
    discovered.write_text("bodyId,type\n2,B\n", encoding="utf-8")
    candidates = nib.metadata_candidates("d:v1", tmp_path)
    # The exact-name file is preferred; the discovered one follows.
    assert candidates[0] == exact
    assert discovered in candidates


def test_metadata_columns_parquet_and_missing_body_id(tmp_path):
    folder = tmp_path / "p_v1"
    folder.mkdir(parents=True)
    parquet = folder / "p_v1_allneurons_neuron_df.parquet"
    pl.DataFrame({"root_id": ["1"], "type": ["A"]}).write_parquet(parquet)
    assert nib.metadata_columns(parquet) == ["bodyId", "type"]

    no_id = folder / "nobody.csv"
    no_id.write_text("type,instance\nA,A_L\n", encoding="utf-8")
    assert nib.metadata_columns(no_id) == []


def test_read_metadata_projection_parquet_source(tmp_path):
    parquet = tmp_path / "q_allneurons_neuron_df.parquet"
    pl.DataFrame(
        {"bodyId": ["7"], "type": ["A"], "instance": ["A_L"], "size": [3]}
    ).write_parquet(parquet)
    projection = nib.read_metadata_projection(parquet)
    assert projection.columns == ["bodyId", "type", "instance", "size"]
    assert projection["bodyId"].to_list() == ["7"]


def test_read_metadata_projection_errors_and_padding(tmp_path):
    missing_id = tmp_path / "no_id.csv"
    missing_id.write_text("size\n1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no body ID column"):
        nib.read_metadata_projection(missing_id)

    bare = tmp_path / "bare.csv"
    bare.write_text("bodyId\n9\n", encoding="utf-8")
    projection = nib.read_metadata_projection(bare)
    # Missing type/instance columns are padded with empty strings.
    assert projection["type"].to_list() == [""]
    assert projection["instance"].to_list() == [""]


# ---------------------------------------------------------------------------
# neuron_search: small text/normalization helpers
# ---------------------------------------------------------------------------


def test_normalize_search_text_and_numeric_detection():
    assert ns.normalize_search_text(None) == ""
    assert ns.normalize_search_text("  x ") == "x"
    assert ns.is_numeric_search(True) is False
    assert ns.is_numeric_search(7) is True
    assert ns.is_numeric_search("123.00") is True
    assert ns.is_numeric_search("123.1") is False
    assert ns.is_numeric_search("abc") is False
    assert ns.is_numeric_search(None) is False


def test_scope_column():
    assert ns._scope_column("BodyID") == "bodyId"
    assert ns._scope_column(" TYPE ") == "type"
    assert ns._scope_column(None) == "auto"


def test_body_id_key_variants():
    assert ns._body_id_key("123.0") == "123"
    assert ns._body_id_key("12a.0") == "12a.0"
    assert ns._body_id_key(None) == ""
    assert ns._body_id_key(" 45.0 ") == "45"


def test_normalized_query_numeric_with_dot():
    # All trailing zero decimals are stripped, mirroring _body_id_key and
    # the numeric-search acceptance pattern \d+(?:\.0+)?
    assert ns._normalized_query("100.0") == "100"
    assert ns._normalized_query("100.00") == "100"
    assert ns._normalized_query("  APL ") == "APL"
    assert ns._normalized_query(None) == ""


def test_prefix_literal_forms():
    assert ns._prefix_literal("aMe.*") == "aMe"
    assert ns._prefix_literal("aMe*") == "aMe"
    assert ns._prefix_literal("a*b*") is None
    assert ns._prefix_literal("plain") is None
    assert ns._prefix_literal("[abc.*") is None
    assert ns._prefix_literal("*") is None
    assert ns._prefix_literal(None) is None


def test_numeric_pattern_detection():
    assert ns._numeric_pattern("") is False
    assert ns._numeric_pattern("123.*") is True
    assert ns._numeric_pattern("^123$") is True
    assert ns._numeric_pattern("aMe.*") is False


def test_legacy_regex_pattern_translation():
    assert ns._legacy_regex_pattern("aMe*") == "^aMe.*"
    assert ns._legacy_regex_pattern(".*aMe") == ".*aMe"
    assert ns._legacy_regex_pattern("^aMe") == "^aMe"
    assert ns._legacy_regex_pattern("abc") == "^abc"
    # A pattern already carrying .* still gets the default ^ anchor.
    assert ns._legacy_regex_pattern("a.*b*") == "^a.*b*"


def test_frame_column_values_fallbacks():
    plain = _PlainColumnFrame({"a": [1, 2]})
    assert ns._frame_column_values(plain, "a") == [1, 2]
    # numpy arrays expose tolist() but not to_list().
    array_frame = _PlainColumnFrame({"a": np.array([3, 4])})
    assert ns._frame_column_values(array_frame, "a") == [3, 4]
    # pandas Series take the to_list() path.
    pandas_frame = pd.DataFrame({"a": [5]})
    assert ns._frame_column_values(pandas_frame, "a") == [5]


def test_display_value_normalization():
    assert ns._display_value(None) == ""
    assert ns._display_value(" NaN ") == ""
    assert ns._display_value("<NA>") == ""
    assert ns._display_value(" x ") == "x"
    assert ns._display_value("100.0", body_id=True) == "100"


def test_normalize_search_operator_aliases():
    assert ns.normalize_search_operator("starts_with") == "prefix"
    assert ns.normalize_search_operator("STARTS WITH") == "prefix"
    assert ns.normalize_search_operator("ends with") == "suffix"
    assert ns.normalize_search_operator("equals") == "exact"
    assert ns.normalize_search_operator("contains") == "substring"
    assert ns.normalize_search_operator("regex") == "regex"
    assert ns.normalize_search_operator("bogus") == "substring"
    assert ns.normalize_search_operator(None) == "substring"


# ---------------------------------------------------------------------------
# neuron_search: search plans and pool matching
# ---------------------------------------------------------------------------


def test_search_plan_branches():
    assert ns.search_plan("", ["type"]) == []
    assert ns.search_plan(None, ["type"]) == []

    # Numeric queries are bodyId-only and vanish without a bodyId pool.
    plan = ns.search_plan("123", ["bodyId", "type"])
    assert plan == [
        ns.SearchStage("prefix", ("bodyId",)),
        ns.SearchStage("substring", ("bodyId",)),
    ]
    assert ns.search_plan("123", ["type"]) == []

    # Scoped plans: available and unavailable scopes.
    scoped = ns.search_plan("x", ["type", "instance"], search_columns="type")
    assert scoped == [
        ns.SearchStage("prefix", ("type",)),
        ns.SearchStage("substring", ("type",)),
    ]
    assert ns.search_plan("x", ["type"], search_columns="instance") == []
    # An unknown scope silently degrades to automatic mode.
    assert ns.search_plan("x", ["type"], search_columns="bogus")

    # All-prefix viewer mode.
    full = ns.search_plan("x", ["type", "instance"], all_prefix_matches=True)
    assert full == [
        ns.SearchStage("prefix", ("type", "instance")),
        ns.SearchStage("substring", ("type", "instance")),
    ]
    assert ns.search_plan("x", [], all_prefix_matches=True) == []

    # Default inline plan: type prefix first, then other prefixes, substring.
    plan = ns.search_plan("x", ["bodyId", "type", "instance"])
    assert plan == [
        ns.SearchStage("prefix", ("type",)),
        ns.SearchStage("prefix", ("bodyId", "instance")),
        ns.SearchStage("substring", ("bodyId", "type", "instance")),
    ]
    assert ns.search_plan("x", []) == []


def test_filter_candidate_entries():
    candidates = [("aMe12", "x"), ("APL", "y"), ("aMe17", "z")]
    assert ns.filter_candidate_entries("", candidates) == []
    assert ns.filter_candidate_entries("aMe", candidates) == [
        ("aMe12", "x"),
        ("aMe17", "z"),
    ]
    # The prefix match is case sensitive.
    assert ns.filter_candidate_entries("ame", candidates) == []


def test_match_search_pools():
    pools = {
        "type": [("aMe12", "t1"), ("APL", "t2")],
        "instance": [("aMe12_L", "i1")],
        "bodyId": [("100", "b1")],
    }
    assert ns.match_search_pools("", pools) == []
    assert ns.match_search_pools("x", {}) == []

    # Prefix wins over substring.
    assert ns.match_search_pools("aMe", pools) == [("aMe12", "t1")]
    # Substring stage is case-insensitive.
    assert ns.match_search_pools("apl", pools) == [("APL", "t2")]
    # Limits truncate both collection and final results.
    assert ns.match_search_pools("aMe", pools, limit=1) == [("aMe12", "t1")]
    many = {"type": [("aMe1", "1"), ("aMe2", "2"), ("aMe3", "3")]}
    assert len(ns.match_search_pools("aMe", many, limit=2)) == 2
    # Numeric queries route to the bodyId pool only (prefix stage).
    assert ns.match_search_pools("10", pools) == [("100", "b1")]
    assert ns.match_search_pools("100", pools) == [("100", "b1")]
    assert ns.match_search_pools("20", many) == []
    # Scoped matching.
    assert ns.match_search_pools("aMe12", pools, search_columns="instance") == [
        ("aMe12_L", "i1")
    ]
    assert ns.match_search_pools("aMe12", pools, search_columns="bodyid") == []
    # No stage matches -> empty.
    assert ns.match_search_pools("zzz", pools) == []


# ---------------------------------------------------------------------------
# neuron_search: structured (NeuronFilter) compatibility surface
# ---------------------------------------------------------------------------


def test_dataframe_search_columns_pairs():
    frame = pd.DataFrame({"root_id": [1], "type": ["A"], "instance": ["A_L"]})
    pairs = ns._dataframe_search_columns(frame)
    assert pairs == [("bodyId", "root_id"), ("type", "type"), ("instance", "instance")]
    assert ns._dataframe_search_columns(pd.DataFrame({"size": [1]})) == []


def test_structured_search_columns_pandas_ordering():
    frame = pd.DataFrame(
        {
            "root_id": [1],
            "type": ["A"],
            "instance": ["A_L"],
            "flywireType": ["F"],
            "class": ["C"],
            "size": [10],
            "notes": ["n"],
        }
    )
    assert ns.structured_search_columns(frame) == [
        "root_id",
        "type",
        "instance",
        "flywireType",
        "class",
        "notes",
    ]


def test_structured_search_columns_non_pandas_frames():
    # Non-dataframe callers keep at least the canonical identity fields.
    assert ns.structured_search_columns(_BrokenFrame()) == ["bodyId"]
    # A frame without any recognizable column yields an empty list.
    assert ns.structured_search_columns(object()) == []


def test_structured_operator_column_all_operators():
    series = pd.Series(["aMe12", "APL", None, "AP[L"], index=[0, 1, 2, 3])

    assert ns._structured_operator_column(series, "exact", ["aMe12"]).tolist() == [
        True,
        False,
        False,
        False,
    ]
    assert ns._structured_operator_column(
        series, "contains", ["Me1"]
    ).tolist() == [True, False, False, False]
    assert ns._structured_operator_column(
        series, "not_contains", ["Me1"]
    ).tolist() == [False, True, False, True]
    assert ns._structured_operator_column(
        series, "startswith", ["aMe"]
    ).tolist() == [True, False, False, False]
    assert ns._structured_operator_column(
        series, "endswith", ["PL"]
    ).tolist() == [False, True, False, False]
    assert ns._structured_operator_column(
        series, "regex", [r"^AP"]
    ).tolist() == [False, True, False, True]
    # Invalid regex falls back to literal equality.
    assert ns._structured_operator_column(
        series, "regex", ["AP[L"]
    ).tolist() == [False, False, False, True]
    assert ns._structured_operator_column(
        series, "not_regex", [r"^AP"]
    ).tolist() == [True, False, False, False]
    assert ns._structured_operator_column(
        series, "not_regex", ["AP[L"]
    ).tolist() == [True, True, False, False]
    # Unknown operators and empty pattern lists match nothing.
    assert ns._structured_operator_column(series, "bogus", ["x"]).tolist() == [
        False,
        False,
        False,
        False,
    ]
    assert ns._structured_operator_column(series, "exact", []).tolist() == [
        False,
        False,
        False,
        False,
    ]


def test_apply_structured_filter_contract():
    frame = pd.DataFrame(
        {
            "bodyId": [100, 200, 300],
            "type": ["aMe12", "APL", "100"],
            "instance": ["aMe12_L", "APL_1", "Other_R"],
        }
    )

    # match_all / empty spec / empty frame return untouched copies.
    assert ns.apply_structured_filter(frame, None, match_all=True) is not frame
    assert len(ns.apply_structured_filter(frame, None, match_all=True)) == 3
    assert len(ns.apply_structured_filter(frame, {})) == 3
    empty = frame.head(0)
    assert len(ns.apply_structured_filter(empty, {"contains": ["x"]})) == 0

    # A frame with no searchable columns passes through unchanged.
    numeric_only = pd.DataFrame({"size": [1, 2]})
    assert len(ns.apply_structured_filter(numeric_only, {"contains": ["1"]})) == 2

    # Bare string patterns become one-element lists.
    result = ns.apply_structured_filter(frame, {"contains": "aMe"})
    assert result["bodyId"].tolist() == [100]

    # Empty pattern lists match nothing.
    assert len(ns.apply_structured_filter(frame, {"exact": []})) == 0

    # Integer exact patterns are restricted to the real body-ID column: the
    # type "100" must not be picked up by the numeric pattern 100.
    result = ns.apply_structured_filter(frame, {"exact": [100]})
    assert result["bodyId"].tolist() == [100]

    # Operators AND together, values OR within one operator.
    result = ns.apply_structured_filter(
        frame, {"startswith": ["aMe", "APL"], "contains": ["1"]}
    )
    assert sorted(result["bodyId"].tolist()) == [100, 200]


def test_apply_structured_filter_none_frame_returns_none():
    # A missing frame is returned as-is (None) instead of crashing on
    # ``None.copy()`` — the fixed graceful behavior.
    assert ns.apply_structured_filter(None, {"contains": ["x"]}) is None


# ---------------------------------------------------------------------------
# neuron_search: polars expression builders
# ---------------------------------------------------------------------------


def test_polars_display_expression():
    frame = pl.DataFrame({"bodyId": ["100.0", None], "type": ["A", None]})
    body = frame.select(ns.polars_display_expression("bodyId").alias("v"))["v"]
    assert body.to_list() == ["100", ""]
    other = frame.select(ns.polars_display_expression("type").alias("v"))["v"]
    assert other.to_list() == ["A", ""]


def test_polars_match_column_expression_modes():
    frame = pl.DataFrame({"instance": ["aMe12_L", "APL_1"]})

    def match(mode, text):
        expr = ns.polars_match_column_expression(frame, "instance", text, mode)
        return frame.select(expr.alias("m"))["m"].to_list()

    assert match("prefix", "aMe") == [True, False]
    assert match("suffix", "_1") == [False, True]
    assert match("exact", "APL_1") == [False, True]
    assert match("regex", r"aMe\d+_L") == [True, False]
    # An invalid regex matches nothing instead of raising; the literal
    # False expression evaluates as a scalar.
    assert frame.select(
        ns.polars_match_column_expression(frame, "instance", "(aMe", "regex").alias("m")
    )["m"].to_list() == [False]
    # Substring matching is case-insensitive.
    assert match("substring", "apl") == [False, True]


def test_polars_body_id_guard():
    frame = pl.DataFrame({"bodyId": ["123", "12a"]})
    guard = ns.polars_body_id_guard(frame, ["bodyId"], "123")
    assert frame.select(guard.alias("g"))["g"].to_list() == [True, False]
    # Non-numeric queries and missing bodyId columns disable the guard
    # (the literal True expression evaluates as a scalar).
    true_guard = ns.polars_body_id_guard(frame, ["bodyId"], "abc")
    assert frame.select(true_guard.alias("g"))["g"].to_list() == [True]
    no_body = ns.polars_body_id_guard(frame, ["type"], "123")
    assert frame.select(no_body.alias("g"))["g"].to_list() == [True]


def test_polars_match_expression():
    frame = pl.DataFrame({"type": ["aMe12", "APL"], "instance": ["x_L", "APL_1"]})

    expr = ns.polars_match_expression(frame, ["type", "instance"], "APL", "exact")
    assert frame.select(expr.alias("m"))["m"].to_list() == [False, True]

    # Columns missing from the frame produce an always-false predicate
    # (the literal False expression evaluates as a scalar).
    expr = ns.polars_match_expression(frame, ["nope"], "APL", "exact")
    assert frame.select(expr.alias("m"))["m"].to_list() == [False]

    # Numeric queries gain the integer-only bodyId guard.
    ids = pl.DataFrame({"bodyId": ["100", "10a"], "type": ["100", "100"]})
    expr = ns.polars_match_expression(ids, ["bodyId", "type"], "100", "exact")
    assert ids.select(expr.alias("m"))["m"].to_list() == [True, False]


# ---------------------------------------------------------------------------
# neuron_search: dataframe resolver edge cases
# ---------------------------------------------------------------------------


def _pandas_frame():
    return pd.DataFrame(
        {
            "bodyId": ["100", "200", "300"],
            "type": ["aMe12", "APL", ""],
            "instance": ["aMe12_L", "APL_1", "Other_R"],
            "flywireType": ["MTe07", "", "MTe27"],
        }
    )


def test_resolve_dataframe_query_empty_query_and_no_pairs():
    ids, info = ns.resolve_dataframe_query(_pandas_frame(), "")
    assert ids == []
    assert info["matched_column"] is None

    ids, info = ns.resolve_dataframe_query(pd.DataFrame({"size": [1]}), "1")
    assert ids == []
    assert info["matched_column"] is None


def test_resolve_dataframe_query_without_body_id_column():
    frame = pd.DataFrame({"type": ["APL", "APL"]})
    ids, info = ns.resolve_dataframe_query(frame, "APL")
    # Rows match, but without a body-ID column nothing can be returned.
    assert ids == []
    assert info["matched_column"] is None
    assert info["match_count"] == 0


def test_resolve_dataframe_query_dedupes_duplicate_body_ids():
    frame = pd.DataFrame({"bodyId": ["5", "5"], "type": ["A", "A"]})
    ids, info = ns.resolve_dataframe_query(frame, "A")
    assert [str(value) for value in ids] == ["5"]
    assert info["match_count"] == 1


def test_resolve_dataframe_query_numeric_scopes():
    frame = pd.DataFrame({"bodyId": ["1"], "type": ["123"]})

    # A type-scoped numeric query searches the type column exactly.
    ids, info = ns.resolve_dataframe_query(frame, "123", search_columns="type")
    assert [str(value) for value in ids] == ["1"]
    assert info["matched_column"] == "type"

    ids, info = ns.resolve_dataframe_query(frame, "999", search_columns="type")
    assert ids == []
    assert info["matched_column"] is None

    # A dotted numeric query normalizes to its integer form.
    ids, _ = ns.resolve_dataframe_query(frame, "1.0", search_columns="bodyId")
    assert [str(value) for value in ids] == ["1"]


def test_resolve_dataframe_query_regex_edges():
    frame = _pandas_frame()

    # An invalid regex yields an empty result, never an exception.
    ids, info = ns.resolve_dataframe_query(frame, "(aMe*")
    assert ids == []
    assert info["matched_column"] is None

    # A valid regex with no hits reports the regex mode empty result.
    ids, info = ns.resolve_dataframe_query(frame, ".*ZZZZ999.*")
    assert ids == []
    assert info["matched_column"] is None

    # Prefix literals stop at the first owning column.
    ids, info = ns.resolve_dataframe_query(frame, "MTe.*")
    assert [str(value) for value in ids] == ["100", "300"]
    assert info["matched_column"] == "flywireType"
    assert info["match_mode"] == "prefix"


def test_resolve_dataframe_query_scopes():
    frame = _pandas_frame()

    ids, info = ns.resolve_dataframe_query(frame, "MTe07", search_columns="type")
    assert ids == []

    ids, info = ns.resolve_dataframe_query(
        frame, "aMe12_L", search_columns="instance"
    )
    assert [str(value) for value in ids] == ["100"]
    assert info["matched_column"] == "instance"

    ids, info = ns.resolve_dataframe_query(frame, "100", search_columns="bodyid")
    assert [str(value) for value in ids] == ["100"]

    # Invalid scope degrades to automatic matching.
    ids, info = ns.resolve_dataframe_query(frame, "MTe07", search_columns="bogus")
    assert [str(value) for value in ids] == ["100"]


# ---------------------------------------------------------------------------
# neuron_search: cache loading
# ---------------------------------------------------------------------------


def test_get_cached_neuron_search_missing_index(tmp_path):
    assert ns.get_cached_neuron_search("absent:v1", index_root=tmp_path) is None


def test_get_cached_neuron_search_builds_missing_sidecar_in_memory(tmp_path):
    root, index_path = _write_index(tmp_path, "mem:v1", _main_frame(), sidecar=False)
    cache = ns.get_cached_neuron_search("mem:v1", index_root=root)
    assert cache is not None
    assert cache.search_path is None
    assert cache.body_ids == ("100", "200", "300", "400")
    assert cache.body_id_keys == frozenset({"100", "200", "300", "400"})
    ids, info = ns.resolve_neuron_query(cache, "aMe12")
    assert ids == ["100"]
    assert info["cache"] is True


def test_get_cached_neuron_search_uses_sidecar_and_memoizes(tmp_path):
    root, index_path = _write_index(tmp_path, "memo:v1", _main_frame())
    cache = ns.get_cached_neuron_search("memo:v1", index_root=root)
    assert cache is not None
    assert cache.search_path == nib.search_cache_path(index_path)
    # The process-local memo returns the identical object.
    assert ns.get_cached_neuron_search("memo:v1", index_root=root) is cache


def test_get_cached_neuron_search_evicts_stale_signatures(tmp_path):
    root, index_path = _write_index(tmp_path, "evict:v1", _main_frame())
    first = ns.get_cached_neuron_search("evict:v1", index_root=root)
    assert first is not None

    # Rewriting the index changes the signature and evicts the old reader.
    bigger = _main_frame().vstack(
        pl.DataFrame({"bodyId": ["500"], "type": ["New"], "instance": ["New_L"]})
    )
    bigger.write_parquet(index_path)
    nib.build_search_cache_frame(bigger).write_parquet(
        nib.search_cache_path(index_path)
    )
    second = ns.get_cached_neuron_search("evict:v1", index_root=root)
    assert second is not first
    assert second.body_ids == ("100", "200", "300", "400", "500")
    matching = [
        key for key in ns._SEARCH_CACHE if key[0][0] == str(index_path)
    ]
    assert len(matching) == 1


def test_get_cached_neuron_search_unreadable_index(tmp_path):
    # An index without a bodyId column cannot build the search projection.
    root, _ = _write_index(
        tmp_path,
        "noID:v1",
        pl.DataFrame({"size": [1]}),
        sidecar=False,
    )
    assert ns.get_cached_neuron_search("noID:v1", index_root=root) is None


def test_get_cached_neuron_search_missing_required_columns(tmp_path, monkeypatch):
    root, _ = _write_index(tmp_path, "req:v1", _main_frame(), sidecar=False)
    monkeypatch.setattr(
        ns, "build_search_cache_frame", lambda frame, columns=None: pl.DataFrame({"x": [1]})
    )
    assert ns.get_cached_neuron_search("req:v1", index_root=root) is None


def test_load_signature_missing_file(tmp_path):
    signature = ns._load_signature(tmp_path / "absent.parquet")
    assert signature == (str(tmp_path / "absent.parquet"), None)


def test_cache_covers_frame_exception_is_false():
    broken = ns.CachedNeuronSearch(
        dataset="x",
        index_path=Path("x"),
        search_path=None,
        search_frame=None,
        body_ids=(),
    )
    frame = pd.DataFrame({"bodyId": ["1"]})
    assert ns._cache_covers_frame(broken, frame) is False


# ---------------------------------------------------------------------------
# neuron_search: cached resolver internals
# ---------------------------------------------------------------------------


def test_entries_for_values_empty_columns_and_no_hits(tmp_path):
    root, _ = _write_index(tmp_path, "entries:v1", _main_frame())
    cache = ns.get_cached_neuron_search("entries:v1", index_root=root)
    assert cache is not None

    empty = ns._entries_for_values(cache.search_frame, [], pl.lit(True))
    assert empty.height == 0

    no_hits = ns._entries_for_values(
        cache.search_frame,
        ["type"],
        pl.col("search_value") == "NOTHING_LIKE_THIS",
    )
    assert no_hits.height == 0


def test_body_ids_for_entries_skips_bad_rows():
    cache = ns.CachedNeuronSearch(
        dataset="x",
        index_path=Path("x"),
        search_path=None,
        search_frame=None,
        body_ids=("1", "2"),
    )
    assert ns._body_ids_for_entries(cache, None) == []

    entries = pl.DataFrame(
        {"__neuron_row": [1, 99, None, 0, 0]},
        schema={"__neuron_row": pl.UInt32},
    )
    # Out-of-range and null rows are skipped; duplicates collapse.
    assert ns._body_ids_for_entries(cache, entries) == ["2", "1"]

    empty = entries.head(0)
    assert ns._body_ids_for_entries(cache, empty) == []


def test_scope_columns_selection(tmp_path):
    root, _ = _write_index(tmp_path, "scope:v1", _main_frame())
    cache = ns.get_cached_neuron_search("scope:v1", index_root=root)
    assert ns._scope_columns(cache.search_frame, "bodyid") == ["bodyId"]
    assert ns._scope_columns(cache.search_frame, "instance") == ["instance"]
    assert ns._scope_columns(cache.search_frame, "auto") == [
        "bodyId",
        "type",
        "instance",
    ]
    # Unknown scopes behave like auto.
    assert ns._scope_columns(cache.search_frame, "bogus") == [
        "bodyId",
        "type",
        "instance",
    ]


# ---------------------------------------------------------------------------
# neuron_search: cached resolver contracts
# ---------------------------------------------------------------------------


def test_resolve_neuron_query_none_cache_and_empty_query(tmp_path):
    assert ns.resolve_neuron_query(None, "aMe12") is None

    root, _ = _write_index(tmp_path, "none:v1", _main_frame())
    cache = ns.get_cached_neuron_search("none:v1", index_root=root)
    ids, info = ns.resolve_neuron_query(cache, "   ")
    assert ids == []
    assert info["cache"] is True
    assert info["matched_column"] is None


def test_resolve_neuron_query_scope_without_column(tmp_path):
    frame = pl.DataFrame({"bodyId": ["1"], "type": ["A"]})
    root, _ = _write_index(tmp_path, "noinst:v1", frame)
    cache = ns.get_cached_neuron_search("noinst:v1", index_root=root)
    assert cache is not None
    ids, info = ns.resolve_neuron_query(cache, "A", search_columns="instance")
    assert ids == []
    assert info["matched_column"] is None


def test_resolve_neuron_query_prefix_and_regex_edges(tmp_path):
    root, _ = _write_index(tmp_path, "edges:v1", _main_frame())
    cache = ns.get_cached_neuron_search("edges:v1", index_root=root)

    # A prefix literal with no hits reports an empty prefix result.
    ids, info = ns.resolve_neuron_query(cache, "QQQ.*")
    assert ids == []
    assert info["matched_column"] is None

    # Regexes that already start with .* keep their form and still match.
    ids, info = ns.resolve_neuron_query(cache, ".*aMe.*")
    assert ids == ["100"]
    assert info["matched_column"] == "type"

    # Invalid regexes return empty instead of raising.
    ids, info = ns.resolve_neuron_query(cache, "(aMe*")
    assert ids == []
    assert info["matched_column"] is None

    # A regex with no hits anywhere reports an empty regex result.
    ids, info = ns.resolve_neuron_query(cache, ".*QQQQ999.*")
    assert ids == []
    assert info["matched_column"] is None


def test_resolve_neuron_query_numeric_scope_edges(tmp_path):
    root, _ = _write_index(tmp_path, "numeric:v1", _main_frame())
    cache = ns.get_cached_neuron_search("numeric:v1", index_root=root)

    # Numeric queries with a type scope search the type column exactly.
    ids, info = ns.resolve_neuron_query(cache, "123", search_columns="type")
    assert ids == ["400"]
    assert info["matched_column"] == "type"

    ids, info = ns.resolve_neuron_query(cache, "999", search_columns="type")
    assert ids == []
    assert info["matched_column"] is None

    # Automatic numeric queries only hit integer-like body IDs.
    ids, info = ns.resolve_neuron_query(cache, "100")
    assert ids == ["100"]
    assert info["matched_column"] == "bodyId"

    # Numeric wildcard forms remain bodyId-only.
    ids, info = ns.resolve_neuron_query(cache, "10.*")
    assert ids == ["100"]
    assert info["matched_column"] == "bodyId"

    # A named scope also allows bodyId-only numeric wildcard misses.
    ids, info = ns.resolve_neuron_query(
        cache, "99.*", search_columns="bodyid"
    )
    assert ids == []


def test_resolve_cached_or_dataframe_query_happy_path(tmp_path):
    root, _ = _write_index(tmp_path, "combo:v1", _main_frame())
    cache = ns.get_cached_neuron_search("combo:v1", index_root=root)
    frame = pd.DataFrame(
        {
            "bodyId": [100, 200, 300, 400],
            "type": ["aMe12", "APL", "", "123"],
            "instance": ["aMe12_L", "APL_1", "Other_R", "123_L"],
        }
    )

    ids, info = ns.resolve_cached_or_dataframe_query(cache, frame, "aMe12")
    # Cached keys map back onto the original source values (ints here).
    assert ids == [100]
    assert info["cache"] is True
    assert info["match_count"] == 1

    ids, info = ns.resolve_cached_or_dataframe_query(cache, frame, "Nope")
    assert ids == []
    assert info["cache"] is True


def test_resolve_cached_or_dataframe_query_fallbacks(tmp_path):
    root, _ = _write_index(tmp_path, "fallback:v1", _main_frame())
    cache = ns.get_cached_neuron_search("fallback:v1", index_root=root)

    # Mismatched body-ID sets force the dataframe path.
    bigger = pd.DataFrame(
        {
            "bodyId": ["100", "200", "300", "400", "999"],
            "type": ["aMe12", "APL", "", "123", "Extra"],
            "instance": ["", "", "", "", ""],
        }
    )
    ids, info = ns.resolve_cached_or_dataframe_query(cache, bigger, "Extra")
    assert ids == ["999"]
    assert info["cache"] is False

    # A cache missing a projection column also falls back.
    narrow = pd.DataFrame(
        {
            "bodyId": ["100", "200", "300", "400"],
            "type": ["aMe12", "APL", "", "123"],
            "instance": ["aMe12_L", "APL_1", "Other_R", "123_L"],
            "flywireType": ["MTe07", "", "", ""],
        }
    )
    ids, info = ns.resolve_cached_or_dataframe_query(cache, narrow, "MTe07")
    assert ids == ["100"]
    assert info["cache"] is False
    assert info["matched_column"] == "flywireType"

    # Without a cache the dataframe resolver answers directly.
    ids, info = ns.resolve_cached_or_dataframe_query(None, narrow, "MTe07")
    assert ids == ["100"]
    assert info["cache"] is False


# ---------------------------------------------------------------------------
# final gap-closing tests
# ---------------------------------------------------------------------------


def test_priority_metadata_columns_rank_three_type_field():
    # Generic *Type fields sort between the explicit promotions and taxonomy.
    assert nib.priority_metadata_columns(["megaClass", "cellType"]) == [
        "cellType",
        "megaClass",
    ]


def test_migrate_legacy_neuron_index_no_op_branches(tmp_path):
    dataset = "single:v1.0"
    index_dir = tmp_path / "idx"
    cache_dir = tmp_path / "cache"

    # No legacy file at all.
    assert (
        nib.migrate_legacy_neuron_index(
            dataset, cache_dir=cache_dir, index_dir=index_dir
        )
        is False
    )

    # An existing target wins even once a legacy file appears.
    target = nib.system_neuron_index_path(dataset, index_dir)
    target.parent.mkdir(parents=True)
    pl.DataFrame({"bodyId": ["new"]}).write_parquet(target)
    legacy = cache_dir / _folder_for(dataset) / "neuron_index.parquet"
    legacy.parent.mkdir(parents=True)
    pl.DataFrame({"bodyId": ["old"]}).write_parquet(legacy)
    assert (
        nib.migrate_legacy_neuron_index(
            dataset, cache_dir=cache_dir, index_dir=index_dir
        )
        is False
    )
    assert legacy.is_file()


def test_read_metadata_projection_parquet_alias_rename(tmp_path):
    parquet = tmp_path / "alias_allneurons_neuron_df.parquet"
    pl.DataFrame({"root_id": ["8"], "type": ["B"]}).write_parquet(parquet)
    projection = nib.read_metadata_projection(parquet)
    assert projection.columns[0] == "bodyId"
    assert projection["bodyId"].to_list() == ["8"]


def test_is_numeric_query_integral_values():
    assert ns._is_numeric_query(5) is True
    assert ns._is_numeric_query(True) is False
    assert ns._is_numeric_query("7") is True
    assert ns._is_numeric_query("7x") is False


def test_structured_search_columns_numeric_priority_skipped():
    # A priority-looking name with a numeric dtype is not a text target.
    frame = pd.DataFrame({"bodyId": [1], "cellType": [5], "notes": ["n"]})
    assert ns.structured_search_columns(frame) == ["bodyId", "notes"]


def test_structured_search_columns_unreadable_priority_column():
    class _PriorityBrokenFrame:
        columns = ["bodyId", "cellType"]

        def __getitem__(self, column):
            if column == "cellType":
                raise RuntimeError("unreadable priority column")

            class _Series:
                dtype = object

            return _Series()

    # The unreadable priority column is skipped, canonical fields survive.
    assert ns.structured_search_columns(_PriorityBrokenFrame()) == ["bodyId"]


def test_structured_search_columns_priority_alias_guard(monkeypatch):
    # Defensive guard: even if the priority projection were extended to
    # promote the body-ID alias itself, it must not be re-added.
    monkeypatch.setattr(
        ns, "priority_metadata_columns", lambda names: ["root_id"]
    )
    frame = pd.DataFrame({"root_id": [1]})
    assert ns.structured_search_columns(frame) == ["root_id"]


def test_resolve_dataframe_query_wildcard_and_int_paths():
    frame = _pandas_frame()

    # Numeric wildcard queries are bodyId-only.
    ids, info = ns.resolve_dataframe_query(frame, "10.*")
    assert [str(value) for value in ids] == ["100"]
    assert info["matched_column"] == "bodyId"

    # A prefix literal with no hits reports an empty prefix result.
    ids, info = ns.resolve_dataframe_query(frame, "QQQ.*")
    assert ids == []
    assert info["matched_column"] is None

    # A regex hit reports the regex mode and its owning column.
    ids, info = ns.resolve_dataframe_query(frame, ".*aMe.*")
    assert [str(value) for value in ids] == ["100"]
    assert info["matched_column"] == "type"
    assert info["match_mode"] == "regex"

    # Integer queries resolve against the body-ID column.
    ids, info = ns.resolve_dataframe_query(frame, 200)
    assert [str(value) for value in ids] == ["200"]
    assert info["matched_column"] == "bodyId"


def test_get_cached_neuron_search_rebuilds_stale_sidecar(tmp_path):
    frame = pl.DataFrame(
        {
            "bodyId": ["1"],
            "type": [""],
            "instance": [""],
            "locationType": ["LocA"],
        }
    )
    root = tmp_path / "neuron_indexes"
    folder = root / "stale_v1"
    folder.mkdir(parents=True)
    index_path = folder / "neuron_index.parquet"
    frame.write_parquet(index_path)
    # A sidecar predating the locationType projection: readable but stale.
    nib.build_search_cache_frame(
        frame, ["bodyId", "type", "instance"]
    ).write_parquet(nib.search_cache_path(index_path))

    cache = ns.get_cached_neuron_search("stale:v1", index_root=root)
    assert cache is not None
    assert cache.search_path is None
    ids, info = ns.resolve_neuron_query(cache, "LocA")
    assert ids == ["1"]
    assert info["matched_column"] == "locationType"


def test_resolve_neuron_query_invalid_scope_and_int_query(tmp_path):
    root, _ = _write_index(tmp_path, "inv:v1", _main_frame())
    cache = ns.get_cached_neuron_search("inv:v1", index_root=root)
    assert cache is not None

    # An unknown scope degrades to automatic matching.
    ids, info = ns.resolve_neuron_query(cache, "aMe12", search_columns="bogus")
    assert ids == ["100"]
    assert info["matched_column"] == "type"

    # Integral queries take the numeric bodyId path.
    ids, info = ns.resolve_neuron_query(cache, 300)
    assert ids == ["300"]
    assert info["matched_column"] == "bodyId"
