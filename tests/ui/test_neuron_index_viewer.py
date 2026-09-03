"""Tests for the cached neuron-index viewer data layer and UI shell."""

from types import SimpleNamespace

import polars as pl
import pytest


def _write_index(tmp_path, dataset="test:v1.0"):
    folder = dataset.replace(":", "_").replace(".", "_")
    cache_dir = tmp_path / "neuron_indexes" / folder
    cache_dir.mkdir(parents=True)
    index_path = cache_dir / "neuron_index.parquet"
    pl.DataFrame(
        {
            "bodyId": ["100", "200", "300", "400"],
            "type": ["", "aMe10", "APL", "aMe12"],
            "instance": ["", "aMe10_R", "APL_1", "aMe12_L"],
            "post": [2, 4, 6, 8],
            "downstream_complete": [True, True, False, True],
        }
    ).write_parquet(index_path)
    return dataset, folder, index_path


def _write_priority_index(tmp_path, dataset="priority:v1.0"):
    folder = dataset.replace(":", "_").replace(".", "_")
    cache_dir = tmp_path / "neuron_indexes" / folder
    cache_dir.mkdir(parents=True)
    index_path = cache_dir / "neuron_index.parquet"
    pl.DataFrame(
        {
            "bodyId": ["aMe-body", "100", "200", "300", "400"],
            "type": ["", "aMe-type", "", "", ""],
            "instance": ["hint", "", "aMe-instance", "", ""],
            "flywireType": ["", "", "", "aMe-other", ""],
            "hemilineage": ["", "", "", "", "aMe-ignored"],
            "post": [1, 2, 3, 4, 5],
        }
    ).write_parquet(index_path)
    return dataset


def _write_paged_index(tmp_path, dataset="paged:v1.0", row_count=60):
    folder = dataset.replace(":", "_").replace(".", "_")
    cache_dir = tmp_path / "neuron_indexes" / folder
    cache_dir.mkdir(parents=True)
    index_path = cache_dir / "neuron_index.parquet"
    pl.DataFrame(
        {
            "bodyId": [str(1000 + i) for i in range(row_count)],
            "type": [f"aMe{i:03d}" for i in range(row_count)],
            "instance": [f"aMe{i:03d}_L" for i in range(row_count)],
            "post": list(range(row_count)),
        }
    ).write_parquet(index_path)
    return dataset


def _write_taxonomy_index(tmp_path, dataset="taxonomy:v1.0"):
    folder = dataset.replace(":", "_").replace(".", "_")
    cache_dir = tmp_path / "neuron_indexes" / folder
    cache_dir.mkdir(parents=True)
    index_path = cache_dir / "neuron_index.parquet"
    pl.DataFrame(
        {
            "bodyId": ["100", "200", "300", "400", "500"],
            "type": ["DN1a", "l-LNv", "s-LNv", "DN1a", ""],
            "instance": ["DN1a_L", "l-LNv_L", "s-LNv_L", "DN1a_R", ""],
            "cell_class": ["circadian", "circadian", "circadian", "other", "circadian"],
            "post": [1, 2, 3, 4, 5],
        }
    ).write_parquet(index_path)
    return dataset


@pytest.fixture
def isolated_index_root(tmp_path, monkeypatch):
    import ui.neuron_index as neuron_index

    monkeypatch.setattr(neuron_index, "PROJECT_ROOT", tmp_path)
    neuron_index.clear_neuron_index_cache()
    yield tmp_path
    neuron_index.clear_neuron_index_cache()


class TestNeuronIndexData:
    def test_hit_rendering_marks_only_matching_characters_and_escapes_values(self):
        from ui.neuron_index import _highlight_text_html

        rendered = _highlight_text_html("MeVPaMe2_L <note>", "aMe", "global")

        assert rendered == (
            "MeVP"
            '<mark class="drocat-neuron-match-text">aMe</mark>'
            "2_L &lt;note&gt;"
        )

    def test_viewer_requires_cached_index_even_when_metadata_table_exists(
        self, isolated_index_root
    ):
        from ui.neuron_index import load_cached_neuron_index

        dataset = "test:v1.0"
        dataset_dir = isolated_index_root / "datasets" / "test_v1_0"
        dataset_dir.mkdir(parents=True)
        pl.DataFrame(
            {"bodyId": ["1"], "type": ["APL"], "instance": ["APL_1"]}
        ).write_parquet(dataset_dir / "test_v1_0_allneurons_neuron_df.parquet")

        with pytest.raises(FileNotFoundError):
            load_cached_neuron_index(dataset)

    def test_load_keeps_cached_rows_and_fills_blank_identifiers(
        self, isolated_index_root
    ):
        from ui.neuron_index import load_cached_neuron_index

        dataset, folder, index_path = _write_index(isolated_index_root)
        dataset_dir = isolated_index_root / "datasets" / folder
        dataset_dir.mkdir(parents=True)
        pl.DataFrame(
            {
                "bodyId": ["100", "200", "999"],
                "type": ["aMe1", "aMe10", "not_in_cache"],
                "instance": ["aMe1_L", "aMe10_R", "not_in_cache_1"],
            }
        ).write_parquet(dataset_dir / f"{folder}_allneurons_neuron_df.parquet")

        index = load_cached_neuron_index(dataset)
        assert index.path == index_path
        assert index.enriched is True
        assert index.frame.height == 4
        assert index.frame.filter(pl.col("bodyId") == "100")["type"].item() == "aMe1"
        assert index.frame.filter(pl.col("bodyId") == "100")["instance"].item() == "aMe1_L"
        assert index.frame.filter(pl.col("bodyId") == "999").height == 0

    def test_query_filters_sorts_full_index_before_paging(self, isolated_index_root):
        from ui.neuron_index import load_cached_neuron_index, query_neuron_index

        dataset, _, _ = _write_index(isolated_index_root)
        index = load_cached_neuron_index(dataset, enrich=False)

        result = query_neuron_index(
            index,
            search="ame",
            sort_by="type",
            page=2,
            page_size=1,
        )
        assert result.total == 2
        assert result.pages == 2
        assert result.page == 2
        assert result.rows[0]["type"] == "aMe12"

        filtered = query_neuron_index(
            index,
            filter_column="bodyId",
            filter_text="30",
            sort_by="bodyId",
        )
        assert filtered.total == 1
        assert filtered.rows[0]["bodyId"] == "300"

        column_filtered = query_neuron_index(
            index,
            filter_column="type",
            filter_text="APL",
        )
        assert column_filtered.rows[0]["match_column"] == "type"

        # A filter value without a selected target must not become a second
        # global search, so it cannot interfere with the main search.
        no_target = query_neuron_index(index, filter_text="APL")
        assert no_target.total == 4

        body_sorted = query_neuron_index(index, sort_by="bodyId", page_size=4)
        assert [row["bodyId"] for row in body_sorted.rows] == ["100", "200", "300", "400"]

    def test_include_all_rows_returns_full_filtered_set(
        self, isolated_index_root
    ):
        """The matched-rows export path returns every matching row."""
        from ui.neuron_index import load_cached_neuron_index, query_neuron_index

        dataset = _write_paged_index(isolated_index_root, row_count=60)
        index = load_cached_neuron_index(dataset, enrich=False)

        all_rows = query_neuron_index(
            index, search="aMe", page_size=10, include_all_rows=True)
        assert all_rows.total == 60
        assert all_rows.page == 1 and all_rows.pages == 1
        assert len(all_rows.rows) == 60
        # identical order to the paged traversal, just without the slice
        paged_keys = [
            row["__neuron_key"]
            for page in range(1, 7)
            for row in query_neuron_index(
                index, search="aMe", page=page, page_size=10).rows
        ]
        assert [row["__neuron_key"] for row in all_rows.rows] == paged_keys

        # the mapped-view query exports its complete type set too
        typed = query_neuron_index(
            index,
            types_include=["aMe001", "aMe002"],
            include_all_rows=True,
        )
        assert typed.total == 2
        assert {row["type"] for row in typed.rows} == {"aMe001", "aMe002"}

        # default behavior unchanged: the page slice still applies
        paged_default = query_neuron_index(index, search="aMe", page_size=10)
        assert len(paged_default.rows) == 10
        assert paged_default.pages == 6

    def test_focus_key_returns_page_for_match_value_jump(self, isolated_index_root):
        from ui.neuron_index import load_cached_neuron_index, query_neuron_index

        dataset = _write_paged_index(isolated_index_root, row_count=60)
        index = load_cached_neuron_index(dataset, enrich=False)
        result = query_neuron_index(
            index,
            search="aMe",
            page_size=10,
            focus_key="1050::50",
        )

        assert result.focus_page == 6
        assert result.page == 1
        focused = query_neuron_index(index, search="aMe", page=6, page_size=10)
        assert focused.rows[0]["bodyId"] == "1050"

    def test_global_search_returns_prefixes_then_substring_matches(
        self, isolated_index_root
    ):
        """The viewer keeps every match, with strict prefixes at the top."""
        from ui.neuron_index import load_cached_neuron_index, query_neuron_index

        dataset = "prefix-and-substring:v1.0"
        folder = dataset.replace(":", "_").replace(".", "_")
        cache_dir = isolated_index_root / "neuron_indexes" / folder
        cache_dir.mkdir(parents=True)
        pl.DataFrame(
            {
                "bodyId": ["100", "200", "300", "400", "500", "600"],
                "type": [
                    "aMe01", "MeVPaMe1", "Other", "aMe02", "NoMatch", "MeVPaMe2",
                ],
                "instance": [
                    "aMe01_L", "MeVPaMe1_R", "aMe03_L", "aMe02_R", "aMe04_L",
                    "MeVPaMe2_L",
                ],
                "flywireType": [
                    "", "", "aMe-taxonomy", "", "aMe-taxonomy-2", "aMe19a",
                ],
            }
        ).write_parquet(cache_dir / "neuron_index.parquet")

        index = load_cached_neuron_index(dataset, enrich=False)
        result = query_neuron_index(index, search="aMe", page_size=20)

        # Results are grouped by bodyId → type → instance → taxonomy. Within
        # each field priority, strict prefixes precede substring matches.
        assert result.total == 6
        assert [row["bodyId"] for row in result.rows] == [
            "100", "400", "200", "600", "300", "500",
        ]
        assert [row["match_column_key"] for row in result.rows] == [
            "type", "type", "type", "type", "instance", "instance",
        ]
        type_substring_row = next(
            row for row in result.rows if row["bodyId"] == "200"
        )
        assert type_substring_row["type"] == "MeVPaMe1"
        assert type_substring_row["match_value"] == "MeVPaMe1"
        secondary_row = next(row for row in result.rows if row["bodyId"] == "600")
        assert secondary_row["match_column_keys"] == [
            "type", "instance", "flywireType",
        ]
        assert secondary_row["secondary_match_column_keys"] == [
            "flywireType",
        ]
        assert secondary_row["secondary_match_values"] == [
            "aMe19a",
        ]
        assert '<mark class="drocat-neuron-match-text">aMe</mark>' in (
            secondary_row["__highlighted_cells"]["instance"]
        )
        assert '<mark class="drocat-neuron-match-text">aMe</mark>' in (
            secondary_row["__highlighted_cells"]["flywireType"]
        )
        match_values = {group["match_value"] for group in result.match_groups}
        assert {"MeVPaMe1", "MeVPaMe2"}.issubset(match_values)
        assert "MeVPaMe2_L" not in match_values
        mevpa2_group = next(
            group for group in result.match_groups
            if group["match_value"] == "MeVPaMe2"
        )
        assert mevpa2_group["match_column_key"] == "type"
        assert mevpa2_group["match_role"] == "primary"
        assert mevpa2_group["first_body_id"] == "600"
        assert mevpa2_group["body_count"] == 1
        assert result.match_group_body_ids["MeVPaMe2"] == ("600",)
        assert result.match_group_related["MeVPaMe2"] == (
            "MeVPaMe2", "aMe19a",
        )
        assert result.match_group_primary["MeVPaMe2"] == ("MeVPaMe2",)
        ordered_match_values = [
            group["match_value"] for group in result.match_groups
        ]
        assert ordered_match_values == [
            "aMe01", "aMe02", "MeVPaMe1", "MeVPaMe2", "aMe19a",
            "aMe03_L", "aMe-taxonomy", "aMe04_L", "aMe-taxonomy-2",
        ]
        assert ordered_match_values.index("aMe19a") == (
            ordered_match_values.index("MeVPaMe2") + 1
        )
        assert next(
            group for group in result.match_groups
            if group["match_value"] == "aMe19a"
        )["match_role"] == "secondary"

        scoped_prefix = query_neuron_index(
            index,
            search="aMe",
            search_column="type",
            search_operator="prefix",
            page_size=20,
        )
        assert scoped_prefix.total == 2
        assert [row["bodyId"] for row in scoped_prefix.rows] == ["100", "400"]
        assert all(row["match_column_key"] == "type" for row in scoped_prefix.rows)

        scoped_contains = query_neuron_index(
            index,
            search="aMe",
            search_column="type",
            search_operator="contains",
            page_size=20,
        )
        assert scoped_contains.total == 4
        assert [row["bodyId"] for row in scoped_contains.rows] == [
            "100", "400", "200", "600",
        ]
        assert all(row["match_column_key"] == "type" for row in scoped_contains.rows)

    def test_column_filter_operators_are_targeted_and_anded_with_global_search(
        self, isolated_index_root
    ):
        from ui.neuron_index import load_cached_neuron_index, query_neuron_index

        dataset, _, _ = _write_index(isolated_index_root, "operators:v1.0")
        index = load_cached_neuron_index(dataset, enrich=False)

        prefix = query_neuron_index(
            index,
            filter_column="type",
            filter_text="aMe",
            filter_operator="starts_with",
            page_size=10,
        )
        assert [row["type"] for row in prefix.rows] == ["aMe10", "aMe12"]

        suffix = query_neuron_index(
            index,
            filter_column="instance",
            filter_text="_L",
            filter_operator="ends with",
            page_size=10,
        )
        assert [row["bodyId"] for row in suffix.rows] == ["400"]

        exact = query_neuron_index(
            index,
            filter_column="type",
            filter_text="APL",
            filter_operator="exact",
            page_size=10,
        )
        assert [row["bodyId"] for row in exact.rows] == ["300"]

        contains = query_neuron_index(
            index,
            filter_column="type",
            filter_text="ME10",
            filter_operator="contains",
            page_size=10,
        )
        assert [row["bodyId"] for row in contains.rows] == ["200"]

        regex = query_neuron_index(
            index,
            filter_column="type",
            filter_text=r"^aMe1[02]$",
            filter_operator="regex",
            page_size=10,
        )
        assert [row["bodyId"] for row in regex.rows] == ["200", "400"]

        combined = query_neuron_index(
            index,
            search="aMe",
            filter_column="instance",
            filter_text="_R",
            filter_operator="suffix",
            page_size=10,
        )
        assert [row["bodyId"] for row in combined.rows] == ["200"]

    def test_search_results_default_to_matched_value_ascending(
        self, isolated_index_root
    ):
        from ui.neuron_index import load_cached_neuron_index, query_neuron_index

        dataset = _write_priority_index(isolated_index_root)
        index = load_cached_neuron_index(dataset, enrich=False)
        result = query_neuron_index(index, search="ame", page_size=10)

        assert [row["bodyId"] for row in result.rows] == [
            "aMe-body", "100", "200", "300",
        ]
        assert [row["match_value"] for row in result.rows] == [
            "aMe-body", "aMe-type", "aMe-instance", "aMe-other",
        ]
        assert [row["match_column"] for row in result.rows] == [
            "hint", "type", "instance", "flywireType",
        ]
        assert [row["match_column_key"] for row in result.rows] == [
            "bodyId", "type", "instance", "flywireType",
        ]
        assert result.total == 4

    def test_explicit_sort_overrides_matched_value_default(
        self, isolated_index_root
    ):
        from ui.neuron_index import load_cached_neuron_index, query_neuron_index

        dataset = _write_priority_index(isolated_index_root)
        index = load_cached_neuron_index(dataset, enrich=False)
        result = query_neuron_index(
            index, search="ame", sort_by="bodyId", page_size=10
        )

        assert [row["bodyId"] for row in result.rows] == [
            "100", "200", "300", "aMe-body",
        ]

    def test_matched_value_sort_groups_priority_then_sorts_each_group(
        self, isolated_index_root
    ):
        from ui.neuron_index import load_cached_neuron_index, query_neuron_index

        dataset = "grouped:v1.0"
        folder = dataset.replace(":", "_").replace(".", "_")
        cache_dir = isolated_index_root / "neuron_indexes" / folder
        cache_dir.mkdir(parents=True)
        pl.DataFrame(
            {
                "bodyId": ["aMe-body", "100", "200", "300", "301"],
                "type": ["", "aMe-z", "", "aMe-a", ""],
                "instance": ["", "", "aMe-z-instance", "", "aMe-a-instance"],
                "flywireType": ["", "", "", "", ""],
            }
        ).write_parquet(cache_dir / "neuron_index.parquet")

        index = load_cached_neuron_index(dataset, enrich=False)
        result = query_neuron_index(index, search="ame", page_size=10)

        assert [row["match_column_key"] for row in result.rows] == [
            "bodyId", "type", "type", "instance", "instance",
        ]
        assert [row["match_value"] for row in result.rows] == [
            "aMe-body", "aMe-a", "aMe-z", "aMe-a-instance", "aMe-z-instance",
        ]

    def test_broad_prefix_group_membership_keeps_large_duplicate_group_complete(
        self, isolated_index_root
    ):
        """A broad prefix must not degrade while deduplicating body membership."""
        from ui.neuron_index import load_cached_neuron_index, query_neuron_index

        dataset = "broad:v1.0"
        folder = dataset.replace(":", "_").replace(".", "_")
        cache_dir = isolated_index_root / "neuron_indexes" / folder
        cache_dir.mkdir(parents=True)
        row_count = 2000
        pl.DataFrame(
            {
                "bodyId": [str(10000 + i) for i in range(row_count)],
                "type": ["a"] * row_count,
                "instance": [""] * row_count,
                "post": list(range(row_count)),
            }
        ).write_parquet(cache_dir / "neuron_index.parquet")

        index = load_cached_neuron_index(dataset, enrich=False)
        result = query_neuron_index(index, search="a", page_size=10)

        assert result.total == row_count
        group = next(
            group for group in result.match_groups
            if group["match_value"] == "a"
        )
        assert group["body_count"] == row_count
        assert len(result.match_group_members["a"]) == row_count
        assert len(result.match_group_body_ids["a"]) == row_count

    def test_shared_match_stage_deduplicates_names_and_verifies_body_ids(
        self, isolated_index_root
    ):
        from ui.neuron_index import load_cached_neuron_index, query_neuron_index

        dataset = "shared:v1.0"
        folder = dataset.replace(":", "_").replace(".", "_")
        cache_dir = isolated_index_root / "neuron_indexes" / folder
        cache_dir.mkdir(parents=True)
        pl.DataFrame(
            {
                "bodyId": ["100", "200", "not-a-body"],
                "type": ["aMeType", "aMeOther", "NoDigits"],
                "instance": ["", "aMeType", "NoDigits_1"],
            }
        ).write_parquet(cache_dir / "neuron_index.parquet")

        index = load_cached_neuron_index(dataset, enrich=False)
        result = query_neuron_index(index, search="ME", page_size=1)

        # The lower/upper-case query has no strict prefix, so the shared
        # matcher uses its case-insensitive substring stage. A matched value
        # is deduplicated by its matched column: a type selection must not
        # absorb a row where the same spelling only occurs in instance.
        assert result.total == 2
        assert {group["match_value"] for group in result.match_groups} == {
            "aMeOther", "aMeType",
        }
        type_group = next(
            group for group in result.match_groups
            if group["match_value"] == "aMeType"
        )
        assert type_group["body_count"] == 1
        assert set(result.match_group_members["aMeType"]) == {
            "100::0",
        }

        strict = query_neuron_index(index, search="aMe", page_size=10)
        strict_group = next(
            group for group in strict.match_groups
            if group["match_value"] == "aMeType"
        )
        assert strict_group["body_count"] == 1
        assert set(strict.match_group_members["aMeType"]) == {
            "100::0",
        }

        # A numeric query is guarded to bodyId; it must not match the text
        # “NoDigits” or a type/instance containing the same digits.
        numeric = query_neuron_index(index, search="1", page_size=10)
        assert [row["bodyId"] for row in numeric.rows] == ["100"]
        assert numeric.rows[0]["match_column_key"] == "bodyId"

    def test_viewer_returns_all_prefix_columns_while_suggestions_stay_type_first(
        self, isolated_index_root
    ):
        from ui.neuron_index import load_cached_neuron_index, query_neuron_index
        from ui.type_suggestions import match_suggestions

        dataset = "cross-fields:v1.0"
        folder = dataset.replace(":", "_").replace(".", "_")
        cache_dir = isolated_index_root / "neuron_indexes" / folder
        cache_dir.mkdir(parents=True)
        frame = pl.DataFrame(
            {
                "bodyId": ["100", "200", "300", "400"],
                "type": ["MTe01a", "", "", "Other"],
                "instance": ["MTe01a_L", "MTe02_L", "", ""],
                "flywireType": ["MTe01a", "MTe02", "MTe03", "MTe04"],
            }
        )
        frame.write_parquet(cache_dir / "neuron_index.parquet")

        index = load_cached_neuron_index(dataset, enrich=False)
        result = query_neuron_index(index, search="MTe", page_size=20)
        assert result.total == 4
        assert [group["match_value"] for group in result.match_groups] == [
            "MTe01a", "MTe02_L", "MTe02", "MTe03", "MTe04",
        ]
        assert [row["match_column_key"] for row in result.rows] == [
            "type", "instance", "flywireType", "flywireType",
        ]

        pools = {
            "type": [("MTe01a", "type"), ("Other", "type")],
            "instance": [("MTe01a_L", "instance"), ("MTe02_L", "instance")],
            "flywireType": [
                ("MTe01a", "flywireType"),
                ("MTe02", "flywireType"),
                ("MTe03", "flywireType"),
                ("MTe04", "flywireType"),
            ],
        }
        assert match_suggestions("MTe", pools, limit=None) == [
            ("MTe01a", "type"),
        ]
        assert match_suggestions(
            "MTe", pools, limit=None, all_prefix_matches=True
        ) == [
            ("MTe01a", "type"),
            ("MTe01a_L", "instance"),
            ("MTe02_L", "instance"),
            ("MTe01a", "flywireType"),
            ("MTe02", "flywireType"),
            ("MTe03", "flywireType"),
            ("MTe04", "flywireType"),
        ]

    def test_viewer_prefix_union_matches_authoritative_metadata_and_exact_resolution(
        self, isolated_index_root
    ):
        import pandas as pd
        from src.statvis import _process_single_neuron
        from ui.neuron_index import load_cached_neuron_index, query_neuron_index
        from ui.type_suggestions import match_suggestions

        dataset = "authoritative:v1.0"
        folder = dataset.replace(":", "_").replace(".", "_")
        cache_dir = isolated_index_root / "neuron_indexes" / folder
        cache_dir.mkdir(parents=True)
        source = pd.DataFrame(
            {
                "bodyId": ["100", "200", "300", "400", "500"],
                "type": ["MTe01a", "", "", "Other", ""],
                "instance": ["", "", "", "", "MTe05_L"],
                "flywireType": ["MTe01a", "MTe02", "MTe03", "MTe04", "MTe05"],
            }
        )
        pl.from_pandas(source).write_parquet(cache_dir / "neuron_index.parquet")

        index = load_cached_neuron_index(dataset, enrich=False)
        viewer = query_neuron_index(index, search="MTe", page_size=20)
        search_columns = ["bodyId", "type", "instance", "flywireType"]
        authoritative_ids = set(
            source.loc[
                source[search_columns].astype(str).apply(
                    lambda column: column.str.startswith("MTe")
                ).any(axis=1),
                "bodyId",
            ]
        )
        viewer_ids = {
            body_id
            for body_ids in viewer.match_group_body_ids.values()
            for body_id in body_ids
        }
        assert viewer_ids == authoritative_ids

        pools = {
            column: [(value, column) for value in sorted(source[column].unique()) if value]
            for column in search_columns[1:]
        }
        assert match_suggestions("MTe", pools, limit=None) == [("MTe01a", "type")]

        prefix_ids, prefix_info = _process_single_neuron(
            "MTe.*", source, source["bodyId"].tolist(),
            verbose=False, search_columns="auto",
        )
        # The viewer remains broad and exposes all authoritative cross-column
        # hits. The analysis resolver intentionally stops at the first type
        # column with a prefix match.
        assert set(str(value) for value in prefix_ids) == {"100"}
        assert prefix_info["matched_column"] == "type"

        # The viewer displays names, but its selection resolution is exact:
        # feeding those resolved IDs through the real metadata resolver gives
        # exactly the same rows, with no column-priority collision.
        resolved_ids = sorted(viewer_ids)
        real_ids = []
        body_ids = source["bodyId"].tolist()
        for body_id in resolved_ids:
            matches, info = _process_single_neuron(
                body_id, source, body_ids, verbose=False, search_columns="auto"
            )
            assert info["matched_column"] == "bodyId"
            real_ids.extend(str(value) for value in matches)
        assert set(real_ids) == authoritative_ids

    def test_load_refreshes_when_progress_sidecar_changes(self, isolated_index_root):
        from ui.neuron_index import (
            load_cached_neuron_index,
            neuron_index_state_path,
        )

        dataset, folder, _ = _write_index(isolated_index_root)
        first = load_cached_neuron_index(dataset, enrich=False)
        state_path = neuron_index_state_path(
            dataset, isolated_index_root / "cache"
        )
        # The state sidecar lives in the cache boundary while the index
        # lives in neuron_indexes/; create the cache folder explicitly.
        state_path.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(
            {
                "bodyId": ["100"],
                "downstream_complete": [False],
                "last_fetched": ["2026-08-12T16:00:00"],
                "connection_count": [17],
            }
        ).write_parquet(state_path)

        second = load_cached_neuron_index(dataset, enrich=False)
        assert second is not first
        row = second.frame.filter(pl.col("bodyId") == "100").row(0, named=True)
        assert row["downstream_complete"] is False
        assert row["connection_count"] == 17

    def test_viewer_loads_presorted_search_sidecar_and_keeps_query_order(
        self, isolated_index_root
    ):
        from src.neuron_index_builder import build_search_cache_frame, search_cache_path
        from ui.neuron_index import load_cached_neuron_index, query_neuron_index

        dataset = _write_priority_index(isolated_index_root)
        folder = dataset.replace(":", "_").replace(".", "_")
        index_path = isolated_index_root / "neuron_indexes" / folder / "neuron_index.parquet"
        source = pl.read_parquet(index_path)
        build_search_cache_frame(source).write_parquet(search_cache_path(index_path))

        index = load_cached_neuron_index(dataset, enrich=False)
        assert "__neuron_rows" in index.search_frame.columns
        result = query_neuron_index(index, search="ame", page_size=10)
        assert [row["match_value"] for row in result.rows] == [
            "aMe-body", "aMe-type", "aMe-instance", "aMe-other",
        ]
        assert [row["match_column_key"] for row in result.rows] == [
            "bodyId", "type", "instance", "flywireType",
        ]

    def test_presorted_search_with_no_hits_keeps_exploded_hit_schema(
        self, isolated_index_root
    ):
        """A valid query with no matches must not crash the sidecar path."""
        from ui.neuron_index import load_cached_neuron_index, query_neuron_index

        dataset = _write_priority_index(isolated_index_root)
        index = load_cached_neuron_index(dataset, enrich=False)

        result = query_neuron_index(index, search="not-present-anywhere")

        assert result.total == 0
        assert result.rows == []
        assert result.match_groups == []

    def test_include_all_keys_materialises_every_matching_row(
        self, isolated_index_root
    ):
        from ui.neuron_index import load_cached_neuron_index, query_neuron_index

        dataset = _write_paged_index(isolated_index_root, row_count=60)
        index = load_cached_neuron_index(dataset, enrich=False)

        result = query_neuron_index(
            index,
            search="aMe",
            page_size=10,
            include_all_keys=True,
        )

        expected = {str(1000 + i) for i in range(60)}
        assert result.total == 60
        assert len(result.all_keys) == 60
        assert set(result.all_keys.values()) == expected
        for body_id in result.all_keys.values():
            assert body_id in expected

        # Without the flag the full map stays empty to keep paged browsing cheap.
        plain = query_neuron_index(index, search="aMe", page_size=10)
        assert plain.all_keys == {}

    def test_match_group_subtypes_expand_a_coarse_entry(
        self, isolated_index_root
    ):
        from ui.neuron_index import (
            load_cached_neuron_index,
            query_match_group_subtypes,
            query_neuron_index,
        )

        dataset = _write_taxonomy_index(isolated_index_root)
        index = load_cached_neuron_index(dataset, enrich=False)

        result = query_neuron_index(index, search="circadian")
        assert [group["match_value"] for group in result.match_groups] == [
            "circadian"
        ]
        group = result.match_groups[0]
        assert group["match_column_key"] == "cell_class"
        assert group["body_count"] == 4

        payload = query_match_group_subtypes(
            index, result.match_group_members["circadian"]
        )
        assert payload["total_types"] == 3
        assert payload["truncated"] is False
        # Sorted by type name; the blank-type body stays out of the list.
        assert [
            subtype["match_value"] for subtype in payload["subtypes"]
        ] == ["DN1a", "l-LNv", "s-LNv"]
        dn1a = payload["subtypes"][0]
        # Membership is restricted to the expanded entry: the DN1a body in
        # the "other" class must not appear.
        assert dn1a["body_ids"] == ("100",)
        assert dn1a["member_keys"] == ("100::0",)
        assert dn1a["first_body_id"] == "100"
        assert dn1a["body_count"] == 1

    def test_match_group_subtypes_ignores_unknown_keys_and_caps_output(
        self, isolated_index_root
    ):
        from ui.neuron_index import (
            load_cached_neuron_index,
            query_match_group_subtypes,
        )

        dataset = _write_taxonomy_index(isolated_index_root)
        index = load_cached_neuron_index(dataset, enrich=False)

        assert query_match_group_subtypes(index, ["nonsense", "", None]) == {
            "subtypes": [],
            "total_types": 0,
            "truncated": False,
        }

        payload = query_match_group_subtypes(
            index,
            ["100::0", "200::1", "300::2", "400::3"],
            limit=2,
        )
        assert payload["total_types"] == 3
        assert payload["truncated"] is True
        assert [subtype["match_value"] for subtype in payload["subtypes"]] == [
            "DN1a",
            "l-LNv",
        ]
        # The cap keeps the DN1a bodies from both taxonomy classes because
        # both rows are part of the requested membership.
        dn1a = payload["subtypes"][0]
        assert dn1a["body_ids"] == ("100", "400")

        typeless = "typeless:v1.0"
        folder = typeless.replace(":", "_").replace(".", "_")
        cache_dir = isolated_index_root / "neuron_indexes" / folder
        cache_dir.mkdir(parents=True)
        pl.DataFrame(
            {
                "bodyId": ["100", "200"],
                "post": [1, 2],
            }
        ).write_parquet(cache_dir / "neuron_index.parquet")
        typeless_index = load_cached_neuron_index(typeless, enrich=False)
        assert query_match_group_subtypes(typeless_index, ["100::0"]) == {
            "subtypes": [],
            "total_types": 0,
            "truncated": False,
        }


class TestNeuronIndexViewer:
    def _click(self, link):
        listener = next(
            listener
            for listener in link._event_listeners.values()
            if listener.type == "click"
        )
        listener.handler(SimpleNamespace())

    def test_link_opens_rendered_cached_index(self, isolated_index_root, monkeypatch):
        from nicegui import Client
        from nicegui.page import page
        import ui.components.neuron_index_viewer as viewer
        from ui.components.neuron_index_viewer import create_neuron_index_viewer_link

        dataset, _, _ = _write_index(isolated_index_root)
        monkeypatch.setattr(viewer, "PROJECT_ROOT", isolated_index_root)

        client = Client(page("/neuron-index-viewer-cached"))
        with client:
            link = create_neuron_index_viewer_link(
                lambda: dataset,
                query_values_getter=lambda: [],
            )
        self._click(link)

        texts = [el.text for el in client.elements.values() if getattr(el, "text", "")]
        assert "See available neurons" in texts
        assert any("Available neurons · test:v1.0" in text for text in texts)
        labels = [
            getattr(el, "_props", {}).get("label") for el in client.elements.values()
        ]
        assert "Search identities & taxonomy" in labels
        assert "Target column" in labels
        assert "Match mode" in labels
        assert "Target column value" not in labels
        assert "Sort by" in labels
        search_fields = [
            element for element in client.elements.values()
            if "drocat-neuron-search-field" in getattr(element, "_classes", set())
        ]
        assert len(search_fields) == 6
        assert all(field._props.get("outlined") is True for field in search_fields)
        toolbar = next(
            element for element in client.elements.values()
            if "drocat-neuron-search-toolbar" in getattr(element, "_classes", set())
        )
        assert toolbar._classes
        header_meta = next(
            element for element in client.elements.values()
            if "drocat-neuron-header-meta" in getattr(element, "_classes", set())
        )
        assert any(
            "indexed rows" in getattr(element, "text", "")
            for element in client.elements.values()
        )
        assert any(
            "Source:" in getattr(element, "text", "")
            for element in client.elements.values()
        )
        assert header_meta._classes
        intro = next(
            element for element in client.elements.values()
            if "drocat-neuron-intro-row" in getattr(element, "_classes", set())
        )
        assert intro._classes
        assert any(
            "drocat-neuron-search-help" in getattr(element, "_classes", set())
            for element in client.elements.values()
        )
        tables = [el for el in client.elements.values() if type(el).__name__ == "Table"]
        assert len(tables) == 2
        match_table = next(
            table for table in tables
            if table._props["columns"][0]["name"] == "match_column"
        )
        full_table = next(
            table for table in tables
            if table._props["columns"][0]["name"] == "bodyId"
        )
        assert match_table._props["columns"][0]["label"] == "Matched by"
        assert match_table._props["columns"][1]["name"] == "match_value"
        assert match_table._props["columns"][1]["label"] == "Matched value"
        assert match_table._props["columns"][2]["name"] == "body_count"
        assert "header" in match_table.slots
        header_template = match_table.slots["header"].template
        assert "drocat-neuron-match-select-cell" in header_template
        assert "v-model=\"props.selected\"" in header_template
        assert ':indeterminate="props.selected === null"' in header_template
        assert "props.multipleSelect" not in header_template
        assert "v-for=\"col in props.cols\"" in header_template
        assert "body" in match_table.slots
        assert "match-value-click" in match_table.slots["body"].template
        assert "secondary" in match_table.slots["body"].template
        assert "drocat-neuron-match-secondary-row" in match_table.slots["body"].template
        assert "arrow_right_alt" in match_table.slots["body"].template
        assert "first_body_id" in match_table.slots["body"].template
        assert full_table._props["columns"][0]["name"] == "bodyId"
        assert full_table._props["selection"] == "multiple"
        assert "match_column" not in {
            column["name"] for column in full_table._props["columns"]
        }
        assert "body" in full_table.slots
        assert "drocat-neuron-hit-cell" in full_table.slots["body"].template
        assert "drocat-neuron-secondary-hit-cell" in full_table.slots["body"].template
        assert "match_column_keys" in full_table.slots["body"].template
        assert "secondary_match_column_keys" in full_table.slots["body"].template
        assert "__highlighted_cells" in full_table.slots["body"].template
        assert "v-html" in full_table.slots["body"].template
        assert "data-neuron-key" in full_table.slots["body"].template
        assert "drocat-neuron-selected-row" in full_table.slots["body"].template
        assert ':props="props"' not in full_table.slots["body"].template.split(
            "<q-tr", 1
        )[1].split(">", 1)[0]
        assert "q-checkbox" in full_table.slots["body"].template

    def test_multi_dataset_picker_is_outlined(self, isolated_index_root, monkeypatch):
        from nicegui import Client
        from nicegui.page import page
        import ui.components.neuron_index_viewer as viewer
        from ui.components.neuron_index_viewer import create_neuron_index_viewer_link

        datasets = ["test:v1.0", "other:v1.0"]
        _write_index(isolated_index_root, datasets[0])
        monkeypatch.setattr(viewer, "PROJECT_ROOT", isolated_index_root)

        client = Client(page("/neuron-index-viewer-multi-dataset"))
        with client:
            link = create_neuron_index_viewer_link(lambda: datasets)
        self._click(link)

        picker = next(
            element
            for element in client.elements.values()
            if getattr(element, "_props", {}).get("label") == "Dataset to view"
        )
        assert picker._props.get("outlined") is True

    def test_match_value_click_scrolls_the_target_row_after_page_jump(
        self, isolated_index_root, monkeypatch
    ):
        from nicegui import Client
        from nicegui.page import page
        import ui.components.neuron_index_viewer as viewer
        from ui.components.neuron_index_viewer import create_neuron_index_viewer_link

        dataset = _write_paged_index(isolated_index_root, row_count=60)
        monkeypatch.setattr(viewer, "PROJECT_ROOT", isolated_index_root)
        scripts = []
        monkeypatch.setattr(viewer.ui, "run_javascript", scripts.append)

        client = Client(page("/neuron-index-viewer-row-jump"))
        with client:
            link = create_neuron_index_viewer_link(lambda: dataset)
        self._click(link)

        search_input = next(
            element for element in client.elements.values()
            if getattr(element, "_props", {}).get("label")
            == "Search identities & taxonomy"
        )
        search_listener = next(iter(search_input._event_listeners.values()))
        search_input._handle_event({
            "listener_id": search_listener.id,
            "args": "aMe",
        })

        tables = [el for el in client.elements.values() if type(el).__name__ == "Table"]
        match_table = next(
            table for table in tables
            if table._props["columns"][0]["name"] == "match_column"
        )
        match_next = next(
            element for element in client.elements.values()
            if getattr(element, "text", "") == "Next matches"
        )
        self._click(match_next)
        target = next(
            row for row in match_table._props["rows"]
            if row["match_value"] == "aMe050"
        )
        click_listener = next(
            listener for listener in match_table._event_listeners.values()
            if listener.type == "matchValueClick"
        )
        match_table._handle_event({
            "listener_id": click_listener.id,
            "args": target,
        })

        assert any("scrollIntoView" in script for script in scripts)
        assert any("1050::50" in script for script in scripts)
        focus_scripts = [script for script in scripts if "scrollIntoView" in script]
        assert len(focus_scripts) == 1
        assert "const signature = anchor" in focus_scripts[0]
        assert "blockedUntil" in focus_scripts[0]
        # A value click followed by the duplicate QTable event must not
        # restart the breathing row notification.
        match_table._handle_event({
            "listener_id": click_listener.id,
            "args": target,
        })
        match_table._selection_handlers[0](SimpleNamespace(selection=[target]))
        assert len([script for script in scripts if "scrollIntoView" in script]) == 1
        # A different matched entry selected immediately afterwards must get
        # its own anchor instead of being swallowed by the duplicate guard.
        another = next(
            row for row in match_table._props["rows"]
            if row["match_value"] == "aMe051"
        )
        match_table._selection_handlers[0](SimpleNamespace(selection=[another]))
        focus_scripts = [script for script in scripts if "scrollIntoView" in script]
        assert len(focus_scripts) == 2
        assert "1051::51" in focus_scripts[-1]
        full_table = next(
            table for table in tables
            if table._props["columns"][0]["name"] == "bodyId"
        )
        assert full_table._props["rows"][0]["bodyId"] == "1050"

    def test_broad_search_bounds_match_panel_payload_without_dropping_rows(
        self, isolated_index_root, monkeypatch
    ):
        from nicegui import Client
        from nicegui.page import page
        import ui.components.neuron_index_viewer as viewer
        from ui.components.neuron_index_viewer import (
            MATCH_GROUP_PAGE_SIZE,
            create_neuron_index_viewer_link,
        )

        dataset = _write_paged_index(isolated_index_root, row_count=300)
        monkeypatch.setattr(viewer, "PROJECT_ROOT", isolated_index_root)

        client = Client(page("/neuron-index-viewer-broad-search"))
        with client:
            link = create_neuron_index_viewer_link(lambda: dataset)
        self._click(link)

        search_input = next(
            element for element in client.elements.values()
            if getattr(element, "_props", {}).get("label")
            == "Search identities & taxonomy"
        )
        search_listener = next(iter(search_input._event_listeners.values()))
        search_input._handle_event({
            "listener_id": search_listener.id,
            "args": "a",
        })

        tables = [el for el in client.elements.values() if type(el).__name__ == "Table"]
        match_table = next(
            table for table in tables
            if table._props["columns"][0]["name"] == "match_column"
        )
        full_table = next(
            table for table in tables
            if table._props["columns"][0]["name"] == "bodyId"
        )
        assert len(match_table._props["rows"]) == MATCH_GROUP_PAGE_SIZE
        assert len(full_table._props["rows"]) == 50
        status_text = [
            getattr(element, "text", "")
            for element in client.elements.values()
            if "matched names" in getattr(element, "text", "")
        ]
        assert status_text == [
            "Showing 1–50 of 300 matched names"
        ]

    def test_match_panel_deduplicates_and_syncs_query_selection(
        self, isolated_index_root, monkeypatch
    ):
        from nicegui import Client
        from nicegui.page import page
        import ui.components.neuron_index_viewer as viewer
        from ui.components.neuron_index_viewer import create_neuron_index_viewer_link

        dataset, _, _ = _write_index(isolated_index_root)
        monkeypatch.setattr(viewer, "PROJECT_ROOT", isolated_index_root)
        scripts = []
        monkeypatch.setattr(viewer.ui, "run_javascript", scripts.append)
        current_query = ["existing"]
        selection_batches = []
        resolution_batches = []
        edited_values = []

        def sync_query(values):
            selection_batches.append(list(values))
            current_query[:] = ["existing", *values]

        def sync_resolution(values):
            resolution_batches.append(list(values))

        client = Client(page("/neuron-index-viewer-selection"))
        with client:
            link = create_neuron_index_viewer_link(
                lambda: dataset,
                query_values_getter=lambda: current_query,
                query_selection=sync_query,
                query_resolution=sync_resolution,
                query_edit=edited_values.append,
                query_label="Source Neurons",
            )
        self._click(link)

        preview_list = next(
            element for element in client.elements.values()
            if "drocat-neuron-query-preview-list" in getattr(element, "_classes", set())
        )
        assert "drocat-neuron-query-preview-collapsed" in preview_list._classes
        preview_expand = next(
            element for element in client.elements.values()
            if "drocat-query-preview-expand-btn" in getattr(element, "_classes", set())
        )
        preview_click = next(
            listener for listener in preview_expand._event_listeners.values()
            if listener.type == "click"
        )
        preview_click.handler(SimpleNamespace())
        assert "drocat-neuron-query-preview-expanded" in preview_list._classes
        preview_click.handler(SimpleNamespace())
        assert "drocat-neuron-query-preview-collapsed" in preview_list._classes
        preview_chip = next(
            element for element in client.elements.values()
            if "drocat-neuron-query-chip-wrap" in getattr(element, "_classes", set())
        )
        preview_dblclick = next(
            listener for listener in preview_chip._event_listeners.values()
            if listener.type == "dblclick"
        )
        preview_dblclick.handler(SimpleNamespace(args=None))
        assert edited_values == ["existing"]

        search_input = next(
            element for element in client.elements.values()
            if getattr(element, "_props", {}).get("label")
            == "Search identities & taxonomy"
        )
        search_listener = next(iter(search_input._event_listeners.values()))
        search_input._handle_event({
            "listener_id": search_listener.id,
            "args": "ame",
        })

        tables = [el for el in client.elements.values() if type(el).__name__ == "Table"]
        match_table = next(
            table for table in tables
            if table._props["columns"][0]["name"] == "match_column"
        )
        assert match_table._props["selection"] == "multiple"
        full_table = next(
            table for table in tables
            if table._props["columns"][0]["name"] == "bodyId"
        )
        assert [row["match_value"] for row in match_table._props["rows"]] == [
            "aMe10", "aMe12",
        ]
        assert [row["body_count"] for row in match_table._props["rows"]] == [1, 1]
        assert not any(
            getattr(el, "text", "") == "Add selected to query"
            for el in client.elements.values()
        )
        assert not any(
            "per visible row" in getattr(el, "text", "")
            for el in client.elements.values()
        )

        selected_groups = [
            match_table._props["rows"][0],
            match_table._props["rows"][1],
        ]
        match_table._selection_handlers[0](SimpleNamespace(selection=selected_groups))

        assert selection_batches[-1] == ["aMe10", "aMe12"]
        assert resolution_batches[-1] == ["200", "400"]
        assert current_query == ["existing", "aMe10", "aMe12"]
        assert {
            str(row["bodyId"]) for row in full_table.selected
        } == {"200", "400"}
        assert any("scrollIntoView" in script for script in scripts)
        assert any(
            getattr(el, "text", "") == "Current query · Source Neurons"
            for el in client.elements.values()
        )
        preview_text = [
            getattr(el, "text", "")
            for el in client.elements.values()
            if "drocat-neuron-query-chip" in getattr(el, "_classes", set())
        ]
        assert preview_text == current_query

        # Deselecting the side-panel groups removes viewer-owned values.
        match_table._selection_handlers[0](SimpleNamespace(selection=[]))
        assert selection_batches[-1] == []
        assert resolution_batches[-1] == []
        assert current_query == ["existing"]

        # A table checkbox selects one bodyId, not the shared matched name;
        # clearing it removes that viewer-owned body ID again.
        full_table._selection_handlers[0](
            SimpleNamespace(selection=[full_table._props["rows"][0]])
        )
        assert selection_batches[-1] == ["200"]
        assert resolution_batches[-1] == ["200"]
        assert [str(row["bodyId"]) for row in full_table.selected] == ["200"]
        full_table._selection_handlers[0](SimpleNamespace(selection=[]))
        assert selection_batches[-1] == []
        assert resolution_batches[-1] == []

    def test_overlapping_match_groups_resolve_to_deduplicated_body_ids(
        self, isolated_index_root, monkeypatch
    ):
        from nicegui import Client
        from nicegui.page import page
        import ui.components.neuron_index_viewer as viewer
        from ui.components.neuron_index_viewer import create_neuron_index_viewer_link

        dataset = "overlapping-groups:v1.0"
        folder = dataset.replace(":", "_").replace(".", "_")
        cache_dir = isolated_index_root / "neuron_indexes" / folder
        cache_dir.mkdir(parents=True)
        pl.DataFrame(
            {
                "bodyId": ["1", "2", "3"],
                "type": ["MeVPaMe2", "MeVPaMe2", "Other"],
                "instance": ["MeVPaMe2_L", "MeVPaMe2_R", ""],
                "flywireType": ["aMe19a", "aMe19a", ""],
            }
        ).write_parquet(cache_dir / "neuron_index.parquet")
        monkeypatch.setattr(viewer, "PROJECT_ROOT", isolated_index_root)
        monkeypatch.setattr(viewer.ui, "run_javascript", lambda script: None)

        current_query = []
        resolved_ids = []
        client = Client(page("/neuron-index-viewer-overlapping-selection"))
        with client:
            link = create_neuron_index_viewer_link(
                lambda: dataset,
                query_selection=lambda values: current_query.__setitem__(
                    slice(None), list(values)
                ),
                query_resolution=lambda values: resolved_ids.__setitem__(
                    slice(None), list(values)
                ),
            )
        self._click(link)

        search_input = next(
            element for element in client.elements.values()
            if getattr(element, "_props", {}).get("label")
            == "Search identities & taxonomy"
        )
        search_listener = next(iter(search_input._event_listeners.values()))
        search_input._handle_event({
            "listener_id": search_listener.id,
            "args": "aMe",
        })

        match_table = next(
            table for table in client.elements.values()
            if type(table).__name__ == "Table"
            and table._props["columns"][0]["name"] == "match_column"
        )
        groups = {
            row["match_value"]: row for row in match_table._props["rows"]
        }
        assert {"aMe19a", "MeVPaMe2"}.issubset(groups)
        # MeVPaMe2 is the primary type match and aMe19a is its secondary
        # flywireType match. They form one selection bundle, but only the
        # primary name is sent to the owning query input.
        match_table._selection_handlers[0](SimpleNamespace(selection=[
            groups["MeVPaMe2"],
        ]))

        assert current_query == ["MeVPaMe2"]
        assert {
            row["match_value"] for row in match_table.selected
        } == {"MeVPaMe2"}
        match_table._selection_handlers[0](SimpleNamespace(selection=[]))

        # A header select-all includes the display-only secondary row in
        # QTable's internal selection so the header remains fully checked,
        # while the owning query still receives only the primary name.
        match_table._selection_handlers[0](SimpleNamespace(
            selection=list(match_table._props["rows"])
        ))
        assert {
            row["match_value"] for row in match_table.selected
        } == {"MeVPaMe2", "aMe19a"}
        assert current_query == ["MeVPaMe2"]
        match_table._selection_handlers[0](SimpleNamespace(selection=[]))
        match_table._selection_handlers[0](SimpleNamespace(selection=[
            groups["aMe19a"],
        ]))
        # Secondary rows are accessory display rows, not independently
        # selectable. A synthetic selection event for one is ignored.
        assert current_query == []
        assert match_table.selected == []
        # The earlier primary selection was explicitly cleared before the
        # synthetic secondary event; no secondary checkbox can restore it.
        assert resolved_ids == []

    def test_independent_primary_types_do_not_cross_select_shared_taxonomy_names(
        self, isolated_index_root, monkeypatch
    ):
        """A type remains independent when its spelling is secondary elsewhere."""
        from nicegui import Client
        from nicegui.page import page
        import ui.components.neuron_index_viewer as viewer
        from ui.components.neuron_index_viewer import create_neuron_index_viewer_link

        dataset = "independent-groups:v1.0"
        folder = dataset.replace(":", "_").replace(".", "_")
        cache_dir = isolated_index_root / "neuron_indexes" / folder
        cache_dir.mkdir(parents=True)
        pl.DataFrame(
            {
                "bodyId": ["1", "2", "3", "4"],
                "type": ["aMe17a", "aMe17e", "aMe17a", "aMe17e"],
                "instance": [
                    "aMe17a_L", "aMe17e_L", "aMe17a_R", "aMe17e_R",
                ],
                "flywireType": [
                    "aMe17a1", "aMe17a2", "aMe17a1", "aMe17a2",
                ],
                "hemibrainType": ["", "aMe17a", "", "aMe17a"],
            }
        ).write_parquet(cache_dir / "neuron_index.parquet")
        monkeypatch.setattr(viewer, "PROJECT_ROOT", isolated_index_root)
        monkeypatch.setattr(viewer.ui, "run_javascript", lambda script: None)

        current_query = []
        resolved_ids = []
        client = Client(page("/neuron-index-viewer-independent-selection"))
        with client:
            link = create_neuron_index_viewer_link(
                lambda: dataset,
                query_selection=lambda values: current_query.__setitem__(
                    slice(None), list(values)
                ),
                query_resolution=lambda values: resolved_ids.__setitem__(
                    slice(None), list(values)
                ),
            )
        self._click(link)

        search_input = next(
            element for element in client.elements.values()
            if getattr(element, "_props", {}).get("label")
            == "Search identities & taxonomy"
        )
        search_listener = next(iter(search_input._event_listeners.values()))
        search_input._handle_event({
            "listener_id": search_listener.id,
            "args": "aMe",
        })

        tables = [el for el in client.elements.values() if type(el).__name__ == "Table"]
        match_table = next(
            table for table in tables
            if table._props["columns"][0]["name"] == "match_column"
        )
        full_table = next(
            table for table in tables
            if table._props["columns"][0]["name"] == "bodyId"
        )
        groups = {
            row["match_value"]: row for row in match_table._props["rows"]
        }
        assert {"aMe17a", "aMe17e", "aMe17a1", "aMe17a2"} <= set(groups)

        match_table._selection_handlers[0](SimpleNamespace(selection=[
            groups["aMe17a"],
        ]))
        assert current_query == ["aMe17a"]
        assert resolved_ids == ["1", "3"]
        assert {
            row["match_value"] for row in match_table.selected
        } == {"aMe17a"}
        assert {
            str(row["bodyId"]) for row in full_table.selected
        } == {"1", "3"}
        assert {
            row["match_value"] for row in match_table.selected
        }.isdisjoint({"aMe17e", "aMe17a2"})

        # Clearing the match panel selection must clear the actual selection,
        # not merely remove chips from the mirrored query.
        match_table._selection_handlers[0](SimpleNamespace(selection=[]))
        assert current_query == []
        assert resolved_ids == []
        assert full_table.selected == []

        match_table._selection_handlers[0](SimpleNamespace(selection=[
            groups["aMe17e"],
        ]))
        assert current_query == ["aMe17e"]
        assert resolved_ids == ["2", "4"]
        assert {
            row["match_value"] for row in match_table.selected
        } == {"aMe17e"}
        assert {
            str(row["bodyId"]) for row in full_table.selected
        } == {"2", "4"}

    def test_match_selection_survives_new_search_and_query_chip_is_removable(
        self, isolated_index_root, monkeypatch
    ):
        from nicegui import Client
        from nicegui.page import page
        import ui.components.neuron_index_viewer as viewer
        from ui.components.neuron_index_viewer import create_neuron_index_viewer_link

        dataset, _, _ = _write_index(isolated_index_root)
        monkeypatch.setattr(viewer, "PROJECT_ROOT", isolated_index_root)
        monkeypatch.setattr(viewer.ui, "run_javascript", lambda script: None)
        current_query = []
        resolved_ids = []

        def sync_query(values):
            current_query[:] = list(values)

        def sync_resolution(values):
            resolved_ids[:] = list(values)

        def remove_query(value):
            current_query[:] = [item for item in current_query if item != value]

        client = Client(page("/neuron-index-viewer-persistent-selection"))
        with client:
            link = create_neuron_index_viewer_link(
                lambda: dataset,
                query_values_getter=lambda: current_query,
                query_selection=sync_query,
                query_resolution=sync_resolution,
                query_remove=remove_query,
            )
        self._click(link)

        search_input = next(
            element for element in client.elements.values()
            if getattr(element, "_props", {}).get("label")
            == "Search identities & taxonomy"
        )
        search_listener = next(iter(search_input._event_listeners.values()))

        search_input._handle_event({
            "listener_id": search_listener.id,
            "args": "ame",
        })
        tables = [el for el in client.elements.values() if type(el).__name__ == "Table"]
        match_table = next(
            table for table in tables
            if table._props["columns"][0]["name"] == "match_column"
        )
        a_me_row = next(
            row for row in match_table._props["rows"]
            if row["match_value"] == "aMe10"
        )
        match_table._selection_handlers[0](SimpleNamespace(selection=[a_me_row]))
        assert current_query == ["aMe10"]
        assert resolved_ids == ["200"]

        # Changing the search replaces the displayed match rows but must not
        # clear the persistent selection or its exact body-ID resolution.
        search_input._handle_event({
            "listener_id": search_listener.id,
            "args": "APL",
        })
        assert current_query == ["aMe10"]
        assert resolved_ids == ["200"]

        chip_remove = next(
            element for element in client.elements.values()
            if "drocat-neuron-query-chip-remove" in getattr(element, "_classes", set())
        )
        click_listener = next(
            listener for listener in chip_remove._event_listeners.values()
            if listener.type == "click"
        )
        click_listener.handler(SimpleNamespace())
        assert current_query == []
        assert resolved_ids == []

    def test_result_page_navigation_preserves_body_selection(
        self, isolated_index_root, monkeypatch
    ):
        from nicegui import Client
        from nicegui.page import page
        import ui.components.neuron_index_viewer as viewer
        from ui.components.neuron_index_viewer import create_neuron_index_viewer_link

        dataset = _write_paged_index(isolated_index_root)
        monkeypatch.setattr(viewer, "PROJECT_ROOT", isolated_index_root)
        current_query = []

        def sync_query(values):
            current_query[:] = list(values)

        client = Client(page("/neuron-index-viewer-page-selection"))
        with client:
            link = create_neuron_index_viewer_link(
                lambda: dataset,
                query_selection=sync_query,
            )
        self._click(link)

        tables = [el for el in client.elements.values() if type(el).__name__ == "Table"]
        full_table = next(
            table for table in tables
            if table._props["columns"][0]["name"] == "bodyId"
        )
        first_row = full_table._props["rows"][0]
        full_table._selection_handlers[0](SimpleNamespace(selection=[first_row]))
        assert current_query == ["1000"]

        next_button = next(
            element for element in client.elements.values()
            if getattr(element, "text", "") == "Next page"
        )
        self._click(next_button)
        assert current_query == ["1000"]
        assert not any(
            str(row.get("bodyId")) == "1000"
            for row in getattr(full_table, "selected", [])
        )

        previous_button = next(
            element for element in client.elements.values()
            if getattr(element, "text", "") == "Previous page"
        )
        self._click(previous_button)
        assert current_query == ["1000"]
        assert any(
            str(row.get("bodyId")) == "1000"
            for row in getattr(full_table, "selected", [])
        )

    def test_full_table_select_all_selects_rows_on_every_page(
        self, isolated_index_root, monkeypatch
    ):
        from nicegui import Client
        from nicegui.page import page
        import ui.components.neuron_index_viewer as viewer
        from ui.components.neuron_index_viewer import create_neuron_index_viewer_link

        dataset = _write_paged_index(isolated_index_root, row_count=60)
        monkeypatch.setattr(viewer, "PROJECT_ROOT", isolated_index_root)
        selection_batches = []
        resolution_batches = []

        def sync_query(values):
            selection_batches.append(list(values))

        def sync_resolution(values):
            resolution_batches.append(list(values))

        client = Client(page("/neuron-index-viewer-select-all"))
        with client:
            link = create_neuron_index_viewer_link(
                lambda: dataset,
                query_selection=sync_query,
                query_resolution=sync_resolution,
            )
        self._click(link)

        tables = [el for el in client.elements.values() if type(el).__name__ == "Table"]
        full_table = next(
            table for table in tables
            if table._props["columns"][0]["name"] == "bodyId"
        )
        assert "header" in full_table.slots
        assert "full-table-select-all" in full_table.slots["header"].template
        select_all_listener = next(
            listener for listener in full_table._event_listeners.values()
            if listener.type == "fullTableSelectAll"
        )

        expected = {str(1000 + i) for i in range(60)}
        # The first click selects every matching row on every page (60 total),
        # not only the 50 rows visible on the current page.
        select_all_listener.handler(SimpleNamespace())
        assert set(selection_batches[-1]) == expected
        assert set(resolution_batches[-1]) == expected

        # A second click toggles the whole cross-page selection back off.
        select_all_listener.handler(SimpleNamespace())
        assert selection_batches[-1] == []
        assert resolution_batches[-1] == []

    def test_link_explains_how_to_build_a_missing_cache(
        self, isolated_index_root, monkeypatch
    ):
        from nicegui import Client
        from nicegui.page import page
        import ui.components.neuron_index_viewer as viewer
        from ui.components.neuron_index_viewer import create_neuron_index_viewer_link

        dataset = "missing:v2.0"
        monkeypatch.setattr(viewer, "PROJECT_ROOT", isolated_index_root)

        client = Client(page("/neuron-index-viewer-missing"))
        with client:
            link = create_neuron_index_viewer_link(lambda: dataset)
        self._click(link)

        texts = [el.text for el in client.elements.values() if getattr(el, "text", "")]
        joined = "\n".join(texts)
        assert "not cached locally" in joined
        assert "Settings → Dataset Cache" in joined
        assert any(
            getattr(el, "_props", {}).get("content")
            == "python src/build_connection_cache.py missing:v2.0"
            for el in client.elements.values()
        )
        assert "The viewer does not open or stream the original dataset file." in joined

    def test_coarse_match_entry_expands_to_selectable_subtypes(
        self, isolated_index_root, monkeypatch
    ):
        from nicegui import Client
        from nicegui.page import page
        import ui.components.neuron_index_viewer as viewer
        from ui.components.neuron_index_viewer import create_neuron_index_viewer_link

        dataset = _write_taxonomy_index(isolated_index_root)
        monkeypatch.setattr(viewer, "PROJECT_ROOT", isolated_index_root)
        current_query = ["existing"]
        selection_batches = []
        resolution_batches = []

        def sync_query(values):
            selection_batches.append(list(values))
            current_query[:] = ["existing", *values]

        def sync_resolution(values):
            resolution_batches.append(list(values))

        client = Client(page("/neuron-index-viewer-subtypes"))
        with client:
            link = create_neuron_index_viewer_link(
                lambda: dataset,
                query_values_getter=lambda: current_query,
                query_selection=sync_query,
                query_resolution=sync_resolution,
            )
        # The trigger presents itself as a loupe.
        assert link._props.get("icon") == "search"
        self._click(link)

        tables = [el for el in client.elements.values() if type(el).__name__ == "Table"]
        match_table = next(
            table for table in tables
            if table._props["columns"][0]["name"] == "match_column"
        )
        body_template = match_table.slots["body"].template
        assert "match-expand-toggle" in body_template
        assert "match-subtype-toggle" in body_template
        assert "drocat-neuron-match-subtype-list" in body_template

        search_input = next(
            element for element in client.elements.values()
            if getattr(element, "_props", {}).get("label")
            == "Search identities & taxonomy"
        )
        search_listener = next(iter(search_input._event_listeners.values()))
        search_input._handle_event({
            "listener_id": search_listener.id,
            "args": "circadian",
        })

        row = match_table._props["rows"][0]
        assert row["match_value"] == "circadian"
        assert row["match_column_key"] == "cell_class"
        assert row["__can_expand"] is True
        assert "__expanded" not in row

        expand_listener = next(
            listener for listener in match_table._event_listeners.values()
            if listener.type == "matchExpandToggle"
        )
        match_table._handle_event({
            "listener_id": expand_listener.id,
            "args": "circadian",
        })

        row = match_table._props["rows"][0]
        assert row["__expanded"] is True
        display = row["__subtypes"]
        assert display["total_types"] == 3
        assert display["truncated"] is False
        assert [s["match_value"] for s in display["subtypes"]] == [
            "DN1a", "l-LNv", "s-LNv",
        ]
        assert [s["body_count"] for s in display["subtypes"]] == [1, 1, 1]
        assert all(not s["selected"] for s in display["subtypes"])

        subtype_listener = next(
            listener for listener in match_table._event_listeners.values()
            if listener.type == "matchSubtypeToggle"
        )
        match_table._handle_event({
            "listener_id": subtype_listener.id,
            "args": {
                "group": "circadian",
                "value": "DN1a",
                "selected": True,
            },
        })

        assert selection_batches[-1] == ["DN1a"]
        assert resolution_batches[-1] == ["100"]
        assert current_query == ["existing", "DN1a"]
        row = match_table._props["rows"][0]
        selected_flags = {
            s["match_value"]: s["selected"] for s in row["__subtypes"]["subtypes"]
        }
        assert selected_flags == {"DN1a": True, "l-LNv": False, "s-LNv": False}
        full_table = next(
            table for table in tables
            if table._props["columns"][0]["name"] == "bodyId"
        )
        assert {str(r["bodyId"]) for r in full_table.selected} == {"100"}

        # Deselecting the subtype removes exactly that value again.
        match_table._handle_event({
            "listener_id": subtype_listener.id,
            "args": {
                "group": "circadian",
                "value": "DN1a",
                "selected": False,
            },
        })
        assert selection_batches[-1] == []
        assert resolution_batches[-1] == []
        assert current_query == ["existing"]
        row = match_table._props["rows"][0]
        assert not any(s["selected"] for s in row["__subtypes"]["subtypes"])

        # Collapsing keeps the cached panel so a re-expand needs no recompute.
        match_table._handle_event({
            "listener_id": expand_listener.id,
            "args": "circadian",
        })
        row = match_table._props["rows"][0]
        assert row["__expanded"] is False
        assert row["__subtypes"]["total_types"] == 3
        match_table._handle_event({
            "listener_id": expand_listener.id,
            "args": "circadian",
        })
        row = match_table._props["rows"][0]
        assert row["__expanded"] is True

        # A type match is already the leaf identity and gets no expander.
        search_input._handle_event({
            "listener_id": search_listener.id,
            "args": "DN1a",
        })
        row = match_table._props["rows"][0]
        assert row["match_value"] == "DN1a"
        assert row["match_column_key"] == "type"
        assert row["__can_expand"] is False


class TestBridgeBodyIdPooling:
    """pool_bridge_body_ids: one-side pooling per standardized linker,
    including the honest N-to-M mismatch between the two sides."""

    def _index(self, tmp_path, dataset, rows):
        import ui.neuron_index as neuron_index

        folder = dataset.replace(":", "_").replace(".", "_")
        cache_dir = tmp_path / "neuron_indexes" / folder
        cache_dir.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(rows).write_parquet(cache_dir / "neuron_index.parquet")
        neuron_index.clear_neuron_index_cache()
        return neuron_index.load_cached_neuron_index(dataset, enrich=False)

    def test_two_to_four_mismatch_and_multipart_cells(
            self, isolated_index_root, tmp_path):
        from ui.neuron_index import pool_bridge_body_ids

        source = self._index(tmp_path, "src:v1.0", {
            "bodyId": ["1", "2", "9"],
            "type": ["A", "A", "B"],
            # bodyId 9 is type B and must NOT leak into A's pool
            "flywireType": ["W", "W", "W"],
        })
        target = self._index(tmp_path, "tgt:v1.0", {
            "bodyId": ["11", "12", "13", "14", "99"],
            "type": ["B", "B", "B", "B", "C"],
            # bodyId 99 is type C: the comma cell matches W but must be
            # excluded by the endpoint type filter
            "additional_type(s)": ["W", "W, V", " W ", "V, W", "W"],
        })
        linkers = [
            {"column": "flywireType", "value": "W",
             "home": "src:v1.0", "kind": "linker"},
            {"column": "additional_type(s)", "value": "W",
             "home": "tgt:v1.0", "kind": "linker"},
        ]
        pool = pool_bridge_body_ids(
            "src:v1.0", "tgt:v1.0", linkers, "A", "B",
            indexes={"src:v1.0": source, "tgt:v1.0": target})

        # honest mismatch: 2 source bodyIds pool to 4 target bodyIds
        assert pool["granularity"] == "2 to 4"
        assert pool["source_body_ids"] == ["1", "2"]
        assert pool["target_body_ids"] == ["11", "12", "13", "14"]
        # comma/space multipart cells match per entry, never per substring
        assert [l["body_ids"] for l in pool["per_linker"]] == [
            ["1", "2"], ["11", "12", "13", "14"]]
        # coverage states the partial target coverage
        assert pool["coverage"] == "covered 4 of 4"

    def test_no_linker_yields_full_endpoint_pools(self, isolated_index_root,
                                                  tmp_path):
        from ui.neuron_index import pool_bridge_body_ids

        source = self._index(tmp_path, "s2:v1.0", {
            "bodyId": ["1"], "type": ["A"], "flywireType": ["A"]})
        target = self._index(tmp_path, "t2:v1.0", {
            "bodyId": ["2"], "type": ["A"]})
        pool = pool_bridge_body_ids(
            "s2:v1.0", "t2:v1.0", [], "A", "A",
            indexes={"s2:v1.0": source, "t2:v1.0": target})
        # a bare same-name chain constrains neither side: both pools are
        # the full endpoint types (the name equality IS the evidence)
        assert pool["granularity"] == "1 to 1"
        assert pool["source_body_ids"] == ["1"]
        assert pool["target_body_ids"] == ["2"]
        assert pool["per_linker"] == []


def test_mapped_csv_extras_dedupe_and_via_note():
    """mapped_csv_extras: bridge-<column> cells per standardized linker,
    deduped values, via-note on indirect (hub) linkers, empty for
    unmapped types (§9.3)."""
    from ui.neuron_index import mapped_csv_extras

    provenance = {
        'CL125': [
            {'foreign_type': 'APDN3', 'matched': "type · 'APDN3'",
             'origins': [
                 {'column': 'flywireType', 'value': 'LMTe01',
                  'home': 'male-cns:v1.0', 'indirect': False,
                  'text': "flywireType 'LMTe01'"},
                 {'column': 'additional_type(s)', 'value': 'LMTe01',
                  'home': 'flywire_FAFB_v783', 'indirect': False,
                  'text': "additional_type(s) 'LMTe01'"},
             ]},
            # same pair via another route: the flywireType value repeats
            {'foreign_type': 'APDN3', 'matched': "label · 'x'",
             'origins': [
                 {'column': 'flywireType', 'value': 'LMTe01',
                  'indirect': False, 'text': "flywireType 'LMTe01'"},
             ]},
        ],
        'SLP250': [
            {'foreign_type': 'LTe71', 'matched': "type · 'LTe71'",
             'origins': [
                 {'column': 'additional_type(s)', 'value': 'APDN3',
                  'indirect': True,
                  'text': "additional_type(s) 'APDN3' (via FAFB)"},
             ]},
        ],
    }
    rows = [{'type': 'CL125', 'bodyId': 1},
            {'type': 'SLP250', 'bodyId': 2},
            {'type': 'UNMAPPED', 'bodyId': 3}]
    fieldnames, extras = mapped_csv_extras(
        rows, provenance, 'flywire_FAFB_v783')
    assert fieldnames == [
        'foreign_dataset', 'foreign_type(s)', 'matched column(s)',
        'bridge-additional_type(s)', 'bridge-flywireType']
    assert extras[0]['foreign_dataset'] == 'flywire_FAFB_v783'
    assert extras[0]['foreign_type(s)'] == 'APDN3'
    assert extras[0]['bridge-flywireType'] == 'LMTe01'  # deduped
    assert extras[0]['bridge-additional_type(s)'] == 'LMTe01'
    assert extras[1]['bridge-additional_type(s)'] == 'APDN3 (via FAFB)'
    assert extras[1]['bridge-flywireType'] == ''
    assert extras[2]['foreign_type(s)'] == ''
    assert extras[2]['bridge-flywireType'] == ''
