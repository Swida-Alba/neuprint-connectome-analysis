"""Performance-path guarantees for the per-chip neuron query resolvers.

The analysis pipeline resolves one query per chip and repeats the whole chip
list for the target side. These tests pin the memoization contracts that make
that loop cheap without changing resolution semantics: the per-frame bodyId
index, the memoized sidecar validation, the O(1) numeric fast path, the
per-token resolution cache in ``statvis`` and the one-shot fallback reason.
"""

import pandas as pd
import polars as pl
import pytest


def _build_cache(tmp_path, frame: pl.DataFrame, dataset="perf-cache:v1.0"):
    from src.neuron_index_builder import build_search_cache_frame, search_cache_path

    folder = dataset.replace(":", "_").replace(".", "_")
    cache_dir = tmp_path / "neuron_indexes" / folder
    cache_dir.mkdir(parents=True)
    index_path = cache_dir / "neuron_index.parquet"
    frame.write_parquet(index_path)
    build_search_cache_frame(frame).write_parquet(search_cache_path(index_path))
    from src.neuron_search import get_cached_neuron_search

    cache = get_cached_neuron_search(dataset, index_root=tmp_path / "neuron_indexes")
    assert cache is not None
    return cache


def _sample_frame() -> pl.DataFrame:
    return pl.DataFrame({
        "bodyId": ["100", "200", "300", "400"],
        "type": ["MTe01a", "Other", "MeVPaMe2", "aMe17a"],
        "instance": ["MTe01a_L", "Other_L", "MeVPaMe2_R", "aMe17a_L"],
    })


def test_numeric_fast_path_matches_dataframe_scan(tmp_path):
    """Hits, misses and float forms resolve identically to the scan path."""
    from src.neuron_search import (
        resolve_cached_or_dataframe_query,
        resolve_dataframe_query,
    )

    frame = _sample_frame().to_pandas()
    cache = _build_cache(tmp_path, _sample_frame())

    for query in ("100", 100, "100.0", "999", "100.5", "0", "200.00"):
        ids, info = resolve_cached_or_dataframe_query(cache, frame, query)
        expected, expected_info = resolve_dataframe_query(frame, query)
        assert [str(value) for value in ids] == [
            str(value) for value in expected
        ], query
        assert info["matched_column"] == expected_info["matched_column"], query
        assert info["match_count"] == expected_info["match_count"], query

    ids, info = resolve_cached_or_dataframe_query(cache, frame, "300")
    assert ids == ["300"]  # raw frame dtype preserved (str bodyIds here)
    assert info["matched_column"] == "bodyId"

    # With no cache at all the fast path still answers, but reports the
    # dataframe surface honestly.
    ids, info = resolve_cached_or_dataframe_query(None, frame, "300")
    assert ids == ["300"]
    assert info["cache"] is False
    assert info["matched_column"] == "bodyId"


def test_numeric_fast_path_respects_scopes_and_wildcards(tmp_path):
    """Only bare numeric queries under auto/bodyid take the O(1) path."""
    from src.neuron_search import (
        resolve_cached_or_dataframe_query,
        resolve_dataframe_query,
    )

    frame = _sample_frame().to_pandas()
    cache = _build_cache(tmp_path, _sample_frame())

    # Numeric wildcard stays regex semantics (no hit in the sample).
    for query, kwargs in (
        ("10.*", {}),
        ("100", {"search_columns": "type"}),
        ("100", {"search_columns": "instance"}),
    ):
        ids, info = resolve_cached_or_dataframe_query(cache, frame, query, **kwargs)
        expected, expected_info = resolve_dataframe_query(
            frame, query, **kwargs
        )
        assert [str(value) for value in ids] == [
            str(value) for value in expected
        ], query
        assert info["matched_column"] == expected_info["matched_column"], query


def test_body_id_keys_memoized_per_cache_instance(tmp_path):
    """Repeated access returns the memo; a fresh instance rebuilds its own."""
    from src.neuron_search import CachedNeuronSearch

    frame = _sample_frame()
    cache = _build_cache(tmp_path, frame)

    assert cache.body_id_keys is cache.body_id_keys
    assert cache.body_id_keys == {"100", "200", "300", "400"}

    # A separately-built instance must not inherit the memo.
    other = CachedNeuronSearch(
        dataset="other",
        index_path=cache.index_path,
        search_path=None,
        search_frame=cache.search_frame,
        body_ids=tuple(cache.body_ids),
    )
    assert other.body_id_keys is not cache.body_id_keys
    assert other.body_id_keys == cache.body_id_keys


def test_frame_index_reused_for_same_frame_rebuilt_for_new_frame():
    """The weak per-frame index survives repeated queries, not frame swaps."""
    from src.neuron_search import frame_body_id_index

    frame = _sample_frame().to_pandas()
    first = frame_body_id_index(frame)
    assert first is not None
    assert frame_body_id_index(frame) is first
    assert first.keys == {"100", "200", "300", "400"}

    reloaded = _sample_frame().to_pandas()  # new object, same content
    second = frame_body_id_index(reloaded)
    assert second is not first

    missing = pd.DataFrame({"type": ["a"], "instance": ["a_L"]})
    assert frame_body_id_index(missing) is None


def test_token_cache_dedupes_repeat_resolutions(tmp_path, monkeypatch):
    """The second pass over the same chips must not re-invoke the resolver."""
    import src.neuron_search as neuron_search
    import src.statvis as statvis

    frame = _sample_frame().to_pandas()
    cache = _build_cache(tmp_path, _sample_frame())

    calls = {"n": 0}
    real_resolve = neuron_search.resolve_cached_or_dataframe_query

    def counting_resolve(*args, **kwargs):
        calls["n"] += 1
        return real_resolve(*args, **kwargs)

    # _resolve_single_neuron imports the symbol from the module at call time.
    monkeypatch.setattr(
        neuron_search, "resolve_cached_or_dataframe_query", counting_resolve
    )
    statvis._TOKEN_RESOLUTION_CACHE.clear()

    for _ in range(2):
        ids, info = statvis._resolve_single_neuron(
            "200", frame, None, dataset="perf-cache:v1.0",
            cached_search=cache, verbose=False,
        )
        assert [str(value) for value in ids] == ["200"]
        assert info["match_count"] == 1

    assert calls["n"] == 1  # second pass hit the token cache


def test_token_cache_invalidated_by_reloaded_frame(tmp_path):
    """A new frame object can never inherit stale per-token results."""
    import src.statvis as statvis

    frame = _sample_frame().to_pandas()
    cache = _build_cache(tmp_path, _sample_frame())
    statvis._TOKEN_RESOLUTION_CACHE.clear()

    first_ids, _ = statvis._resolve_single_neuron(
        "MTe01a", frame, None, dataset="perf-cache:v1.0",
        cached_search=cache, verbose=False,
    )
    reloaded = _sample_frame().to_pandas()
    second_ids, _ = statvis._resolve_single_neuron(
        "MTe01a", reloaded, None, dataset="perf-cache:v1.0",
        cached_search=cache, verbose=False,
    )
    assert [str(value) for value in first_ids] == ["100"]
    assert [str(value) for value in second_ids] == ["100"]


def test_fallback_reason_travels_in_search_info(tmp_path, capsys):
    """A rejected sidecar explains itself via search_info, not stdout.

    The shared resolver must stay silent — the run layer turns the reason
    into one console line and a user_warning_notes.txt entry.
    """
    from src.neuron_search import resolve_cached_or_dataframe_query

    # Sidecar lacks flywireType, so the frame's extra column forces the
    # dataframe fallback; the reason must ride along on every result.
    narrow = _sample_frame()
    frame = narrow.with_columns(
        pl.Series("flywireType", ["MTe07", "MTe12", "MTe27", "MTe99"])
    ).to_pandas()
    cache = _build_cache(tmp_path, narrow, dataset="narrow-cache:v1.0")

    statvis_module = pytest.importorskip("src.statvis")
    statvis_module._TOKEN_RESOLUTION_CACHE.clear()
    reasons = set()
    for _ in range(3):
        ids, info = resolve_cached_or_dataframe_query(cache, frame, "MTe07")
        assert [str(value) for value in ids] == ["100"]
        assert info["cache"] is False
        reasons.add(info.get("cache_fallback_reason"))

    assert reasons == {"sidecar lacks searchable column(s) ['flywireType']"}
    # Validated queries carry no fallback reason.
    _, clean_info = resolve_cached_or_dataframe_query(cache, _sample_frame().to_pandas(), "MTe07")
    assert "cache_fallback_reason" not in clean_info
    # The library itself printed nothing.
    assert "sidecar" not in capsys.readouterr().out


def _patch_get_neurons_fixtures(monkeypatch, statvis, frame, cache, dataset):
    """Serve a synthetic dataset table + sidecar to getNeurons offline."""
    roi = pd.DataFrame({"bodyId": frame["bodyId"]})
    monkeypatch.setattr(
        statvis, "_ensure_local_dataset_files",
        lambda *args, **kwargs: (dataset.replace(":", "_").replace(".", "_"), "unused"),
    )
    monkeypatch.setattr(
        statvis, "_get_cached_neuron_df",
        lambda *args, **kwargs: (frame.copy(), roi.copy()),
    )
    monkeypatch.setattr(statvis, "_get_cached_neuron_search", lambda _dataset: cache)


def test_chip_prints_capped_above_ten(tmp_path, monkeypatch, capsys):
    """Past 10 chips one totals summary replaces the per-chip prints."""
    import src.statvis as statvis

    n = 15
    frame = pd.DataFrame({
        "bodyId": [1000 + i for i in range(n)],
        "type": [f"T{i:02d}" for i in range(n)],
        "instance": [f"T{i:02d}_L" for i in range(n)],
    })
    cache = _build_cache(tmp_path, pl.DataFrame(frame), dataset="cap-test:v1.0")
    _patch_get_neurons_fixtures(monkeypatch, statvis, frame, cache, "cap-test:v1.0")
    statvis._TOKEN_RESOLUTION_CACHE.clear()

    chips = [str(1000 + i) for i in range(n)]
    neurons, _, _, _ = statvis.getNeurons(
        chips, dataset="cap-test:v1.0", verbose=True
    )
    assert len(neurons) == n

    out = capsys.readouterr().out
    assert "Found" not in out  # every per-chip line suppressed
    assert f"Resolved {n} queries → {n} neurons" in out
    assert "(cached search: 15; dataframe search: 0)" in out


def test_chip_prints_kept_below_cap(tmp_path, monkeypatch, capsys):
    """Small queries keep the familiar per-chip lines and print no summary."""
    import src.statvis as statvis

    frame = pd.DataFrame({
        "bodyId": ["100", "200", "300"],
        "type": ["A", "B", "C"],
        "instance": ["A_L", "B_L", "C_L"],
    })
    cache = _build_cache(tmp_path, pl.DataFrame(frame), dataset="small-test:v1.0")
    _patch_get_neurons_fixtures(monkeypatch, statvis, frame, cache, "small-test:v1.0")
    statvis._TOKEN_RESOLUTION_CACHE.clear()

    neurons, _, _, _ = statvis.getNeurons(
        ["100", "200", "300"], dataset="small-test:v1.0", verbose=True
    )
    assert len(neurons) == 3

    out = capsys.readouterr().out
    assert out.count("Found 1 neurons for") == 3
    assert "Resolved" not in out


def test_capped_summary_reports_missed_chips(tmp_path, monkeypatch, capsys):
    """Chips that matched nothing stay visible in the capped summary."""
    import src.statvis as statvis

    n = 12
    frame = pd.DataFrame({
        "bodyId": [2000 + i for i in range(n)],
        "type": [f"U{i:02d}" for i in range(n)],
        "instance": [f"U{i:02d}_L" for i in range(n)],
    })
    cache = _build_cache(tmp_path, pl.DataFrame(frame), dataset="miss-test:v1.0")
    _patch_get_neurons_fixtures(monkeypatch, statvis, frame, cache, "miss-test:v1.0")
    statvis._TOKEN_RESOLUTION_CACHE.clear()

    chips = [str(2000 + i) for i in range(n)] + ["999999999", "888888888"]
    neurons, _, _, _ = statvis.getNeurons(
        chips, dataset="miss-test:v1.0", verbose=True
    )
    assert len(neurons) == n

    out = capsys.readouterr().out
    assert f"Resolved {len(chips)} queries → {n} neurons" in out
    assert "2 of 14 queries matched no neurons: 999999999, 888888888" in out


def test_run_layer_records_fallback_note_once():
    """coana turns the resolver's fallback reason into the run channels."""
    coana = pytest.importorskip("coana")

    fc = object.__new__(coana.FindNeuronConnection)
    fc._warn_notes = []
    fc.verbose_mode = "full"
    printed = []
    fc._vprint = lambda message="", level="full", end="\n", flush=False: printed.append(message)

    reason = "sidecar lacks searchable column(s) ['flywireType']"
    infos = [{"cache_fallback_reason": reason}, {"cache_fallback_reason": reason}]
    fc._record_cache_fallback_notes("source", infos)
    fc._record_cache_fallback_notes("source", infos)  # same role+reason again

    assert len(fc._warn_notes) == 1
    assert "[search fallback]" in fc._warn_notes[0]
    assert reason in fc._warn_notes[0]
    assert "source" in fc._warn_notes[0]
    assert len(printed) == 1  # one console line, not one per query

    # The target side with the same reason gets its own note.
    fc._record_cache_fallback_notes("target", infos)
    assert len(fc._warn_notes) == 2
    assert any("target" in note for note in fc._warn_notes)

    # Infos without a reason are ignored.
    fc._record_cache_fallback_notes("source", [{"matched_column": "type"}])
    assert len(fc._warn_notes) == 2


def test_sidecar_coverage_verdict_is_deterministic(tmp_path):
    """Column-order ties must not flip the cache acceptance between calls."""
    from src.neuron_search import _cache_covers_frame

    # Rank ties (every *Type field ranks together) in a specific source order.
    frame = pl.DataFrame({
        "bodyId": ["100", "200"],
        "type": ["A", ""],
        "instance": ["A_L", "B_L"],
        "supertype": ["", "sup"],
        "locationType": ["loc", ""],
        "receptorType": ["rec", "rec2"],
    })
    cache = _build_cache(tmp_path, frame, dataset="tie-order:v1.0")

    # Build order is the stable order recorded in the sidecar.
    order = cache.priority_column_order
    assert order.index("supertype") < order.index("locationType")
    assert order.index("locationType") < order.index("receptorType")
    assert cache.priority_column_order is cache.priority_column_order

    verdicts = {_cache_covers_frame(cache, frame.to_pandas()) for _ in range(10)}
    assert verdicts == {True}

    # A column reorder that changes tie priority is a genuine mismatch and
    # must be rejected deterministically (never flip-flop between calls).
    shuffled = frame.select(
        "bodyId", "type", "instance",
        "receptorType", "locationType", "supertype",
    ).to_pandas()
    shuffled_verdicts = {
        _cache_covers_frame(cache, shuffled) for _ in range(10)
    }
    assert len(shuffled_verdicts) == 1
