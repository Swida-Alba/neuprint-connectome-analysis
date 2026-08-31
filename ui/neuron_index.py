"""Cached neuron-index loading and querying for the UI viewer.

The auto-suggestion backend and the available-neurons viewer intentionally
share the same local index boundary: a viewer is available only when
``neuron_indexes/<dataset>/neuron_index.parquet`` exists.  That app-owned
"system files" directory persists across ``cache/`` cleanups - bundled
datasets ship committed seeds there and the pull pipeline builds every other
dataset's index into the same place.  The viewer never serves the raw dataset
file to the browser.  The index is built from the materialized projection of
the prepared local neuron table; an older/partial index can still be enriched
from that table to fill blank ``type``/``instance`` values.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from datetime import date, datetime, time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import PROJECT_ROOT
from .dataset_service import dataset_to_folder
from .search_logic import (
    SearchStage,
    is_numeric_search,
    normalize_search_operator,
    normalize_search_text,
    ordered_search_columns,
    polars_body_id_guard,
    polars_display_expression,
    polars_match_column_expression,
    polars_match_expression,
    search_plan,
)

try:
    from src.neuron_index_builder import (
        build_search_cache_frame,
        is_search_cache_compatible,
        metadata_columns,
        ordered_projection_columns,
        read_metadata_projection,
        search_cache_path,
    )
except ImportError:
    from neuron_index_builder import (
        build_search_cache_frame,
        is_search_cache_compatible,
        metadata_columns,
        ordered_projection_columns,
        read_metadata_projection,
        search_cache_path,
    )


@dataclass(frozen=True)
class CachedNeuronIndex:
    """A cached neuron index plus the path used to load it."""

    dataset: str
    path: Path
    frame: Any  # polars.DataFrame; kept Any so importing the UI does not require Polars eagerly
    columns: Tuple[str, ...]
    enriched: bool = False
    search_frame: Any = None  # compact presorted Polars search sidecar


@dataclass(frozen=True)
class NeuronIndexPage:
    """One server-side page of a filtered/sorted neuron index."""

    rows: List[Dict[str, Any]]
    total: int
    page: int
    pages: int
    page_size: int
    sort_by: str
    descending: bool
    match_groups: List[Dict[str, Any]] = field(default_factory=list)
    match_group_members: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    match_group_body_ids: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    # Related match names share at least one exact result row.  The primary
    # values are the canonical query tokens for a selected related group.
    match_group_related: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    match_group_primary: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    focus_page: Optional[int] = None
    # Every matching ``__neuron_key`` -> user-facing bodyId across all pages.
    # Populated only when ``include_all_keys`` is requested, so normal paged
    # browsing never pays to materialise the full result set.
    all_keys: Dict[str, str] = field(default_factory=dict)


# The largest local indexes are only a few megabytes as Parquet but are read
# by several UI clients.  Keep one process-local copy and invalidate it when
# either the cache index or its optional metadata table changes.
_INDEX_CACHE: Dict[Tuple, CachedNeuronIndex] = {}

def clear_neuron_index_cache() -> None:
    """Clear the process-local viewer cache (primarily useful for tests)."""
    _INDEX_CACHE.clear()


def neuron_index_path(dataset: str, cache_dir: Optional[Path] = None) -> Path:
    """Return the app-owned neuron-index path for *dataset*.

    The index is a persistent "system file" outside ``cache/``: shipped seeds
    and pull-built indexes share this location, so clearing the cache never
    removes the metadata behind auto-suggestions and the viewer.
    """
    root = Path(cache_dir) if cache_dir is not None else PROJECT_ROOT / "neuron_indexes"
    return root / dataset_to_folder(str(dataset).strip()) / "neuron_index.parquet"


def neuron_index_state_path(dataset: str, cache_dir: Optional[Path] = None) -> Path:
    """Return the optional progress sidecar for a cached index."""
    root = Path(cache_dir) if cache_dir is not None else PROJECT_ROOT / "cache"
    return root / dataset_to_folder(str(dataset).strip()) / "neuron_index_state.parquet"


def _metadata_candidates(dataset: str, datasets_dir: Optional[Path]) -> List[Path]:
    """Find generated metadata files, preferring the pulled CSV source."""
    if datasets_dir is None:
        datasets_dir = PROJECT_ROOT / "datasets"
    folder = Path(datasets_dir) / dataset_to_folder(str(dataset).strip())
    if not folder.is_dir():
        return []

    safe = folder.name
    exact = [
        folder / f"{safe}_allneurons_neuron_df.csv",
        folder / f"{safe}_neuron_df.csv",
        folder / f"{safe}_allneurons_neuron_df.parquet",
        folder / f"{safe}_neuron_df.parquet",
    ]
    discovered = sorted(
        [
            p
            for pattern in ("*_allneurons_neuron_df.csv", "*_neuron_df.csv",
                            "*_allneurons_neuron_df.parquet", "*_neuron_df.parquet")
            for p in folder.glob(pattern)
        ],
        key=lambda p: p.name,
    )
    result: List[Path] = []
    for path in exact + discovered:
        if path.is_file() and path not in result:
            result.append(path)
    return result


def _metadata_signature(dataset: str, datasets_dir: Optional[Path]) -> Optional[Tuple[str, int]]:
    candidates = _metadata_candidates(dataset, datasets_dir)
    if not candidates:
        return None
    path = candidates[0]
    try:
        return str(path), path.stat().st_mtime_ns
    except OSError:
        return None


def _is_blank(expression):
    """Return a Polars expression identifying null/empty display values."""
    import polars as pl

    return (
        expression.is_null()
        | (
            expression.cast(pl.Utf8, strict=False)
            .fill_null("")
            .str.strip_chars()
            == ""
        )
    )


def _read_metadata_table(path: Path):
    """Read the materialized projection from a generated local neuron table."""
    import polars as pl

    try:
        frame = read_metadata_projection(path)
    except Exception:
        return None
    if "bodyId" not in frame.columns:
        return None
    frame = frame.with_columns(
        pl.col("bodyId").cast(pl.Utf8, strict=False).fill_null("").alias("bodyId")
    )
    rename = {
        column: f"__metadata_{column}"
        for column in frame.columns
        if column != "bodyId"
    }
    return frame.rename(rename).unique(subset=["bodyId"], keep="first")


def _enrich_identifiers(frame, dataset: str, datasets_dir: Optional[Path]):
    """Fill blank type/instance values from local generated metadata."""
    import polars as pl

    if "bodyId" not in frame.columns:
        frame = frame.with_columns(pl.lit("").alias("bodyId"))

    display_columns = ["bodyId", "type", "instance"]
    expressions = []
    for column in display_columns:
        if column in frame.columns:
            expressions.append(
                pl.col(column).cast(pl.Utf8, strict=False).fill_null("").alias(column)
            )
        else:
            expressions.append(pl.lit("").alias(column))
    frame = frame.with_columns(expressions)

    metadata_signature = _metadata_signature(dataset, datasets_dir)
    if metadata_signature is None:
        return frame, False

    metadata_file = Path(metadata_signature[0])
    # A freshly generated rich index already contains the same projection.
    # Avoid opening the hundreds-of-MiB CSV again just to fill values it has.
    try:
        import polars as pl
        source_columns = set(metadata_columns(metadata_file))
        if (
            source_columns.issubset(set(frame.columns))
            and {"downstream_complete", "last_fetched", "connection_count"}
            .issubset(set(frame.columns))
        ):
            return frame, False
    except Exception:
        pass

    metadata = _read_metadata_table(metadata_file)
    if metadata is None:
        return frame, False

    frame = frame.join(metadata, on="bodyId", how="left")
    for metadata_column in metadata.columns:
        if not metadata_column.startswith("__metadata_"):
            continue
        column = metadata_column.removeprefix("__metadata_")
        if column not in frame.columns:
            frame = frame.with_columns(
                pl.col(metadata_column).alias(column)
            ).drop(metadata_column)
            continue
        metadata_column = f"__metadata_{column}"
        frame = frame.with_columns(
            pl.when(_is_blank(pl.col(column)))
            .then(pl.col(metadata_column).cast(pl.Utf8, strict=False).fill_null(""))
            .otherwise(pl.col(column))
            .alias(column)
        ).drop(metadata_column)
    return frame, True


def load_cached_neuron_index(
    dataset: str,
    *,
    cache_dir: Optional[Path] = None,
    datasets_dir: Optional[Path] = None,
    enrich: bool = True,
) -> CachedNeuronIndex:
    """Load a cached neuron index for display.

    Raises:
        FileNotFoundError: when the cached ``neuron_index.parquet`` is absent.
        ValueError: when the file cannot be read as a usable table.
    """
    import polars as pl

    dataset = str(dataset or "").strip()
    if not dataset:
        raise FileNotFoundError("No dataset was selected")

    path = neuron_index_path(dataset, cache_dir)
    if not path.is_file():
        raise FileNotFoundError(str(path))

    metadata_signature = _metadata_signature(dataset, datasets_dir) if enrich else None
    state_path = neuron_index_state_path(dataset, cache_dir)
    search_path = search_cache_path(path)
    try:
        state_signature = (
            (str(state_path), state_path.stat().st_mtime_ns)
            if state_path.is_file()
            else (str(state_path), None)
        )
        signature = (
            str(path),
            path.stat().st_mtime_ns,
            metadata_signature,
            state_signature,
            (
                str(search_path),
                search_path.stat().st_mtime_ns,
            ) if search_path.is_file() else (str(search_path), None),
            bool(enrich),
        )
    except OSError as exc:
        raise FileNotFoundError(str(path)) from exc
    if signature in _INDEX_CACHE:
        return _INDEX_CACHE[signature]

    try:
        frame = pl.read_parquet(path)
    except Exception as exc:
        raise ValueError(f"Could not read cached neuron index: {exc}") from exc

    # Overlay progress written by an in-flight connection pull.  The sidecar
    # contains only cache flags/counts and never changes the metadata columns.
    if state_path.is_file():
        try:
            state = pl.read_parquet(state_path)
            if "bodyId" in state.columns and "bodyId" in frame.columns:
                frame = frame.with_columns(
                    pl.col("bodyId").cast(pl.Utf8, strict=False).fill_null("").alias("bodyId")
                )
                state = state.with_columns(
                    pl.col("bodyId").cast(pl.Utf8, strict=False).fill_null("").alias("bodyId")
                )
                frame = frame.join(state, on="bodyId", how="left", suffix="__state")
                for column in ("downstream_complete", "last_fetched", "connection_count"):
                    state_column = f"{column}__state"
                    if state_column not in frame.columns:
                        continue
                    if column not in frame.columns:
                        frame = frame.with_columns(
                            pl.col(state_column).alias(column)
                        )
                    else:
                        frame = frame.with_columns(
                            pl.when(pl.col(state_column).is_not_null())
                            .then(pl.col(state_column))
                            .otherwise(pl.col(column))
                            .alias(column)
                        )
                    frame = frame.drop(state_column)
        except Exception:
            # A partially written sidecar must not make the viewer unusable;
            # the canonical index remains readable on its own.
            pass

    frame, enriched = _enrich_identifiers(frame, dataset, datasets_dir) if enrich else (frame, False)
    if not frame.columns:
        raise ValueError("The cached neuron index has no columns")

    # Legacy indexes may have been enriched from a source table after load;
    # apply the same order as newly generated caches before exposing columns.
    frame = frame.select(ordered_projection_columns(frame.columns))

    # Keep the schema stable and JSON-friendly for the UI.  Index values are
    # displayed as strings for bodyId/type/instance so large IDs are never
    # rounded by browser JavaScript.
    frame = frame.with_columns(
        pl.col("bodyId").cast(pl.Utf8, strict=False).fill_null("").alias("bodyId")
        if "bodyId" in frame.columns
        else pl.lit("").alias("bodyId")
    )
    for column in ("type", "instance"):
        if column in frame.columns:
            frame = frame.with_columns(
                pl.col(column).cast(pl.Utf8, strict=False).fill_null("").alias(column)
            )

    # Prefer the materialized compact sidecar. Older caches and test fixtures
    # transparently receive the same structure in memory, so the matcher has
    # one implementation and can be upgraded without rewriting the full
    # metadata parquet from the UI process.
    search_frame = None
    try:
        if search_path.is_file():
            search_frame = pl.read_parquet(search_path)
            if not is_search_cache_compatible(search_frame, frame.columns):
                search_frame = None
    except Exception:
        search_frame = None
    if search_frame is None:
        search_frame = build_search_cache_frame(frame)

    result = CachedNeuronIndex(
        dataset=dataset,
        path=path,
        frame=frame,
        columns=tuple(frame.columns),
        enriched=enriched,
        search_frame=search_frame,
    )
    _INDEX_CACHE[signature] = result
    # Remove old versions of this path so a rebuilt index does not accumulate
    # unbounded DataFrames in a long-running UI process.
    for old_key in list(_INDEX_CACHE):
        if old_key != signature and old_key[0] == str(path):
            _INDEX_CACHE.pop(old_key, None)
    return result


def _match_expression(frame, columns: List[str], text: str, mode: str):
    """Return the Polars expression for one shared search stage."""
    return polars_match_expression(frame, columns, text, mode)


def _match_column_expression(frame, column: str, text: str, mode: str):
    """Return the boolean expression for one column in a search stage."""
    return polars_match_column_expression(frame, column, text, mode)


def _match_hit_lists(
    frame,
    columns: List[str],
    text: str,
    mode: str,
    *,
    suppress_instance_if_type: bool = False,
):
    """Return ordered lists of matched columns and their displayed values.

    The primary match remains in ``match_column_key``/``match_value`` for
    compatibility with the pinned hint. These lists preserve every field that
    matched the same row, allowing the table to highlight a prefix hit and a
    secondary substring hit simultaneously. ``suppress_instance_if_type`` is
    retained for call-site compatibility, but suppression belongs only to the
    deduplicated match-panel lists; row-level hit lists must keep instances.
    """
    import polars as pl

    available = [column for column in columns if column in frame.columns]
    empty = pl.lit([], dtype=pl.List(pl.Utf8))
    if not available:
        return empty, empty

    needle = normalize_search_text(text)
    body_guard = _body_id_guard(frame, available, needle, mode)
    key_items = []
    value_items = []
    for column in available:
        matched = _match_column_expression(frame, column, needle, mode) & body_guard
        display_value = (
            _display_expression(column, frame)
            .cast(pl.Utf8, strict=False)
            .fill_null("")
        )
        key_items.append(
            pl.when(matched).then(pl.lit(column)).otherwise(pl.lit(""))
        )
        value_items.append(
            pl.when(matched).then(display_value).otherwise(pl.lit(""))
        )

    def compact(items):
        return pl.concat_list(items).list.eval(
            pl.element().filter(pl.element() != "")
        )

    return compact(key_items), compact(value_items)


def _secondary_hit_lists(
    frame,
    columns: List[str],
    text: str,
    prefix_mode: str = "prefix",
    substring_mode: str = "substring",
):
    """Return useful secondary hits after applying field suppression rules.

    A type name is the canonical identity for a row.  If the same query also
    matches that row's instance, the instance is redundant and is omitted
    from the secondary display.  This keeps values such as ``MeVPaMe2_L``
    and ``MeVPaMe2_R`` from competing with their matched type ``MeVPaMe2``.
    """
    import polars as pl

    available = [column for column in columns if column in frame.columns]
    empty = pl.lit([], dtype=pl.List(pl.Utf8))
    if not available:
        return empty, empty

    needle = normalize_search_text(text)
    body_guard = _body_id_guard(frame, available, needle, substring_mode)
    type_matches = pl.lit(False)
    if "type" in frame.columns:
        type_matches = (
            (
                _match_column_expression(frame, "type", needle, prefix_mode)
                | _match_column_expression(frame, "type", needle, substring_mode)
            )
            & _body_id_guard(frame, ["type"], needle, substring_mode)
        )

    key_items = []
    value_items = []
    for column in available:
        prefix_match = _match_column_expression(
            frame, column, needle, prefix_mode
        ) & body_guard
        substring_match = _match_column_expression(
            frame, column, needle, substring_mode
        ) & body_guard
        matched = substring_match & ~prefix_match
        if column == "instance":
            matched = matched & ~type_matches
        display_value = (
            _display_expression(column, frame)
            .cast(pl.Utf8, strict=False)
            .fill_null("")
        )
        key_items.append(
            pl.when(matched).then(pl.lit(column)).otherwise(pl.lit(""))
        )
        value_items.append(
            pl.when(matched).then(display_value).otherwise(pl.lit(""))
        )

    def compact(items):
        return pl.concat_list(items).list.eval(
            pl.element().filter(pl.element() != "")
        )

    return compact(key_items), compact(value_items)


def _contains_expression(frame, columns: List[str], text: str):
    """Case-insensitive substring expression for an explicit column filter."""
    return _match_expression(frame, columns, text, "substring")


def _display_expression(column: str, frame):
    """Return a safe string expression for a retained metadata column.

    Body IDs are always matched and displayed as text.  Numeric-looking body
    IDs are normalized so a parquet reader that inferred ``123.0`` cannot
    produce a suggestion/query value that does not verify against the cached
    identifier ``123``.
    """
    return polars_display_expression(column)


def _body_id_guard(frame, columns: List[str], text: str, mode: str):
    """Prevent numeric searches from matching a non-bodyId display field."""
    return polars_body_id_guard(frame, columns, text)


def _ordered_match_columns(columns: List[str]) -> List[str]:
    """Return only the viewer's identity/taxonomy search scope.

    The full cache remains available for display and explicit column filters,
    but global viewer search must not scan operational, measurement, notes, or
    other arbitrary metadata. Keeping this list small also avoids repeatedly
    decoding large non-search fields for every keystroke.
    """
    return ordered_search_columns(columns)


def _normalize_filter_operator(value: str) -> str:
    """Normalize the viewer's explicit column-filter operator."""
    return normalize_search_operator(value)


def _highlight_text_html(value: Any, needle: str, mode: str) -> Optional[str]:
    """Return escaped cell text with only the matched spans marked.

    The table keeps its normal white/blue cell background and uses the cell
    outline for hit ownership.  This helper deliberately returns escaped HTML
    so the viewer can render a small ``<mark>`` around the matching characters
    without exposing metadata values through ``v-html``.
    """
    text = "" if value is None else str(value)
    needle = normalize_search_text(needle)
    if not text or not needle:
        return None

    spans: List[Tuple[int, int]] = []
    if mode == "prefix":
        if text.startswith(needle):
            spans = [(0, len(needle))]
    elif mode == "suffix":
        if text.endswith(needle):
            spans = [(len(text) - len(needle), len(text))]
    elif mode == "exact":
        if text == needle:
            spans = [(0, len(text))]
    elif mode == "regex":
        try:
            spans = [match.span() for match in re.finditer(needle, text)]
        except re.error:
            spans = []
    elif mode == "global":
        # A global row can have been admitted by either stage. Preserve the
        # strict-prefix visual cue when that exact prefix exists; substring
        # rows use the same case-insensitive rule as the search backend.
        if text.startswith(needle):
            spans = [(0, len(needle))]
        else:
            mode = "substring"

    if mode in {"substring", "contains"} and not spans:
        folded_text = text.casefold()
        folded_needle = needle.casefold()
        start = 0
        while folded_needle:
            position = folded_text.find(folded_needle, start)
            if position < 0:
                break
            spans.append((position, position + len(needle)))
            start = position + max(1, len(needle))

    if not spans:
        return None

    # Normalize overlapping/adjacent spans before escaping the source text.
    merged: List[Tuple[int, int]] = []
    for start, end in sorted(spans):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    if not merged:
        return None

    parts: List[str] = []
    cursor = 0
    for start, end in merged:
        parts.append(html.escape(text[cursor:start], quote=False))
        parts.append(
            '<mark class="drocat-neuron-match-text">'
            f"{html.escape(text[start:end], quote=False)}"
            "</mark>"
        )
        cursor = end
    parts.append(html.escape(text[cursor:], quote=False))
    return "".join(parts)


def _highlighted_cells(
    row: Dict[str, Any],
    needle: str,
    mode: str,
    columns: List[str],
) -> Dict[str, str]:
    """Build safe HTML only for the searchable cells hit by this row."""
    hit_columns: List[str] = []
    for column in (
        *(row.get("match_column_keys") or []),
        *(row.get("secondary_match_column_keys") or []),
    ):
        column = str(column or "").strip()
        if column and column in columns and column not in hit_columns:
            hit_columns.append(column)
    if not hit_columns:
        column = str(row.get("match_column_key") or "").strip()
        if column in columns:
            hit_columns.append(column)

    highlighted: Dict[str, str] = {}
    for column in hit_columns:
        marked = _highlight_text_html(row.get(column), needle, mode)
        if marked is not None:
            highlighted[column] = marked
    return highlighted


def _presorted_search_matches(
    source,
    search_cache,
    columns: List[str],
    text: str,
    *,
    include_substrings: bool = True,
):
    """Return rows in canonical match order using the presorted sidecar.

    The sidecar is sorted by column priority and value.  For each column we
    therefore append its strict-prefix slice, then its substring-only slice,
    and remove row keys already claimed by a higher-priority column.  No
    result sort or full metadata-column scan is needed for this path.

    ``None`` means the requested mode is not representable by this cache
    (regex, suffix, and exact searches continue through the general matcher).
    """
    import polars as pl

    if search_cache is None or not columns:
        return None
    needle = normalize_search_text(text)
    if not needle:
        return None

    # The sidecar stores source row ordinals as a list.  The two hit frames
    # returned by this function are the exploded form used by the downstream
    # joins.  Keep that schema even when a valid query has no matching value;
    # returning ``search_cache.head(0)`` here leaves only ``__neuron_rows``
    # and causes the later aggregation to look for a missing
    # ``__neuron_row`` column.
    # Do not derive this schema with ``head(0).explode(...)``.  Polars keeps
    # an empty list column un-exploded in some releases, which leaves
    # ``__neuron_rows`` in the result and makes the later group/aggregation
    # fail with ``ColumnNotFoundError: __neuron_row`` on a valid no-hit query.
    empty_hits = pl.DataFrame(
        {
            "__neuron_row": pl.Series([], dtype=pl.UInt32),
            "search_column": pl.Series([], dtype=pl.Utf8),
            "search_priority": pl.Series([], dtype=pl.UInt16),
            "search_value": pl.Series([], dtype=pl.Utf8),
        }
    )
    empty_candidates = source.filter(pl.lit(False)).with_columns(
        pl.lit(0).alias("__match_priority"),
        pl.lit(0).alias("__match_kind_priority"),
        pl.lit("").alias("__candidate_column"),
        pl.lit("").alias("__candidate_value"),
    )

    available_columns = set(search_cache["search_column"].unique().to_list())
    ordered = [column for column in columns if column in available_columns]
    if not ordered:
        return empty_candidates, empty_hits, empty_hits
    numeric = is_numeric_search(needle)

    def _explode_combined_cells(column_frame):
        """Split comma-joined cells into one searchable name per part.

        FlyWire datasets pack several alternative type names into one
        ``additional_type(s)`` cell (``'vDeltaB, vDeltaC, ...'``).  The
        joined string is not a real neuron name, so match and report each
        part individually — otherwise match groups (and any query value a
        user selects from them) would carry the raw joined cell.
        """
        if not column_frame.height:
            return column_frame
        if not column_frame["search_value"].str.contains(",").any():
            return column_frame
        return (
            column_frame
            .with_columns(pl.col("search_value").str.split(","))
            .explode("search_value")
            .with_columns(pl.col("search_value").str.strip_chars())
            .filter(pl.col("search_value") != "")
            .with_columns(
                pl.col("search_value").str.to_lowercase().alias("search_value_folded")
            )
        )

    claimed = None
    chunks = []
    hit_parts = []
    for priority, column in enumerate(ordered):
        if numeric and column != "bodyId":
            continue
        column_frame = _explode_combined_cells(
            search_cache.filter(pl.col("search_column") == column)
        )
        if numeric:
            column_frame = column_frame.filter(
                pl.col("search_value").str.contains(r"^\d+$")
            )
        prefix_values = column_frame.filter(
            pl.col("search_value").str.starts_with(needle)
        )
        raw_prefix = (
            prefix_values
            .explode("__neuron_rows")
            .rename({"__neuron_rows": "__neuron_row"})
            if prefix_values.height else column_frame.head(0)
        )
        prefix_keys = raw_prefix.select("__neuron_row").unique(
            subset=["__neuron_row"], maintain_order=True
        ) if raw_prefix.height else None
        if raw_prefix.height:
            hit_parts.append(raw_prefix.select(
                "__neuron_row", "search_column", "search_priority", "search_value"
            ))
        prefix = raw_prefix
        if claimed is not None and prefix.height:
            prefix = prefix.join(claimed, on="__neuron_row", how="anti")
        prefix = prefix.with_columns(
            pl.lit(priority).alias("__candidate_priority"),
            pl.lit(0).alias("__candidate_kind"),
        )
        raw_substring_values = column_frame.filter(
            pl.col("search_value_folded").str.contains(
                needle.casefold(), literal=True
            )
        ) if include_substrings else column_frame.head(0)
        if prefix_values.height and raw_substring_values.height:
            raw_substring_values = raw_substring_values.join(
                prefix_values.select("search_value"),
                on="search_value",
                how="anti",
            )
        raw_substring = (
            raw_substring_values
            .explode("__neuron_rows")
            .rename({"__neuron_rows": "__neuron_row"})
            if raw_substring_values.height else column_frame.head(0)
        )
        if prefix_keys is not None and raw_substring.height:
            raw_substring = raw_substring.join(
                prefix_keys, on="__neuron_row", how="anti"
            )
        if raw_substring.height:
            hit_parts.append(raw_substring.select(
                "__neuron_row", "search_column", "search_priority", "search_value"
            ))
        substring = raw_substring
        excluded = claimed
        if prefix_keys is not None:
            excluded = prefix_keys if excluded is None else pl.concat(
                [excluded, prefix_keys], how="vertical"
            ).unique(subset=["__neuron_row"], maintain_order=True)
        if excluded is not None and substring.height:
            substring = substring.join(excluded, on="__neuron_row", how="anti")
        substring = substring.with_columns(
            pl.lit(priority).alias("__candidate_priority"),
            pl.lit(1).alias("__candidate_kind"),
        )

        chunk_parts = [part for part in (prefix, substring) if part.height]
        if not chunk_parts:
            continue
        chunk = pl.concat(chunk_parts, how="vertical_relaxed")
        # One searchable value exists per row/column, but keep this guard for
        # legacy sidecars built before that invariant was enforced.
        chunk = chunk.unique(subset=["__neuron_row"], maintain_order=True)
        chunks.append(chunk)
        keys = chunk.select("__neuron_row")
        claimed = keys if claimed is None else pl.concat(
            [claimed, keys], how="vertical"
        ).unique(subset=["__neuron_row"], maintain_order=True)

    if not chunks:
        return empty_candidates, empty_hits, empty_hits
    candidate = pl.concat(chunks, how="vertical_relaxed").select(
        "__neuron_row",
        pl.col("__candidate_priority").alias("__match_priority"),
        pl.col("__candidate_kind").alias("__match_kind_priority"),
        pl.col("search_column").alias("__candidate_column"),
        pl.col("search_value").alias("__candidate_value"),
    )
    raw_hits = pl.concat(hit_parts, how="vertical_relaxed")
    all_hits = raw_hits
    type_keys = all_hits.filter(
        pl.col("search_column") == "type"
    ).select("__neuron_row").unique(
        subset=["__neuron_row"], maintain_order=True
    )
    if type_keys.height:
        all_hits = all_hits.filter(
            ~(
                (pl.col("search_column") == "instance")
                & pl.col("__neuron_row").is_in(
                    type_keys["__neuron_row"].implode()
                )
            )
        )
    return source.join(
        candidate,
        on="__neuron_row",
        how="inner",
        maintain_order="right",
    ), all_hits, raw_hits


def _presorted_match_groups(
    filtered,
    hit_entries,
    search_columns: List[str],
    json_value,
    membership_entries=None,
):
    """Build the match panel from compact searchable-cell hits.

    This is the broad-query companion to :func:`_presorted_search_matches`.
    It keeps full metadata out of the group pass, but retains exact row and
    body membership for selection and focus.  ``None`` requests the legacy
    path (for example, when a caller supplies a hand-built frame without the
    sidecar hit columns).
    """
    import polars as pl

    required = {
        "__neuron_row", "search_column", "search_priority", "search_value",
    }
    if hit_entries is None or not required.issubset(set(hit_entries.columns)):
        return None
    if filtered.is_empty() or hit_entries.is_empty():
        return ([], {}, {}, {}, {})

    row_columns = [
        "__neuron_row", "__neuron_key", "bodyId", "match_column",
        "match_column_key", "match_value", "__match_priority",
        "__match_kind_priority", "__candidate_column", "__candidate_value",
    ]
    row_columns = [column for column in row_columns if column in filtered.columns]
    rows = filtered.select(row_columns)
    hits = hit_entries.join(
        rows.select("__neuron_row"), on="__neuron_row", how="inner"
    )
    if hits.is_empty():
        return ([], {}, {}, {}, {})
    hits = hits.join(rows, on="__neuron_row", how="inner", maintain_order="left")
    hits = hits.with_columns(
        (
            (pl.col("search_column") == pl.col("__candidate_column"))
            & (pl.col("search_value") == pl.col("__candidate_value"))
        ).cast(pl.Int8).alias("__is_primary_hit"),
        pl.col("search_value").str.to_lowercase().alias("__match_norm"),
    )
    # The first hit for a group is chosen after primary status, canonical
    # column priority, mode, and source row. The group itself is then sorted
    # by bodyId/type/instance/taxonomy priority and prefix-vs-substring.
    hits = hits.sort(
        [
            "search_value", "__is_primary_hit", "search_priority",
            "__match_kind_priority", "__neuron_row",
        ],
        descending=[False, True, False, False, False],
    )
    summary = hits.group_by("search_value", maintain_order=True).agg(
        pl.col("search_column").first().alias("__group_column"),
        pl.col("search_priority").first().alias("__group_priority"),
        pl.col("__match_kind_priority").first().alias("__group_kind"),
        pl.col("__is_primary_hit").max().alias("__group_is_primary"),
        pl.col("__neuron_key").unique(maintain_order=True).alias("__members"),
        pl.col("bodyId").unique(maintain_order=True).alias("__body_ids"),
    ).sort(
        ["__group_priority", "__group_kind", "search_value"],
        descending=[False, False, False],
    )

    # Case-insensitive exact-name membership mirrors the normal viewer
    # selection rule, while the group key preserves the visible spelling.
    # Membership is column-specific: if a value is a type, selecting it must
    # select the rows whose *type* is that value, not unrelated rows where
    # the same spelling happens to occur in hemibrainType/flywireType.
    # Use raw hits so a secondary group still retains every row sharing that
    # exact secondary value.
    membership_hits = membership_entries if membership_entries is not None else hit_entries
    membership_hits = membership_hits.join(
        rows.select("__neuron_row"), on="__neuron_row", how="inner"
    ).join(rows, on="__neuron_row", how="inner", maintain_order="left")
    membership_hits = membership_hits.with_columns(
        pl.col("search_value").str.to_lowercase().alias("__match_norm")
    )
    membership_map = {}
    if not membership_hits.is_empty():
        grouped_members = membership_hits.group_by(
            ["search_column", "__match_norm"], maintain_order=True
        ).agg(
            pl.col("__neuron_key").unique(maintain_order=True).alias("__members"),
            pl.col("bodyId").unique(maintain_order=True).alias("__body_ids"),
        )
        membership_map = {
            (
                str(raw.get("search_column") or ""),
                str(raw.get("__match_norm") or ""),
            ): (
                [str(value) for value in (raw.get("__members") or [])],
                [str(value or "") for value in (raw.get("__body_ids") or [])],
            )
            for raw in grouped_members.to_dicts()
        }

    match_groups: List[Dict[str, Any]] = []
    group_members: Dict[str, Tuple[str, ...]] = {}
    group_body_ids: Dict[str, Tuple[str, ...]] = {}
    group_related_sets: Dict[str, set[str]] = {}
    group_order: Dict[str, Tuple[int, int, str]] = {}
    group_rows: Dict[str, List[str]] = {}
    for raw in summary.to_dicts():
        value = str(json_value(raw.get("search_value")) or "").strip()
        if not value:
            continue
        column = str(raw.get("__group_column") or "")
        try:
            priority = int(raw.get("__group_priority") or len(search_columns))
        except (TypeError, ValueError):
            priority = len(search_columns)
        try:
            kind = int(raw.get("__group_kind") or 1)
        except (TypeError, ValueError):
            kind = 1
        members, body_ids = membership_map.get(
            (column, value.casefold()),
            ([str(item) for item in (raw.get("__members") or [])],
             [str(item or "") for item in (raw.get("__body_ids") or [])]),
        )
        members = tuple(member for member in members if member)
        body_ids = tuple(
            item[:-2] if item.endswith(".0") and item[:-2].isdigit() else item
            for item in body_ids if item
        )
        role = "primary" if int(raw.get("__group_is_primary") or 0) else "secondary"
        match_groups.append({
            "__match_group_key": value,
            "match_column": column,
            "match_column_key": column,
            "match_value": value,
            "body_count": len(body_ids),
            "match_role": role,
            "first_body_id": body_ids[0] if body_ids else "",
        })
        group_members[value] = members
        group_body_ids[value] = body_ids
        group_order[value] = (priority, kind, value)
        group_related_sets[value] = {value}
        group_rows[value] = list(members)

    group_rank = {
        str(group["__match_group_key"]): position
        for position, group in enumerate(match_groups)
    }
    # Link each row's canonical primary value only to its own secondary
    # values.  A value can be primary on one row and secondary on another;
    # that must not merge two independent primary entries.  In particular,
    # male-cns rows can have type=aMe17e and hemibrainType=aMe17a while other
    # rows have type=aMe17a: selecting either type must stay independent.
    primary_values = {
        str(json_value(raw.get("search_value")) or "").strip()
        for raw in summary.to_dicts()
        if int(raw.get("__group_is_primary") or 0)
    }
    row_hits = hits.select(
        "__neuron_row", "__candidate_value", "search_value"
    ).unique(maintain_order=True)
    owner_values: Dict[str, set[str]] = {}
    for raw in row_hits.to_dicts():
        primary = str(json_value(raw.get("__candidate_value")) or "").strip()
        secondary = str(json_value(raw.get("search_value")) or "").strip()
        if (
            not primary
            or not secondary
            or primary == secondary
            or primary not in group_rank
            or secondary not in group_rank
            or secondary in primary_values
        ):
            continue
        group_related_sets[primary].add(secondary)
        group_related_sets[secondary].add(primary)
        owner_values.setdefault(secondary, set()).add(primary)

    match_group_related = {
        key: tuple(sorted(values, key=lambda value: group_rank.get(value, len(group_rank))))
        for key, values in group_related_sets.items()
    }
    # Keep the relationship local to one primary/secondary bundle.  There is
    # deliberately no transitive component collapse: two primary names that
    # happen to share a secondary metadata spelling remain separate query
    # entries.
    primary_by_value = {}
    for key in group_rank:
        if key in primary_values:
            primary_by_value[key] = (key,)
            continue
        owners = sorted(
            owner_values.get(key, ()),
            key=lambda value: group_rank.get(value, len(group_rank)),
        )
        primary_by_value[key] = tuple(owners or (key,))

    return (
        _order_match_groups_with_secondaries(match_groups, primary_by_value),
        group_members,
        group_body_ids,
        match_group_related,
        primary_by_value,
    )


def _order_match_groups_with_secondaries(
    match_groups: List[Dict[str, Any]],
    match_group_primary: Dict[str, Tuple[str, ...]],
) -> List[Dict[str, Any]]:
    """Place each secondary match directly after its owning primary.

    Match groups are initially ordered for search relevance.  That order is
    useful for choosing the first page, but it can leave a taxonomy alias far
    away from the name that owns it.  The panel is easier to read when a
    secondary is rendered as an accessory row immediately below its primary.
    The relationship map remains unchanged, so selection continues to lock
    the whole primary/secondary bundle together.

    A secondary can technically be shared by more than one primary.  Display
    it after the first owner in the existing priority order and leave the
    relationship map responsible for the synchronized selection semantics.
    Any orphaned secondary is retained in its original position at the end
    rather than being dropped.
    """
    if not match_groups:
        return []

    by_key = {
        str(group.get("__match_group_key") or group.get("match_value") or ""): group
        for group in match_groups
    }
    secondary_by_owner: Dict[str, List[Dict[str, Any]]] = {}
    attached_secondary_keys: set[str] = set()
    for group in match_groups:
        if str(group.get("match_role") or "") != "secondary":
            continue
        key = str(group.get("__match_group_key") or group.get("match_value") or "")
        owners = match_group_primary.get(key, ())
        owner = next(
            (
                str(candidate or "")
                for candidate in owners
                if str(candidate or "") in by_key
                and str(by_key[str(candidate)].get("match_role") or "") == "primary"
            ),
            "",
        )
        if owner:
            secondary_by_owner.setdefault(owner, []).append(group)
            attached_secondary_keys.add(key)

    ordered: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for group in match_groups:
        key = str(group.get("__match_group_key") or group.get("match_value") or "")
        if key in seen:
            continue
        if (
            str(group.get("match_role") or "") == "secondary"
            and key in attached_secondary_keys
        ):
            continue
        ordered.append(group)
        seen.add(key)
        for secondary in secondary_by_owner.get(key, ()):
            secondary_key = str(
                secondary.get("__match_group_key")
                or secondary.get("match_value")
                or ""
            )
            if secondary_key not in seen:
                ordered.append(secondary)
                seen.add(secondary_key)

    # Preserve every group even when malformed/legacy cache data has a
    # missing relationship entry or a duplicate display key.
    for group in match_groups:
        key = str(group.get("__match_group_key") or group.get("match_value") or "")
        if key not in seen:
            ordered.append(group)
            seen.add(key)
    return ordered


def _match_metadata(
    frame,
    columns: List[str],
    text: str,
    stage: SearchStage | None = None,
):
    """Build match-priority and display metadata expressions for a query.

    The first matching column in the ordered scope wins.  For bodyId matches,
    the displayed hint mirrors auto-suggestion: show the corresponding
    instance when one exists, otherwise show ``bodyId``.  The match key and
    value are kept separately so the viewer can highlight the actual source
    cell while showing the compact hint in its pinned info columns.
    """
    import polars as pl

    ordered = [
        column
        for column in (stage.columns if stage is not None else _ordered_match_columns(columns))
        if column in frame.columns
    ]
    if not ordered:
        empty = pl.lit("")
        return pl.lit(0), empty, empty, empty

    needle = normalize_search_text(text)
    priority = pl.lit(len(ordered))
    match_column = pl.lit("")
    match_column_key = pl.lit("")
    match_value = pl.lit("")
    for rank, column in reversed(list(enumerate(ordered))):
        display_value = _display_expression(column, frame).cast(
            pl.Utf8, strict=False
        ).fill_null("")
        mode = stage.mode if stage is not None else "substring"
        if mode == "prefix":
            matched = display_value.str.starts_with(needle)
        elif mode == "suffix":
            matched = display_value.str.ends_with(needle)
        elif mode == "exact":
            matched = display_value == needle
        elif mode == "regex":
            import re

            try:
                re.compile(needle)
            except re.error:
                matched = pl.lit(False)
            else:
                matched = display_value.str.contains(needle, literal=False)
        else:
            matched = display_value.str.to_lowercase().str.contains(
                needle.casefold(), literal=True
            )
        if column == "bodyId" and is_numeric_search(needle):
            matched = matched & display_value.str.contains(r"^\d+$")
        if column == "bodyId" and "instance" in frame.columns:
            instance = (
                pl.col("instance")
                .cast(pl.Utf8, strict=False)
                .fill_null("")
                .str.strip_chars()
            )
            hint = pl.when(instance != "").then(instance).otherwise(pl.lit("bodyId"))
        else:
            hint = pl.lit(column)
        priority = pl.when(matched).then(pl.lit(rank)).otherwise(priority)
        match_column = pl.when(matched).then(hint).otherwise(match_column)
        match_column_key = pl.when(matched).then(pl.lit(column)).otherwise(match_column_key)
        match_value = pl.when(matched).then(display_value).otherwise(match_value)
    return priority, match_column, match_column_key, match_value


def _global_match_metadata(frame, columns: List[str], text: str):
    """Build canonical match metadata for the full viewer search.

    The viewer returns the union of prefix and substring matches, so the
    stage that admitted a row is not necessarily the row's best match.  For
    example, ``MeVPaMe2`` can contain ``aMe`` in ``type`` while ``aMe19a`` is
    a strict prefix in ``flywireType``.  The canonical column order must win
    first; prefix-vs-substring is only the tie-breaker inside that column.

    Returns expressions for the primary match, its match mode, all useful
    hits, and the lower-priority secondary hits.  Instance hits are omitted
    from the deduplicated match panel whenever the row has a type hit, because
    the type already identifies the neuron more precisely.  They remain in
    the row-level hit list so the corresponding instance cell can still be
    outlined and its matching characters highlighted.
    """
    import polars as pl

    ordered = [column for column in columns if column in frame.columns]
    empty = pl.lit([], dtype=pl.List(pl.Utf8))
    if not ordered:
        return (
            pl.lit(0), pl.lit(1), pl.lit(""), pl.lit(""), pl.lit(""),
            empty, empty, empty, empty,
        )

    needle = normalize_search_text(text)
    numeric = is_numeric_search(needle)
    body_guard = (
        _body_id_guard(frame, ["bodyId"], needle, "substring")
        if numeric and "bodyId" in frame.columns
        else pl.lit(True)
    )
    column_matches = {}
    type_matches = pl.lit(False)
    for column in ordered:
        allowed = (not numeric) or column == "bodyId"
        guard = body_guard if allowed else pl.lit(False)
        prefix = (
            _match_column_expression(frame, column, needle, "prefix") & guard
        )
        substring = (
            _match_column_expression(frame, column, needle, "substring") & guard
        )
        column_matches[column] = (prefix, substring, prefix | substring)
        if column == "type":
            type_matches = prefix | substring

    # Reverse assignment makes the first column in canonical order win.  A
    # prefix and substring match in the same column are resolved by the mode
    # expression, with strict prefix ranked first.
    priority = pl.lit(len(ordered))
    kind_priority = pl.lit(1)
    match_column = pl.lit("")
    match_column_key = pl.lit("")
    match_value = pl.lit("")
    for rank, column in reversed(list(enumerate(ordered))):
        prefix, _substring, matched = column_matches[column]
        display_value = (
            _display_expression(column, frame)
            .cast(pl.Utf8, strict=False)
            .fill_null("")
        )
        if column == "bodyId" and "instance" in frame.columns:
            instance = (
                _display_expression("instance", frame)
                .cast(pl.Utf8, strict=False)
                .fill_null("")
                .str.strip_chars()
            )
            hint = pl.when(instance != "").then(instance).otherwise(pl.lit("bodyId"))
        else:
            hint = pl.lit(column)
        priority = pl.when(matched).then(pl.lit(rank)).otherwise(priority)
        kind_priority = pl.when(matched).then(
            pl.when(prefix).then(pl.lit(0)).otherwise(pl.lit(1))
        ).otherwise(kind_priority)
        match_column = pl.when(matched).then(hint).otherwise(match_column)
        match_column_key = pl.when(matched).then(
            pl.lit(column)
        ).otherwise(match_column_key)
        match_value = pl.when(matched).then(display_value).otherwise(match_value)

    hit_column_items = []
    hit_value_items = []
    secondary_column_items = []
    secondary_value_items = []
    for rank, column in enumerate(ordered):
        _prefix, _substring, matched = column_matches[column]
        all_matched = matched
        if column == "instance":
            panel_matched = matched & ~type_matches
        else:
            panel_matched = matched
        display_value = (
            _display_expression(column, frame)
            .cast(pl.Utf8, strict=False)
            .fill_null("")
        )
        hit_column_items.append(
            pl.when(all_matched).then(pl.lit(column)).otherwise(pl.lit(""))
        )
        hit_value_items.append(
            pl.when(all_matched).then(display_value).otherwise(pl.lit(""))
        )
        # The primary priority is row-dependent.  Keep only lower-priority
        # fields in the secondary lists so the match panel can label them
        # without exposing an instance already covered by type.
        lower_priority = panel_matched & (pl.lit(rank) > priority)
        secondary_column_items.append(
            pl.when(lower_priority).then(pl.lit(column)).otherwise(pl.lit(""))
        )
        secondary_value_items.append(
            pl.when(lower_priority).then(display_value).otherwise(pl.lit(""))
        )

    def compact(items):
        if not items:
            return empty
        return pl.concat_list(items).list.eval(
            pl.element().filter(pl.element() != "")
        ).list.unique(maintain_order=True)

    return (
        priority,
        kind_priority,
        match_column,
        match_column_key,
        match_value,
        compact(hit_column_items),
        compact(hit_value_items),
        compact(secondary_column_items),
        compact(secondary_value_items),
    )


def _collect_all_keys(source, json_value) -> Dict[str, str]:
    """Map every matching ``__neuron_key`` to its user-facing bodyId.

    Used by the select-all action, which needs the complete result set across
    all pages without sending full row data to the browser.  The bodyId is
    normalised exactly like the viewer's per-row helper so that selecting all
    rows and selecting a single row resolve to the same query-safe value.
    """
    keys: Dict[str, str] = {}
    for key, raw_body_id in source.select(["__neuron_key", "bodyId"]).iter_rows():
        body_id = str(json_value(raw_body_id) or "").strip()
        if not body_id:
            keys[str(key or "")] = ""
            continue
        integer, dot, fraction = body_id.partition(".")
        if dot and integer.isdigit() and fraction and set(fraction) == {"0"}:
            body_id = integer
        keys[str(key or "")] = body_id
    return keys


def query_neuron_index(
    index: CachedNeuronIndex,
    *,
    search: str = "",
    search_column: Optional[str] = None,
    search_operator: str = "contains",
    filter_column: Optional[str] = None,
    filter_text: str = "",
    filter_operator: str = "contains",
    sort_by: Optional[str] = None,
    descending: bool = False,
    page: int = 1,
    page_size: int = 50,
    focus_key: Optional[str] = None,
    include_all_keys: bool = False,
) -> NeuronIndexPage:
    """Filter, sort, and page a cached index without sending all rows to JS.

    Global search follows the shared matcher: every strict case-sensitive
    prefix match is returned first, followed by substring-only matches. This
    means a query such as ``aMe`` also returns ``MeVPaMe*`` values while
    keeping true prefixes at the top. Numeric input is verified against the
    real bodyId column only. An explicit column filter may target any retained
    metadata column and uses its selected operator directly on the main search
    text. With no selected search column, the global prefix-first behavior is
    used. The legacy ``filter_column``/``filter_text`` pair remains supported
    as an additional AND restriction for callers that still use it. When a
    query is present and no explicit sort is supplied, rows are grouped by
    matched-column priority (bodyId → type → instance → taxonomy), then by
    strict prefix versus substring, and finally matched value. The result
    also contains deduplicated matched-value groups plus primary/secondary
    relationships for the viewer's selection panel.
    """
    import polars as pl

    page_size = max(1, min(int(page_size), 500))
    frame = index.frame.with_row_index("__neuron_row")
    columns = list(index.columns)
    # The visible bodyId is the user-facing identity. The private key keeps
    # table selection safe even if a legacy index accidentally contains a
    # duplicate or blank bodyId.
    frame = frame.with_columns(
        pl.concat_str(
            [
                _display_expression("bodyId", frame)
                if "bodyId" in frame.columns
                else pl.lit(""),
                pl.lit("::"),
                pl.col("__neuron_row").cast(pl.Utf8),
            ]
        ).alias("__neuron_key")
    )
    filtered = frame

    search_text = normalize_search_text(search)
    search_columns = _ordered_match_columns(columns)
    search_column = str(search_column or "").strip()
    search_mode = _normalize_filter_operator(search_operator)
    filter_text = normalize_search_text(filter_text)
    filter_column = str(filter_column or "").strip()
    filter_mode = _normalize_filter_operator(filter_operator)
    if search_column in filtered.columns:
        search_target_columns = [search_column]
    else:
        search_target_columns = []
    if filter_column in {"__all__", "All columns", "all searchable fields"}:
        filter_columns = search_columns
    elif filter_column in filtered.columns:
        filter_columns = [filter_column]
    else:
        filter_columns = []
    active_column_filter = bool(filter_text and filter_columns)
    if active_column_filter:
        filtered = filtered.filter(
            _match_expression(filtered, filter_columns, filter_text, filter_mode)
        )

    def all_viewer_matches(source, text: str):
        """Return prefix rows followed by substring-only rows.

        The inline suggestion menu intentionally stops at the first useful
        stage. The full viewer has room for the broader result set, so it
        unions both stages and marks the row kind for stable priority sorting.
        """
        stages = search_plan(
            text,
            columns,
            "auto",
            all_prefix_matches=True,
        )
        if not stages:
            return source.filter(pl.lit(False)), None, None
        prefix_stage = stages[0]
        substring_stage = stages[1] if len(stages) > 1 else None
        prefix_rows = source.filter(
            _match_expression(source, list(prefix_stage.columns), text, "prefix")
        ).with_columns(pl.lit(0).alias("__match_kind_priority"))
        if substring_stage is None:
            return prefix_rows, prefix_stage, None
        substring_rows = source.filter(
            _match_expression(source, list(substring_stage.columns), text, "substring")
        )
        if prefix_rows.height:
            substring_rows = substring_rows.join(
                prefix_rows.select("__neuron_key").unique(),
                on="__neuron_key",
                how="anti",
            )
        substring_rows = substring_rows.with_columns(
            pl.lit(1).alias("__match_kind_priority")
        )
        if not prefix_rows.height:
            return substring_rows, prefix_stage, substring_stage
        if not substring_rows.height:
            return prefix_rows, prefix_stage, substring_stage
        return (
            pl.concat([prefix_rows, substring_rows], how="vertical"),
            prefix_stage,
            substring_stage,
        )

    scoped_search = bool(search_text and search_target_columns)
    match_text = search_text or (filter_text if active_column_filter else "")
    match_stage: SearchStage | None = None
    prefix_stage: SearchStage | None = None
    substring_stage: SearchStage | None = None
    staged_search = False
    presorted_search = False
    fast_hit_entries = None
    fast_membership_entries = None
    if scoped_search:
        match_stage = SearchStage(search_mode, tuple(search_target_columns))
        if search_mode == "substring":
            # A targeted "contains" search still presents strict,
            # case-sensitive prefixes first.  The operator controls which
            # rows qualify; the two stages control their display priority.
            prefix_stage = SearchStage("prefix", tuple(search_target_columns))
            substring_stage = SearchStage(
                "substring", tuple(search_target_columns)
            )
            prefix_rows = filtered.filter(
                _match_expression(
                    filtered,
                    search_target_columns,
                    search_text,
                    "prefix",
                )
            ).with_columns(pl.lit(0).alias("__match_kind_priority"))
            substring_rows = filtered.filter(
                _match_expression(
                    filtered,
                    search_target_columns,
                    search_text,
                    "substring",
                )
            )
            if prefix_rows.height:
                substring_rows = substring_rows.join(
                    prefix_rows.select("__neuron_key").unique(),
                    on="__neuron_key",
                    how="anti",
                )
            substring_rows = substring_rows.with_columns(
                pl.lit(1).alias("__match_kind_priority")
            )
            if prefix_rows.height and substring_rows.height:
                filtered = pl.concat([prefix_rows, substring_rows], how="vertical")
            elif prefix_rows.height:
                filtered = prefix_rows
            else:
                filtered = substring_rows
            staged_search = True
        else:
            filtered = filtered.filter(
                _match_expression(
                    filtered,
                    search_target_columns,
                    search_text,
                    search_mode,
                )
            )
    elif search_text:
        # The default viewer search can use the compact sidecar. It is
        # already ordered by the exact match priority used by the UI, so the
        # first page is available without sorting the full metadata frame.
        fast_matches = None
        if not active_column_filter:
            fast_matches = _presorted_search_matches(
                filtered,
                index.search_frame,
                search_columns,
                search_text,
                include_substrings=True,
            )
        if fast_matches is not None:
            filtered, fast_hit_entries, fast_membership_entries = fast_matches
            presorted_search = True
        else:
            filtered, prefix_stage, substring_stage = all_viewer_matches(
                filtered, search_text
            )
        staged_search = True
    elif active_column_filter:
        match_stage = SearchStage(filter_mode, tuple(filter_columns))

    if match_text:
        empty_hit_list = pl.lit([], dtype=pl.List(pl.Utf8))
        global_search = bool(search_text and not scoped_search)
        if global_search:
            if fast_hit_entries is not None:
                # The sidecar already knows every eligible cell. Aggregate
                # those narrow rows instead of re-evaluating every search
                # expression against the full metadata table.
                # ``fast_hit_entries`` is the panel projection: it omits an
                # instance value when the row already has a type match.  The
                # raw membership entries retain that value for the table's
                # cell-level outline/highlight.
                row_hit_entries = (
                    fast_membership_entries
                    if fast_membership_entries is not None
                    else fast_hit_entries
                )
                hit_lists = row_hit_entries.group_by(
                    "__neuron_row", maintain_order=True
                ).agg(
                    pl.col("search_column").alias("__fast_hit_columns"),
                    pl.col("search_value").alias("__fast_hit_values"),
                )
                secondary_entries = fast_hit_entries.join(
                    filtered.select(["__neuron_row", "__match_priority"]),
                    on="__neuron_row",
                    how="inner",
                ).filter(
                    pl.col("search_priority") > pl.col("__match_priority")
                )
                secondary_lists = secondary_entries.group_by(
                    "__neuron_row", maintain_order=True
                ).agg(
                    pl.col("search_column").alias("__fast_secondary_columns"),
                    pl.col("search_value").alias("__fast_secondary_values"),
                ) if secondary_entries.height else None
                filtered = filtered.join(
                    hit_lists,
                    on="__neuron_row",
                    how="left",
                    maintain_order="left",
                )
                if secondary_lists is not None:
                    filtered = filtered.join(
                        secondary_lists,
                        on="__neuron_row",
                        how="left",
                        maintain_order="left",
                    )
                else:
                    filtered = filtered.with_columns(
                        empty_hit_list.alias("__fast_secondary_columns"),
                        empty_hit_list.alias("__fast_secondary_values"),
                    )
                filtered = filtered.with_columns(
                    pl.when(pl.col("__fast_secondary_columns").is_null())
                    .then(empty_hit_list)
                    .otherwise(pl.col("__fast_secondary_columns"))
                    .alias("__fast_secondary_columns"),
                    pl.when(pl.col("__fast_secondary_values").is_null())
                    .then(empty_hit_list)
                    .otherwise(pl.col("__fast_secondary_values"))
                    .alias("__fast_secondary_values"),
                )
                match_priority = pl.col("__match_priority")
                global_kind_priority = pl.col("__match_kind_priority")
                match_column_key = pl.col("__candidate_column")
                match_value = pl.col("__candidate_value")
                match_column = pl.when(
                    pl.col("__candidate_column") == "bodyId"
                ).then(
                    pl.when(
                        _display_expression("instance", filtered)
                        .cast(pl.Utf8, strict=False)
                        .fill_null("")
                        .str.strip_chars() != ""
                    ).then(
                        _display_expression("instance", filtered)
                        .cast(pl.Utf8, strict=False)
                        .fill_null("")
                        .str.strip_chars()
                    ).otherwise(pl.lit("bodyId"))
                ).otherwise(pl.col("__candidate_column"))
                all_hit_columns = pl.col("__fast_hit_columns")
                all_hit_values = pl.col("__fast_hit_values")
                secondary_hit_columns = pl.col("__fast_secondary_columns")
                secondary_hit_values = pl.col("__fast_secondary_values")
                prefix_hit_columns = all_hit_columns
                prefix_hit_values = all_hit_values
                substring_hit_columns = empty_hit_list
                substring_hit_values = empty_hit_list
                secondary_metadata = (
                    secondary_hit_columns.list.first().fill_null(""),
                    secondary_hit_columns.list.first().fill_null(""),
                    secondary_hit_values.list.first().fill_null(""),
                )
            else:
                (
                    match_priority,
                    global_kind_priority,
                    match_column,
                    match_column_key,
                    match_value,
                    all_hit_columns,
                    all_hit_values,
                    secondary_hit_columns,
                    secondary_hit_values,
                ) = _global_match_metadata(filtered, search_columns, match_text)
                # Global search already has one canonical, column-ordered hit
                # list. The common metadata block below can materialize it
                # directly without reconstructing separate stage lists.
                prefix_hit_columns = all_hit_columns
                prefix_hit_values = all_hit_values
                substring_hit_columns = empty_hit_list
                substring_hit_values = empty_hit_list
                secondary_metadata = (
                    secondary_hit_columns.list.first().fill_null(""),
                    secondary_hit_columns.list.first().fill_null(""),
                    secondary_hit_values.list.first().fill_null(""),
                )
        elif staged_search:
            prefix_metadata = _match_metadata(
                filtered, columns, match_text, stage=prefix_stage
            )
            substring_metadata = _match_metadata(
                filtered, columns, match_text,
                stage=substring_stage or SearchStage("substring", tuple(search_columns)),
            )
            is_substring = pl.col("__match_kind_priority") == 1
            match_priority = pl.when(is_substring).then(
                substring_metadata[0]
            ).otherwise(prefix_metadata[0])
            match_column = pl.when(is_substring).then(
                substring_metadata[1]
            ).otherwise(prefix_metadata[1])
            match_column_key = pl.when(is_substring).then(
                substring_metadata[2]
            ).otherwise(prefix_metadata[2])
            match_value = pl.when(is_substring).then(
                substring_metadata[3]
            ).otherwise(prefix_metadata[3])
            prefix_hit_columns, prefix_hit_values = _match_hit_lists(
                filtered,
                list(prefix_stage.columns) if prefix_stage is not None else [],
                match_text,
                "prefix",
                suppress_instance_if_type=bool(search_text and not scoped_search),
            )
            substring_hit_columns, substring_hit_values = _match_hit_lists(
                filtered,
                list(substring_stage.columns)
                if substring_stage is not None
                else search_columns,
                match_text,
                "substring",
                suppress_instance_if_type=bool(search_text and not scoped_search),
            )
            secondary_hit_columns, secondary_hit_values = _secondary_hit_lists(
                filtered,
                list(substring_stage.columns)
                if substring_stage is not None
                else search_columns,
                match_text,
            )
            secondary_metadata = (
                substring_metadata[1],
                substring_metadata[2],
                substring_metadata[3],
            )
        else:
            match_priority, match_column, match_column_key, match_value = _match_metadata(
                filtered, columns, match_text, stage=match_stage
            )
            prefix_hit_columns, prefix_hit_values = _match_hit_lists(
                filtered,
                list(match_stage.columns) if match_stage is not None else [],
                match_text,
                match_stage.mode if match_stage is not None else "substring",
                suppress_instance_if_type=False,
            )
            substring_hit_columns = empty_hit_list
            substring_hit_values = empty_hit_list
            secondary_hit_columns = empty_hit_list
            secondary_hit_values = empty_hit_list
            secondary_metadata = (pl.lit(""), pl.lit(""))

        if not global_search:
            filtered = filtered.with_columns(
                prefix_hit_columns.alias("__prefix_match_columns"),
                prefix_hit_values.alias("__prefix_match_values"),
                substring_hit_columns.alias("__substring_match_columns"),
                substring_hit_values.alias("__substring_match_values"),
            )
            all_hit_columns = pl.concat_list(
                [pl.col("__prefix_match_columns"), pl.col("__substring_match_columns")]
            ).list.unique(maintain_order=True)
            all_hit_values = pl.concat_list(
                [pl.col("__prefix_match_values"), pl.col("__substring_match_values")]
            ).list.unique(maintain_order=True)
        secondary_condition = (
            pl.lit(True)
            if global_search
            else pl.lit(False)
        )
        secondary_hit_columns = pl.when(secondary_condition).then(
            secondary_hit_columns
        ).otherwise(empty_hit_list)
        secondary_hit_values = pl.when(secondary_condition).then(
            secondary_hit_values
        ).otherwise(empty_hit_list)
        filtered = filtered.with_columns(
            match_priority.alias("__match_priority"),
            match_column.alias("match_column"),
            match_column_key.alias("match_column_key"),
            match_value.alias("match_value"),
            all_hit_columns.alias("match_column_keys"),
            all_hit_values.alias("match_values"),
            secondary_hit_columns.alias("secondary_match_column_keys"),
            secondary_hit_values.alias("secondary_match_values"),
            (
                global_kind_priority
                if global_search
                else (
                    pl.col("__match_kind_priority")
                    if staged_search
                    else pl.lit(0)
                )
            ).alias("__match_kind_priority"),
            (
                pl.lit(len(search_columns))
                if global_search
                else (
                    substring_metadata[0]
                    if staged_search
                    else pl.lit(len(search_columns))
                )
            ).alias("__secondary_match_priority"),
            (
                secondary_metadata[0]
                if global_search or staged_search
                else pl.lit("")
            ).alias("__secondary_match_column"),
            (
                secondary_metadata[1]
                if global_search or staged_search
                else pl.lit("")
            ).alias("__secondary_match_column_key"),
            (
                secondary_metadata[2]
                if global_search or staged_search
                else pl.lit("")
            ).alias("__secondary_match_value"),
        )
    else:
        filtered = filtered.with_columns(
            pl.lit(len(search_columns)).alias("__match_priority"),
            pl.lit(1).alias("__match_kind_priority"),
            pl.lit("").alias("match_column"),
            pl.lit("").alias("match_column_key"),
            pl.lit("").alias("match_value"),
            pl.lit([], dtype=pl.List(pl.Utf8)).alias("match_column_keys"),
            pl.lit([], dtype=pl.List(pl.Utf8)).alias("match_values"),
            pl.lit([], dtype=pl.List(pl.Utf8)).alias("secondary_match_column_keys"),
            pl.lit([], dtype=pl.List(pl.Utf8)).alias("secondary_match_values"),
        )

    manual_match_value_sort = sort_by == "__match_value__"
    manual_sort = manual_match_value_sort or sort_by in columns
    selected_sort = "match_value" if manual_match_value_sort else (
        sort_by if manual_sort else ("bodyId" if "bodyId" in columns else columns[0])
    )
    sort_columns = []
    sort_directions = []
    if match_text and (not manual_sort or manual_match_value_sort):
        # Keep each match-column subset together in bodyId → type → instance →
        # taxonomy order. Within each column subset, strict prefixes precede
        # case-insensitive substring matches, then values are alphabetical.
        # This keeps a type substring such as MeVPaMe2 ahead of lower-priority
        # *type/taxonomy prefixes while still behind aMe* type prefixes.
        sort_columns.extend(
            ("__match_priority", "__match_kind_priority", "match_value")
        )
        sort_directions.extend(
            (
                False,
                False,
                bool(descending) if manual_match_value_sort else False,
            )
        )
    else:
        if selected_sort == "bodyId":
            # Body IDs are stored as strings in the cache to preserve large
            # values in the browser.  Sort them numerically so 10001 follows
            # 9999 instead of appearing between 10000 and 100011.
            filtered = filtered.with_columns(
                pl.col(selected_sort).cast(pl.UInt64, strict=False).alias("__body_id_sort")
            )
            sort_columns.append("__body_id_sort")
        else:
            sort_columns.append(selected_sort)
        sort_directions.append(bool(descending))
        if match_text:
            # An explicit sort column is primary; matching priority only
            # resolves rows with the same selected-column value.
            sort_columns.append("__match_priority")
            sort_directions.append(False)
    if not (
        presorted_search
        and match_text
        and not manual_sort
    ):
        try:
            try:
                filtered = filtered.sort(
                    sort_columns,
                    descending=sort_directions,
                    nulls_last=[True] * len(sort_columns),
                )
            except TypeError:  # compatibility with older Polars releases
                filtered = filtered.sort(sort_columns, descending=sort_directions)
        except TypeError:  # compatibility with older Polars releases
            filtered = filtered.sort(sort_columns, descending=sort_directions)

    # A match-group click should reveal the first corresponding metadata row.
    # Compute the page after the server-side sort so the UI does not need to
    # download or scan the full table in Python. This is optional and only
    # runs for an explicit focus request.
    focus_page = None
    if focus_key:
        focus_positions = (
            filtered
            .select("__neuron_key")
            .with_row_index("__sorted_position")
            .filter(pl.col("__neuron_key") == str(focus_key))
            .select("__sorted_position")
            .to_series()
            .to_list()
        )
        if focus_positions:
            focus_page = int(focus_positions[0]) // page_size + 1

    total = int(filtered.height)
    pages = max(1, (total + page_size - 1) // page_size)
    current_page = max(1, min(int(page or 1), pages))
    output_columns = [
        "__neuron_key",
        "match_column",
        "match_value",
        "match_column_key",
        "match_column_keys",
        "match_values",
        "secondary_match_column_keys",
        "secondary_match_values",
        *columns,
    ]
    rows = (
        filtered
        .select(output_columns)
        .slice((current_page - 1) * page_size, page_size)
        .to_dicts()
    )

    def json_value(value):
        if value is None:
            return ""
        if isinstance(value, (datetime, date, time)):
            return value.isoformat()
        return value

    safe_rows = [
        {column: json_value(value) for column, value in row.items()}
        for row in rows
    ]

    def highlight_mode() -> str:
        if search_text and not scoped_search:
            return "global"
        if search_text:
            return search_mode
        return filter_mode if active_column_filter else "global"

    if match_text:
        mode_for_highlight = highlight_mode()
        for row in safe_rows:
            row["__highlighted_cells"] = _highlighted_cells(
                row,
                match_text,
                mode_for_highlight,
                columns,
            )
    else:
        for row in safe_rows:
            row["__highlighted_cells"] = {}

    all_keys: Dict[str, str] = {}
    if include_all_keys:
        all_keys = _collect_all_keys(filtered, json_value)

    # Broad global searches can build their complete match panel directly
    # from the compact sidecar. This avoids re-evaluating every searchable
    # expression against the full metadata frame and, importantly, avoids a
    # Python pass over every wide row before the first page is rendered.
    if fast_hit_entries is not None:
        fast_groups = _presorted_match_groups(
            filtered,
            fast_hit_entries,
            search_columns,
            json_value,
            membership_entries=fast_membership_entries,
        )
        if fast_groups is not None:
            (
                fast_match_groups,
                fast_group_members,
                fast_group_body_ids,
                fast_group_related,
                fast_group_primary,
            ) = fast_groups
            return NeuronIndexPage(
                rows=safe_rows,
                total=total,
                page=current_page,
                pages=pages,
                page_size=page_size,
                sort_by=(
                    "match_value"
                    if match_text and (not manual_sort or manual_match_value_sort)
                    else selected_sort
                ),
                descending=(
                    bool(descending)
                    if match_text and manual_match_value_sort
                    else (False if match_text and not manual_sort else bool(descending))
                ),
                match_groups=fast_match_groups,
                match_group_members={
                    key: tuple(values)
                    for key, values in fast_group_members.items()
                },
                match_group_body_ids={
                    key: tuple(values)
                    for key, values in fast_group_body_ids.items()
                },
                match_group_related=fast_group_related,
                match_group_primary=fast_group_primary,
                focus_page=focus_page,
                all_keys=all_keys,
            )

    # Deduplicate by the exact matched name. A selection of one group means
    # “all rows sharing this name”, regardless of whether the name was found
    # in type, instance, class, or another useful taxonomy field.
    match_groups: List[Dict[str, Any]] = []
    group_members: Dict[str, List[str]] = {}
    group_body_ids: Dict[str, List[str]] = {}
    group_member_sets: Dict[str, set[str]] = {}
    group_body_id_sets: Dict[str, set[str]] = {}
    group_related_sets: Dict[str, set[str]] = {}
    group_primary_sets: Dict[str, set[str]] = {}
    group_index: Dict[str, Dict[str, Any]] = {}
    group_columns = [
        "__neuron_key",
        "bodyId",
        "match_column",
        "match_column_key",
        "match_value",
    ]
    if staged_search:
        group_columns.extend(
            [
                "__match_kind_priority",
                "__match_priority",
                "__secondary_match_column",
                "__secondary_match_column_key",
                "__secondary_match_value",
                "__secondary_match_priority",
                "secondary_match_column_keys",
                "secondary_match_values",
            ]
        )
    group_source = filtered.select(group_columns).to_dicts()
    ordered_filtered_keys = [
        str(json_value(raw.get("__neuron_key")) or "").strip()
        for raw in group_source
    ]
    filtered_key_order = {
        key: position for position, key in enumerate(ordered_filtered_keys)
    }
    group_order: Dict[str, Tuple[int, int, str]] = {}
    for raw in group_source:
        candidates = [
            (
                raw.get("match_column"),
                raw.get("match_column_key"),
                raw.get("match_value"),
                raw.get("__match_kind_priority", 1),
                raw.get("__match_priority", len(search_columns)),
                "primary",
            )
        ]
        if search_text and not scoped_search:
            primary_value = str(json_value(raw.get("match_value")) or "").strip()
            primary_column_key = str(raw.get("match_column_key") or "").strip()
            secondary_column_keys = [
                str(value or "").strip()
                for value in (raw.get("secondary_match_column_keys") or [])
            ]
            secondary_values = [
                str(json_value(value) or "").strip()
                for value in (raw.get("secondary_match_values") or [])
            ]
            secondary_candidates = list(zip(secondary_column_keys, secondary_values))
            if not secondary_candidates:
                # Keep compatibility with older cached rows that only carry
                # the original single-secondary fields.
                secondary_candidates = [
                    (
                        str(raw.get("__secondary_match_column_key") or "").strip(),
                        str(json_value(raw.get("__secondary_match_value")) or "").strip(),
                    )
                ]
            for secondary_column_key, secondary_value in secondary_candidates:
                if not secondary_value:
                    continue
                if (
                    secondary_value == primary_value
                    and secondary_column_key == primary_column_key
                ):
                    continue
                try:
                    secondary_priority = search_columns.index(secondary_column_key)
                except ValueError:
                    secondary_priority = len(search_columns)
                candidates.append(
                    (
                        secondary_column_key,
                        secondary_column_key,
                        secondary_value,
                        1,
                        secondary_priority,
                        "secondary",
                    )
                )

        member_key = str(json_value(raw.get("__neuron_key")) or "")
        primary_candidate_value = str(
            json_value(raw.get("match_value")) or ""
        ).strip()
        row_candidate_values = []
        for candidate in candidates:
            candidate_value = str(json_value(candidate[2]) or "").strip()
            if candidate_value and candidate_value not in row_candidate_values:
                row_candidate_values.append(candidate_value)
        body_id = json_value(raw.get("bodyId"))
        body_id = str(body_id or "").strip()
        if body_id.endswith(".0") and body_id[:-2].isdigit():
            body_id = body_id[:-2]
        for column, column_key, raw_value, raw_kind, raw_priority, match_role in candidates:
            value = str(json_value(raw_value) or "").strip()
            if not value:
                continue
            key = value
            try:
                kind_priority = int(raw_kind)
            except (TypeError, ValueError):
                kind_priority = 1
            try:
                column_priority = int(raw_priority)
            except (TypeError, ValueError):
                column_priority = len(search_columns)
            candidate_order = (column_priority, kind_priority, value)
            if key not in group_index:
                group_index[key] = {
                    "__match_group_key": key,
                    "match_column": str(column or ""),
                    "match_column_key": str(column_key or ""),
                    "match_value": value,
                    "body_count": 0,
                    "match_role": match_role,
                }
                match_groups.append(group_index[key])
                group_members[key] = []
                group_body_ids[key] = []
                group_member_sets[key] = set()
                group_body_id_sets[key] = set()
                group_related_sets[key] = set()
                group_primary_sets[key] = set()
                group_order[key] = candidate_order
            elif candidate_order < group_order[key]:
                group_order[key] = candidate_order
                group_index[key]["match_column"] = str(column or "")
                group_index[key]["match_column_key"] = str(column_key or "")
                group_index[key]["match_role"] = match_role
            if member_key and member_key not in group_member_sets[key]:
                group_members[key].append(member_key)
                group_member_sets[key].add(member_key)
                group_index[key]["body_count"] += 1
            if body_id and body_id not in group_body_id_sets[key]:
                group_body_ids[key].append(body_id)
                group_body_id_sets[key].add(body_id)
            # Keep the row's candidate values for the direct primary/secondary
            # relationship pass below.  Do not union every value into one
            # connected component: a name can be primary on one row and a
            # secondary taxonomy value on another row.
            if primary_candidate_value:
                group_primary_sets[key].add(primary_candidate_value)

    match_groups.sort(
        key=lambda group: group_order.get(
            str(group.get("__match_group_key") or ""),
            (len(search_columns), 1, str(group.get("match_value") or "")),
        )
    )
    group_rank = {
        str(group.get("__match_group_key") or ""): position
        for position, group in enumerate(match_groups)
    }
    primary_values = {
        key for key, group in group_index.items()
        if group.get("match_role") == "primary"
    }
    owner_values: Dict[str, set[str]] = {}
    for key, owners in group_primary_sets.items():
        if key not in group_rank:
            continue
        for owner in owners:
            owner = str(owner or "").strip()
            if (
                not owner
                or owner == key
                or owner not in group_rank
                or key in primary_values
            ):
                continue
            group_related_sets.setdefault(owner, {owner}).add(key)
            group_related_sets.setdefault(key, {key}).add(owner)
            owner_values.setdefault(key, set()).add(owner)

    match_group_related = {
        key: tuple(
            sorted(
                group_related_sets.get(key, {key}),
                key=lambda value: group_rank.get(
                    value, len(group_rank)
                ),
            )
        )
        for key in group_rank
    }
    # Keep each primary value independent. A pure secondary value points back
    # to its owning primary value; a value that is itself primary does not
    # become an alias for another primary just because it also appears in a
    # taxonomy column on that row.
    match_group_primary = {}
    for key in group_rank:
        if key in primary_values:
            match_group_primary[key] = (key,)
        else:
            owners = sorted(
                owner_values.get(key, ()),
                key=lambda value: group_rank.get(value, len(group_rank)),
            )
            match_group_primary[key] = tuple(owners or (key,))

    # Keep the relevance order for primary names, but render each secondary
    # taxonomy match as an accessory row immediately below its owner.
    match_groups = _order_match_groups_with_secondaries(
        match_groups,
        match_group_primary,
    )

    # A row's displayed match is intentionally only its highest-priority
    # match, but selecting a name should include rows where that same name is
    # present in another searched field as well. Build that membership map
    # once from the active shared stage instead of making the UI guess from a
    # page-sized subset.
    if match_text and match_groups:
        # A strict type stage determines which rows are returned, but a
        # matched-name selection should still cover the same name in the other
        # searchable identity/taxonomy fields. Numeric input remains bodyId
        # only so a number in a taxonomy field can never bypass the safeguard.
        if scoped_search:
            membership_columns = list(search_target_columns)
        elif search_text:
            membership_columns = (
                ["bodyId"] if is_numeric_search(match_text) else search_columns
            )
        elif active_column_filter:
            membership_columns = list(match_stage.columns)
        else:
            membership_columns = search_columns
        scope_columns = [
            column for column in membership_columns if column in filtered.columns
        ]
        # A broad one-character prefix can match most of a large index. Do
        # not materialize every searchable cell as Python dictionaries here:
        # that blocks NiceGUI's event loop long enough for the browser socket
        # to disconnect. Polars performs the same exact-name membership join
        # column-wise and only returns the compact groups needed by selection.
        group_names_by_column: Dict[str, set[str]] = {}
        for group in match_groups:
            column = str(group.get("match_column_key", "") or "").strip()
            value = str(group.get("match_value", "") or "").strip()
            if column in scope_columns and value:
                group_names_by_column.setdefault(column, set()).add(
                    value.casefold()
                )
        same_name_members: Dict[Tuple[str, str], set[str]] = {}
        for column in scope_columns:
            group_norms = sorted(group_names_by_column.get(column, set()))
            if not group_norms:
                continue
            display_value = (
                _display_expression(column, filtered)
                .cast(pl.Utf8, strict=False)
                .fill_null("")
                .str.strip_chars()
            )
            column_matches = (
                filtered
                .select(
                    pl.col("__neuron_key"),
                    display_value.alias("__match_name"),
                )
                .with_columns(
                    pl.col("__match_name")
                    .str.to_lowercase()
                    .alias("__match_norm")
                )
                .filter(pl.col("__match_norm").is_in(group_norms))
            )
            grouped = column_matches.group_by("__match_norm").agg(
                pl.col("__neuron_key").alias("__members")
            )
            for raw in grouped.to_dicts():
                name = str(raw.get("__match_norm") or "")
                members = {
                    str(member)
                    for member in (raw.get("__members") or [])
                    if str(member)
                }
                if name and members:
                    same_name_members[(column, name)] = members

        body_ids_by_key = {}
        for key, raw_body_id in filtered.select(
            ["__neuron_key", "bodyId"]
        ).iter_rows():
            body_id = str(json_value(raw_body_id) or "").strip()
            if body_id.endswith(".0") and body_id[:-2].isdigit():
                body_id = body_id[:-2]
            body_ids_by_key[str(key or "")] = body_id
        for group in match_groups:
            value = str(group.get("match_value", "") or "").strip()
            column = str(group.get("match_column_key", "") or "").strip()
            members = same_name_members.get((column, value.casefold()))
            if members:
                ordered_members = sorted(
                    members,
                    key=lambda key: filtered_key_order.get(
                        key, len(filtered_key_order)
                    ),
                )
                group_members[value] = ordered_members
                group["body_count"] = len(members)
                group_body_ids[value] = [
                    body_ids_by_key[key]
                    for key in ordered_members
                    if body_ids_by_key.get(key)
                ]

    for group in match_groups:
        group_key = str(group.get("__match_group_key") or "")
        body_ids = group_body_ids.get(group_key, [])
        group["first_body_id"] = body_ids[0] if body_ids else ""

    return NeuronIndexPage(
        rows=safe_rows,
        total=total,
        page=current_page,
        pages=pages,
        page_size=page_size,
        sort_by=(
            "match_value"
            if match_text and (not manual_sort or manual_match_value_sort)
            else selected_sort
        ),
        descending=(
            bool(descending)
            if match_text and manual_match_value_sort
            else (False if match_text and not manual_sort else bool(descending))
        ),
        match_groups=match_groups,
        match_group_members={
            key: tuple(members) for key, members in group_members.items()
        },
        match_group_body_ids={
            key: tuple(body_ids) for key, body_ids in group_body_ids.items()
        },
        match_group_related=match_group_related,
        match_group_primary=match_group_primary,
        focus_page=focus_page,
        all_keys=all_keys,
    )


def query_match_group_subtypes(
    index: CachedNeuronIndex,
    member_keys,
    *,
    type_column: str = "type",
    limit: int = 500,
) -> Dict[str, Any]:
    """Return the distinct type values inside one match group's member rows.

    The viewer calls this lazily when a coarse taxonomy entry (for example a
    ``cell_type`` value such as ``circadian_clock``) is expanded, so a broad
    query never pays the subtype fan-out for every group. Member keys are the
    private ``bodyId::<row ordinal>`` keys used by the match panel; the
    ordinals are positions in the cached frame and stay stable for the
    lifetime of the loaded index, so filtering by them reproduces exactly the
    rows behind the clicked entry.

    The result is sorted by type name (case-insensitive). ``body_ids`` and
    ``member_keys`` per subtype are the exact selection payloads for the
    viewer; the client only receives the compact display fields.
    """
    import polars as pl

    empty = {"subtypes": [], "total_types": 0, "truncated": False}
    ordinals: List[int] = []
    key_by_ordinal: Dict[int, str] = {}
    for raw in member_keys or ():
        key = str(raw or "").strip()
        if not key:
            continue
        suffix = key.rpartition("::")[2]
        try:
            ordinal = int(suffix)
        except ValueError:
            continue
        if ordinal < 0 or ordinal in key_by_ordinal:
            continue
        ordinals.append(ordinal)
        key_by_ordinal[ordinal] = key
    if not ordinals or type_column not in index.columns:
        return empty

    frame = index.frame.with_row_index("__neuron_row").filter(
        pl.col("__neuron_row").is_in(
            pl.Series(ordinals, dtype=pl.UInt32).implode()
        )
    )
    frame = frame.with_columns(
        pl.col(type_column)
        .cast(pl.Utf8, strict=False)
        .fill_null("")
        .str.strip_chars()
        .alias("__subtype_value")
    ).filter(pl.col("__subtype_value") != "")
    if frame.is_empty():
        return empty

    grouped = frame.group_by("__subtype_value").agg(
        pl.col("bodyId").cast(pl.Utf8, strict=False).fill_null("").alias(
            "__subtype_body_ids"
        ),
        pl.col("__neuron_row").alias("__subtype_rows"),
    )
    subtypes: List[Dict[str, Any]] = []
    for raw in grouped.to_dicts():
        value = str(raw.get("__subtype_value") or "")
        keys: List[str] = []
        for ordinal in raw.get("__subtype_rows") or ():
            key = key_by_ordinal.get(int(ordinal))
            if key and key not in keys:
                keys.append(key)
        body_ids: List[str] = []
        for raw_body_id in raw.get("__subtype_body_ids") or ():
            body_id = str(raw_body_id or "").strip()
            if body_id.endswith(".0") and body_id[:-2].isdigit():
                body_id = body_id[:-2]
            if body_id and body_id not in body_ids:
                body_ids.append(body_id)
        subtypes.append({
            "match_value": value,
            "body_count": len(body_ids),
            "first_body_id": body_ids[0] if body_ids else "",
            "body_ids": tuple(body_ids),
            "member_keys": tuple(keys),
        })
    subtypes.sort(
        key=lambda item: (item["match_value"].casefold(), item["match_value"])
    )
    total_types = len(subtypes)
    return {
        "subtypes": subtypes[: max(0, int(limit))],
        "total_types": total_types,
        "truncated": total_types > int(limit),
    }


# ---------------------------------------------------------------------------
# Cross-dataset alias matches (expanded search for zero-hit viewer queries)
# ---------------------------------------------------------------------------

# Loaded indexes for alias lookups, keyed by (dataset, index mtime) so a
# Settings-side force rebuild invalidates the cache naturally.
_ALIAS_INDEX_CACHE: Dict[Tuple[str, int], "CachedNeuronIndex"] = {}


def datasets_with_cached_indexes(cache_dir: Optional[Path] = None) -> List[str]:
    """Datasets from the static list that have a local neuron index."""
    from .config import DATASETS

    return [
        dataset
        for dataset in DATASETS
        if neuron_index_path(dataset, cache_dir).is_file()
    ]


def count_type_in_index(index: "CachedNeuronIndex", type_name: str) -> Optional[int]:
    """Exact neuron count for one ``type`` value in a cached index."""
    if index is None or "type" not in index.frame.columns:
        return None
    import polars as pl

    return int(index.frame.filter(pl.col("type") == type_name).height)


def _load_alias_index(dataset: str) -> Optional["CachedNeuronIndex"]:
    """Load (and memoize) the cached index used for alias counting."""
    path = neuron_index_path(dataset)
    if not path.is_file():
        return None
    key = (dataset, path.stat().st_mtime_ns)
    cached = _ALIAS_INDEX_CACHE.get(key)
    if cached is None:
        try:
            cached = load_cached_neuron_index(dataset)
        except Exception:
            return None
        _ALIAS_INDEX_CACHE[key] = cached
    return cached


def collect_alias_matches(
    dataset: str,
    search: str,
    datasets: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Expand a zero-hit viewer search into cross-dataset alias matches.

    The search name is resolved through the cross-dataset type mapper into
    per-dataset alias candidates (kind + orthogonal aggregation annotation,
    see ``CrossDatasetTypeMapper.get_alias_candidates``), then each alias is
    counted in that dataset's cached neuron index.  Only locally cached
    indexes are consulted — nothing is fetched.

    The result is strictly informational: callers must never merge these
    rows into the selected dataset's table or selection, because the bodyIds
    belong to a different dataset.
    """
    search = str(search or "").strip()
    if not search or search.isdigit() or "*" in search or len(search) < 2:
        return []
    try:
        from comparison.cross_dataset_type_mapper import get_type_mapper

        mapper = get_type_mapper()
    except Exception:
        return []
    if mapper is None or not getattr(mapper, "_loaded", False):
        return []

    if datasets is None:
        datasets = datasets_with_cached_indexes()
    datasets = [ds for ds in datasets if neuron_index_path(ds).is_file()]
    if not datasets:
        return []
    try:
        outcomes = mapper.get_alias_candidates(search, datasets)
    except Exception:
        return []

    matches: List[Dict[str, Any]] = []
    for ds in datasets:
        info = outcomes.get(ds) or {}
        outcome = info.get("outcome")
        if outcome in (None, "not applicable", "mapper unavailable"):
            continue
        entry: Dict[str, Any] = {
            "dataset": ds,
            "is_selected": ds == dataset,
            "outcome": outcome,
            "candidates": [],
        }
        if outcome == "matched":
            index = _load_alias_index(ds)
            if index is None:
                continue
            for cand in info.get("candidates", []):
                entry["candidates"].append({
                    "name": cand["name"],
                    "kind": cand["kind"],
                    "aggregates": cand.get("aggregates"),
                    "count": count_type_in_index(index, cand["name"]),
                })
        matches.append(entry)

    matches.sort(key=lambda entry: not entry["is_selected"])
    return matches
