"""Normalized comparison-point identities shared by exports and reports.

Standard comparisons use one scalar threshold as their comparison point.
Custom combination comparisons use one stable query row.  Raw execution keys
remain dataset/threshold pairs; this module only describes the point that
consumes those raw cells.
"""

from dataclasses import dataclass
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional


def safe_point_id(value: Any, fallback: str = "query") -> str:
    """Return a deterministic filesystem/DOM-safe identifier."""
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or fallback))
    text = text.strip(" ._-")
    return text or fallback


@dataclass(frozen=True)
class ComparisonPoint:
    """One Standard scalar point or one Custom combination query."""

    point_id: str
    label: str
    mode: str
    thresholds_by_dataset: Dict[str, int]
    display_order: int = 1

    @property
    def query_id(self) -> Optional[str]:
        return self.point_id if self.mode == "combinations" else None

    @property
    def raw_thresholds(self) -> List[int]:
        return sorted({int(value) for value in self.thresholds_by_dataset.values()})

    @property
    def file_stem(self) -> str:
        if self.mode == "combinations":
            return f"query_{safe_point_id(self.point_id)}"
        value = next(iter(self.thresholds_by_dataset.values()), self.point_id)
        return f"minsyn_{int(value)}"

    @property
    def display_label(self) -> str:
        if self.mode != "combinations":
            return self.label
        return f"{self.point_id}: {self.label}"

    @classmethod
    def from_query(
        cls,
        query: Mapping[str, Any],
        dataset_order: Iterable[str],
        display_order: int = 1,
    ) -> "ComparisonPoint":
        query_id = str(query.get("id") or query.get("query_id") or "query")
        thresholds = query.get("thresholds") or query.get(
            "thresholds_by_dataset", {}
        )
        ordered = {
            dataset: int(thresholds[dataset])
            for dataset in dataset_order
            if dataset in thresholds
        }
        label = str(query.get("label") or query_id)
        return cls(query_id, label, "combinations", ordered, display_order)

    @classmethod
    def from_standard(
        cls,
        threshold: int,
        dataset_order: Iterable[str],
        display_order: int = 1,
    ) -> "ComparisonPoint":
        threshold = int(threshold)
        ordered = {dataset: threshold for dataset in dataset_order}
        return cls(
            f"threshold_{threshold}",
            f"N={threshold}",
            "standard",
            ordered,
            display_order,
        )


def points_from_parameters(parameters: Any) -> List[ComparisonPoint]:
    """Build stable points without changing the parameter model."""
    datasets = list(parameters.get_dataset_names())
    if getattr(parameters, "threshold_mode", "standard") == "combinations":
        getter = getattr(parameters, "get_threshold_queries", None)
        if callable(getter):
            queries = getter()
        else:
            # Lightweight parameter doubles and older integrations may only
            # expose the serialized combination rows.  Keep point identity
            # available for those callers as well.
            queries = []
            for index, row in enumerate(
                    getattr(parameters, "threshold_combinations", []) or [],
                    start=1):
                if not isinstance(row, Mapping):
                    continue
                thresholds = row.get("thresholds") or row.get(
                    "thresholds_by_dataset", {})
                queries.append({
                    "id": row.get("id", f"query_{index:03d}"),
                    "label": row.get("label", row.get(
                        "id", f"query_{index:03d}")),
                    "thresholds": thresholds,
                })
        return [
            ComparisonPoint.from_query(query, datasets, index)
            for index, query in enumerate(queries, start=1)
        ]
    thresholds = list(getattr(parameters, "thresholds", []) or [])
    return [
        ComparisonPoint.from_standard(threshold, datasets, index)
        for index, threshold in enumerate(thresholds, start=1)
    ]


def point_from_value(
    value: Any,
    parameters: Any,
    points: Optional[List[ComparisonPoint]] = None,
) -> ComparisonPoint:
    """Resolve a point object, query ID, or Standard scalar threshold."""
    if isinstance(value, ComparisonPoint):
        return value
    points = points if points is not None else points_from_parameters(parameters)
    if isinstance(value, Mapping):
        query_id = value.get("id") or value.get("query_id")
        for point in points:
            if point.point_id == str(query_id):
                return point
    if isinstance(value, str):
        for point in points:
            if point.point_id == value:
                return point
        if getattr(parameters, "threshold_mode", "standard") == "standard":
            try:
                value = int(value)
            except ValueError:
                pass
    if isinstance(value, (int, float)):
        threshold = int(value)
        for point in points:
            if point.mode == "standard" and next(
                iter(point.thresholds_by_dataset.values()), None
            ) == threshold:
                return point
    raise KeyError(f"Unknown comparison point: {value!r}")
