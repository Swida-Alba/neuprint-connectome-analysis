"""Persistent, standalone query history for the Type Mapping panel.

This module deliberately uses a different file from :mod:`history_store` so
the cross-dataset Type Mapping dialog keeps its own Recent/Frequent list:
panel searches never appear in the analysis tabs' neuron-query history, and
tab searches never pollute the panel's list. Unlike the line history, the
selected datasets ARE recorded as provenance (the panel always knows its
dataset selection), so future consumers can scope or tag panel entries.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

from . import history_store as _store


_HISTORY_PATH = Path(__file__).resolve().parent / "type_mapping_history.json"
_LIMIT_RECENT = _store._LIMIT_RECENT
_LIMIT_FREQUENT = _store._LIMIT_FREQUENT


def record(values: List[str], now: Optional[str] = None,
           custom_values: Optional[Iterable[str]] = None,
           datasets=None) -> None:
    return _store.record(
        values,
        now=now,
        custom_values=custom_values,
        datasets=datasets,
        _history_path=_HISTORY_PATH,
    )


def datasets_of(value: str) -> List[str]:
    return _store.datasets_of(value, _history_path=_HISTORY_PATH)


def mark_custom(values: Iterable[str]) -> None:
    return _store.mark_custom(values, _history_path=_HISTORY_PATH)


def prune_orphaned_custom(valid_values: Iterable[str]) -> List[str]:
    return _store.prune_orphaned_custom(
        valid_values, _history_path=_HISTORY_PATH
    )


def recent(limit: int = _LIMIT_RECENT, datasets=None) -> List[str]:
    return _store.recent(
        limit=limit, datasets=datasets, _history_path=_HISTORY_PATH
    )


def frequent(limit: int = _LIMIT_FREQUENT, datasets=None) -> List[str]:
    return _store.frequent(
        limit=limit, datasets=datasets, _history_path=_HISTORY_PATH
    )


def remove(value: str) -> bool:
    return _store.remove(value, _history_path=_HISTORY_PATH)


def clear() -> None:
    return _store.clear(_history_path=_HISTORY_PATH)
