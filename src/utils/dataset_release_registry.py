"""Release-aware dataset metadata shared by the mapper and the UI.

The registry deliberately describes *policy* rather than pretending that
every release pair is a safe body-level mapping.  A recommendation means
that the newer release is the preferred user-facing choice; a mapping alias
is a separate, explicitly named strategy used by the type mapper.
"""

from __future__ import annotations

from typing import Iterable, Optional


VERSION_ALIAS_REGISTRY = {
    "male-cns:v0.9": {
        "family": "male-cns",
        "release": "v0.9",
        "newer": "male-cns:v1.0",
        "newer_available": True,
        "strategy": "same_name_release_alias",
        "mapping_strategy": "same_name_release_alias",
        "recommend": True,
        "recommend_if_selected": True,
        "provenance": (
            "11,597 shared primary names; 99.1768% agreement on common "
            "typed body rows"
        ),
        "message": (
            "male-cns:v1.0 is the newer supported release. Use it for "
            "current type mapping; keep v0.9 for legacy or reproducible runs."
        ),
    },
    # This is a known newer release, but the current plan does not certify a
    # safe MANC v1.2.1 -> v1.2.3 type/body bridge.  Keep the metadata here so
    # the UI can remain explicit rather than accidentally recommending it.
    "manc:v1.0": {
        "family": "manc",
        "release": "v1.0",
        "newer": "manc:v1.2.1",
        "newer_available": True,
        "strategy": "candidate_only",
        "mapping_strategy": "candidate_only",
        "recommend": False,
        "recommend_if_selected": False,
        "provenance": "version-specific type changes are too large for a blind replacement",
        "message": (
            "manc:v1.2.1 is available, but this release pair is not yet "
            "certified for automatic type mapping."
        ),
    },
    "manc:v1.2.1": {
        "family": "manc",
        "release": "v1.2.1",
        "newer": "manc:v1.2.3",
        "newer_available": True,
        "strategy": "candidate_only",
        "mapping_strategy": "candidate_only",
        "recommend": False,
        "recommend_if_selected": False,
        "provenance": "99.873% same-type agreement, but the direct release overlay is not certified",
        "message": (
            "manc:v1.2.3 is available, but this release pair is not yet "
            "certified for automatic type mapping."
        ),
    },
}


def get_release_alias(source: str, target: str) -> Optional[dict]:
    """Return the configured alias record for an ordered release pair."""

    source = str(source or "").strip()
    target = str(target or "").strip()
    record = VERSION_ALIAS_REGISTRY.get(source)
    if not record or record.get("newer") != target:
        return None
    return {"source": source, "target": target, **record}


def get_release_recommendation(
    dataset: str,
    available_datasets: Optional[Iterable[str]] = None,
) -> Optional[dict]:
    """Return a UI-safe recommendation for one selected release.

    ``available_datasets`` is optional because the static selector catalog is
    useful before a server refresh.  When supplied, it controls whether the
    replacement can be actioned immediately; a missing replacement produces a
    clear access/preparation message instead of a misleading button.
    """

    key = str(dataset or "").strip()
    record = VERSION_ALIAS_REGISTRY.get(key)
    if not record or not record.get(
            "recommend_if_selected", record.get("recommend")):
        return None

    available = None
    if available_datasets is not None:
        available = {str(value).strip() for value in available_datasets}
    newer = record["newer"]
    actionable = available is None or newer in available
    result = {"source": key, "target": newer, **record}
    result["actionable"] = actionable
    if not actionable:
        result["message"] = (
            f"{newer} is the newer supported release, but it is not currently "
            "available in this dataset list. Prepare or enable it before switching."
        )
    return result


def recommended_release_for(dataset: str) -> Optional[str]:
    """Return the recommended replacement, if the registry endorses one."""

    record = VERSION_ALIAS_REGISTRY.get(str(dataset or "").strip())
    if not record or not record.get(
            "recommend_if_selected", record.get("recommend")):
        return None
    return str(record["newer"])


__all__ = [
    "VERSION_ALIAS_REGISTRY",
    "get_release_alias",
    "get_release_recommendation",
    "recommended_release_for",
]
