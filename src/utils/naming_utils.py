"""
Shared output-folder naming helpers for DROCAT.

Every main function creates one top-level, timestamped run folder under the
user's output directory, named:

    {tool}_{dataset_abbreviation}_{detail}_{timestamp}

Examples:
    find-paths-complete_MCNS_aMe12_to_aMe10_L2w3r0p0_20260801_183000
    finddirect_MCNS_aMe12_to_aMe10_L2w3r0p0_20260801_183005
    profiling_MCNS_aMe12_aMe10_aMe9_20260801_183010
    homologs_MCNS_to_HEMI_aMe12_20260801_183015
    similar-morphology_MCNS_aMe12_20260801_183018
    similar-connectivity_MCNS_to_HEMI_aMe12_20260801_183019
    NB-find-lines_MCNS_aMe12_20260801_183020
    plot-3d_MCNS_aMe12_20260801_183025
"""

from collections import Counter
import re


DATASET_ABBREVIATIONS = {
    "male-cns": "MCNS",
    "male_cns": "MCNS",
    "hemibrain": "HEMI",
    "optic-lobe": "OL",
    "optic_lobe": "OL",
    "manc": "MANC",
    "banc": "BANC",
    "fib19": "FIB",
    "mushroombody": "MB",
    "flywire_fafb": "FAFB",
    "fafb": "FAFB",
    "flywire_banc": "BANC",
    # bare flywire identifiers refer to the FAFB dataset in DROCAT
    "flywire": "FAFB",
}


_DATASET_VERSION_SUFFIX_RE = re.compile(
    r"(?:^|[:_\-\s])v?(\d+(?:[._]\d+)*)$",
    re.IGNORECASE,
)

# Legacy BANC identifiers carried the ``flywire_`` prefix while BANC was
# handled as a FlyWire release.  BANC is now analyzed from its own public
# release data (not through FlyWire), so the canonical names drop the
# prefix; the legacy spellings stay accepted as aliases everywhere a
# dataset name enters the app.
_BANC_LEGACY_NAME_RE = re.compile(
    r"^(?:flywire[_-]?)?banc(?:[_-](v\d+(?:[._]\d+)*))?$",
    re.IGNORECASE,
)


def canonical_dataset_name(dataset) -> str:
    """Return the canonical dataset identifier for *dataset*.

    Legacy ``flywire_BANC_v626``-style names map to ``banc_v626`` /``banc_v888``
    (per release).  Bare ``flywire_BANC`` / ``banc`` pin to the historical
    default BANC release ``banc_v626`` — the same pin the cross-dataset type
    mapper applies — so the unversioned alias can never straddle the two
    BANC releases (v626 and v888 are distinct datasets with distinct id
    spaces).  Every other identifier — including NeuPrint colon forms and
    the FAFB release — passes through unchanged.
    """
    text = str(dataset or "").strip()
    match = _BANC_LEGACY_NAME_RE.match(text)
    if match:
        version = match.group(1)
        return f"banc_{version.lower()}" if version else "banc_v626"
    return text


def dataset_version(dataset) -> str | None:
    """Return a normalized version token from a dataset identifier.

    Examples:
        ``male-cns:v1.0`` -> ``v1.0``
        ``male_cns_v0_9`` -> ``v0.9``
        ``flywire_BANC_v888`` -> ``v888``

    Dataset versions are intentionally extracted from the original identifier;
    callers can therefore distinguish releases that share a family abbreviation.
    """
    if not dataset:
        return None

    match = _DATASET_VERSION_SUFFIX_RE.search(str(dataset).strip())
    if not match:
        return None
    return f"v{match.group(1).replace('_', '.')}"


def make_unique_dataset_labels(datasets, labels=None) -> list[str]:
    """Make display labels unique without discarding dataset identity.

    The normal label remains unchanged when it is unique.  When two selected
    datasets share the same family label (for example ``MCNS`` or ``BANC``),
    their version is appended using a filename-safe separator:
    ``MCNS_v1_0`` and ``MCNS_v0_9``.

    ``datasets`` may contain strings or objects exposing a ``dataset``
    attribute.  ``labels`` is optional and defaults to :func:`dataset_abbrev`.
    """
    dataset_names = [
        getattr(dataset, "dataset", str(dataset))
        for dataset in (datasets or [])
    ]
    base_labels = []
    for index, dataset_name in enumerate(dataset_names):
        label = labels[index] if labels is not None and index < len(labels) else None
        label = str(label).strip() if label is not None else ""
        base_labels.append(label or dataset_abbrev(dataset_name))

    counts = Counter(label.casefold() for label in base_labels)
    result = []
    used = set()

    for index, (dataset_name, base_label) in enumerate(zip(dataset_names, base_labels)):
        candidate = base_label
        if counts[base_label.casefold()] > 1:
            version = dataset_version(dataset_name)
            if version:
                candidate = f"{base_label}_{version.replace('.', '_')}"
            else:
                candidate = f"{base_label}_{index + 1}"

        # Protect against duplicate release identifiers or user-provided labels
        # that still collide after the version suffix is added.
        stem = candidate
        suffix = 2
        while candidate.casefold() in used:
            candidate = f"{stem}_{suffix}"
            suffix += 1

        result.append(candidate)
        used.add(candidate.casefold())

    return result


def dataset_abbrev(dataset) -> str:
    """Return a short, folder-safe abbreviation for a dataset identifier."""
    if not dataset:
        return "UNKN"
    ds = str(dataset).lower()
    for key, abbrev in DATASET_ABBREVIATIONS.items():
        if key in ds:
            return abbrev
    letters = "".join(c for c in ds.split(":")[0] if c.isalpha())
    return (letters[:4] or "DS").upper()
