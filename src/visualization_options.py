"""Shared options for analysis-generated visualizations.

This module intentionally stays lightweight so analysis backends can resolve
dataset-aware rendering defaults without importing the full visualization
stack.
"""

try:
    from .flywire_ids import is_flywire_dataset
except ImportError:  # pragma: no cover - direct/script imports
    from flywire_ids import is_flywire_dataset


def _is_flywire_family(dataset: str) -> bool:
    """Return whether *dataset* belongs to the FlyWire render family.

    Mirrors ``flywire_ids.is_flywire_dataset``: FAFB and BANC releases both
    use the FlyWire-family render defaults.
    """
    return is_flywire_dataset(dataset)


def default_skeleton_tab_simplification(
        dataset: str, neuprint_skeleton_pipeline: str = "fast") -> float:
    """Return the default target for the dedicated Skeleton tab.

    The fast/direct pipeline removes 90% of tube-mesh faces and fine/artistic
    pipelines remove 95%. The same method-specific defaults apply to
    NeuPrint and FlyWire/FAFB tube renders.

    BANC needs no special default: the slider only ever drives the
    full-resolution sources (FAFB-style 90%), while L2 tubes skip the
    decimation stage structurally in the BANC render processor — a 0.0
    default here would leave full-res-only neurons (they exist, e.g. one
    l-LNv) undecimated.
    """
    pipeline = str(neuprint_skeleton_pipeline or "fast").strip().lower()
    return 0.90 if pipeline in {"fast", "direct"} else 0.95


def default_analysis_skeleton_mesh_simplification(
        dataset: str, neuprint_skeleton_pipeline: str = "fine") -> float:
    """Return the default tube-mesh simplification for analysis renders.

    Analysis renders use the same method defaults as the dedicated Skeleton
    tab: fast/direct removes 90% of faces and fine/artistic removes 95%.
    """
    pipeline = str(neuprint_skeleton_pipeline or "fine").strip().lower()
    return 0.90 if pipeline in {"fast", "direct"} else 0.95
