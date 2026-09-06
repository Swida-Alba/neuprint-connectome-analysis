"""Shared options for analysis-generated visualizations.

This module intentionally stays lightweight so analysis backends can resolve
dataset-aware rendering defaults without importing the full visualization
stack.
"""


def _is_flywire_family(dataset: str) -> bool:
    """Return whether *dataset* uses the FlyWire/FAFB render family."""
    normalized = str(dataset or "").strip().lower()
    return normalized.startswith("flywire_") or "fafb" in normalized


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
