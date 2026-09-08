"""Shared options for analysis-generated visualizations.

This module intentionally stays lightweight so analysis backends can resolve
dataset-aware rendering defaults without importing the full visualization
stack.
"""

try:
    from .flywire_ids import is_fafb_dataset
except ImportError:  # pragma: no cover - direct/script imports
    from flywire_ids import is_fafb_dataset


def _is_flywire_family(dataset: str) -> bool:
    """Compatibility helper for FAFB-specific render defaults.

    The historical helper name remains public to avoid breaking callers, but
    BANC is a standalone source and is not part of the FlyWire render family.
    """
    return is_fafb_dataset(dataset)


def default_skeleton_tab_simplification(
        dataset: str, neuprint_skeleton_pipeline: str = "fast") -> float:
    """Return the default target for the dedicated Skeleton tab.

    The fast/direct pipeline removes 90% of tube-mesh faces and fine/artistic
    pipelines remove 95%. The same method-specific defaults apply to
    NeuPrint, FAFB, and standalone BANC tube renders.

    BANC uses the same numeric default at the renderer boundary, while its
    public-release SWC/L2 source selection remains separate from FAFB.
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
