"""In-memory topology-preserving TreeNeuron node simplification.

Render-time node reduction shared by the FAFB fast tube pipeline (25%
node retention) and the line-mode defaults (NeuPrint 50% reduction,
FAFB 90% reduction).  Simplification always operates on a copy so raw
SWC sources (healed ZIP, shared raw cache) stay canonical; nothing in
this module touches disk.

``navis.downsample_neuron`` has no target-node-count parameter, only a
factor.  The relationship between factor and retained nodes is monotonic,
so this module calibrates the factor with a short bracketed search until
the achieved count is close enough to the requested target.  Roots,
branch points and terminal nodes are preserved by navis, so very dense
neurons may keep more nodes than the nominal target — the reported
``achieved_nodes`` is authoritative.
"""

from __future__ import annotations

import time

MIN_RETAINED_NODES = 100
"""Minimum retained node count; below this a neuron is returned as-is."""

TOLERANCE = 0.05
"""Accept an achieved count within 5% of the requested target."""

MAX_SEARCH_STEPS = 6
"""Refinement passes of the factor search (each pass downsamples once)."""

FACTOR_MIN = 1.0001
"""Downsampling factor lower bound (navis requires > 1)."""

FACTOR_MAX = 1000.0
"""Downsampling factor upper bound (safety clamp)."""

FAFB_FAST_NODE_RETENTION = 0.25
"""FAFB fast tube mode retains 25% of raw SWC nodes."""

FAFB_LINE_NODE_REDUCTION = 0.90
"""FAFB line-mode default removes 90% of nodes (factor ~10)."""

NEUPRINT_LINE_NODE_REDUCTION = 0.50
"""NeuPrint line-mode default removes 50% of nodes (factor ~2)."""

BANC_LINE_FULL_NODE_REDUCTION = 0.50
"""BANC line mode removes 50% of nodes — and only on full-resolution
sources; the coarse L2 skeletons keep every node."""

BANC_L2_EQUIVALENT_SIMPLIFICATION = 0.90
"""The BANC L2 skeleton product is treated as already ~90% simplified
relative to the full-resolution skeletons (measured: ~3.3 um mean edge
spacing vs ~0.34 um, i.e. ~10x coarser node density).  A requested
skeleton_mesh_simplification ABOVE this baseline applies to L2 tubes with
the excess scaled onto the L2 density: asked 0.95 -> remove 50% of the L2
tube faces; asked 0.90 or below leaves L2 tubes untouched."""

BANC_FULL_MIN_KEEP_FACES = 4_000
"""Plan §B safety floor: BANC full-resolution tube decimation never keeps
fewer than this many faces, however aggressive the simplification slider
is. L2 sources skip the decimation stage entirely, so the floor only
bounds the full-resolution stage."""


def _neuron_node_ids(neuron):
    """Node ids eligible for extra preservation (soma nodes)."""
    ids = []
    soma = getattr(neuron, "soma", None)
    if soma is None:
        return ids
    try:
        for node_id in soma:
            if node_id is not None:
                ids.append(int(node_id))
    except (TypeError, ValueError):
        pass
    return ids


def simplify_skeleton_nodes(neuron, reduction, preserve_nodes=None):
    """Reduce a TreeNeuron to a node-count target, in memory.

    Parameters
    ----------
    neuron : navis.TreeNeuron
        Source skeleton.  Never mutated; simplification runs on a copy.
    reduction : float
        Fraction of nodes to remove (0.0-1.0).  ``0.75`` retains 25%.
    preserve_nodes : list of int, optional
        Extra node ids to keep (e.g. soma nodes).  Roots, branch points
        and terminal nodes are preserved by navis regardless.

    Returns
    -------
    tuple
        (simplified TreeNeuron, stats).  ``stats`` carries
        ``raw_nodes``, ``target_nodes``, ``achieved_nodes``, ``factor``
        and ``elapsed`` for progress/diagnostic reporting.
    """
    import navis

    if not isinstance(neuron, navis.TreeNeuron):
        raise TypeError(
            f"simplify_skeleton_nodes expects a TreeNeuron, got "
            f"{type(neuron).__name__}"
        )

    started = time.perf_counter()
    raw_nodes = int(getattr(neuron, "n_nodes", 0) or 0)
    reduction = float(min(max(reduction, 0.0), 1.0))
    target_nodes = max(
        MIN_RETAINED_NODES, int(round(raw_nodes * (1.0 - reduction)))
    )

    stats = {
        "raw_nodes": raw_nodes,
        "target_nodes": target_nodes,
        "achieved_nodes": raw_nodes,
        "factor": None,
        "elapsed": 0.0,
    }

    if raw_nodes <= MIN_RETAINED_NODES or raw_nodes <= target_nodes:
        stats["elapsed"] = time.perf_counter() - started
        return neuron.copy(), stats

    preserved = list(preserve_nodes or []) + _neuron_node_ids(neuron)
    if not preserved:
        preserved = None

    def count_at(factor):
        out = navis.downsample_neuron(
            neuron,
            downsampling_factor=float(factor),
            inplace=False,
            preserve_nodes=preserved,
        )
        return out, int(getattr(out, "n_nodes", 0) or 0)

    def close_enough(count):
        return abs(count - target_nodes) <= max(
            1, int(target_nodes * TOLERANCE)
        )

    # navis walks up to ``factor`` parents per kept node, so the effective
    # stride is ~factor + 1 and the initial estimate is raw/target - 1.
    estimate = max(FACTOR_MIN, min(
        FACTOR_MAX, raw_nodes / max(target_nodes, 1) - 1.0))
    best_neuron, best_count = count_at(estimate)
    best_factor = estimate

    if not close_enough(best_count):
        # Monotonic count(factor) with quantized plateaus: expand a bracket
        # until it straddles the target, then bisect.  When the target lies
        # outside the reachable range (topology preservation floor/ceiling),
        # the closest achievable count wins.
        low_factor = max(FACTOR_MIN, estimate / 4.0)
        low_neuron, low_count = count_at(low_factor)
        high_factor = min(FACTOR_MAX, estimate * 4.0)
        high_neuron, high_count = count_at(high_factor)

        # Expand outward when the estimate missed the target regime
        # entirely (e.g. heavily topology-preserved neurons).
        for _ in range(MAX_SEARCH_STEPS):
            if low_count < target_nodes and low_factor > FACTOR_MIN:
                low_factor = max(FACTOR_MIN, low_factor / 4.0)
                low_neuron, low_count = count_at(low_factor)
            elif high_count > target_nodes and high_factor < FACTOR_MAX:
                high_factor = min(FACTOR_MAX, high_factor * 4.0)
                high_neuron, high_count = count_at(high_factor)
            else:
                break

        # Bisect only while the target stays bracketed; quantized plateaus
        # or the factor bounds can make it unreachable.
        for _ in range(MAX_SEARCH_STEPS):
            if close_enough(best_count):
                break
            if low_count < target_nodes or high_count > target_nodes:
                break
            mid_factor = (low_factor + high_factor) / 2.0
            mid_neuron, mid_count = count_at(mid_factor)

            if abs(mid_count - target_nodes) < abs(best_count - target_nodes):
                best_neuron, best_count, best_factor = (
                    mid_neuron, mid_count, mid_factor
                )
            if mid_count > target_nodes:
                low_factor, low_neuron, low_count = (
                    mid_factor, mid_neuron, mid_count
                )
            else:
                high_factor, high_neuron, high_count = (
                    mid_factor, mid_neuron, mid_count
                )

    stats["achieved_nodes"] = best_count
    stats["factor"] = round(best_factor, 4)
    stats["elapsed"] = time.perf_counter() - started
    return best_neuron, stats
