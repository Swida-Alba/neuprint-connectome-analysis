"""
Shared neuron-label predicates for DROCAT.

This module is the single source of truth for the "untyped neuron"
definition used by pathfinding (``FindNeuronConnection(drop_untyped=...)``)
and Cross-Dataset Comparison (``ComparisonParameters.drop_untyped``).
Both surfaces must agree on which resolved type labels count as untyped so
a neuron dropped in one tool would be dropped in the other.
"""


def is_untyped_type_label(value) -> bool:
    """True when a resolved type label means 'untyped'.

    A label is untyped when it is
    - empty after whitespace stripping,
    - one of the explicit sentinel strings ``Unknown`` / ``None`` / ``NaN``
      (case-insensitive), or
    - the numeric bodyId fallback (the neuron's own id used as its type
      when no name resolved — an all-digit string).

    Accepts any scalar; values are stringified first, so pandas ``NaN``
    floats stringify to ``'nan'`` and are caught by the sentinel branch.
    """
    s = str(value).strip()
    if not s or s.lower() in {"unknown", "nan", "none"}:
        return True
    return s.isdigit()


def untyped_side(pre_untyped: bool, post_untyped: bool) -> str:
    """Record flag naming which side(s) of an edge are untyped."""
    if pre_untyped and post_untyped:
        return 'pre+post'
    if pre_untyped:
        return 'pre'
    if post_untyped:
        return 'post'
    return ''
