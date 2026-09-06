"""Edge-Budget floor (Fix D, w0 = w1 + 1) — unit tests.

The floor is a pure threshold raise applied to the lossless-pruned cone
when it still exceeds the budget. Guarantees under test:

- w0 = (N-th strongest edge weight) + 1 — boundary-tie mass at w1 is
  excluded, so the kept-edge count is STRICTLY below N (the tie-mass
  blowup the top-N landing alone would suffer).
- No floor when the cone fits the budget; the floor never returns an
  empty graph (degenerate single-tier cones / below-support budgets
  revert with an honest note).
- Inputs are never mutated (cached graph entries stay shareable).
"""

import os
import sys

import polars as pl

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import coana  # noqa: E402


def _cone(rows):
    return [pl.DataFrame({
        "bodyId_pre": [r[0] for r in rows],
        "bodyId_post": [r[1] for r in rows],
        "weight": [r[2] for r in rows],
    })]


def _total_rows(tables):
    return sum(f.height for f in tables)


def test_floor_excludes_tie_mass_and_keeps_below_budget():
    """A large tie tier at the landing weight is excluded: kept edges are
    strictly below the budget even when the boundary tier alone exceeds
    it (the top-N landing without +1 would keep all of it)."""
    rows = [
        ("S", "A", 5), ("A", "T", 5), ("S", "B", 5), ("B", "T", 5),
        # tie tier at 5 (four live edges)
        ("S", "C", 4), ("C", "T", 4),                  # tier at 4
        ("S", "D", 3), ("D", "T", 2),                  # weaker tiers
    ]
    tables = _cone(rows)
    budget = 5
    notes = []
    out, stats = coana.apply_edge_budget_floor(
        tables, budget, ["S"], ["T"], 3, warn_notes=notes)

    # weights desc: 5,5,5,5,4,4,3,2 -> N-th (5th) strongest w1 = 4,
    # w0 = 5: the four 5-weights survive (4 < 5); the tie tier at 4
    # (which the plain top-N landing would have kept, 6 rows >= N) is
    # excluded.
    assert stats["applied"] is True
    assert stats["landing"] == 4.0
    assert stats["floor"] == 5.0
    kept = sorted(w for f in out for w in f["weight"].to_list())
    assert kept == [5, 5, 5, 5], kept
    assert len(kept) < budget
    assert any("edge budget" in n for n in notes)


def test_no_floor_when_cone_fits_budget():
    tables = _cone([("S", "A", 3), ("A", "T", 4)])
    out, stats = coana.apply_edge_budget_floor(
        tables, 10, ["S"], ["T"], 2)
    assert stats["applied"] is False
    assert _total_rows(out) == 2
    assert stats["landing"] is None and stats["floor"] is None


def test_degenerate_single_tier_cone_applies_empty_floor():
    """Production semantics (no degenerate revert): every edge in one
    weight tier with total > budget — the floor APPLIES at w0 and the
    cone is legitimately EMPTY (equivalent to a complete run at w0
    finding no paths). The notice says the cone is empty."""
    rows = [("S", "A", 3), ("A", "T", 3), ("S", "B", 3), ("B", "T", 3)]
    tables = _cone(rows)
    notes = []
    out, stats = coana.apply_edge_budget_floor(
        tables, 2, ["S"], ["T"], 3, warn_notes=notes)
    assert stats["applied"] is True
    assert stats["floor"] == 4.0  # w1 = 3 -> w0 = 4
    assert _total_rows(out) == 0  # restrictive: the empty result stands
    assert any("EMPTY" in n for n in notes)
    assert not any("NOT applied" in n for n in notes)


def test_floor_never_mutates_inputs():
    rows = [("S", "A", 10), ("A", "T", 9), ("S", "B", 8), ("B", "T", 2),
            ("S", "C", 1), ("C", "T", 1)]
    tables = _cone(rows)
    before = tables[0].clone()
    out, stats = coana.apply_edge_budget_floor(
        tables, 3, ["S"], ["T"], 3)
    assert stats["applied"] is True
    assert tables[0].equals(before)          # input untouched
    assert _total_rows(out) < 6              # floored


def test_floor_is_pure_threshold_raise():
    """Every surviving edge has weight >= w0, and w0 > every dropped
    weight — the floored graph is exactly the cone filtered at w0."""
    rows = [("S", "A", 10), ("A", "T", 9), ("S", "B", 8), ("B", "T", 2),
            ("S", "C", 1), ("C", "T", 1)]
    tables = _cone(rows)
    out, stats = coana.apply_edge_budget_floor(
        tables, 3, ["S"], ["T"], 3)
    w0 = stats["floor"]
    dropped = {w for f in tables for w in f["weight"].to_list()
               if w < w0}
    kept = {w for f in out for w in f["weight"].to_list()}
    assert kept & dropped == set()
    assert all(w >= w0 for w in kept)
