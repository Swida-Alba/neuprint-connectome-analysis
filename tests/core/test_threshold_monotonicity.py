"""Fix A cap-policy tests (plan-cross-dataset-report-fixes.md §2).

With `max_paths_bodyid` unset/0, legacy complete enumerators run UNBOUNDED
and per-threshold path sets are nested (paths(t=3) ⊆ paths(t=2) ⊆
paths(t=1)) — the failure mode behind the evidence run's non-monotone
`threshold_sensitivity.csv` cannot recur.
"""

import os
import sys
from pathlib import Path

import polars as pl
import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
for p in (PROJECT_ROOT, PROJECT_ROOT / "src", PROJECT_ROOT / "vispath-subproject" / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from tests.core.test_pathfinding import _make_pipeline_fc  # noqa: E402

_DIAMOND = [("S", "A", 5), ("S", "B", 2), ("A", "T", 3), ("B", "T", 6)]


def _type_paths(fc):
    import polars as pl
    type_csv = os.path.join(fc.allpath_folder, "src_to_tgt_allpaths_type.csv")
    return set(pl.read_csv(type_csv)["path"].to_list())


def test_unbounded_legacy_runs_are_nested(monkeypatch, tmp_path):
    sets = {}
    for t in [1, 2, 3]:
        fc, _calls, _logs = _make_pipeline_fc(
            monkeypatch, tmp_path / f"t{t}", _DIAMOND, max_interlayer=2,
            min_synapse=t)
        fc.max_paths_bodyid = 0  # unlimited (explicit)
        fc.FindAllPath()
        sets[t] = _type_paths(fc)
    assert sets[3] <= sets[2] <= sets[1]
    assert sets[1], 'sanity'


def test_no_truncation_note_with_unbounded_cap(monkeypatch, tmp_path):
    fc, _calls, _logs = _make_pipeline_fc(
        monkeypatch, tmp_path, _DIAMOND, max_interlayer=2, min_synapse=1)
    fc.max_paths_bodyid = 0
    fc.FindAllPath()
    assert not any("TRUNCATED" in n for n in fc._warn_notes)


def test_graph_edge_limit_is_the_edge_budget_above_cone_size(monkeypatch, tmp_path):
    """Fix D: graph_edge_limit_bodyid is the EDGE BUDGET — above the cone
    size it is a no-op (no floor, no deprecation notice) and results stay
    complete."""
    fc, _calls, _logs = _make_pipeline_fc(
        monkeypatch, tmp_path, _DIAMOND, max_interlayer=2, min_synapse=1)
    fc.graph_edge_limit_bodyid = 5  # above this tiny cone: no floor
    fc.edge_weight_floor = None
    fc.edge_budget_landing = None
    fc.FindAllPath()
    assert not any("DEPRECATED" in n for n in fc._warn_notes)
    assert not any("edge budget" in n for n in fc._warn_notes)
    assert fc.edge_weight_floor is None
    import os
    import polars as pl
    type_csv = os.path.join(
        fc.allpath_folder, "src_to_tgt_allpaths_type.csv")
    assert len(pl.read_csv(type_csv)) == 2
