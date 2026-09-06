"""Unit tests for the applied-threshold banner (concern 2, report-fixes plan).

The banner states the APPLIED minimal synapse threshold per dataset
(Feature G τ collapse) and is shown only when at least one dataset was
budget-bitten.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
for p in (PROJECT_ROOT, PROJECT_ROOT / "src", PROJECT_ROOT / "vispath-subproject" / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from comparison.html_report_generator import _generate_applied_threshold_banner  # noqa: E402


class _FakeAnalyzer:
    def __init__(self, run_meta, dataset_names):
        self._path_run_meta = run_meta
        self.dataset_names = list(dataset_names)
        self.parameters = type('P', (), {
            'get_dataset_names': staticmethod(lambda names=list(dataset_names): list(names))})()


def _meta(entries):
    return {(ds, t): {'tau': tau, 'budget_bitten': bitten}
            for (ds, t), (tau, bitten) in entries.items()}


def test_banner_lists_bitten_datasets():
    an = _FakeAnalyzer(_meta({
        ('male-cns:v1.0', 3): (16.0, True),
        ('male-cns:v1.0', 10): (16.0, True),
        ('flywire_FAFB_v783', 3): (12.0, True),
    }), ['male-cns:v1.0', 'flywire_FAFB_v783'])
    html = _generate_applied_threshold_banner(an, an.dataset_names)
    assert 'Applied minimal thresholds' in html
    assert '16' in html and '12' in html


def test_no_banner_when_nothing_bitten():
    an = _FakeAnalyzer(_meta({
        ('male-cns:v1.0', 3): (None, False),
        ('flywire_FAFB_v783', 3): (None, False),
    }), ['male-cns:v1.0', 'flywire_FAFB_v783'])
    assert _generate_applied_threshold_banner(an, an.dataset_names) == ''
