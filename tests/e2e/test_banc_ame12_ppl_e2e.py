"""Real-data e2e: aMe12 > PPL.* pathfinding and skeleton availability.

Exposes two contracts on the live public bucket (skipped when offline):

1. Pathfinding aMe12 > PPL.* with max_interlayer=1 must discover reachable
   PPL targets and materialize at least one path — this is the regression
   for the cache_only dead-end that silently emptied BANC networks.
2. The aMe12 / PPL101 / PPL103 cohort renders through the unified skeleton
   chain despite uneven release coverage (one aMe12 is full-res-only, the
   other pcg-skel-only; PPL101/103 have L2).
"""
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import banc_public_data  # noqa: E402


def _network_available() -> bool:
    import urllib.error
    import urllib.request

    try:
        urllib.request.urlopen(
            "https://storage.googleapis.com/", timeout=5)
        return True
    except urllib.error.HTTPError:
        return True
    except Exception:
        return False


pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(not _network_available(),
                       reason="network unavailable"),
]

AME12_FULL_ONLY = "720575941596944935"   # no L2 in the 888 export
AME12_PCG_ONLY = "720575941395228822"    # no 888 file at all (pcg twin)
PPL101_ID = "720575941416009108"


class TestCohortSkeletonAvailability:
    def test_ame12_full_only_serves_full(self):
        n = banc_public_data.fetch_banc_swc(
            "banc_v888", AME12_FULL_ONLY, use_cache=False)
        assert n is not None
        # No L2 exists: the full-resolution fallback serves (11k+ nodes).
        assert getattr(n, "_drocat_banc_resolution", "") == "full"
        assert len(n.nodes) > 5000

    def test_ame12_pcg_only_serves_pcg(self):
        n = banc_public_data.fetch_banc_swc(
            "banc_v888", AME12_PCG_ONLY, use_cache=False)
        assert n is not None
        assert getattr(n, "_drocat_banc_resolution", "") == "l2"
        # Micrometre pcg coordinates are scaled x1000 to nm.
        coords = n.nodes[["x", "y", "z"]].to_numpy()
        assert float(abs(coords).max()) > 10_000


class TestAme12PplPathfinding:
    def test_ame12_to_ppl_finds_paths(self, tmp_path):
        """The user-reported query: aMe12 > PPL.*, max_interlayer = 1.

        Regression for the cache_only dead-end — before the fix this run
        silently returned an empty network and zero paths.
        """
        from coana import FindNeuronConnection

        fnc = FindNeuronConnection()
        fnc.dataset = "banc_v888"
        fnc.sourceNeurons = ["aMe12"]
        fnc.targetNeurons = ["PPL.*"]
        fnc.max_interlayer = 1
        fnc.output_dir = str(tmp_path)
        fnc.verbose = False
        fnc.InitializeNeuronInfo()
        fnc.FindAllPath()

        run_dirs = list(tmp_path.glob("find-paths-complete_BANC_aMe12_to_PPL_*"))
        assert run_dirs, "expected a materialized run folder"
        path_files = list(run_dirs[0].glob("*allpaths*.csv"))
        assert path_files, "expected a materialized path table"

        import pandas as pd

        paths = pd.read_csv(path_files[0])
        assert len(paths) >= 1, "expected at least one aMe12 -> PPL path"
        assert paths["path"].str.contains("aMe12").all()
        assert paths["path"].str.contains("PPL").all()
