"""BANC documentation example-ID membership (BANC-DOC-008).

Every 19-digit BANC bodyId shown in a BANC-scoped example block of the
integration and FlyWire guides must exist in the prepared neuron table
of the release the example targets, so copy-paste examples cannot rot
when the bucket snapshot is refreshed. Skips when the local prepared
tables are absent.
"""

import re
from pathlib import Path

import pytest

pytest.importorskip("pandas")

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 19-digit BANC-style bodyIds appear only inside example code blocks.
BODY_ID_RE = re.compile(r"\b(7205759[0-9]{11})\b")

# BANC bodyIds share the 72057594 prefix family with FAFB root IDs; a
# block is BANC-scoped only when its dataset= assignment names a banc
# dataset, so FAFB examples in the same guides are not policed here.
DATASET_ASSIGN_RE = re.compile(r"dataset\s*=\s*['\"]([^'\"]+)['\"]")

DOCS = ["docs/BANC_INTEGRATION.md", "docs/FLYWIRE_USAGE.md"]


def _release_of(dataset_token):
    """Map a dataset= token to its prepared-table release, or None."""
    if not dataset_token.lower().startswith("banc"):
        return None
    digits = re.search(r"v?(\d+)", dataset_token)
    return f"banc_v{digits.group(1)}" if digits else None


def _banc_blocks(rel):
    """Yield (release, block) for every fenced code block whose dataset=
    assignment names a BANC release."""
    text = (PROJECT_ROOT / rel).read_text(encoding="utf-8")
    for block in re.findall(r"```[a-z]*\n(.*?)```", text, re.DOTALL):
        assignment = DATASET_ASSIGN_RE.search(block)
        if not assignment:
            continue
        release = _release_of(assignment.group(1))
        if release:
            yield release, block


def _prepared_ids(dataset):
    table = (PROJECT_ROOT / "datasets" / dataset /
             f"{dataset}_allneurons_neuron_df.parquet")
    if not table.exists():
        pytest.skip(f"{table} not prepared")
    import pandas as pd
    return set(pd.read_parquet(table, columns=["bodyId"])["bodyId"]
               .astype(str))


@pytest.mark.parametrize("rel", DOCS)
def test_documented_banc_body_ids_exist(rel):
    checked = []
    known = {}
    for release, block in _banc_blocks(rel):
        ids = BODY_ID_RE.findall(block)
        if not ids:
            continue
        if release not in known:
            known[release] = _prepared_ids(release)
        absent = [bid for bid in set(ids) if bid not in known[release]]
        assert not absent, (
            f"{rel} documents bodyIds absent from the current {release} "
            f"table: {absent}. Replace them with IDs from the current "
            "prepared artifact and re-run the BANC pathfinding E2E test.")
        checked.extend(ids)
    assert checked, f"{rel} no longer carries a BANC bodyId example"
