"""UI-side standalone-BANC categorization regressions.

- ``progress_steps_for``: a BANC find-similar run resolves to the backend's
  4-step cache-direct checklist (it used to get the 6-step ROI checklist via
  a stale ``startswith("flywire_")`` test).
- The Dataset Availability card must render BANC's own source badge from the
  resolved ``DatasetInfo.source`` — ``banc_v888`` used to render a NeuPrint or
  FlyWire badge depending on the call site.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nicegui import Client  # noqa: E402
from nicegui.page import page  # noqa: E402

from ui.components.page_progress import (  # noqa: E402
    METHOD_PROGRESS_STEPS,
    progress_steps_for,
)


# ---------------------------------------------------------------------------
# find_similar_morphology progress protocol
# ---------------------------------------------------------------------------

def _steps(source_key):
    return list(METHOD_PROGRESS_STEPS[("find_similar_morphology", source_key)])


def test_find_similar_progress_maps_banc_to_cache_protocol():
    steps = progress_steps_for(
        "find_similar_morphology", context={"dataset": "banc_v888"})
    assert steps == _steps("cache")
    assert len(steps) == 4


def test_find_similar_progress_maps_legacy_banc_name_identically():
    steps = progress_steps_for(
        "find_similar_morphology", context={"dataset": "flywire_BANC_v888"})
    assert steps == _steps("cache")


def test_find_similar_progress_keeps_fafb_and_neuprint_protocols():
    fafb = progress_steps_for(
        "find_similar_morphology", context={"dataset": "flywire_FAFB_v783"})
    assert fafb == _steps("cache")
    roi = progress_steps_for(
        "find_similar_morphology", context={"dataset": "male-cns:v1.0"})
    assert roi == _steps("roi")
    assert len(roi) == 6


# ---------------------------------------------------------------------------
# Dataset Availability source badge
# ---------------------------------------------------------------------------

def _badge_texts(client):
    return [
        el.text
        for el in client.elements.values()
        if type(el).__name__ == "Badge" and getattr(el, "text", "")
    ]


def test_availability_badge_labels_banc_as_standalone_source(monkeypatch):
    import ui.dataset_service as ds_mod
    from ui.components.common import dataset_status_card
    from ui.dataset_service import DatasetInfo

    class _FakeService:
        def get_cached_availability(self):
            banc = DatasetInfo(
                name="banc_v888", source="banc", local_prepared=True,
                display_name="BANC v888")
            neuprint = DatasetInfo(
                name="male-cns:v1.0", source="neuprint", available=True)
            return {"banc_v888": banc, "male-cns:v1.0": neuprint}, None

    monkeypatch.setattr(ds_mod, "get_dataset_service", _FakeService)

    client = Client(page("/banc-availability-badge"))
    with client:
        dataset_status_card()

    texts = _badge_texts(client)
    assert "BANC" in texts, texts
    assert "NeuPrint" in texts, texts
    assert texts.index("BANC") < texts.index("NeuPrint")
