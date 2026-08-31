"""Tab-level query-history recording tests.

Every UI tab records its query inputs after a successful run. Neuron queries
use the neuron history store, while NeuronBridge / FlyLight driver lines use a
separate line history store. Failed or cancelled runs never record.
"""

import asyncio
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from nicegui import Client  # noqa: E402
from nicegui.page import page  # noqa: E402

import ui.history_store as hs  # noqa: E402
import ui.line_history_store as lhs  # noqa: E402
from ui.components.output_panel import OutputPanel  # noqa: E402
from ui.tabs.connectivity import (  # noqa: E402
    create_connectivity_tab,
)
from ui.tabs.flylight import create_flylight_tab  # noqa: E402
from ui.tabs.morphology import create_morphology_tab  # noqa: E402
from ui.tabs.nb_colabel import create_nb_colabel_tab  # noqa: E402
from ui.tabs.nb_find_neuron import create_nb_find_neuron_tab  # noqa: E402


@pytest.fixture
def isolated_history(tmp_path, monkeypatch):
    monkeypatch.setattr(hs, "_HISTORY_PATH", tmp_path / "neuron_history.json")
    monkeypatch.setattr(lhs, "_HISTORY_PATH", tmp_path / "line_history.json")
    return hs


def _mock_output_panel_run(monkeypatch, returncode=0):
    """Stub the panel runner and its UI-touching methods.

    The run handlers are driven through ``asyncio.run``, which runs on a
    different task than the page build; the panel methods that create
    elements or emit JavaScript therefore get no-ops so the test focuses on
    the history recording.
    """
    captured = []

    async def fake_run(self, runner, tool_name, constructor_params,
                       method_name, method_params=None, output_dir=None):
        captured.append((tool_name, constructor_params, method_params))
        return {"returncode": returncode, "files": [], "duration": 0,
                "cancelled": False, "output_folder": None,
                "neuron_match": None}

    monkeypatch.setattr(OutputPanel, "run", fake_run)
    monkeypatch.setattr(OutputPanel, "set_running", lambda self, value: None)
    monkeypatch.setattr(OutputPanel, "set_status",
                        lambda self, status, color="grey": None)
    monkeypatch.setattr(OutputPanel, "clear", lambda self: None)
    monkeypatch.setattr(OutputPanel, "show_files",
                        lambda self, files, output_dir=None: None)
    monkeypatch.setattr(OutputPanel, "log",
                        lambda self, message, level="stdout": None)
    return captured


def _chip_inputs(client):
    return [
        element for element in client.elements.values()
        if getattr(element, "chip_input", None) is not None
    ]


def _chip_input(client, label):
    return next(
        element for element in _chip_inputs(client)
        if element.chip_input._props.get("label") == label
    )


def _click_run(client, label):
    """Invoke the tab's async run handler directly (NiceGUI wraps the raw
    callback in a ``lambda _: handle_event(...)``; cell 0 of that closure
    holds the original handler)."""
    button = next(
        element for element in client.elements.values()
        if type(element).__name__ == "Button"
        and getattr(element, "text", "") == label
    )
    click = next(
        listener for listener in button._event_listeners.values()
        if listener.type == "click"
    )
    handler = click.handler.__closure__[0].cell_contents
    result = handler()
    if asyncio.iscoroutine(result):
        asyncio.run(result)


class TestComparisonHistory:
    def test_comparison_records_query_with_selected_datasets(
        self, isolated_history, monkeypatch
    ):
        captured = _mock_output_panel_run(monkeypatch)
        client = Client(page("/history-comparison"))
        with client:
            create_connectivity_tab()
        _chip_input(client, "Neurons to Compare").add_values(["aMe12", "aMe10"])
        _click_run(client, "Run Comparison")

        assert captured and captured[0][0] == "connectivity_profiling"
        assert isolated_history.recent() == ["aMe12", "aMe10"]
        assert isolated_history.datasets_of("aMe12") == ["male-cns:v1.0"]
        assert isolated_history.datasets_of("aMe10") == ["male-cns:v1.0"]

    def test_comparison_failed_run_leaves_history_untouched(
        self, isolated_history, monkeypatch
    ):
        _mock_output_panel_run(monkeypatch, returncode=1)
        client = Client(page("/history-comparison-fail"))
        with client:
            create_connectivity_tab()
        _chip_input(client, "Neurons to Compare").add_values(["aMe12"])
        _click_run(client, "Run Comparison")

        assert isolated_history.recent() == []


class TestSimilarHistory:
    def test_similar_records_each_source_with_source_dataset(
        self, isolated_history, monkeypatch
    ):
        captured = _mock_output_panel_run(monkeypatch)
        client = Client(page("/history-similar"))
        with client:
            create_connectivity_tab()
        _chip_input(client, "Source Neuron(s) (type or bodyId)").add_values(
            ["aMe12", "aMe10"]
        )
        _click_run(client, "Find Similar Neurons")

        assert captured and captured[0][0] == "find_homologs"
        assert set(isolated_history.recent()) == {"aMe12", "aMe10"}
        recorded = isolated_history.datasets_of("aMe12")
        assert recorded and recorded == isolated_history.datasets_of("aMe10")


class TestMorphologyComparisonHistory:
    def test_comparison_records_query_with_dataset(
        self, isolated_history, monkeypatch
    ):
        captured = _mock_output_panel_run(monkeypatch)
        client = Client(page("/history-morph-comparison"))
        with client:
            create_morphology_tab()
        _chip_input(client, "Neurons to Compare").add_values(
            ["aMe12", "aMe10"]
        )
        _click_run(client, "Run Comparison")

        assert captured and captured[0][0] == "morphology_comparison"
        assert isolated_history.recent() == ["aMe12", "aMe10"]
        recorded = isolated_history.datasets_of("aMe12")
        assert recorded and recorded == isolated_history.datasets_of("aMe10")

    def test_comparison_failed_run_leaves_history_untouched(
        self, isolated_history, monkeypatch
    ):
        _mock_output_panel_run(monkeypatch, returncode=1)
        client = Client(page("/history-morph-comparison-fail"))
        with client:
            create_morphology_tab()
        _chip_input(client, "Neurons to Compare").add_values(
            ["aMe12", "aMe10"]
        )
        _click_run(client, "Run Comparison")

        assert isolated_history.recent() == []


class TestColabelHistory:
    def test_colabel_records_driver_lines_without_dataset_scope(
        self, isolated_history, monkeypatch
    ):
        captured = _mock_output_panel_run(monkeypatch)
        client = Client(page("/history-colabel"))
        with client:
            create_nb_colabel_tab()
        _chip_input(client, "Driver Line Names").add_values(
            ["R10A06", "R10A07"]
        )
        _click_run(client, "Run Co-Labeling")

        assert captured and captured[0][0] == "nb_colabel"
        assert isolated_history.recent() == []
        assert lhs.recent() == ["R10A06", "R10A07"]
        assert lhs.datasets_of("R10A06") == []


class TestFindNeuronHistory:
    def test_find_neurons_records_driver_lines_in_line_history(
        self, isolated_history, monkeypatch
    ):
        captured = _mock_output_panel_run(monkeypatch)
        client = Client(page("/history-find-neuron"))
        with client:
            create_nb_find_neuron_tab()
        _chip_input(client, "Driver Line Names").add_values(
            ["VT037867", "SS01015"]
        )
        _click_run(client, "Find EM Neurons")

        assert captured and captured[0][0] == "nb_find_neuron"
        assert isolated_history.recent() == []
        assert lhs.recent() == ["VT037867", "SS01015"]


class TestFlylightHistory:
    def test_flylight_shares_global_line_history(
        self, isolated_history, monkeypatch
    ):
        captured = _mock_output_panel_run(monkeypatch)
        client = Client(page("/history-flylight"))
        with client:
            create_flylight_tab()
        _chip_input(client, "Driver line name(s)").add_values(["R10A06"])
        _click_run(client, "Download Images")

        assert captured and captured[0][0] == "flylight_download"
        assert isolated_history.recent() == []
        assert lhs.recent() == ["R10A06"]


class TestLineHistoryMenu:
    def test_driver_line_input_shows_line_history_only(
        self, isolated_history
    ):
        lhs.record(["R10A06"], now="2026-08-11T10:00:00")
        isolated_history.record(["aMe12"], now="2026-08-11T10:01:00")

        client = Client(page("/history-line-menu"))
        with client:
            from ui.components.common import neuron_list_input

            box = neuron_list_input(
                label="Driver Line Names",
                show_filter=False,
                show_upload=False,
                history_kind="line",
            )

        focus = next(
            listener for listener in box.chip_input._event_listeners.values()
            if listener.type == "focus"
        )
        box.chip_input._handle_event({"listener_id": focus.id, "args": None})
        texts = [
            element.text for element in client.elements.values()
            if getattr(element, "text", "")
        ]
        assert "R10A06" in texts
        assert "aMe12" not in texts
