"""Regression tests for live dataset-status labels in the Settings flow."""

from nicegui import Client
from nicegui.elements.timer import Timer
from nicegui.page import page


def test_selector_status_refreshes_after_a_pull_creates_local_cache(
    monkeypatch, tmp_path
):
    import ui.dataset_service as dataset_service_module
    from ui.components.common import (
        dataset_selector,
        refresh_dataset_selector_statuses,
    )
    from ui.dataset_service import DatasetInfo, DatasetService

    service = DatasetService()
    service._datasets_dir = tmp_path / "datasets"
    service._cache_dir = tmp_path / "cache"
    dataset = "manc:v1.2.3"
    service._cache[dataset] = DatasetInfo(
        name=dataset,
        source="neuprint",
        available=True,
    )
    monkeypatch.setattr(dataset_service_module, "_dataset_service", service)

    client = Client(page("/dataset-status-selector-refresh"))
    with client:
        selector = dataset_selector(datasets=[dataset])

    assert "☁ server" in selector.options[dataset]

    cache_dir = service._cache_dir / "manc_v1_2_3"
    cache_dir.mkdir(parents=True)
    (cache_dir / "connections.parquet").touch()

    refresh_dataset_selector_statuses(service)

    assert "◐ cached" in selector.options[dataset]
    assert "☁ server" not in selector.options[dataset]

    (cache_dir / "connections.parquet").unlink()
    refresh_dataset_selector_statuses(service)

    assert "☁ server" in selector.options[dataset]
    assert "◐ cached" not in selector.options[dataset]


def test_status_card_prefers_cached_icon_when_server_is_also_available(
    monkeypatch, tmp_path
):
    import ui.dataset_service as dataset_service_module
    from ui.components.common import dataset_status_card
    from ui.dataset_service import DatasetService, dataset_to_folder

    service = DatasetService()
    service._datasets_dir = tmp_path / "datasets"
    service._cache_dir = tmp_path / "cache"
    service._availability_loaded = True  # skip disk load; keep _server_rows
    dataset = "manc:v1.2.3"
    service._server_rows = {
        dataset: {"state": "available", "checked_at": None, "metadata": {}},
    }
    # A local connection cache exists but no metadata table.
    cache_dir = service._cache_dir / dataset_to_folder(dataset)
    cache_dir.mkdir(parents=True)
    (cache_dir / "connections.parquet").touch()
    monkeypatch.setattr(dataset_service_module, "_dataset_service", service)

    client = Client(page("/dataset-status-card-priority"))
    with client:
        dataset_status_card()

    icons = {
        element._props.get("name")
        for element in client.elements.values()
        if getattr(element, "_props", {}).get("name")
    }
    assert "cached" in icons
    assert "cloud_done" not in icons


def test_settings_watcher_updates_selector_and_card_on_create_and_remove(
    monkeypatch, tmp_path
):
    import ui.dataset_service as dataset_service_module
    from ui.components.common import dataset_selector, dataset_status_card
    from ui.dataset_service import DatasetService, dataset_to_folder

    service = DatasetService()
    service._datasets_dir = tmp_path / "datasets"
    service._cache_dir = tmp_path / "cache"
    service._availability_loaded = True
    dataset = "manc:v1.2.3"
    service._server_rows = {
        dataset: {"state": "available", "checked_at": None, "metadata": {}},
    }
    monkeypatch.setattr(dataset_service_module, "_dataset_service", service)

    client = Client(page("/dataset-status-auto-watcher"))
    with client:
        selector = dataset_selector(datasets=[dataset])
        dataset_status_card()

    timer = next(
        element
        for element in client.elements.values()
        if isinstance(element, Timer)
    )
    cache_file = service._cache_dir / dataset_to_folder(dataset) / "connections.parquet"
    cache_file.parent.mkdir(parents=True)
    cache_file.touch()

    with client:
        timer.callback()

    assert "◐ cached" in selector.options[dataset]
    icons = {
        element._props.get("name")
        for element in client.elements.values()
        if getattr(element, "_props", {}).get("name")
    }
    assert "cached" in icons

    cache_file.unlink()
    with client:
        timer.callback()

    assert "☁ server" in selector.options[dataset]
    icons = {
        element._props.get("name")
        for element in client.elements.values()
        if getattr(element, "_props", {}).get("name")
    }
    assert "cloud_done" in icons
