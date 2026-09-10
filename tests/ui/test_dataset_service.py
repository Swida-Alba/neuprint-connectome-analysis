"""Tests for DatasetService: dataset lists, name conversions, live-list
hidden-dataset filtering, and token file precedence."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import ui.dataset_service as ds_mod
from ui.config import (
    BANC_DATASETS,
    DATASETS,
    DEFAULTS,
    FLYWIRE_DATASETS,
    NEUPRINT_DATASETS,
)
from ui.dataset_service import (
    DatasetInfo,
    DatasetService,
    dataset_to_folder,
    folder_to_dataset,
    is_banc_dataset,
    is_flywire_dataset,
)

NEUPRINT_EXPECTED = {
    "male-cns:v1.0",
    "male-cns:v0.9",
    "hemibrain:v1.2.1",
    "hemibrain:v1.1",
    "optic-lobe:v1.1",
    "optic-lobe:v1.0.1",
    "manc:v1.2.3",
    "manc:v1.2.1",
    "manc:v1.0",
    "fib19:v1.0",
    "mushroombody",
}
FLYWIRE_EXPECTED = {"flywire_FAFB_v783"}
BANC_EXPECTED = {"banc_v888", "banc_v626"}


class _PermissiveTokenManager:
    """Stand-in for the shared token manager: never refuses a token, so
    precedence tests stay hermetic (no neuprint.janelia.org calls)."""

    def neuprint_token_rejected(self, token, server=None):
        return False


@pytest.fixture(autouse=True)
def _no_network_probe(monkeypatch):
    monkeypatch.setattr(
        ds_mod, "_shared_token_manager", _PermissiveTokenManager())


class TestDatasetLists:
    def test_neuprint_lists_are_complete_and_consistent(self):
        assert set(NEUPRINT_DATASETS) == NEUPRINT_EXPECTED
        assert set(NEUPRINT_DATASETS) <= set(DATASETS)
        assert set(FLYWIRE_DATASETS) == FLYWIRE_EXPECTED
        assert set(BANC_DATASETS) == BANC_EXPECTED
        assert not set(FLYWIRE_DATASETS) & set(BANC_DATASETS)

    def test_banc_v888_not_supported_via_neuprint(self):
        # The NeuPrint server lists banc:v888 as hidden and not queryable;
        # BANC is served by its own public-release source instead.
        assert "banc:v888" not in NEUPRINT_DATASETS
        assert "banc:v888" not in DATASETS
        assert "banc:v888" not in DatasetService.NEUPRINT_CANDIDATES
        assert "banc_v888" in DATASETS

    def test_defaults_contain_core_parameters(self):
        for key in ("min_synapse_num", "min_ratio", "min_traversal_probability",
                    "max_interlayer", "output_format"):
            assert key in DEFAULTS


class TestDatasetNameConversion:
    def test_roundtrip_neuprint(self):
        for ds in NEUPRINT_EXPECTED:
            assert folder_to_dataset(dataset_to_folder(ds)) == ds

    def test_flywire_passthrough(self):
        for ds in FLYWIRE_EXPECTED:
            assert dataset_to_folder(ds) == ds
            assert folder_to_dataset(ds) == ds

    def test_examples(self):
        assert folder_to_dataset("hemibrain_v1_2_1") == "hemibrain:v1.2.1"
        assert folder_to_dataset("male-cns_v0_9") == "male-cns:v0.9"
        assert folder_to_dataset("manc_v1_2_3") == "manc:v1.2.3"
        assert dataset_to_folder("manc:v1.2.3") == "manc_v1_2_3"
        assert dataset_to_folder("hemibrain:v1.2.1") == "hemibrain_v1_2_1"

    def test_legacy_banc_alias_uses_standalone_folder(self):
        assert dataset_to_folder("flywire_BANC_v888") == "banc_v888"
        assert folder_to_dataset("flywire_BANC_v888") == "banc_v888"
        assert dataset_to_folder("banc:v888") == "banc_v888"


class TestIsFlywireDataset:
    def test_positive(self):
        for ds in list(FLYWIRE_EXPECTED) + ["fafb", "flywire_fafb:v783"]:
            assert is_flywire_dataset(ds)

    def test_negative(self):
        # BANC must never be classified as FAFB/FlyWire.
        for ds in ["banc_v626", "banc_v888", "flywire_BANC_v888",
                   "banc:v888", "male-cns:v0.9", "hemibrain:v1.2.1",
                   "manc:v1.0"]:
            assert not is_flywire_dataset(ds)
            if "banc" in ds.lower():
                assert is_banc_dataset(ds)


class TestHiddenDatasetFilter:
    """The live /api/dbmeta/datasets listing is filtered: hidden entries such
    as banc:v888 are listed by the server but are not queryable."""

    SERVER_META = {
        "banc:v888": {"hidden": "True", "description": "BANC"},
        "male-cns:v0.9": {"hidden": "False", "description": "MaleCNS"},
        "mushroombody": {"hidden": "False"},
        "hemibrain:v1.2.1": {},
    }

    class _FakeResponse:
        status_code = 200

        def __init__(self, payload):
            self._payload = payload

        def json(self):
            return self._payload

    def test_hidden_datasets_excluded_from_available(self, monkeypatch):
        monkeypatch.setattr(
            "requests.get",
            lambda *a, **k: self._FakeResponse(dict(self.SERVER_META)),
        )
        svc = DatasetService()
        svc._token = "tok"
        svc._cave_token = "cav"
        available = svc.fetch_neuprint_datasets()
        assert "banc:v888" not in available
        assert available == ["hemibrain:v1.2.1", "male-cns:v0.9", "mushroombody"]

    def test_full_metadata_still_stored(self, monkeypatch):
        monkeypatch.setattr(
            "requests.get",
            lambda *a, **k: self._FakeResponse(dict(self.SERVER_META)),
        )
        svc = DatasetService()
        svc._token = "tok"
        svc._cave_token = "cav"
        svc.fetch_neuprint_datasets()
        assert "banc:v888" in svc._server_datasets  # kept for status display

    def test_api_failure_falls_back_to_candidates(self, monkeypatch):
        def boom(*a, **k):
            raise OSError("network down")

        monkeypatch.setattr("requests.get", boom)
        svc = DatasetService()
        svc._token = "tok"
        svc._cave_token = "cav"
        monkeypatch.setattr(
            svc, "_probe_neuprint_dataset",
            lambda ds: type("Info", (), {"available": True})(),
        )
        available = svc.fetch_neuprint_datasets()
        assert set(available) == NEUPRINT_EXPECTED

    def test_probe_marks_hidden_dataset_unavailable(self):
        svc = DatasetService()
        svc._token = "tok"
        svc._cave_token = "cav"
        svc._server_datasets = dict(self.SERVER_META)

        hidden = svc._probe_neuprint_dataset("banc:v888")
        assert hidden.available is False
        assert hidden.error

        visible = svc._probe_neuprint_dataset("male-cns:v0.9")
        assert visible.available is True
        assert visible.error is None


class TestTokenConfigJson:
    """_load_tokens reads the tokens section of config.json (the only
    token file); placeholders and empty values are treated as unset."""

    def _svc(self, monkeypatch, tmp_path, config=None):
        if config is not None:
            (tmp_path / "config.json").write_text(config, encoding="utf-8")
        monkeypatch.setattr(ds_mod, "PROJECT_ROOT", tmp_path)
        return DatasetService()

    def test_reads_both_tokens(self, monkeypatch, tmp_path):
        svc = self._svc(
            monkeypatch, tmp_path,
            '{"tokens": {"neuprint": "cfg-np", "cave": "cfg-cave"}}\n',
        )
        assert svc.get_token() == "cfg-np"
        assert svc.get_cave_token() == "cfg-cave"

    def test_empty_value_is_unset(self, monkeypatch, tmp_path):
        svc = self._svc(
            monkeypatch, tmp_path,
            '{"tokens": {"neuprint": "", "cave": "cfg-cave"}}\n',
        )
        monkeypatch.delenv("NEUPRINT_APPLICATION_CREDENTIALS", raising=False)
        monkeypatch.delenv("NEUPRINT_TOKEN", raising=False)
        assert svc.get_token() is None
        assert svc.get_cave_token() == "cfg-cave"

    def test_placeholder_ignored(self, monkeypatch, tmp_path):
        svc = self._svc(
            monkeypatch, tmp_path,
            '{"tokens": {"neuprint": "YOUR_NEUPRINT_TOKEN_HERE"}}\n',
        )
        monkeypatch.delenv("NEUPRINT_APPLICATION_CREDENTIALS", raising=False)
        monkeypatch.delenv("NEUPRINT_TOKEN", raising=False)
        assert svc.get_token() is None

    def test_legacy_token_files_ignored(self, monkeypatch, tmp_path):
        (tmp_path / "token_info_local.txt").write_text(
            "NEUPRINT_TOKEN='legacy-tok'\n", encoding="utf-8"
        )
        svc = self._svc(monkeypatch, tmp_path, None)
        monkeypatch.delenv("NEUPRINT_APPLICATION_CREDENTIALS", raising=False)
        monkeypatch.delenv("NEUPRINT_TOKEN", raising=False)
        assert svc.get_token() is None

    def test_config_json_wins_over_config_local(self, monkeypatch, tmp_path):
        (tmp_path / "config.json").write_text(
            '{"tokens": {"neuprint": "cfg-np", "cave": "cfg-cave"}}\n',
            encoding="utf-8",
        )
        (tmp_path / "config_local.json").write_text(
            '{"tokens": {"neuprint": "local-np"}}\n', encoding="utf-8"
        )
        svc = self._svc(monkeypatch, tmp_path, None)
        assert svc.get_token() == "cfg-np"
        assert svc.get_cave_token() == "cfg-cave"
    
    def test_config_local_fills_empty_config_json_entry(self, monkeypatch, tmp_path):
        (tmp_path / "config.json").write_text(
            '{"tokens": {"neuprint": "", "cave": "cfg-cave"}}\n',
            encoding="utf-8",
        )
        (tmp_path / "config_local.json").write_text(
            '{"tokens": {"neuprint": "local-np"}}\n', encoding="utf-8"
        )
        svc = self._svc(monkeypatch, tmp_path, None)
        assert svc.get_token() == "local-np"
        assert svc.get_cave_token() == "cfg-cave"

    def test_no_config_returns_none(self, monkeypatch, tmp_path):
        svc = self._svc(monkeypatch, tmp_path, None)
        monkeypatch.delenv("NEUPRINT_APPLICATION_CREDENTIALS", raising=False)
        monkeypatch.delenv("NEUPRINT_TOKEN", raising=False)
        monkeypatch.delenv("CAVE_TOKEN", raising=False)
        assert svc.get_token() is None
        assert svc.get_cave_token() is None

    def test_env_fallback_when_no_config(self, monkeypatch, tmp_path):
        """No config files: the environment is the last chain link."""
        svc = self._svc(monkeypatch, tmp_path, None)
        monkeypatch.setenv("NEUPRINT_APPLICATION_CREDENTIALS", "env-np")
        monkeypatch.setenv("CAVE_TOKEN", "env-cave")
        try:
            assert svc.get_token() == "env-np"
            assert svc.get_cave_token() == "env-cave"
        finally:
            monkeypatch.delenv("NEUPRINT_APPLICATION_CREDENTIALS")
            monkeypatch.delenv("CAVE_TOKEN")

    def test_config_update_overrides_env(self, monkeypatch, tmp_path):
        """A config token wins over a shell-exported env var."""
        svc = self._svc(
            monkeypatch, tmp_path,
            '{"tokens": {"neuprint": "cfg-np"}}\n',
        )
        monkeypatch.setenv("NEUPRINT_APPLICATION_CREDENTIALS", "env-np")
        try:
            assert svc.get_token() == "cfg-np"
        finally:
            monkeypatch.delenv("NEUPRINT_APPLICATION_CREDENTIALS")

    def test_config_update_overrides_env_even_if_env_var_differs(
            self, monkeypatch, tmp_path):
        """Both config entries precede the env var in the chain."""
        (tmp_path / "config.json").write_text(
            '{"tokens": {"neuprint": "cfg-np"}}\n', encoding="utf-8")
        (tmp_path / "config_local.json").write_text(
            '{"tokens": {"neuprint": "local-np"}}\n', encoding="utf-8")
        monkeypatch.setattr(ds_mod, "PROJECT_ROOT", tmp_path)
        monkeypatch.setenv("NEUPRINT_APPLICATION_CREDENTIALS", "env-np")
        try:
            svc = DatasetService()
            assert svc.get_token() == "cfg-np"
        finally:
            monkeypatch.delenv("NEUPRINT_APPLICATION_CREDENTIALS")


class TestSkipInvalidTokenChain:
    """A NeuPrint-rejected candidate is skipped to the next location
    instead of failing the request."""

    def _svc(self, monkeypatch, tmp_path, config):
        (tmp_path / "config.json").write_text(config, encoding="utf-8")
        monkeypatch.setattr(ds_mod, "PROJECT_ROOT", tmp_path)
        monkeypatch.delenv("NEUPRINT_APPLICATION_CREDENTIALS", raising=False)
        monkeypatch.delenv("NEUPRINT_TOKEN", raising=False)
        return DatasetService()

    @staticmethod
    def _probe_rejecting(*rejected):
        class _Fake:
            def neuprint_token_rejected(self, token, server=None):
                return token in rejected
        return _Fake()

    def test_rejected_config_token_falls_back_to_env(
            self, monkeypatch, tmp_path):
        svc = self._svc(
            monkeypatch, tmp_path, '{"tokens": {"neuprint": "revoked-tok"}}\n')
        monkeypatch.setenv("NEUPRINT_APPLICATION_CREDENTIALS", "env-tok")
        monkeypatch.setattr(
            ds_mod, "_shared_token_manager", self._probe_rejecting("revoked-tok"))
        try:
            assert svc.get_token() == "env-tok"
        finally:
            monkeypatch.delenv("NEUPRINT_APPLICATION_CREDENTIALS")

    def test_rejected_config_json_falls_back_to_config_local(
            self, monkeypatch, tmp_path):
        (tmp_path / "config.json").write_text(
            '{"tokens": {"neuprint": "revoked-tok"}}\n', encoding="utf-8")
        (tmp_path / "config_local.json").write_text(
            '{"tokens": {"neuprint": "fresh-tok"}}\n', encoding="utf-8")
        monkeypatch.setattr(ds_mod, "PROJECT_ROOT", tmp_path)
        monkeypatch.delenv("NEUPRINT_APPLICATION_CREDENTIALS", raising=False)
        monkeypatch.delenv("NEUPRINT_TOKEN", raising=False)
        monkeypatch.setattr(
            ds_mod, "_shared_token_manager", self._probe_rejecting("revoked-tok"))
        assert DatasetService().get_token() == "fresh-tok"

    def test_all_candidates_rejected_returns_first(
            self, monkeypatch, tmp_path):
        svc = self._svc(
            monkeypatch, tmp_path, '{"tokens": {"neuprint": "revoked-tok"}}\n')
        monkeypatch.setenv("NEUPRINT_TOKEN", "also-revoked")
        monkeypatch.setattr(
            ds_mod, "_shared_token_manager", self._probe_rejecting(
                "revoked-tok", "also-revoked"))
        try:
            assert svc.get_token() == "revoked-tok"
        finally:
            monkeypatch.delenv("NEUPRINT_TOKEN")

    def test_single_candidate_used_without_probe(
            self, monkeypatch, tmp_path):
        """With one candidate there is nowhere to fall back, so no probe
        runs and the token is returned as-is."""
        probe_calls = []

        class _Counting:
            def neuprint_token_rejected(self, token, server=None):
                probe_calls.append(token)
                return True

        svc = self._svc(monkeypatch, tmp_path, '{"tokens": {}}\n')
        svc._token = "revoked-tok"
        svc._cave_token = "cave"
        monkeypatch.setattr(ds_mod, "_shared_token_manager", _Counting())
        assert svc.get_token() == "revoked-tok"
        assert probe_calls == []


class TestBancFallbackCounts:
    """BANC last-resort display counts must track the release-count
    contract: the 2026-09-04 public-bucket snapshot (v888 as served,
    v626 after root_626 dedup). When the local prepared tables are
    present, the constants must equal their actual row counts so the
    UI never shows a stale no-data estimate."""

    RELEASE_CONTRACT = {"banc_v888": 188508, "banc_v626": 185165}

    def test_constants_match_release_contract(self):
        for dataset, expected in self.RELEASE_CONTRACT.items():
            entry = DatasetService.CODEX_DATASETS.get(dataset) or {}
            assert entry.get("neurons") == expected, (
                f"{dataset} fallback count drifted from the release "
                f"contract ({expected}); refresh it from the current "
                "prepared tables and update the comment in "
                "ui/dataset_service.py")

    def test_constants_match_local_prepared_tables(self):
        pytest.importorskip("pandas")
        import pandas as pd

        project_root = Path(__file__).resolve().parents[2]
        checked = 0
        for dataset, expected in self.RELEASE_CONTRACT.items():
            table = (project_root / "datasets" / dataset /
                     f"{dataset}_allneurons_neuron_df.parquet")
            if not table.exists():
                continue
            actual = len(pd.read_parquet(table, columns=["bodyId"]))
            assert actual == expected, (
                f"{dataset}: fallback constant {expected} != prepared "
                f"table rows {actual}; refresh both together")
            checked += 1
        if not checked:
            pytest.skip("no local prepared BANC tables")


class TestSnapshotNameCanonicalization:
    """A persisted availability file written before the ``flywire_BANC_*``
    -> ``banc_*`` rename must not surface a legacy name or a ``flywire``
    source for a standalone BANC release."""

    def test_legacy_v1_local_rows_are_dropped_and_rederived(self, monkeypatch, tmp_path):
        """A v1 file's local rows must not surface; only server rows survive,
        and local state is re-derived from disk."""
        import json

        snapshot_path = tmp_path / "dataset_availability.json"
        snapshot_path.write_text(json.dumps({
            "format": "drocat_dataset_availability/v1",
            "updated_at": "2026-08-18T00:11:08+08:00",
            "datasets": {
                "flywire_BANC_v888": {
                    "name": "flywire_BANC_v888", "source": "flywire",
                    "available": True, "local_prepared": True,
                    "neuron_count": 158262,
                },
                "hemibrain:v1.2.1": {
                    "name": "hemibrain:v1.2.1", "source": "neuprint",
                    "available": True,
                },
            },
        }), encoding="utf-8")

        svc = DatasetService()
        monkeypatch.setattr(
            DatasetService, "availability_cache_path",
            property(lambda self: snapshot_path))
        # Hermetic: keep the legacy row out of the row set entirely; the
        # catalog still exposes the canonical local release.
        monkeypatch.setattr(DatasetService, "get_local_datasets", lambda self: [])
        svc._load_persisted_availability()

        # The legacy local row is dropped; the server row survives.
        assert "flywire_BANC_v888" not in svc._server_rows
        assert svc._server_rows["hemibrain:v1.2.1"]["state"] == "available"

    def test_local_release_row_is_rederived_not_frozen(self, monkeypatch):
        """A local-release snapshot row must not win over the live local
        check, or a re-prepared release would show stale counts."""
        import json

        stale = {
            "banc_v626": {
                "name": "banc_v626", "source": "flywire",
                "available": True, "local_prepared": True,
                "neuron_count": 115151, "typed_count": 115151,
            },
        }
        # A v1 local row is discarded outright by the server-row reader.
        rows = DatasetService._read_server_rows({"datasets": stale})
        assert "banc_v626" not in rows

        # And the composed row derives current counts from disk.
        svc = DatasetService()
        info = svc._compose("banc_v626")
        assert info.family == "banc"
        assert info.source == "banc"
        # The stale 115151/115151 must never come from a persisted local row.
        assert info.neuron_count != 115151 or info.typed_count != 115151


class TestSourceVocabulary:
    """``DatasetInfo.source`` keeps its established values ('neuprint' /
    'flywire' / 'banc'); ``family`` carries the precise 'fafb' spelling."""

    def test_source_of(self):
        assert DatasetService.source_of("hemibrain:v1.2.1") == "neuprint"
        assert DatasetService.source_of("flywire_FAFB_v783") == "flywire"
        assert DatasetService.source_of("banc_v888") == "banc"

    def test_composed_sources(self, tmp_path):
        svc = DatasetService()
        svc._datasets_dir = tmp_path / "datasets"
        svc._cache_dir = tmp_path / "cache"
        svc._index_dir = tmp_path / "neuron_indexes"
        svc._availability_loaded = True
        assert svc._compose("flywire_FAFB_v783").source == "flywire"
        assert svc._compose("flywire_FAFB_v783").family == "fafb"
        assert svc._compose("banc_v888").source == "banc"
        assert svc._compose("hemibrain:v1.2.1").source == "neuprint"


class TestCatalogComposition:
    """The composed catalog must list every known dataset on a fresh machine
    (no file, no token, empty ``datasets/``), not just the local releases."""

    def test_fresh_machine_lists_neuprint_candidates_and_locals(self, tmp_path):
        svc = DatasetService()
        svc._cache_dir = tmp_path / "cache"      # no availability file
        svc._datasets_dir = tmp_path / "datasets"  # does not exist
        svc._index_dir = tmp_path / "neuron_indexes"
        results, _updated = svc.get_cached_availability()

        # Every NeuPrint candidate is listed, even though none is checked.
        for cand in svc.NEUPRINT_CANDIDATES:
            assert cand in results, f"{cand} missing from fresh-machine catalog"
        # Local releases are listed too.
        assert "flywire_FAFB_v783" in results
        assert "banc_v888" in results

        # No network was attempted, so every server state is 'unknown' and
        # nothing is falsely reported available.
        from ui.dataset_service import SERVER_UNKNOWN
        for info in results.values():
            assert info.server_state == SERVER_UNKNOWN
            assert info.available is False
        # A download-required release is not analyzable without local tables.
        assert results["banc_v888"].access_mode == "download_required"
        assert results["hemibrain:v1.2.1"].access_mode == "streaming"


class TestDimensionStatus:
    """The four-dimensional status model: server / metadata / connectivity /
    visualization, plus access mode."""

    def _svc(self, tmp_path):
        svc = DatasetService()
        svc._datasets_dir = tmp_path / "datasets"
        svc._cache_dir = tmp_path / "cache"
        svc._index_dir = tmp_path / "neuron_indexes"
        return svc

    def test_families_and_access_modes(self):
        from ui.dataset_service import ACCESS_DOWNLOAD_REQUIRED, ACCESS_STREAMING
        assert DatasetService.family_of("hemibrain:v1.2.1") == "neuprint"
        assert DatasetService.family_of("flywire_FAFB_v783") == "fafb"
        assert DatasetService.family_of("banc_v888") == "banc"
        assert DatasetService.access_mode_of("hemibrain:v1.2.1") == ACCESS_STREAMING
        assert DatasetService.access_mode_of("banc_v888") == ACCESS_DOWNLOAD_REQUIRED
        assert DatasetService.access_mode_of("flywire_FAFB_v783") == \
            ACCESS_DOWNLOAD_REQUIRED

    def test_banc_not_available_until_local_tables_exist(self, tmp_path):
        """BANC is download-required: server reachability alone is not ready."""
        from ui.dataset_service import CAP_MISSING, CAP_READY, SERVER_AVAILABLE
        svc = self._svc(tmp_path)
        svc._server_rows = {"banc_v888": {
            "state": SERVER_AVAILABLE, "checked_at": None, "metadata": {}}}
        svc._availability_loaded = True

        info = svc.check_dataset_availability("banc_v888")
        assert info.access_mode == "download_required"
        assert info.server_state == SERVER_AVAILABLE
        assert info.connectivity_state == CAP_MISSING
        assert info.available is False  # reachable but not analyzable

        # Now create the local tables.
        ds_dir = tmp_path / "datasets" / "banc_v888"
        ds_dir.mkdir(parents=True)
        (ds_dir / "banc_v888_allneurons_neuron_df.parquet").write_bytes(b"x")
        (ds_dir / "banc_v888_merged_connections.parquet").write_bytes(b"x")
        info2 = svc.check_dataset_availability("banc_v888")
        assert info2.connectivity_state == CAP_READY
        assert info2.metadata_state == CAP_READY
        assert info2.available is True

    def test_banc_available_requires_both_tables(self, tmp_path):
        """One table alone must not make a release analyzable."""
        from ui.dataset_service import CAP_MISSING, CAP_READY
        svc = self._svc(tmp_path)
        svc._availability_loaded = True
        ds_dir = tmp_path / "datasets" / "banc_v888"
        ds_dir.mkdir(parents=True)

        # Connections only -> not available (the neuron table is missing).
        (ds_dir / "banc_v888_merged_connections.parquet").write_bytes(b"x")
        info = svc.check_dataset_availability("banc_v888")
        assert info.connectivity_state == CAP_READY
        assert info.metadata_state == CAP_MISSING
        assert info.available is False
        assert info.error

        # Neuron table only -> not available either.
        (ds_dir / "banc_v888_merged_connections.parquet").unlink()
        (ds_dir / "banc_v888_allneurons_neuron_df.parquet").write_bytes(b"x")
        info = svc.check_dataset_availability("banc_v888")
        assert info.metadata_state == CAP_READY
        assert info.connectivity_state == CAP_MISSING
        assert info.available is False

        # Both -> available.
        (ds_dir / "banc_v888_merged_connections.parquet").write_bytes(b"x")
        assert svc.check_dataset_availability("banc_v888").available is True

    def test_neuprint_metadata_requires_table_roi_and_index(self, tmp_path):
        """Metadata readiness = neuron table + ROI table + neuron index."""
        from ui.dataset_service import CAP_MISSING, CAP_PARTIAL, CAP_READY
        svc = self._svc(tmp_path)
        ds_dir = tmp_path / "datasets" / "hemibrain_v1_2_1"
        ds_dir.mkdir(parents=True)
        (ds_dir / "hemibrain_v1_2_1_allneurons_neuron_df.parquet").write_bytes(b"x")
        assert svc.probe_metadata("hemibrain:v1.2.1") == CAP_PARTIAL

        (ds_dir / "hemibrain_v1_2_1_allneurons_roi_count_df.parquet").write_bytes(b"x")
        assert svc.probe_metadata("hemibrain:v1.2.1") == CAP_PARTIAL

        idx = tmp_path / "neuron_indexes" / "hemibrain_v1_2_1"
        idx.mkdir(parents=True)
        (idx / "neuron_index.parquet").write_bytes(b"x")
        assert svc.probe_metadata("hemibrain:v1.2.1") == CAP_READY

    def test_neuprint_connectivity_streams_without_cache(self, tmp_path):
        from ui.dataset_service import CAP_ON_DEMAND, CAP_READY
        svc = self._svc(tmp_path)
        assert svc.probe_connectivity("hemibrain:v1.2.1") == CAP_ON_DEMAND
        cache_conn = tmp_path / "cache" / "hemibrain_v1_2_1" / "connections.parquet"
        cache_conn.parent.mkdir(parents=True)
        cache_conn.touch()
        assert svc.probe_connectivity("hemibrain:v1.2.1") == CAP_READY

    def test_fafb_visualization_local_zip_vs_cave(self, tmp_path, monkeypatch):
        from ui.dataset_service import CAP_MISSING, CAP_ON_DEMAND, CAP_READY
        svc = self._svc(tmp_path)
        monkeypatch.setattr(DatasetService, "get_cave_token", lambda self: None)
        assert svc.probe_visualization("flywire_FAFB_v783")[0] == CAP_MISSING

        monkeypatch.setattr(DatasetService, "get_cave_token", lambda self: "cav")
        state, source = svc.probe_visualization("flywire_FAFB_v783")
        assert (state, source) == (CAP_ON_DEMAND, "cave")

        ds_dir = tmp_path / "datasets" / "flywire_FAFB_v783"
        ds_dir.mkdir(parents=True)
        (ds_dir / "sk_lod1_783_healed.zip").write_bytes(b"x")
        state, source = svc.probe_visualization("flywire_FAFB_v783")
        assert (state, source) == (CAP_READY, "local")

    def test_banc_visualization_on_demand_from_bucket(self, tmp_path):
        from ui.dataset_service import CAP_ON_DEMAND, CAP_READY
        svc = self._svc(tmp_path)
        assert svc.probe_visualization("banc_v888") == (CAP_ON_DEMAND, "bucket")
        skel = (tmp_path / "cache" / "banc_v888" / "skeletons" / "raw_skeletons")
        skel.mkdir(parents=True)
        (skel / "1.swc.zst").write_bytes(b"x")
        assert svc.probe_visualization("banc_v888") == (CAP_READY, "local")


class TestServerStates:
    """Server failures are classified, not collapsed to one bool."""

    def _svc(self, tmp_path):
        svc = DatasetService()
        svc._datasets_dir = tmp_path / "datasets"
        svc._cache_dir = tmp_path / "cache"
        svc._index_dir = tmp_path / "neuron_indexes"
        return svc

    def test_neuprint_no_token(self, tmp_path, monkeypatch):
        from ui.dataset_service import SERVER_NO_TOKEN
        svc = self._svc(tmp_path)
        # Force the no-token branch (a developer config may carry a token).
        monkeypatch.setattr(DatasetService, "get_token", lambda self: None)
        assert svc._probe_neuprint_server("hemibrain:v1.2.1")["state"] == SERVER_NO_TOKEN

    def test_neuprint_hidden(self, tmp_path, monkeypatch):
        from ui.dataset_service import SERVER_HIDDEN
        svc = self._svc(tmp_path)
        monkeypatch.setattr(DatasetService, "get_token", lambda self: "tok")
        svc._server_datasets = {"banc:v888": {"hidden": "true"}}
        assert svc._probe_neuprint_server("banc:v888")["state"] == SERVER_HIDDEN

    def test_neuprint_unreachable_vs_timeout(self, tmp_path, monkeypatch):
        from ui.dataset_service import SERVER_TIMEOUT, SERVER_UNREACHABLE
        svc = self._svc(tmp_path)
        monkeypatch.setattr(DatasetService, "get_token", lambda self: "tok")
        svc._server_datasets = {}

        def _raise(exc):
            def _inner(dataset, with_state=False):
                return (0, 0, exc) if with_state else (0, 0)
            return _inner

        monkeypatch.setattr(svc, "_fetch_neuprint_counts",
                            _raise(SERVER_TIMEOUT))
        assert svc._probe_neuprint_server("fib19:v1.0")["state"] == SERVER_TIMEOUT

        monkeypatch.setattr(svc, "_fetch_neuprint_counts",
                            _raise(SERVER_UNREACHABLE))
        assert svc._probe_neuprint_server("fib19:v1.0")["state"] == SERVER_UNREACHABLE

    def test_fafb_server_state_tracks_cave_token(self, tmp_path, monkeypatch):
        from ui.dataset_service import SERVER_AVAILABLE, SERVER_NO_TOKEN
        svc = self._svc(tmp_path)
        monkeypatch.setattr(DatasetService, "get_cave_token", lambda self: None)
        assert svc._probe_server("flywire_FAFB_v783")["state"] == SERVER_NO_TOKEN
        monkeypatch.setattr(DatasetService, "get_cave_token", lambda self: "cav")
        assert svc._probe_server("flywire_FAFB_v783")["state"] == SERVER_AVAILABLE

    def test_banc_bucket_probe_states(self, tmp_path, monkeypatch):
        from ui.dataset_service import SERVER_AVAILABLE, SERVER_UNREACHABLE
        import banc_public_data
        svc = self._svc(tmp_path)

        monkeypatch.setattr(banc_public_data, "_remote_size", lambda url: 123)
        assert svc._probe_banc_bucket()["state"] == SERVER_AVAILABLE

        svc._banc_bucket_probe_cache = None
        monkeypatch.setattr(banc_public_data, "_remote_size", lambda url: None)
        assert svc._probe_banc_bucket()["state"] == SERVER_UNREACHABLE

    def test_banc_bucket_probe_timeout(self, tmp_path, monkeypatch):
        """A socket timeout is classified as timeout, not unreachable."""
        from ui.dataset_service import SERVER_TIMEOUT
        import banc_public_data
        svc = self._svc(tmp_path)

        def _boom(url):
            raise TimeoutError("timed out")

        monkeypatch.setattr(banc_public_data, "_remote_size", _boom)
        assert svc._probe_banc_bucket()["state"] == SERVER_TIMEOUT
