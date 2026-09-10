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
    """A persisted availability snapshot written before the
    ``flywire_BANC_*`` -> ``banc_*`` rename must not surface a legacy name
    or a ``flywire`` source for a standalone BANC release."""

    def test_deserialize_canonicalizes_legacy_banc_name(self):
        info = DatasetService._deserialize_info(
            "flywire_BANC_v888",
            {"name": "flywire_BANC_v888", "source": "flywire",
             "available": True, "neuron_count": 158262},
        )
        assert info.name == "banc_v888"
        assert info.source == "banc"
        assert info.neuron_count == 158262

    def test_deserialize_keeps_fafb_and_neuprint_sources(self):
        fafb = DatasetService._deserialize_info(
            "flywire_FAFB_v783",
            {"name": "flywire_FAFB_v783", "source": "flywire"})
        assert fafb.name == "flywire_FAFB_v783"
        assert fafb.source == "flywire"

        np_info = DatasetService._deserialize_info(
            "hemibrain:v1.2.1",
            {"name": "hemibrain:v1.2.1", "source": "neuprint"})
        assert np_info.name == "hemibrain:v1.2.1"
        assert np_info.source == "neuprint"

    def test_persisted_snapshot_self_heals_on_load(self, monkeypatch, tmp_path):
        import json

        snapshot_path = tmp_path / "dataset_availability.json"
        snapshot_path.write_text(json.dumps({
            "format": DatasetService.AVAILABILITY_CACHE_FORMAT,
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
        # A local release is re-derived from disk, so stub the offline check
        # to keep the test hermetic and to prove the persisted counts lose.
        monkeypatch.setattr(
            DatasetService, "check_dataset_availability",
            lambda self, ds: DatasetInfo(name="banc_v888", source="banc",
                                         available=True, local_prepared=True,
                                         neuron_count=188508, typed_count=118748))

        results, _updated = svc.get_cached_availability()
        assert "banc_v888" in results
        assert "flywire_BANC_v888" not in results
        assert results["banc_v888"].source == "banc"
        assert results["banc_v888"].neuron_count == 188508
        assert results["banc_v888"].typed_count == 118748
        # Non-local rows keep their persisted values.
        assert results["hemibrain:v1.2.1"].source == "neuprint"

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
        captured = {}

        def _fake_check(self, dataset):
            captured["called_with"] = dataset
            return DatasetInfo(name=dataset, source="banc", available=True,
                               local_prepared=True, neuron_count=185165,
                               typed_count=116906)

        svc = DatasetService()
        monkeypatch.setattr(DatasetService, "check_dataset_availability", _fake_check)
        # Drive the same loop _load_persisted_availability runs.
        info = svc._deserialize_info("banc_v626", stale["banc_v626"])
        assert is_banc_dataset(info.name)
        refreshed = svc.check_dataset_availability(info.name)
        assert captured["called_with"] == "banc_v626"
        assert refreshed.neuron_count == 185165
        assert refreshed.typed_count == 116906
