"""Tests for DatasetService: dataset lists, name conversions, live-list
hidden-dataset filtering, and token file precedence."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import ui.dataset_service as ds_mod
from ui.config import DATASETS, DEFAULTS, FLYWIRE_DATASETS, NEUPRINT_DATASETS
from ui.dataset_service import (
    DatasetService,
    dataset_to_folder,
    folder_to_dataset,
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
FLYWIRE_EXPECTED = {"flywire_FAFB_v783", "banc_v888", "banc_v626"}


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

    def test_banc_v888_not_supported_via_neuprint(self):
        # The NeuPrint server lists banc:v888 as hidden and not queryable;
        # BANC is served through FlyWire/Codex instead.
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


class TestIsFlywireDataset:
    def test_positive(self):
        for ds in list(FLYWIRE_EXPECTED) + ["fafb", "flywire_fafb:v783"]:
            assert is_flywire_dataset(ds)

    def test_negative(self):
        # banc:v888 must never be classified as the FlyWire BANC release.
        for ds in ["banc:v888", "male-cns:v0.9", "hemibrain:v1.2.1", "manc:v1.0"]:
            assert not is_flywire_dataset(ds)


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
