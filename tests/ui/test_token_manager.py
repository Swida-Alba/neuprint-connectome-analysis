"""Tests for TokenManager: config.json loading, precedence, env fallback,
server-verified skip-invalid chain, and token type auto-detection."""
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.token_manager import TokenManager


class _FakeResponse:
    def __init__(self, status_code):
        self.status_code = status_code


class TestTokenManager:
    @pytest.fixture(autouse=True)
    def _no_network_probe(self, monkeypatch):
        """Precedence tests must not hit neuprint.janelia.org; every token
        counts as accepted unless a test overrides the probe."""
        monkeypatch.setattr(
            TokenManager, "neuprint_token_rejected",
            lambda self, token, server=None: False)

    def test_loads_from_config_json(self, tmp_path, monkeypatch):
        (tmp_path / "config.json").write_text(
            '{"tokens": {"neuprint": "cfg-np", "cave": "cfg-cave"}}\n',
            encoding="utf-8",
        )
        monkeypatch.chdir(tmp_path)
        manager = TokenManager()
        assert manager.tokens.get("NEUPRINT_TOKEN") == "cfg-np"
        assert manager.tokens.get("CAVE_TOKEN") == "cfg-cave"

    def test_direct_input_wins_over_config(self):
        manager = TokenManager()
        token = manager.get_token("NEUPRINT_TOKEN", direct_input="direct-tok")
        assert token == "direct-tok"

    def test_placeholder_token_ignored(self, monkeypatch):
        manager = TokenManager()
        manager._token_sources = {
            "NEUPRINT_TOKEN": [("config.json", "YOUR_NEUPRINT_TOKEN_HERE")]}
        monkeypatch.delenv("NEUPRINT_TOKEN", raising=False)
        monkeypatch.delenv("NEUPRINT_APPLICATION_CREDENTIALS", raising=False)
        assert manager.get_token("NEUPRINT_TOKEN") is None

    def test_env_fallback(self, monkeypatch):
        manager = TokenManager()
        manager._token_sources = {}
        monkeypatch.setenv("NEUPRINT_TOKEN", "env-tok")
        assert manager.get_token("NEUPRINT_TOKEN") == "env-tok"
        monkeypatch.delenv("NEUPRINT_TOKEN")

    def test_env_fallback_reads_canonical_application_credentials(
            self, monkeypatch):
        """NEUPRINT_APPLICATION_CREDENTIALS (the variable neuprint-python
        itself reads) is honored even when NEUPRINT_TOKEN is unset."""
        manager = TokenManager()
        manager._token_sources = {}
        monkeypatch.delenv("NEUPRINT_TOKEN", raising=False)
        monkeypatch.setenv("NEUPRINT_APPLICATION_CREDENTIALS", "canonical-tok")
        try:
            assert manager.get_token("NEUPRINT_TOKEN") == "canonical-tok"
            assert manager.get_neuprint_token() == "canonical-tok"
        finally:
            monkeypatch.delenv("NEUPRINT_APPLICATION_CREDENTIALS")

    def test_config_json_wins_over_config_local(self, tmp_path, monkeypatch):
        """config.json wins per key; config_local.json only fills empties."""
        (tmp_path / "config.json").write_text(
            '{"tokens": {"neuprint": "cfg-np", "cave": "cfg-cave"}}\n',
            encoding="utf-8",
        )
        (tmp_path / "config_local.json").write_text(
            '{"tokens": {"neuprint": "local-np"}}\n', encoding="utf-8"
        )
        monkeypatch.chdir(tmp_path)
        manager = TokenManager()
        assert manager.tokens.get("NEUPRINT_TOKEN") == "cfg-np"
        assert manager.tokens.get("CAVE_TOKEN") == "cfg-cave"

    def test_config_local_fills_empty_config_json_entry(self, tmp_path, monkeypatch):
        """An empty config.json entry falls back to config_local.json."""
        (tmp_path / "config.json").write_text(
            '{"tokens": {"neuprint": "", "cave": "cfg-cave"}}\n',
            encoding="utf-8",
        )
        (tmp_path / "config_local.json").write_text(
            '{"tokens": {"neuprint": "local-np"}}\n', encoding="utf-8"
        )
        monkeypatch.chdir(tmp_path)
        manager = TokenManager()
        assert manager.tokens.get("NEUPRINT_TOKEN") == "local-np"
        assert manager.tokens.get("CAVE_TOKEN") == "cfg-cave"

    def test_config_json_empty_values_fall_back_to_env(self, tmp_path, monkeypatch):
        (tmp_path / "config.json").write_text(
            '{"tokens": {"neuprint": "", "cave": ""}}\n', encoding="utf-8"
        )
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("CAVE_TOKEN", "env-cave")
        try:
            manager = TokenManager(project_root=str(tmp_path))
            assert manager.tokens.get("NEUPRINT_TOKEN") is None
            assert manager.get_token("CAVE_TOKEN") == "env-cave"
        finally:
            monkeypatch.delenv("CAVE_TOKEN")

    def test_legacy_token_files_are_not_read(self, tmp_path, monkeypatch):
        """token_info files are deprecated; only config.json is read."""
        (tmp_path / "token_info_local.txt").write_text(
            "NEUPRINT_TOKEN='legacy-np'\nCAVE_TOKEN='legacy-cave'\n", encoding="utf-8"
        )
        monkeypatch.chdir(tmp_path)
        manager = TokenManager(project_root=str(tmp_path))
        assert manager.tokens.get("NEUPRINT_TOKEN") is None
        assert manager.tokens.get("CAVE_TOKEN") is None

    def test_detect_token_type(self):
        tm = TokenManager()
        long_jwt = "x" * 120 + ".y" * 10
        assert tm.detect_token_type(long_jwt) == "neuprint"
        assert tm.detect_token_type("a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4") == "cave"
        assert tm.detect_token_type("") == "unknown"
        assert tm.detect_token_type("not-a-token!") == "unknown"

    def test_require_both_tokens_raises_when_missing(self, monkeypatch):
        manager = TokenManager()
        manager._token_sources = {}
        monkeypatch.delenv("NEUPRINT_TOKEN", raising=False)
        monkeypatch.delenv("NEUPRINT_APPLICATION_CREDENTIALS", raising=False)
        monkeypatch.delenv("CAVE_TOKEN", raising=False)
        with pytest.raises(ValueError, match="NEUPRINT_TOKEN"):
            manager.require_both_tokens()

    def test_direct_input_detects_neuprint(self, monkeypatch):
        """A long JWT-like direct input detects as neuprint; CAVE may be absent."""
        manager = TokenManager()
        monkeypatch.delenv("NEUPRINT_TOKEN", raising=False)
        monkeypatch.delenv("NEUPRINT_APPLICATION_CREDENTIALS", raising=False)
        monkeypatch.delenv("CAVE_TOKEN", raising=False)
        result = manager.get_auto_token(direct_input="x" * 150 + ".y" * 10)
        assert result["neuprint"]
        assert result["detected_type"] == "neuprint"

    def test_get_auto_token_no_direct_input(self, monkeypatch):
        manager = TokenManager()
        manager._token_sources = {}
        monkeypatch.setenv("NEUPRINT_TOKEN", "env-np")
        monkeypatch.setenv("CAVE_TOKEN", "env-cave")
        try:
            result = manager.get_auto_token()
        finally:
            monkeypatch.delenv("NEUPRINT_TOKEN")
            monkeypatch.delenv("CAVE_TOKEN")
        assert result["neuprint"] == "env-np"
        assert result["cave"] == "env-cave"


class TestSkipInvalidTokenChain:
    """A server-rejected candidate is skipped to the next location instead
    of interrupting the run."""

    def _manager(self, tmp_path, monkeypatch, neuprint_chain):
        (tmp_path / "config.json").write_text(
            '{"tokens": {"neuprint": "", "cave": "cfg-cave"}}\n',
            encoding="utf-8",
        )
        (tmp_path / "config_local.json").write_text(
            '{"tokens": {"neuprint": "%s"}}\n' % neuprint_chain,
            encoding="utf-8",
        )
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("NEUPRINT_TOKEN", raising=False)
        monkeypatch.delenv("NEUPRINT_APPLICATION_CREDENTIALS", raising=False)
        return TokenManager(project_root=str(tmp_path))

    def test_rejected_config_local_token_falls_back_to_env(
            self, tmp_path, monkeypatch):
        manager = self._manager(tmp_path, monkeypatch, "revoked-tok")
        monkeypatch.setenv("NEUPRINT_TOKEN", "env-tok")
        monkeypatch.setattr(
            TokenManager, "neuprint_token_rejected",
            lambda self, token, server=None: token == "revoked-tok")
        try:
            assert manager.get_token("NEUPRINT_TOKEN") == "env-tok"
        finally:
            monkeypatch.delenv("NEUPRINT_TOKEN")

    def test_rejected_env_token_cannot_shadow_valid_direct_input(
            self, tmp_path, monkeypatch):
        manager = self._manager(tmp_path, monkeypatch, "local-tok")
        monkeypatch.setenv("NEUPRINT_TOKEN", "env-tok")
        monkeypatch.setattr(
            TokenManager, "neuprint_token_rejected",
            lambda self, token, server=None: token == "env-tok")
        try:
            assert manager.get_token(
                "NEUPRINT_TOKEN", direct_input="env-tok") == "local-tok"
        finally:
            monkeypatch.delenv("NEUPRINT_TOKEN")

    def test_duplicate_candidates_probed_once(self, tmp_path, monkeypatch):
        """The same token listed by two sources is verified a single time."""
        manager = self._manager(tmp_path, monkeypatch, "same-tok")
        manager._token_sources["NEUPRINT_TOKEN"] = [
            ("config.json", "same-tok"), ("config_local.json", "same-tok")]
        probes = []

        def _probe(self, token, server=None):
            probes.append(token)
            return False

        monkeypatch.setattr(TokenManager, "neuprint_token_rejected", _probe)
        assert manager.get_token("NEUPRINT_TOKEN") == "same-tok"
        assert probes == ["same-tok"]

    def test_all_candidates_rejected_returns_first_for_caller_guidance(
            self, tmp_path, monkeypatch):
        """With every candidate refused there is no valid fallback; the
        first one is returned so callers show rejected-token guidance."""
        manager = self._manager(tmp_path, monkeypatch, "revoked-tok")
        monkeypatch.setenv("NEUPRINT_TOKEN", "also-revoked")
        monkeypatch.setattr(
            TokenManager, "neuprint_token_rejected",
            lambda self, token, server=None: True)
        try:
            assert manager.get_token("NEUPRINT_TOKEN") == "revoked-tok"
        finally:
            monkeypatch.delenv("NEUPRINT_TOKEN")

    def test_non_neuprint_tokens_skip_probe(self, tmp_path, monkeypatch):
        """Only NEUPRINT_TOKEN is server-verified; other names keep the
        first-candidate behavior."""
        manager = self._manager(tmp_path, monkeypatch, "np-tok")
        manager._token_sources["CAVE_TOKEN"] = [("config.json", "cave-tok")]
        probes = []

        def _probe(self, token, server=None):
            probes.append(token)
            return False

        monkeypatch.setattr(TokenManager, "neuprint_token_rejected", _probe)
        assert manager.get_token("CAVE_TOKEN") == "cave-tok"
        assert probes == []


class TestNeuprintProbe:
    """neuprint_token_rejected: 401/403 reject, everything else (including
    probe failures) never disqualifies a candidate."""

    def _patch_requests(self, monkeypatch, handler):
        calls = []

        def fake_get(url, headers=None, timeout=None):
            calls.append((url, headers, timeout))
            return handler()

        monkeypatch.setitem(
            sys.modules, "requests", types.SimpleNamespace(get=fake_get))
        return calls

    def test_401_marks_token_rejected(self, monkeypatch):
        calls = self._patch_requests(
            monkeypatch, lambda: _FakeResponse(401))
        manager = TokenManager()
        assert manager.neuprint_token_rejected("bad-tok") is True
        url, headers, timeout = calls[0]
        assert url.endswith("/api/version")
        assert headers["Authorization"] == "Bearer bad-tok"
        assert timeout == manager.NEUPRINT_PROBE_TIMEOUT

    def test_403_marks_token_rejected(self, monkeypatch):
        self._patch_requests(monkeypatch, lambda: _FakeResponse(403))
        manager = TokenManager()
        assert manager.neuprint_token_rejected("bad-tok") is True

    def test_200_keeps_token(self, monkeypatch):
        self._patch_requests(monkeypatch, lambda: _FakeResponse(200))
        manager = TokenManager()
        assert manager.neuprint_token_rejected("good-tok") is False

    def test_unexpected_status_keeps_token(self, monkeypatch):
        """Only an explicit refusal invalidates; odd statuses do not."""
        self._patch_requests(monkeypatch, lambda: _FakeResponse(503))
        manager = TokenManager()
        assert manager.neuprint_token_rejected("tok") is False

    def test_network_error_keeps_token(self, monkeypatch):
        """An unreachable server must not knock tokens out of the chain."""
        def _raise():
            raise ConnectionError("offline")

        self._patch_requests(monkeypatch, _raise)
        manager = TokenManager()
        assert manager.neuprint_token_rejected("tok") is False

    def test_probe_result_is_cached_per_token(self, monkeypatch):
        counter = {"n": 0}

        def handler():
            counter["n"] += 1
            return _FakeResponse(401)

        calls = self._patch_requests(monkeypatch, handler)
        manager = TokenManager()
        assert manager.neuprint_token_rejected("tok") is True
        assert manager.neuprint_token_rejected("tok") is True
        assert len(calls) == 1
        assert counter["n"] == 1

    def test_distinct_tokens_probed_separately(self, monkeypatch):
        calls = self._patch_requests(
            monkeypatch, lambda: _FakeResponse(200))
        manager = TokenManager()
        manager.neuprint_token_rejected("tok-a")
        manager.neuprint_token_rejected("tok-b")
        assert len(calls) == 2
