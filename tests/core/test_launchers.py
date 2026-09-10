"""Structural tests for the launchers and installers: the unified
mac_DROCAT.command / windows_DROCAT.bat files, the packed installers, and the
token workflow they implement. These guard the first-run / self-healing /
token UX contract."""
import importlib.util
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

# The shell launchers only run on macOS/Linux and on Windows via Git Bash;
# skip the executable parser checks where no bash is available.
requires_bash = pytest.mark.skipif(
    shutil.which("bash") is None, reason="bash not available"
)


class TestRunDrocatLaunchers:
    def test_macos_launcher_is_self_healing(self):
        text = (ROOT / "mac_DROCAT.command").read_text(encoding="utf-8")
        # installs on demand from the packed location
        assert "archive/install/install.sh" in text
        assert "Conda is not installed" in text
        # port-conflict prompt with the DROCAT-ownership guard
        assert "Your choice [1-3]" in text
        assert "Kill the existing DROCAT process" in text
        assert "is not DROCAT" in text
        # token reminder for users who skipped the installer prompt
        assert "config.json" in text
        assert "config_local.json" in text
        assert "NeuPrint token is not configured yet" in text
        assert "CAVE token is optional" in text
        # token acquisition links are part of the notice
        assert "neuprint.janelia.org/account" in text
        assert "codex.flywire.ai/auth_token" in text
        # version-specific custom env override: config.json wins per key
        assert "json_value envs" in text
        assert "ENV_OVERRIDE" in text
        assert "json_value tokens" in text
        assert 'for cfg in "$CONFIG_FILE" "$CONFIG_LOCAL"' in text
        # the resolved environment is written back into config_local.json
        assert "update_config_env" in text
        assert "CONFIG_LOCAL" in text
        # double-click UX: keep the window open on failure
        assert "Press Return to close" in text

    def test_windows_launcher_mirrors(self):
        text = (ROOT / "windows_DROCAT.bat").read_text(encoding="utf-8")
        assert "archive\\install\\install.ps1" in text
        assert "Your choice [1-3]" in text
        assert "taskkill /PID" in text
        assert "netstat -ano" in text
        assert "NeuPrint token is not configured yet" in text
        assert "CAVE token is optional" in text
        # token acquisition links are part of the notice
        assert "neuprint.janelia.org/account" in text
        assert "codex.flywire.ai/auth_token" in text
        # version-specific custom env override: config.json wins per key;
        # the gitignored config_local.json is the fallback for empty entries
        assert "config.json" in text
        assert "config_local.json" in text
        assert "ENV_OVERRIDE" in text
        assert "envs.'!DROCAT_VERSION!'" in text
        assert text.count("LiteralPath '!CONFIG_LOCAL!'") >= 2
        # the resolved environment is written back into config_local.json
        assert "Set-Content -LiteralPath '!CONFIG_LOCAL!'" in text
        # conda is resolved to its FULL path at both detection sites: a
        # quoted bare name is not resolved through PATH/PATHEXT by `call`
        assert text.count("for /f \"delims=\" %%c in ('where conda')") == 2
        # /C: keeps the port regex together (findstr splits on spaces),
        # on all three port-check sites
        assert text.count("findstr /R /C:\":!APP_PORT! .*LISTENING\"") == 2
        assert "findstr /R /C:\":!NEW_PORT! .*LISTENING\"" in text
        # System-owned PIDs (4) cannot be inspected; fallback label shown
        assert "-ErrorAction SilentlyContinue" in text
        assert "(command line unavailable)" in text
        assert "pause" in text

    def test_ui_prints_browser_fallback_after_nicegui_ready(self):
        text = (ROOT / "ui" / "app.py").read_text(encoding="utf-8")
        assert "NiceGUI prints its ready line" in text
        assert "browser did not open automatically" in text
        assert "copy and open" in text


class TestInstallers:
    def test_install_sh_resolves_project_root_and_verifies(self):
        text = (ROOT / "archive/install/install.sh").read_text(encoding="utf-8")
        assert 'PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"' in text
        assert 'verify_install.py --project "$PROJECT_ROOT"' in text
        assert 'cd "$PROJECT_ROOT"' in text
        assert "mac_DROCAT.command" in text
        # config_local.json is never auto-created (developer-managed); the
        # versioned env override is consulted before the default auto-find
        assert 'cp "$CONFIG_FILE" "$CONFIG_LOCAL"' not in text
        assert "config_local.json" in text
        assert 'json_value envs "$DROCAT_VERSION"' in text
        assert "ENV_OVERRIDE" in text
        assert "CONFIG_LOCAL" in text
        # the auto-fill write-back targets the LOCAL config only
        assert 'update_config_env "$DROCAT_VERSION" "$ENV_NAME" "$CONFIG_LOCAL"' in text
        # a configured custom env is strict: wrong Python aborts instead of
        # silently switching to a default name, and the entry is never
        # overwritten by the auto-fill write-back
        assert "exists but is not Python" in text
        assert 'if [[ -z "$ENV_OVERRIDE" ]]' in text
        # token_info files are removed: the notice must not reference them
        assert "token_info" not in text
        # verification failure retries once after clearing the pip wheel cache
        assert "VERIFY_OK" in text
        assert 'rm -rf "$pip_cache_dir/wheels"' in text
        assert "rebuilding dependencies once" in text

    def test_install_sh_token_notice_has_no_terminal_prompt(self):
        """Tokens are NOT collected in the terminal: the installer only prints
        a status notice pointing to the UI Settings tab / config.json,
        with the CAVE token marked optional."""
        text = (ROOT / "archive/install/install.sh").read_text(encoding="utf-8")
        assert "UI Settings tab" in text
        assert "config.json" in text
        assert "config_local.json" in text
        assert "required for NeuPrint datasets" in text
        assert "only needed for FlyWire FAFB online fetching" in text
        # token acquisition links are part of the notice
        assert "neuprint.janelia.org/account" in text
        assert "codex.flywire.ai/auth_token" in text
        # no interactive paste prompt remains
        assert "read -r -p" not in text
        assert "Paste them here in the terminal" not in text
        assert "Non-interactive: skipping" not in text
        # token_info files are removed: the notice must not reference them
        assert "token_info" not in text

    def test_install_ps1_token_notice_has_no_terminal_prompt(self):
        text = (ROOT / "archive/install/install.ps1").read_text(encoding="utf-8")
        assert "UI Settings tab" in text
        assert "required for NeuPrint datasets" in text
        assert "only needed for FlyWire FAFB online fetching" in text
        # token acquisition links are part of the notice
        assert "neuprint.janelia.org/account" in text
        assert "codex.flywire.ai/auth_token" in text
        assert "Read-Host" not in text
        assert "IsInputRedirected" not in text
        assert "Paste them here in the terminal" not in text
        # token_info files are removed: the notice must not reference them
        assert "token_info" not in text

    def test_install_ps1_config_json_env_override(self):
        text = (ROOT / "archive/install/install.ps1").read_text(encoding="utf-8")
        # config.json ships clean and wins per key; the gitignored
        # config_local.json fallback is developer-managed (never generated)
        assert 'Copy-Item $ConfigFile $ConfigLocal' not in text
        assert "$ConfigLocal" in text
        assert "$ConfigLocalData" in text
        assert "$EnvOverride" in text
        assert "config.json" in text
        # the resolved environment is written back into config_local.json
        assert "Set-ConfigEnvOverride" in text
        assert "Set-Content $ConfigLocal" in text
        # a configured custom env is strict: wrong Python throws instead of
        # silently switching to a default name, and the entry is never
        # overwritten by the auto-fill write-back
        assert "exists but is not Python" in text
        assert "if (-not $EnvOverride) { Set-ConfigEnvOverride" in text
        # token_info files are removed: the notice must not reference them
        assert "token_info" not in text

    def test_install_ps1_windows_powershell_51_hardening(self):
        """Windows PowerShell 5.1 turns native stderr into a terminating
        NativeCommandError under EAP Stop; the installer must not abort on
        pip warnings (pip show / pip install)."""
        text = (ROOT / "archive/install/install.ps1").read_text(encoding="utf-8")
        # legacy neuronbridge-python probe is guarded (missing = desired)
        assert "$LegacyNeuronbridge" in text
        assert "$LASTEXITCODE -eq 0" in text
        # native commands run with EAP Continue and exit-code checks
        assert '$ErrorActionPreference = "Continue"' in text
        assert "$PreviousErrorActionPreference" in text
        assert "$LASTEXITCODE -ne 0" in text
        # unwritable shared pip cache is redirected to a project-local dir,
        # with PIP_NO_CACHE_DIR as the final fallback
        assert "PIP_CACHE_DIR" in text
        assert "cache\\pip" in text
        assert "PIP_NO_CACHE_DIR" in text
        # verification failure retries once after clearing the pip wheel cache
        assert "$Verified" in text
        assert 'Remove-Item (Join-Path $PipCacheDir "wheels")' in text
        assert "rebuilding dependencies once" in text

    def test_install_bat_wraps_ps1(self):
        text = (ROOT / "archive/install/install.bat").read_text(encoding="utf-8")
        assert "install.ps1" in text
        assert "-ExecutionPolicy Bypass" in text


# The shell launchers/installer parse config.json with a hand-rolled awk
# reader (`json_value`). These tests actually execute it against both JSON
# layouts users write: the pretty-printed default and the single-line nested
# form shown in the docs. A regression here silently produces a corrupt env
# name (a stray trailing quote) instead of failing loudly.

_JSON_VALUE_RE = re.compile(
    r"^([ \t]*)json_value\(\)\s*\{(?P<body>.*?)^\1\}", re.S | re.M
)


def _json_value_source(path: Path) -> str:
    match = _JSON_VALUE_RE.search(path.read_text(encoding="utf-8"))
    assert match, f"json_value() not found in {path}"
    return match.group(0)


def _run_json_value(path: Path, section: str, key: str, config: Path) -> str:
    script = f"{_json_value_source(path)}\nprintf '%s' \"$(json_value {section} {key} '{config}')\"\n"
    proc = subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, check=True
    )
    return proc.stdout


@requires_bash
@pytest.mark.parametrize(
    "shell_file",
    ["archive/install/install.sh", "mac_DROCAT.command"],
)
class TestConfigJsonValueParser:
    _CASES = {
        "pretty": (
            '{\n  "tokens": {\n    "neuprint": "NPTOKEN",\n    "cave": ""\n  },\n'
            '  "envs": {\n    "4.5.0": "my-custom-env"\n  }\n}\n'
        ),
        # The exact single-line nested form in docs/INSTALLATION.md: the
        # closing brace follows the value with no space.
        "single_line": (
            '{\n  "tokens": { "neuprint": "NPTOKEN", "cave": "" },\n'
            '  "envs": { "4.5.0": "my-custom-env" }\n}\n'
        ),
        "single_line_spaced": (
            '{\n  "tokens": { "neuprint": "NPTOKEN" },\n'
            '  "envs": { "4.5.0": "my-custom-env" }\n}\n'
        ),
    }

    def test_env_override_has_no_stray_quote(self, shell_file, tmp_path):
        path = ROOT / shell_file
        for label, text in self._CASES.items():
            cfg = tmp_path / f"{label}.json"
            cfg.write_text(text, encoding="utf-8")
            assert _run_json_value(path, "envs", "4.5.0", cfg) == "my-custom-env", label

    def test_token_value_has_no_stray_quote(self, shell_file, tmp_path):
        path = ROOT / shell_file
        cfg = tmp_path / "c.json"
        cfg.write_text(self._CASES["single_line"], encoding="utf-8")
        assert _run_json_value(path, "tokens", "neuprint", cfg) == "NPTOKEN"

    def test_empty_values_stay_empty(self, shell_file, tmp_path):
        # An empty entry means "auto-find": it must not become a bare quote.
        path = ROOT / shell_file
        cfg = tmp_path / "c.json"
        cfg.write_text('{\n  "envs": { "4.5.0": "" }\n}\n', encoding="utf-8")
        assert _run_json_value(path, "envs", "4.5.0", cfg) == ""


def _load_verify_install():
    path = ROOT / "skills/drocat-install/scripts/verify_install.py"
    spec = importlib.util.spec_from_file_location("drocat_verify_install", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestVerifyInstallConfigRead:
    """The verifier must read the BOM that Windows PowerShell writes."""

    def test_bom_config_reports_configured_token(self, tmp_path):
        module = _load_verify_install()
        token = b"a" * 64
        # Windows PowerShell `Set-Content -Encoding UTF8` prepends a BOM.
        (tmp_path / "config.json").write_bytes(
            b"\xef\xbb\xbf"
            + b'{\n  "tokens": {\n    "neuprint": "'
            + token
            + b'"\n  },\n  "envs": {\n    "4.5.0": "drocat-4.5.0"\n  }\n}\n'
        )
        has_token, source = module.config_neuprint_token(tmp_path)
        assert has_token is True
        assert source and source.endswith("config.json")

    def test_missing_token_is_advisory(self, tmp_path):
        module = _load_verify_install()
        (tmp_path / "config.json").write_text(
            '{\n  "tokens": {\n    "neuprint": ""\n  }\n}\n', encoding="utf-8"
        )
        has_token, _source = module.config_neuprint_token(tmp_path)
        assert has_token is False

    def test_placeholder_token_is_not_configured(self, tmp_path):
        module = _load_verify_install()
        (tmp_path / "config.json").write_text(
            '{\n  "tokens": {\n    "neuprint": "YOUR_NEUPRINT_TOKEN_HERE"\n  }\n}\n',
            encoding="utf-8",
        )
        has_token, _source = module.config_neuprint_token(tmp_path)
        assert has_token is False

