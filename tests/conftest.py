"""Shared pytest configuration.

Resolves the NeuPrint token through the standard DROCAT fallback chain
(an explicitly exported NEUPRINT_APPLICATION_CREDENTIALS env var wins,
then config.json, then the gitignored config_local.json) and exposes it
as the env var neuprint-python itself reads.  Tests that create neuprint
clients therefore behave exactly like the runtime without requiring the
shell to export the token.
"""
import os
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))


@pytest.fixture(scope="session", autouse=True)
def _neuprint_token_from_config_chain():
    """Seed NEUPRINT_APPLICATION_CREDENTIALS from config files, if absent.

    Tests that need to simulate a tokenless environment still can:
    ``monkeypatch.delenv('NEUPRINT_APPLICATION_CREDENTIALS')`` removes the
    value for that test and pytest restores it afterwards.
    """
    if os.environ.get("NEUPRINT_APPLICATION_CREDENTIALS"):
        return
    try:
        from utils.token_manager import token_manager
        token = token_manager.get_neuprint_token()
    except Exception:
        token = None
    if token:
        os.environ["NEUPRINT_APPLICATION_CREDENTIALS"] = token


@pytest.fixture(scope="session", autouse=True)
def _neuprint_default_client_session_guard():
    """Snapshot and restore neuprint's process-global default client.

    The default client is a module global; a test (or a module-scoped
    fixture) that sets one leaks it into every later test in the same
    process, which makes order-dependent assertions fail.  Capture the
    ambient value once per session so it can be restored at the very end.
    """
    try:
        import neuprint as _np
    except Exception:
        yield
        return
    get_default = getattr(_np, "default_client", None)
    ambient = None
    if callable(get_default):
        try:
            ambient = get_default()
        except Exception:
            ambient = None
    yield
    try:
        if ambient is not None:
            _np.set_default_client(ambient)
        else:
            _np.clear_default_client()
    except Exception:
        pass


@pytest.fixture(autouse=True)
def _isolate_neuprint_default_client(_neuprint_default_client_session_guard):
    """Clear neuprint's default client after every test.

    Regardless of where a leak came from (in-test code, a module-scoped
    fixture, or production ``set_default_client`` calls), the next test must
    start from the same process-global state as the first — no default
    client.  The session guard restores any genuine ambient client at the
    end of the run.  Guarded so a test that swaps ``sys.modules['neuprint']``
    for a fake cannot raise here.
    """
    try:
        import neuprint as _np
    except Exception:
        yield
        return
    yield
    clear_default = getattr(_np, "clear_default_client", None)
    if callable(clear_default):
        try:
            clear_default()
        except Exception:
            pass
