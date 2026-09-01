"""Tests for the actionable warning shown when NeuPrint rejects a token."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from visualize_skeleton import VisualizeSkeleton  # noqa: E402


def test_warn_neuprint_token_rejected_prints_guidance(capsys):
    visualizer = object.__new__(VisualizeSkeleton)
    visualizer._warn_neuprint_token_rejected()
    out = capsys.readouterr().out
    assert '401' in out
    assert 'https://neuprint.janelia.org/account' in out
    assert 'config_local.json' in out
    assert 'aborted' in out


def test_warn_neuprint_token_rejected_env_variant_points_at_env(capsys):
    visualizer = object.__new__(VisualizeSkeleton)
    visualizer._warn_neuprint_token_rejected(from_env=True)
    out = capsys.readouterr().out
    assert 'NEUPRINT_APPLICATION_CREDENTIALS' in out
