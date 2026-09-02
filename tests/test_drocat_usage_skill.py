"""Offline checks for the layered DROCAT skills (install / tab-matched / backend)."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
INSTALL = ROOT / "skills" / "drocat-install"
USAGE = ROOT / "skills" / "drocat-usage"
BACKEND = ROOT / "skills" / "drocat-backend"

# The 13 UI tabs (find_similar and visualization drive two backend tools each).
TAB_RECIPES = [
    "find-path.md",
    "find-shortest.md",
    "network.md",
    "connectivity-profiling.md",
    "find-homologs.md",
    "find-similar.md",
    "inter-dataset.md",
    "nb-find-lines.md",
    "nb-find-neuron.md",
    "nb-colabel.md",
    "flylight.md",
    "visualization.md",
    "settings.md",
]

# Layer-2 module guides and the real backend classes they must reference.
BACKEND_MODULES = {
    "coana-connectivity.md": "FindNeuronConnection",
    "morphology-similarity.md": "MorphologyComparer",
    "comparison.md": "ComparisonAnalyzer",
    "profile-comparator.md": "HomologFinder",
    "neuronbridge.md": "NeuronBridgeFinder",
    "flylight.md": "FlyLightDownloader",
    "visualize-skeleton.md": "VisualizeSkeleton",
    "vispath.md": "VisualizePath",
}


def test_layer0_install_bundle() -> None:
    skill_file = INSTALL / "SKILL.md"
    text = skill_file.read_text(encoding="utf-8")
    assert text.startswith("---\nname: drocat-install\n")
    assert "one-click installer" in text or "installer" in text
    assert "troubleshooting.md" in text
    assert "custom-installation.md" in text
    assert (INSTALL / "agents" / "openai.yaml").is_file()
    assert (INSTALL / "scripts" / "verify_install.py").is_file()
    assert (INSTALL / "references" / "troubleshooting.md").is_file()
    assert (INSTALL / "references" / "custom-installation.md").is_file()
    # no raw fetch of usage skill
    assert "raw.githubusercontent.com" not in text
    assert "drocat-backend/SKILL.md" in text


def test_layer1_usage_bundle_and_tabs() -> None:
    skill_file = USAGE / "SKILL.md"
    text = skill_file.read_text(encoding="utf-8")
    assert text.startswith("---\nname: drocat-usage\n")
    assert "Layer 1" in text
    assert "drocat-backend" in text
    assert (USAGE / "agents" / "openai.yaml").is_file()
    assert (USAGE / "scripts" / "run_direct.py").is_file()
    assert (USAGE / "references" / "combinations.md").is_file()
    for ref in (
        "tool-catalog.md",
        "workflow-recipes.md",
        "datasets-and-auth.md",
    ):
        assert (USAGE / "references" / ref).is_file()
    for name in TAB_RECIPES:
        assert (USAGE / "tabs" / name).is_file(), f"missing tab recipe {name}"
    assert "raw.githubusercontent.com" not in text


def test_layer2_backend_bundle() -> None:
    skill_file = BACKEND / "SKILL.md"
    text = skill_file.read_text(encoding="utf-8")
    assert text.startswith("---\nname: drocat-backend\n")
    assert "Layer 2" in text
    assert (BACKEND / "agents" / "openai.yaml").is_file()
    assert (BACKEND / "scripts" / "run_module.py").is_file()
    for ref in ("module-index.md", "util-support.md", "combinations.md"):
        assert (BACKEND / "references" / ref).is_file()
    for name, class_name in BACKEND_MODULES.items():
        mod = (BACKEND / "modules" / name)
        assert mod.is_file(), f"missing backend module {name}"
        assert class_name in mod.read_text(encoding="utf-8"), (
            f"{name} does not reference {class_name}"
        )
    assert "raw.githubusercontent.com" not in text


def test_direct_launcher_dry_run_resolves_script_directory() -> None:
    launcher = USAGE / "scripts" / "run_direct.py"
    result = subprocess.run(
        [
            sys.executable,
            str(launcher),
            "--repo",
            str(ROOT),
            "--script",
            "scripts/FindPath.py",
            "--dry-run",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "[DROCAT] cwd:" in result.stdout
    assert str(ROOT / "scripts") in result.stdout
    assert str(ROOT / "scripts" / "FindPath.py") in result.stdout


def test_module_launcher_dry_run_resolves_repo_root() -> None:
    launcher = BACKEND / "scripts" / "run_module.py"
    result = subprocess.run(
        [
            sys.executable,
            str(launcher),
            "--repo",
            str(ROOT),
            "--script",
            "scripts/FindPath.py",
            "--dry-run",
        ],
        cwd=str(BACKEND / "scripts"),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "[DROCAT] cwd:" in result.stdout
    assert str(ROOT / "scripts") in result.stdout
    assert str(ROOT / "scripts" / "FindPath.py") in result.stdout


def test_readme_exposes_install_prompt_and_local_analysis_skills() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    install = (ROOT / "docs" / "INSTALLATION.md").read_text(encoding="utf-8")
    # local analysis-skill paths are referenced
    assert "skills/drocat-usage/SKILL.md" in readme
    assert "skills/drocat-backend/SKILL.md" in readme
    assert "skills/drocat-install/SKILL.md" in readme
    assert "docs/INSTALLATION.md" in readme
    assert "mac_DROCAT.command" in readme  # one-click install & launch
    # Option 2 is a copyable prompt that starts with Fetching the install skill;
    # both the lead-in and the prompt state it finishes cloning, installing,
    # verifying, and launching
    assert "Copy the following prompt to your AI agent" in readme
    assert "let it finish cloning the repo, installing, verifying, and launching DROCAT" in readme
    assert "Fetch https://raw.githubusercontent.com/Swida-Alba/Drosophila-cross-dataset-connectome-analysis/v4.5.0/skills/drocat-install/SKILL.md" in readme
    assert "follow it to finish cloning the repo, installing, verifying, and launching DROCAT" in readme
    assert (
        "https://raw.githubusercontent.com/Swida-Alba/Drosophila-cross-dataset-connectome-analysis/v4.5.0/skills/drocat-install/SKILL.md"
        in readme
    )
    # the analysis skills are checked-in, so no fetch is required for analysis
    assert "no fetch is required" in readme
    # INSTALLATION has no raw fetch URL
    assert "raw.githubusercontent.com" not in install


def test_skills_are_checked_in_no_fetch_required() -> None:
    usage = (USAGE / "SKILL.md").read_text(encoding="utf-8")
    install = (INSTALL / "SKILL.md").read_text(encoding="utf-8")
    backend = (BACKEND / "SKILL.md").read_text(encoding="utf-8")
    for text in (usage, install, backend):
        assert "raw.githubusercontent.com" not in text
        assert "mkdir -p ~/.codex/skills" not in text


def test_agent_workflows_have_completion_contracts() -> None:
    usage = (USAGE / "SKILL.md").read_text(encoding="utf-8")
    install = (INSTALL / "SKILL.md").read_text(encoding="utf-8")
    backend = (BACKEND / "SKILL.md").read_text(encoding="utf-8")

    for text in (usage, install, backend):
        assert "## Completion contract" in text, "missing completion contract"
    assert "Do not stop after reading source files" in usage
    assert "end-to-end" in backend
