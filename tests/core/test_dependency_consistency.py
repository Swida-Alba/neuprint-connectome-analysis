"""Keep DROCAT's duplicated dependency declarations in lockstep."""

from __future__ import annotations

import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NAME_RE = re.compile(r"^\s*([A-Za-z0-9_.-]+)(.*)$")


def _canonical_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _requirement_map(requirements: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for requirement in requirements:
        requirement = requirement.split("#", 1)[0].strip()
        if not requirement:
            continue
        match = NAME_RE.match(requirement)
        assert match, f"Cannot parse requirement: {requirement}"
        parsed[_canonical_name(match.group(1))] = match.group(2).strip()
    return parsed


def _requirements_file(path: Path) -> dict[str, str]:
    return _requirement_map(path.read_text(encoding="utf-8-sig").splitlines())


def _toml(path: Path) -> dict:
    with path.open("rb") as stream:
        return tomllib.load(stream)


def _assert_declared_dependencies_match(
    declared: dict[str, str], requirements: dict[str, str], source: str
) -> None:
    missing = sorted(set(declared) - set(requirements))
    mismatched = {
        name: (specifier, requirements.get(name))
        for name, specifier in declared.items()
        if name in requirements and requirements[name] != specifier
    }
    assert not missing, f"{source} is missing dependencies: {missing}"
    assert not mismatched, f"{source} has version drift: {mismatched}"


def test_root_runtime_requirements_match_package_metadata():
    project = _toml(PROJECT_ROOT / "pyproject.toml")
    declared = _requirement_map(
        project["project"]["dependencies"]
        + project["project"]["optional-dependencies"]["viz"]
        + project["project"]["optional-dependencies"]["gui"]
    )
    for filename in ("requirements.txt", "requirements-windows.txt"):
        requirements = _requirements_file(PROJECT_ROOT / filename)
        _assert_declared_dependencies_match(
            declared, requirements, filename
        )
        unexpected = sorted(set(requirements) - set(declared))
        assert not unexpected, f"{filename} has undeclared dependencies: {unexpected}"


def test_vispath_manifests_and_root_extra_match():
    project = _toml(PROJECT_ROOT / "pyproject.toml")
    vispath = _toml(PROJECT_ROOT / "vispath-subproject" / "pyproject.toml")
    standalone = _requirement_map(vispath["project"]["dependencies"])
    requirements = _requirements_file(
        PROJECT_ROOT / "vispath-subproject" / "requirements.txt"
    )
    _assert_declared_dependencies_match(
        standalone, requirements, "vispath-subproject/requirements.txt"
    )

    standalone_with_gui = _requirement_map(
        vispath["project"]["dependencies"]
        + vispath["project"]["optional-dependencies"]["gui"]
    )
    root_extra = _requirement_map(
        project["project"]["optional-dependencies"]["vispath"]
    )
    assert root_extra == standalone_with_gui


def test_optional_runtime_extras_match_full_requirements():
    project = _toml(PROJECT_ROOT / "pyproject.toml")
    requirements = _requirements_file(PROJECT_ROOT / "requirements.txt")
    for extra in ("viz", "gui"):
        declared = _requirement_map(
            project["project"]["optional-dependencies"][extra]
        )
        _assert_declared_dependencies_match(
            declared, requirements, f"requirements.txt ({extra} extra)"
        )


def test_ui_requirements_match_ui_extra_when_present():
    ui_requirements = PROJECT_ROOT / "ui" / "requirements.txt"
    if not ui_requirements.exists():
        return
    project = _toml(PROJECT_ROOT / "pyproject.toml")
    declared = _requirement_map(
        project["project"]["optional-dependencies"]["ui"]
    )
    assert declared == _requirements_file(ui_requirements)


def test_declared_py_modules_exist_in_src():
    """Every py-modules entry must map to an existing src/<name>.py.

    A stale entry (the module was removed) makes the build metadata lie and
    breaks `import <name>` for anyone who trusted it; it is otherwise inert,
    so only an explicit check catches it.
    """
    project = _toml(PROJECT_ROOT / "pyproject.toml")
    declared = project["tool"]["setuptools"]["py-modules"]
    missing = [name for name in declared if not (PROJECT_ROOT / "src" / f"{name}.py").exists()]
    assert not missing, f"pyproject.toml declares missing py-modules: {missing}"


def test_supported_python_window_and_removed_conflicts():
    project = _toml(PROJECT_ROOT / "pyproject.toml")
    vispath = _toml(PROJECT_ROOT / "vispath-subproject" / "pyproject.toml")
    assert project["project"]["requires-python"] == ">=3.10,<3.12"
    assert vispath["project"]["requires-python"] == ">=3.10,<3.12"

    prohibited = {"neuronbridge-python", "ray", "memray", "python-rapidjson"}
    manifests = [
        _requirement_map(project["project"]["dependencies"]),
        _requirements_file(PROJECT_ROOT / "requirements.txt"),
        _requirements_file(PROJECT_ROOT / "requirements-windows.txt"),
    ]
    for manifest in manifests:
        assert prohibited.isdisjoint(manifest)
