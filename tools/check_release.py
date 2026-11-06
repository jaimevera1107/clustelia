"""Release checklist: version consistency, changelog entry and required files.

Run before tagging. Exit code 0 means every check passed.
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION_FILE = ROOT / "src" / "clustelia" / "_version.py"
VERSION_PATTERN = re.compile(
    r'^__version__(?:: str)? = "([^"]+)"$', re.MULTILINE
)
REQUIRED_FILES = (
    "LICENSE",
    "README.md",
    "CHANGELOG.md",
    "CONTRIBUTING.md",
    "src/clustelia/py.typed",
    "scripts/release.ps1",
    "tools/bump_version.py",
    "tools/open_changelog_section.py",
    ".github/workflows/tests.yml",
    ".github/workflows/release.yml",
)


def check() -> None:
    """Raise ``AssertionError`` on the first failed check."""
    metadata = tomllib.loads(
        (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    project = metadata["project"]
    assert "version" in project.get("dynamic", []), (
        "pyproject.toml must declare version as dynamic "
        "(single source: _version.py)"
    )
    assert "version" not in project, (
        "pyproject.toml must not hard-code the version"
    )
    match = VERSION_PATTERN.search(VERSION_FILE.read_text(encoding="utf-8"))
    assert match, f"Missing __version__ in {VERSION_FILE}"
    version = match.group(1)
    release = r"\d+\.\d+\.\d+(?:(?:a|b|rc)\d+|\.post\d+)?"
    assert re.fullmatch(release, version), f"Invalid version {version!r}"
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    heading = rf"^## {re.escape(version)}\b"
    assert re.search(heading, changelog, re.MULTILINE), (
        f"CHANGELOG.md has no section for {version}"
    )
    unreleased = r"^## Unreleased\s*\n+\s*[-#]"
    pending = re.search(unreleased, changelog, re.MULTILINE)
    assert not pending, (
        "CHANGELOG.md still has entries under Unreleased; "
        "move them to the version section"
    )
    for item in REQUIRED_FILES:
        assert (ROOT / item).exists(), f"Missing artifact: {item}"
    release_yml = ROOT / ".github/workflows/release.yml"
    workflow = release_yml.read_text(encoding="utf-8")
    assert "id-token: write" in workflow
    assert "pypa/gh-action-pypi-publish" in workflow
    assert "environment: pypi" in workflow or "name: pypi" in workflow
    print("clustelia release checklist: PASS")
    print(f"Version: {version}")
    print("Note: CI must be green on the exact commit being released.")


if __name__ == "__main__":
    try:
        check()
    except AssertionError as exc:
        print(f"clustelia release checklist: FAIL — {exc}", file=sys.stderr)
        sys.exit(1)
