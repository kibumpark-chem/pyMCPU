"""scripts/check_release.py, the guard wheels.yml runs before it publishes.

It runs only on a release, so a bug in it would show up on release day; these
tests run it on small copies of the three files it reads.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_release.py"


@pytest.fixture(scope="module")
def check_release():
    spec = importlib.util.spec_from_file_location("check_release", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tree(root: Path, version="0.1.0", changelog_date="2026-10-12",
          cff_version="0.1.0", cff_date='"2026-10-12"') -> Path:
    (root / "pymcpu").mkdir()
    (root / "pymcpu" / "__init__.py").write_text(f'__version__ = "{version}"\n')
    (root / "CHANGELOG.md").write_text(
        f"# Changelog\n\n## [{version}] - {changelog_date}\n\nFirst release.\n"
    )
    (root / "CITATION.cff").write_text(
        f"cff-version: 1.2.0\nversion: {cff_version}\ndate-released: {cff_date}\n"
    )
    return root


def test_a_ready_release_passes(check_release, tmp_path):
    root = _tree(tmp_path)
    assert check_release.problems(root, "v0.1.0") == []
    assert check_release.main(["--root", str(root), "--tag", "v0.1.0"]) == 0
    assert check_release.problems(root, None) == []  # the TestPyPI dry run


@pytest.mark.parametrize(
    ("tree", "tag", "complaint"),
    [
        ({}, "v0.1.1", "the tag must be 'v0.1.0'"),
        ({}, "0.1.0", "the tag must be 'v0.1.0'"),
        ({"changelog_date": "YYYY-MM-DD"}, "v0.1.0", "CHANGELOG.md gives 0.1.0"),
        ({"cff_date": '"YYYY-MM-DD"'}, "v0.1.0", "CITATION.cff has date-released"),
        ({"cff_version": "0.0.9"}, "v0.1.0", "CITATION.cff has version"),
        ({"cff_date": "2026-10-13"}, "v0.1.0", "the release dates differ"),
    ],
)
def test_an_unready_release_is_refused(check_release, tmp_path, tree, tag, complaint):
    root = _tree(tmp_path, **tree)
    found = check_release.problems(root, tag)
    assert any(complaint in problem for problem in found), found
    assert check_release.main(["--root", str(root), "--tag", tag]) == 1


def test_a_changelog_without_the_version_heading_is_refused(check_release, tmp_path):
    root = _tree(tmp_path)
    (root / "CHANGELOG.md").write_text("# Changelog\n\n## [Unreleased]\n")
    assert check_release.problems(root, None) == [
        "CHANGELOG.md has no '## [0.1.0] - YYYY-MM-DD' heading"
    ]
