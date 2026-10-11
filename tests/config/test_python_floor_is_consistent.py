"""The declared Python floor must be real, and stated the same everywhere.

``requires-python = ">=3.12"`` is a promise. Two things can break it
silently:

1. Someone bumps the floor in one file and not in the others that state it:
   the ruff ``target-version``, the conda recipe, and the "Python X.Y or
   newer" lines of the README and the installation docs.
2. Someone writes syntax newer than the floor. ``target-version`` in the ruff
   config catches this, so this module just asserts the two numbers agree.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent.parent
PYPROJECT = (REPO / "pyproject.toml").read_text(encoding="utf-8")


def _declared_floor() -> tuple[int, int]:
    match = re.search(r'^requires-python\s*=\s*"[><=~!]*(\d+)\.(\d+)', PYPROJECT, re.M)
    assert match, "could not find requires-python in pyproject.toml"
    return int(match.group(1)), int(match.group(2))


def test_ruff_target_version_matches_requires_python() -> None:
    major, minor = _declared_floor()
    match = re.search(r'^target-version\s*=\s*"py(\d)(\d+)"', PYPROJECT, re.M)
    assert match, "ruff target-version is not set; the floor is unenforced"
    assert (int(match.group(1)), int(match.group(2))) == (major, minor), (
        f"ruff target-version is py{match.group(1)}{match.group(2)} but "
        f"requires-python is >={major}.{minor}"
    )


def test_conda_recipe_states_the_same_floor() -> None:
    recipe = REPO / "conda-recipe" / "meta.yaml"
    if not recipe.exists():
        pytest.skip("no conda recipe in this tree")
    floors = re.findall(r"^\s*- python >=(\d+)\.(\d+)", recipe.read_text(), re.M)
    assert floors, "conda-recipe/meta.yaml states no python floor"
    assert {(int(a), int(b)) for a, b in floors} == {_declared_floor()}


def test_every_tracked_module_parses_under_the_declared_floor() -> None:
    """Catches syntax newer than the floor, in the files that ship.

    ``ast.parse`` uses the running interpreter's grammar, so this is only a
    real check when the test runs on a version at or near the floor -- in CI's
    lowest matrix entry. It still catches a hard syntax error anywhere, which
    is worth having on every run.
    """
    failures = []
    for path in sorted((REPO / "pymcpu").rglob("*.py")):
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            failures.append(f"{path.relative_to(REPO)}: {exc}")
    assert not failures, "modules failed to parse:\n" + "\n".join(failures)


_FLOOR_CLAIM = re.compile(r"Python (\d+)\.(\d+) or newer")


@pytest.mark.parametrize(
    "relpath",
    ["docs/installation.rst", "README.md"],
)
def test_user_facing_docs_state_the_declared_floor(relpath: str) -> None:
    """A higher claim turns people away; a lower one fails their install."""
    path = REPO / relpath
    if not path.exists():
        pytest.skip(f"{relpath} not present")
    claims = {(int(a), int(b)) for a, b in _FLOOR_CLAIM.findall(path.read_text("utf-8"))}
    assert claims == {_declared_floor()}, (
        f"{relpath} states the floor as {sorted(claims)}; requires-python is "
        f">={'.'.join(map(str, _declared_floor()))}"
    )
