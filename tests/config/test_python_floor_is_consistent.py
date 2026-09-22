"""The declared Python floor must be real, and stated once.

``requires-python = ">=3.9"`` is a promise. Three things can break it
silently:

1. Someone bumps the floor in one file and not the five others that state it.
2. Someone writes 3.10+ syntax. ``target-version`` in the ruff config catches
   this, so this module just asserts the two numbers agree.
3. Someone writes a runtime-evaluated PEP 604 union (``x: int | None`` in a
   module without ``from __future__ import annotations``). That is valid 3.9
   *syntax* that raises ``TypeError`` on 3.9 at import, so neither the parser
   nor ``target-version`` sees it -- ruff's FA102 does, and it is enabled.

This module covers (1) and the agreement half of (2). It deliberately does not
re-implement (3).
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
        f"requires-python is >={major}.{minor} -- ruff would accept syntax the "
        f"declared floor cannot run"
    )


def test_ruff_enables_the_future_annotations_check() -> None:
    """FA is the only guard against a runtime-evaluated PEP 604 union."""
    lint = PYPROJECT.split("[tool.ruff.lint]", 1)
    assert len(lint) == 2, "no [tool.ruff.lint] section"
    section = lint[1].split("\n[", 1)[0]
    assert re.search(r'"FA', section), (
        "ruff does not select FA; a PEP 604 union without "
        "`from __future__ import annotations` would pass lint and then fail at "
        "import on the declared floor"
    )


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

# A claim is about something OTHER than pyMCPU if its paragraph says so.
_SCOPED = ("extra", "pymcpu-westpa", "separate package")


def unscoped_floor_claims(text: str) -> set[tuple[int, int]]:
    """Floor claims in ``text`` that are about **pyMCPU itself**.

    A claim scoped to an optional extra or a separate companion package is
    excluded, because those legitimately can require more than pyMCPU does:
    ``pymcpu-westpa`` needs 3.10, since westpa 2022.15 declares
    ``requires-python >=3.10``, while pyMCPU still supports 3.9. Saying so
    is accurate and useful, and is not the mistake this guard exists for --
    which is a doc turning 3.9 users away by overstating what pyMCPU needs.

    Scope is decided per PARAGRAPH, not per line. A per-line rule was the
    first attempt and it was wrong: prose wraps wherever it happens to
    wrap, so whether the guard fired depended on which line the phrase
    landed on rather than on what the sentence said. That is a coin flip
    dressed up as a check.
    """
    claims: set[tuple[int, int]] = set()
    for para in re.split(r"\n\s*\n", text):
        if any(marker in para.lower() for marker in _SCOPED):
            continue
        for m in _FLOOR_CLAIM.finditer(para):
            claims.add((int(m.group(1)), int(m.group(2))))
    return claims


@pytest.mark.parametrize(
    "relpath",
    ["docs/installation.rst", "README.md"],
)
def test_user_facing_docs_do_not_claim_a_higher_floor(relpath: str) -> None:
    """A doc promising 3.11 while the package accepts 3.9 turns people away."""
    path = REPO / relpath
    if not path.exists():
        pytest.skip(f"{relpath} not present")
    major, minor = _declared_floor()
    claims = unscoped_floor_claims(path.read_text(encoding="utf-8"))
    too_high = {c for c in claims if c > (major, minor)}
    assert not too_high, (
        f"{relpath} claims a floor of {sorted(too_high)} but requires-python "
        f"is >={major}.{minor}. If the claim is about an optional extra "
        f"rather than about pyMCPU, say so on the same line."
    )


def test_an_unscoped_over_claim_is_still_caught() -> None:
    """The exemption above must not be a hole.

    Written because the exemption exists to let a true statement about the
    ``pymcpu-westpa`` package through, and an exemption added to unblock
    something is exactly the kind that quietly stops the guard working.
    """
    assert unscoped_floor_claims("pyMCPU needs Python 3.11 or newer.") == {(3, 11)}
    assert unscoped_floor_claims("Install it; Python 3.12 or newer.") == {(3, 12)}


def test_a_claim_scoped_to_a_companion_package_is_allowed() -> None:
    assert unscoped_floor_claims(
        "The ``westpa`` extra requires Python 3.10 or newer."
    ) == set()
    assert unscoped_floor_claims(
        "WESTPA support is a separate package, not an extra:\n"
        "pip install pymcpu-westpa. It requires Python 3.10 or newer."
    ) == set()


def test_scoping_does_not_leak_between_paragraphs() -> None:
    """The important half: one exempt paragraph must not excuse the rest of
    the document. A whole-text keyword search would have passed this."""
    text = (
        "The ``westpa`` extra requires Python 3.10 or newer.\n"
        "\n"
        "pyMCPU requires Python 3.13 or newer.\n"
    )
    assert unscoped_floor_claims(text) == {(3, 13)}
