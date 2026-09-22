"""The core package must not depend on any external sampling framework.

The WESTPA integration is a separate distribution in a separate
repository (published as ``pymcpu-westpa``), and the whole claim being made
about it --
that adopting pyMCPU from another framework needs no change to pyMCPU -- is
only true while this holds. It is the kind of property that erodes one
convenient import at a time, so it is asserted rather than trusted.

Checked by parsing, not by grepping for the word: ``pymcpu/`` legitimately
*mentions* WESTPA in comments, as the worked example. A dependency is an
import.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
_FORBIDDEN = {"westpa", "pymcpu_westpa"}


def _imported_top_level_modules(path: Path) -> set[str]:
    names: set[str] = set()
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_no_module_under_pymcpu_imports_an_external_framework() -> None:
    offenders = []
    for path in sorted((REPO / "pymcpu").rglob("*.py")):
        hits = _imported_top_level_modules(path) & _FORBIDDEN
        if hits:
            offenders.append(f"{path.relative_to(REPO)} imports {sorted(hits)}")
    assert not offenders, (
        "core must not import an external sampling framework; that code "
        "belongs in the pymcpu-westpa repository:\n" + "\n".join(offenders)
    )


def test_the_core_cli_does_not_advertise_westpa_subcommands() -> None:
    """``mcpu --help`` must not offer what it cannot run.

    argparse cannot register a subcommand lazily, so a westpa subcommand on
    the core CLI is advertised to every user, almost none of whom have the
    add-on installed. The add-on ships its own ``mcpu-westpa`` console
    script instead, which appears exactly when it is installed.
    """
    from pymcpu.cli import build_parser

    assert "westpa" not in build_parser().format_help().lower()


def test_pymcpu_does_not_declare_a_westpa_extra() -> None:
    """A ``[westpa]`` extra would put the dependency back in core's metadata.

    Also rejects the tempting `westpa = ["pymcpu-westpa"]` convenience
    alias, which would create a core -> add-on -> core metadata cycle.
    """
    text = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    body = text.split("[project.optional-dependencies]", 1)
    if len(body) == 1:
        pytest.skip("no optional-dependencies table")
    extras = body[1].split("\n[", 1)[0]
    assert "westpa" not in extras, (
        "pyproject.toml still declares a westpa extra; the integration is a "
        "separate distribution installed as `pip install pymcpu-westpa`"
    )
