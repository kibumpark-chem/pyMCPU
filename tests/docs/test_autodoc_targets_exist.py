"""Every ``.. auto*::`` target in ``docs/`` must resolve against the package.

Why this test exists: at the time it was written, **18 of 22** autodoc targets
named objects that did not exist (``MuContactForce``, ``MCIntegrator``,
``DCDReporter``, ``ConstraintForce``, all five ``analysis`` functions, ...).
Sphinx reports those as warnings, not errors, so ``docs/api/analysis.rst`` and
``reporters.rst`` rendered as *empty pages* and nobody noticed. Autodoc silence
is the failure mode this test converts into a red build.

It also means the next rename that forgets the docs fails here rather than
shipping a page documenting a class that no longer exists.

Deliberately independent of Sphinx: it resolves targets with ``importlib`` and
``hasattr``, so it runs in milliseconds and needs no docs toolchain. It does
require the compiled extension, because most targets are pybind11 classes.
"""

from __future__ import annotations

import importlib
import inspect
import re
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parent.parent.parent / "docs"

#: ``.. autoclass:: pymcpu.System``, ``.. automethod:: pymcpu.Context.step`` ...
_DIRECTIVE = re.compile(
    r"^\s*\.\.\s+auto(class|function|method|module|attribute|data|exception)::\s+(\S+)",
    re.MULTILINE,
)

#: Inline cross-reference roles pointing into the pymcpu namespace, e.g.
#: ``:class:`~pymcpu.Integrator```.
#:
#: These need their own check because Sphinx resolves roles SILENTLY: a role
#: naming a class that does not exist renders as plain text and emits no
#: warning at all unless ``nitpicky`` is on. That is how
#: ``:class:`~pymcpu.MCIntegrator```, ``~pymcpu.QBiasForce`` and
#: ``~pymcpu.ConstraintForce`` sat in the published physics notes through a
#: clean ``sphinx-build -W``. Turning ``nitpicky`` on is not the fix -- it
#: floods the build with unresolved type annotations (``Path``, ``optional``,
#: ``array-like``) and with real classes that simply are not autodoc'd, which
#: is a different problem. This targets the one case that matters: a name that
#: does not exist.
_ROLE = re.compile(
    r":(?:class|func|meth|attr|obj|exc|data|mod):`~?(pymcpu(?:\.[A-Za-z_][A-Za-z0-9_]*)+)`"
)


def _iter_targets() -> list[tuple[str, str, str]]:
    """Yield ``(relpath, kind, dotted_target)`` for all of ``docs/``.

    Covers both ``.. auto*::`` directives and inline ``:class:``-style roles
    that name something under ``pymcpu``.
    """
    out: list[tuple[str, str, str]] = []
    for path in sorted(DOCS.rglob("*.rst")) + sorted(DOCS.rglob("*.md")):
        text = path.read_text(encoding="utf-8")
        rel = str(path.relative_to(DOCS.parent))
        for kind, target in _DIRECTIVE.findall(text):
            out.append((rel, kind, target))
        for target in _ROLE.findall(text):
            out.append((rel, "role", target))
    return out


def _is_instance_attribute(owner: object, attr: str) -> bool:
    """True when ``attr`` is assigned on ``self`` in ``owner``'s source.

    Instance attributes are absent from the class object, so ``hasattr`` says
    False for perfectly real documented members like ``Simulation.context`` or
    ``MCPUForceField.coords``. Checking the source keeps those passing without
    weakening the test against names that genuinely do not exist.
    """
    try:
        source = inspect.getsource(owner)  # type: ignore[arg-type]
    except (TypeError, OSError):
        return False
    return (
        f"self.{attr} =" in source
        or f"self.{attr}:" in source
        or f"self.{attr}," in source
    )


def _resolve(dotted: str) -> bool:
    """True when ``dotted`` names a real module/attribute chain.

    Walks the dotted path from the longest importable module prefix inward, so
    ``pymcpu.Context.set_positions`` resolves even though ``pymcpu.Context`` is
    a class rather than a module.
    """
    parts = dotted.split(".")
    for i in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:i]))
        except Exception:  # noqa: BLE001 -- any import failure: try a shorter prefix
            continue
        for attr in parts[i:]:
            if not hasattr(obj, attr):
                return _is_instance_attribute(obj, attr)
            obj = getattr(obj, attr)
        return True
    return False


TARGETS = _iter_targets()


def test_docs_contain_autodoc_targets() -> None:
    """Guard the guard: if the regex or the docs layout changes, fail loudly
    rather than passing an empty parametrization."""
    assert TARGETS, f"no autodoc directives found under {DOCS} -- regex or layout changed?"


@pytest.mark.parametrize(
    ("relpath", "kind", "target"),
    TARGETS,
    ids=[f"{t[2]}" for t in TARGETS],
)
def test_autodoc_target_resolves(relpath: str, kind: str, target: str) -> None:
    assert _resolve(target), (
        f"{relpath} documents `auto{kind}:: {target}`, which does not exist. "
        f"Sphinx renders this as an empty section with only a warning, so fix "
        f"the target or delete the directive -- do not leave it."
    )
