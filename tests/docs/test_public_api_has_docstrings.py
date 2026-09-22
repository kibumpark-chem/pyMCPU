"""Every name in ``pymcpu.__all__`` must carry a real docstring.

This is the interactive path: ``help(pymcpu.System)`` and a notebook's ``?``
read ``__doc__``, not the Sphinx pages. 13 of the 18 top-level exports used to
have nothing there, because pybind11 only attaches a class docstring if
``py::class_`` is given a third argument and none of them were.

"Real" needs defining, because pybind11 never leaves ``__doc__`` empty: it
synthesizes a signature line such as
``__init__(self: pymcpu.mcpu_core.System, arg0: int, arg1: int) -> None``.
That is not documentation, so a docstring that is only a synthesized signature
counts as missing here.

Chosen over a docstring *linter* deliberately: the point is that the handful of
names a user actually reaches for are explained, not that 358 internal
functions grow section headers.
"""

from __future__ import annotations

import inspect

import pytest

import pymcpu

#: Module-level data and dunders, not documentable objects.
_NOT_DOCUMENTABLE = {"PACKAGE_ROOT", "mcpu_core", "__version__"}

EXPORTS = sorted(n for n in pymcpu.__all__ if n not in _NOT_DOCUMENTABLE)


def _is_synthesized_signature(name: str, doc: str) -> bool:
    """True when pybind11's auto-generated signature is all there is."""
    head = doc.strip()
    if not head:
        return True
    first = head.splitlines()[0].strip()
    # e.g. "System(arg0: int, ...)" or "__init__(self: pymcpu.mcpu_core..., ...)"
    return first.startswith(f"{name}(") or first.startswith("__init__(self")


def test_exports_is_not_empty() -> None:
    """Guard the guard: an empty parametrization would pass silently."""
    assert len(EXPORTS) >= 15, f"only {len(EXPORTS)} documentable exports found"


@pytest.mark.parametrize("name", EXPORTS)
def test_export_has_a_real_docstring(name: str) -> None:
    obj = getattr(pymcpu, name)
    doc = inspect.getdoc(obj) or ""

    assert not _is_synthesized_signature(name, doc), (
        f"pymcpu.{name} has no real docstring -- `help(pymcpu.{name})` shows "
        f"only a synthesized signature. For a compiled class, pass a third "
        f"argument to its py::class_ in src/bindings/bindings.cpp."
    )
    # A one-line summary is fine; a bare word is not.
    assert len(doc.strip()) >= 40, (
        f"pymcpu.{name}'s docstring is {len(doc.strip())} characters, which is "
        f"too short to say anything useful: {doc.strip()!r}"
    )
