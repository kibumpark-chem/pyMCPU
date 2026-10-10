"""Build / ABI smoke tests for the compiled ``mcpu_core`` extension.

Attribute-surface and freshness checks -- these confirm the compiled binary
matches the Python-side API the rest of the suite depends on, not any
physics behavior. Functional behavior (the RNG-state round trip, for one)
is tested under ``tests/physics/``.

These tests exist to make a stale build fail *here*, loudly and once, rather
than as dozens of ``AttributeError``s scattered across the physics suite. A
real instance: HEAD added ``System::setRamaMixtureLibrary`` while every build
in the tree predated it, and 65 tests + 12 errors all traced back to one
missing symbol -- which the previous version of this file did not catch,
because it only checked the RNG-state pair.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SURFACE_FILE = Path(__file__).with_name("abi_surface.txt")
# Source whose modification should invalidate a built extension. Restricted to
# translation units and headers: the parameter data under src/pymcpu/parameters
# is not compiled in, so touching it must not trip the freshness check.
_SOURCE_GLOBS = ("src/**/*.cpp", "src/**/*.c", "include/**/*.h")


def _expected_surface() -> list[str]:
    return [
        line.strip()
        for line in _SURFACE_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def test_compiled_surface_matches_snapshot() -> None:
    """Every ``Class.attr`` in ``abi_surface.txt`` must exist on the built
    extension.

    This is the broad staleness canary. If it fails, rebuild first
    (``module load gcc/14.2.0-fasrc01 && pip install --no-build-isolation -e .``)
    and only then investigate. If you *added* a binding, regenerate the
    snapshot with ``tests/extension_abi/_regen_abi_surface.py``.
    """
    from pymcpu import mcpu_core

    expected = _expected_surface()
    assert expected, f"{_SURFACE_FILE.name} is empty -- regenerate it"

    missing: list[str] = []
    for entry in expected:
        class_name, _, attr = entry.partition(".")
        cls = getattr(mcpu_core, class_name, None)
        if cls is None:
            missing.append(f"{class_name} (whole class)")
        elif not hasattr(cls, attr):
            missing.append(entry)

    assert not missing, (
        "Rebuild required: the compiled mcpu_core is missing "
        f"{len(missing)} of {len(expected)} expected symbols.\n"
        "  Rebuild: module load gcc/14.2.0-fasrc01 && "
        "pip install --no-build-isolation -e .\n"
        f"  Missing: {', '.join(missing[:12])}"
        + (" ..." if len(missing) > 12 else "")
    )


def test_extension_is_not_older_than_sources() -> None:
    """The built extension must not predate the C++ it is built from.

    Catches the stale-build failure mode *before* any physics test runs. Skips
    when the source tree is absent (i.e. when testing an installed wheel).
    """
    from pymcpu import mcpu_core

    sources = [
        path
        for pattern in _SOURCE_GLOBS
        for path in _REPO_ROOT.glob(pattern)
        if path.is_file()
    ]
    if not sources:
        pytest.skip("C++ source tree not present (installed-wheel run)")

    so_path = Path(mcpu_core.__file__)
    newest = max(sources, key=lambda p: p.stat().st_mtime)
    if so_path.stat().st_mtime >= newest.stat().st_mtime:
        return

    pytest.fail(
        "Rebuild required: the compiled extension is older than the C++ sources.\n"
        f"  extension: {so_path} ({so_path.stat().st_mtime:.0f})\n"
        f"  newer src: {newest.relative_to(_REPO_ROOT)} "
        f"({newest.stat().st_mtime:.0f})\n"
        "  Rebuild: module load gcc/14.2.0-fasrc01 && "
        "pip install --no-build-isolation -e ."
    )
