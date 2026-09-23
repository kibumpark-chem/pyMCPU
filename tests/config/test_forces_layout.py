"""The ``forces/`` tree's layout rules, asserted rather than trusted.

``include/pymcpu/forces/README.md`` states where a new potential goes and what
namespace it opens. Both rules are the kind that erode one convenient edit at a
time -- a fit directory quietly including its sibling's header is a working
build and a broken abstraction -- so they are checked here.

Checked by reading the source, not the build: these are source-layout claims,
and they must fail in a checkout that has never been compiled.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
INC = REPO / "include" / "pymcpu" / "forces"
SRC = REPO / "src" / "pymcpu" / "forces"

#: Directories directly under a family that are NOT a fit.
_NOT_A_FIT = {"common"}


def _families() -> list[Path]:
    return [d for d in sorted(INC.iterdir()) if d.is_dir() and d.name != "bias"]


def _fit_dirs() -> list[Path]:
    out = []
    for fam in _families():
        out += [d for d in sorted(fam.iterdir()) if d.is_dir() and d.name not in _NOT_A_FIT]
    return out


def _sources() -> list[Path]:
    return sorted([*INC.rglob("*.h"), *SRC.rglob("*.cpp")])


def _rel(path: Path) -> str:
    return str(path.relative_to(REPO))


def test_no_fit_directory_includes_another_fits_header() -> None:
    """The whole point of the layout: mcpu08 and mcpu26 never reach into each other."""
    tags = {d.name for d in _fit_dirs()}
    offenders = []
    for path in _sources():
        parts = path.relative_to(REPO).parts
        own = next((t for t in tags if t in parts), None)
        if own is None:
            continue
        for inc in re.findall(r'#include\s+"([^"]+)"', path.read_text(encoding="utf-8")):
            foreign = [t for t in tags if t != own and f"/{t}/" in inc]
            if foreign:
                offenders.append(f"{_rel(path)} (fit {own!r}) includes {inc!r}")
    assert not offenders, "a fit directory reached into another fit:\n  " + "\n  ".join(offenders)


def test_common_never_includes_a_fit() -> None:
    """``common/`` is shared by every fit, so it cannot depend on one of them."""
    tags = {d.name for d in _fit_dirs()}
    offenders = []
    for root in (INC, SRC):
        for path in sorted(root.rglob("*")):
            if path.is_dir() or "common" not in path.parts:
                continue
            for inc in re.findall(r'#include\s+"([^"]+)"', path.read_text(encoding="utf-8")):
                if any(f"/{t}/" in inc for t in tags):
                    offenders.append(f"{_rel(path)} includes {inc!r}")
    assert not offenders, "common/ depends on a specific fit:\n  " + "\n  ".join(offenders)


def test_namespace_matches_the_directory_tier() -> None:
    """A fit directory opens ``mcpu::forces::<tag>``; everything above it ``mcpu::forces``."""
    tags = {d.name for d in _fit_dirs()}
    offenders = []
    for path in _sources():
        text = path.read_text(encoding="utf-8")
        opened = re.findall(r"^namespace\s+(mcpu::forces[A-Za-z_:]*)\s*\{", text, re.MULTILINE)
        if not opened:
            continue
        parts = path.relative_to(REPO).parts
        tag = next((t for t in tags if t in parts), None)
        want = f"mcpu::forces::{tag}" if tag else "mcpu::forces"
        for got in opened:
            if got != want:
                offenders.append(f"{_rel(path)} opens {got!r}, expected {want!r}")
    assert not offenders, "namespace does not match the directory tier:\n  " + "\n  ".join(offenders)


def test_every_potential_lives_under_a_recognised_tier() -> None:
    """No stray ``*Potential.h`` directly under a family directory."""
    offenders = [
        _rel(p)
        for fam in _families()
        for p in fam.glob("*.h")
    ]
    assert not offenders, (
        "a header sits directly under a family directory; it belongs in "
        "common/ or a fit directory (see include/pymcpu/forces/README.md):\n  "
        + "\n  ".join(offenders)
    )


def test_the_readme_documenting_all_of_this_exists() -> None:
    assert (INC / "README.md").is_file(), "include/pymcpu/forces/README.md is the rule; keep it"
