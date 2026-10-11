"""Build a wheel and assert what is (and is not) inside it.

This is the gate for the packaging failure mode that is easiest to ship by
accident: **scikit-build-core filters wheel contents through .gitignore**, and
it **dereferences symlinks** when copying ``wheel.packages``. Neither is
visible from the source tree, so only inspecting a built wheel catches it.

Both have already bitten this project:

* a ``!pymcpu/data/params/**`` negation placed mid-file was overridden by a
  later ``*.npz`` rule (git honors the LAST match), which would have shipped a
  wheel with no parameter tables at all;
* ``pymcpu/parameters`` was a symlink to ``../src/pymcpu/parameters``, and the
  wheel build followed it -- pulling in both raw parameter sets, so the 633 MB
  sidechain table appeared **twice**: 1,357 MiB of a 1,371 MiB wheel.

Marked ``slow`` because it compiles the extension (~1-2 min).
"""

from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Uncompressed budget. Code + the ~2 MiB compact parameters + the extension is
# well under this; the symlink-deref bug was 1,371 MiB.
_MAX_UNCOMPRESSED_MIB = 60.0

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def built_wheel(tmp_path_factory) -> Path:
    if not (_REPO_ROOT / "pyproject.toml").is_file():
        pytest.skip("not a source checkout")
    out = tmp_path_factory.mktemp("wheel")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "wheel", str(_REPO_ROOT),
         "--no-deps", "--no-build-isolation", "-w", str(out)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.skip(
            "wheel build unavailable in this environment "
            f"(exit {result.returncode}):\n{result.stdout[-1500:]}"
        )
    wheels = list(out.glob("*.whl"))
    assert len(wheels) == 1, f"expected exactly one wheel, got {wheels}"
    return wheels[0]


def _members(wheel: Path) -> dict[str, int]:
    with zipfile.ZipFile(wheel) as archive:
        return {info.filename: info.file_size for info in archive.infolist()}


def test_wheel_contains_the_compact_parameters(built_wheel: Path) -> None:
    """Without this the installed package cannot build a force field at all."""
    members = _members(built_wheel)
    archive = "pymcpu/data/params/mcpu08/tables.npz"
    assert archive in members, (
        f"{archive} is missing from the wheel. Most likely a .gitignore rule "
        "matches it -- add a negation to the MUST SHIP block at the END of "
        ".gitignore (an earlier negation is overridden by later broad rules)."
    )
    assert members[archive] > 100_000, "tables.npz looks truncated"

    from pymcpu.params import constants_files

    for rel in constants_files("mcpu08"):
        member = f"pymcpu/data/params/mcpu08/{rel}"
        if rel.endswith("rama_mixture.json"):
            continue  # optional
        assert member in members, f"required constant missing from wheel: {rel}"


def test_wheel_has_exactly_one_extension_module(built_wheel: Path) -> None:
    shared = [name for name in _members(built_wheel) if name.endswith(".so")]
    assert shared == ["pymcpu/mcpu_core.cpython-"
                      f"{sys.version_info.major}{sys.version_info.minor}"
                      "-x86_64-linux-gnu.so"] or len(shared) == 1, shared


def test_wheel_does_not_ship_the_raw_parameter_tree(built_wheel: Path) -> None:
    """The 685 MB of raw .bin tables must never reach the wheel."""
    offenders = [
        name for name, size in _members(built_wheel).items()
        if name.startswith("pymcpu/parameters/") or (
            name.endswith(".bin") and "data/params" not in name
        )
    ]
    assert not offenders, (
        "the raw parameter tree leaked into the wheel -- check that "
        "pymcpu/parameters is not a symlink and that wheel.exclude covers it:\n  "
        + "\n  ".join(sorted(offenders)[:10])
    )


def test_wheel_does_not_ship_headers_or_notebooks(built_wheel: Path) -> None:
    members = _members(built_wheel)
    # Eigen's install() rules come along via FetchContent unless excluded.
    assert not [n for n in members if n.startswith("include/")], "C++ headers in wheel"
    assert not [n for n in members if n.endswith((".ipynb", ".slurm"))], (
        "docs notebooks / HPC job scripts in wheel"
    )


def test_wheel_uncompressed_size_is_sane(built_wheel: Path) -> None:
    total_mib = sum(_members(built_wheel).values()) / (1 << 20)
    assert total_mib < _MAX_UNCOMPRESSED_MIB, (
        f"wheel expands to {total_mib:.1f} MiB (budget {_MAX_UNCOMPRESSED_MIB} MiB). "
        "A symlink is probably being dereferenced into the package."
    )

def test_wheel_does_not_ship_a_cmake_package(built_wheel: Path) -> None:
    """No ``share/`` -- Eigen's install() rules leak a CMake package.

    This is not hypothetical tidiness. FetchContent makes Eigen's own
    ``install()`` rules part of the build, so ``cmake --install`` wrote
    ``share/eigen3/cmake/Eigen3Config.cmake`` and ``share/pkgconfig/eigen3.pc``
    into site-packages, listed in pymcpu's own RECORD.

    Once ``wheel.exclude`` stopped shipping Eigen's *headers*, what remained
    was a HALF-INSTALLED Eigen3: a CMake package whose imported target pointed
    at an include directory that no longer existed. A later
    ``find_package(Eigen3)`` then selected it in preference to a real Eigen and
    failed the configure with "Imported target Eigen3::Eigen includes
    non-existent path" -- a Python wheel breaking an unrelated C++ build.
    """
    offenders = [
        name
        for name in _members(built_wheel)
        if name.startswith(("share/", "lib/", "bin/", "cmake/"))
    ]
    assert not offenders, (
        f"wheel installs non-package paths: {offenders}. A Python wheel must "
        f"not place CMake packages or pkg-config files in the environment."
    )


def test_wheel_ships_the_example_structure(built_wheel: Path) -> None:
    """The documented quickstart loads this file.

    ``default_example_pdb()`` used to resolve ``<package parent>/examples/...``,
    which is the repo root only for an editable install; from a wheel it
    pointed at a nonexistent ``site-packages/examples/`` and the README
    quickstart raised ``OSError: No such file`` for every pip user.
    """
    members = _members(built_wheel)
    assert "pymcpu/data/1uao.pdb" in members, (
        "pymcpu/data/1uao.pdb is missing, so pymcpu.runners.default_example_pdb() "
        "cannot resolve and the documented quickstart fails for a wheel install"
    )
    assert members["pymcpu/data/1uao.pdb"] > 1000, "1uao.pdb looks truncated"


def test_wheel_top_level_is_only_the_package(built_wheel: Path) -> None:
    """A wheel should own exactly its package plus its dist-info."""
    tops = {name.split("/", 1)[0] for name in _members(built_wheel)}
    unexpected = tops - {"pymcpu", "pymcpu-0.1.0.dist-info"}
    unexpected = {t for t in unexpected if not t.endswith(".dist-info")}
    assert not unexpected, f"unexpected top-level entries in wheel: {sorted(unexpected)}"


def test_wheel_records_its_cpu_baseline(built_wheel: Path) -> None:
    """pymcpu/_build_arch.py names the baseline the extension was built for;
    the import checks the CPU against it before loading the extension (see
    pymcpu._cpu_check). The published wheels are built for x86-64-v3."""
    import os
    import re

    with zipfile.ZipFile(built_wheel) as archive:
        source = archive.read("pymcpu/_build_arch.py").decode()
    tier = re.search(r'^TIER = "([^"]*)"', source, re.MULTILINE)
    assert tier, source
    assert tier.group(1) in ("v2", "v3", "v4", "none")
    if not os.environ.get("MCPU_ARCH"):
        assert tier.group(1) == "v3"
