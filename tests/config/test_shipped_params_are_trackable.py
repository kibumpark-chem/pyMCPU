"""The shipped compact parameter sets must not be gitignored.

This guards the single most dangerous failure mode in the packaging work:
**scikit-build-core filters wheel contents through .gitignore**. Its wheel
path (``build/_pathutil.py:packages_to_file_mapping``) calls
``each_unignored_file(..., mode=mode)``, and ``build/_file_processor.py``
sets ``reads_gitignore = mode in {"classic", "default"}`` -- ``"default"``
being the resolved default. So an over-broad ignore rule does not merely keep
a file out of git, it silently ships a wheel that cannot run.

That is not hypothetical here. It has happened twice:

* ``*.bin`` and ``*.csv`` rules hid ``constants/atom_types.csv`` and every
  potential table, so a fresh clone could not build a force field.
* A ``!pymcpu/data/params/**`` negation placed next to the section it related
  to was silently overridden by a later ``*.npz`` rule, because **git honors
  the last matching pattern**. That would have shipped a parameter-less wheel.

Hence the must-ship negations live at the very end of ``.gitignore``, and this
test pins it, for every registered set. The other way round, each set's raw
``.bin`` tables (678 MiB) must stay ignored: only the compact archive ships.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from pymcpu.params import available_sets, required_files, shipped_constants, table_layout

_REPO_ROOT = Path(__file__).resolve().parents[2]
SETS = available_sets()


def _shipped_dir(set_name: str) -> Path:
    return _REPO_ROOT / "pymcpu" / "data" / "params" / set_name


def _raw_root(set_name: str) -> Path:
    return _REPO_ROOT / "src" / "pymcpu" / "parameters" / "pretrained" / set_name


def _is_git_ignored(path: Path) -> bool:
    """True if git would ignore ``path`` (exit 0 from ``git check-ignore``)."""
    result = subprocess.run(
        ["git", "check-ignore", "-q", str(path)],
        cwd=_REPO_ROOT,
        capture_output=True,
    )
    return result.returncode == 0


def _require_git_checkout() -> None:
    if not (_REPO_ROOT / ".git").exists():
        pytest.skip("not a git checkout (installed-wheel run)")


def test_every_set_is_registered() -> None:
    assert {"mcpu08", "mcpu26"} <= set(SETS)


@pytest.mark.parametrize("set_name", SETS)
def test_compact_archive_exists(set_name: str) -> None:
    _require_git_checkout()
    archive = _shipped_dir(set_name) / "tables.npz"
    assert archive.is_file(), (
        f"{archive} is missing -- regenerate with scripts/encode_params.py --set {set_name}"
    )
    # ~2 MiB for mcpu08's integer tables, ~8 MiB for mcpu26's float32 ones. A
    # wildly different size means the encoder changed behavior.
    size_mib = archive.stat().st_size / (1 << 20)
    assert 0.5 < size_mib < 20, f"unexpected archive size: {size_mib:.2f} MiB"


@pytest.mark.parametrize("set_name", SETS)
def test_no_shipped_parameter_file_is_gitignored(set_name: str) -> None:
    """Every file under a shipped params dir must reach git (and the wheel)."""
    _require_git_checkout()
    shipped_dir = _shipped_dir(set_name)
    shipped = sorted(p for p in shipped_dir.rglob("*") if p.is_file())
    assert shipped, f"nothing under {shipped_dir}"
    ignored = [str(p.relative_to(_REPO_ROOT)) for p in shipped if _is_git_ignored(p)]
    assert not ignored, (
        "these shipped parameter files are gitignored, so scikit-build-core "
        "would omit them from the wheel:\n  " + "\n  ".join(ignored)
        + "\nAdd a negation to the MUST SHIP block at the END of .gitignore "
        "(a negation placed earlier is overridden by later broad rules)."
    )


@pytest.mark.parametrize("set_name", SETS)
def test_dev_tree_required_constants_are_trackable(set_name: str) -> None:
    """The raw dev tree's constants (``constants/atom_types.csv`` and the
    rest, and the set's ``constants/paramset.json``) must be trackable too.

    A bare ``*.csv`` rule (intended for simulation output) used to hide
    ``atom_types.csv``, which is why a fresh clone could not build a force
    field.
    """
    _require_git_checkout()
    raw_root = _raw_root(set_name)
    if not raw_root.is_dir():
        pytest.skip("raw parameter tree not present")
    offenders = []
    for rel in shipped_constants(set_name):
        path = raw_root / rel
        if path.exists() and _is_git_ignored(path):
            offenders.append(rel)
    assert not offenders, f"gitignored required constants: {offenders}"


@pytest.mark.parametrize("set_name", SETS)
def test_dev_tree_tables_stay_gitignored(set_name: str) -> None:
    """A set's raw ``.bin`` tables are large and regenerated offline; only the
    compact archive is committed. Each set keeps them in a folder of its own
    (``mcpu_params/``, ``mcpu26_params/``), which ``.gitignore`` matches by a
    glob, not by one set's spelling."""
    _require_git_checkout()
    tracked = [
        rel for rel in table_layout(set_name).values()
        if not _is_git_ignored(_raw_root(set_name) / rel)
    ]
    assert not tracked, (
        f"{set_name}'s raw tables are not gitignored, so they would be committed "
        f"and shipped in the sdist: {tracked}"
    )


@pytest.mark.parametrize("set_name", SETS)
def test_shipped_set_covers_every_required_role(set_name: str) -> None:
    """The shipped dir plus the compact archive must satisfy the layout."""
    _require_git_checkout()
    from pymcpu.paramcodec import read_header

    shipped_dir = _shipped_dir(set_name)
    tables = set(read_header(shipped_dir / "tables.npz")["tables"])
    layout = table_layout(set_name)
    table_rels = set(layout.values())
    required = set(required_files(set_name).values())
    # every required .bin is carried by a table in the archive
    carried = {rel for name, rel in layout.items() if name in tables}
    assert required & table_rels <= carried, (
        f"required tables not in archive: {(required & table_rels) - carried}"
    )
    # every other required file is present verbatim
    missing = [rel for rel in sorted(required - table_rels) if not (shipped_dir / rel).is_file()]
    assert not missing, f"required constants missing from the shipped set: {missing}"
