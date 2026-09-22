"""The shipped compact parameter set must not be gitignored.

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
test pins it.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from pymcpu.params import constants_files, required_files

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SHIPPED_DIR = _REPO_ROOT / "pymcpu" / "data" / "params" / "mcpu_v1"


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


def test_compact_archive_exists() -> None:
    _require_git_checkout()
    archive = _SHIPPED_DIR / "tables.npz"
    assert archive.is_file(), (
        f"{archive} is missing -- regenerate with scripts/encode_params.py"
    )
    # ~2 MiB. A wildly different size means the encoder changed behavior.
    size_mib = archive.stat().st_size / (1 << 20)
    assert 0.5 < size_mib < 20, f"unexpected archive size: {size_mib:.2f} MiB"


def test_no_shipped_parameter_file_is_gitignored() -> None:
    """Every file under the shipped params dir must reach git (and the wheel)."""
    _require_git_checkout()
    shipped = sorted(p for p in _SHIPPED_DIR.rglob("*") if p.is_file())
    assert shipped, f"nothing under {_SHIPPED_DIR}"
    ignored = [str(p.relative_to(_REPO_ROOT)) for p in shipped if _is_git_ignored(p)]
    assert not ignored, (
        "these shipped parameter files are gitignored, so scikit-build-core "
        "would omit them from the wheel:\n  " + "\n  ".join(ignored)
        + "\nAdd a negation to the MUST SHIP block at the END of .gitignore "
        "(a negation placed earlier is overridden by later broad rules)."
    )


def test_dev_tree_required_constants_are_trackable() -> None:
    """``constants/atom_types.csv`` in the raw dev tree must be trackable too.

    A bare ``*.csv`` rule (intended for simulation output) used to hide it,
    which is why a fresh clone could not build a force field.
    """
    _require_git_checkout()
    raw_root = _REPO_ROOT / "src" / "pymcpu" / "parameters" / "pretrained" / "mcpu08"
    if not raw_root.is_dir():
        pytest.skip("raw parameter tree not present")
    offenders = []
    for rel in constants_files("mcpu_v1"):
        path = raw_root / rel
        if path.exists() and _is_git_ignored(path):
            offenders.append(rel)
    assert not offenders, f"gitignored required constants: {offenders}"


def test_shipped_set_covers_every_required_role() -> None:
    """The shipped dir plus the compact archive must satisfy the layout."""
    _require_git_checkout()
    from pymcpu.paramcodec import read_header

    archive = _SHIPPED_DIR / "tables.npz"
    tables = set(read_header(archive)["tables"])
    table_rels = {
        rel for rel in required_files("mcpu_v1").values() if rel.startswith("mcpu_params/")
    }
    const_rels = {
        rel for rel in required_files("mcpu_v1").values() if rel.startswith("constants/")
    }
    # every required .bin is carried by a table in the archive
    from pymcpu.params import table_layout

    carried = {rel for name, rel in table_layout("mcpu_v1").items() if name in tables}
    assert table_rels <= carried, f"required tables not in archive: {table_rels - carried}"
    # every required constant is present verbatim
    missing = [rel for rel in const_rels if not (_SHIPPED_DIR / rel).is_file()]
    assert not missing, f"required constants missing from the shipped set: {missing}"
