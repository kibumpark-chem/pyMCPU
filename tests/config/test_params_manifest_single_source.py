"""The registry ``layout`` block is the single source of truth for what a
parameter set contains.

Several places used to hand-maintain that list and had already drifted apart:

* ``pymcpu/data/params_registry.json`` (``files`` / ``core_files``) omitted
  ``mcpu_params/hbond_seq_dep.bin``, so ``ensure_params`` could report a
  directory complete and ``MCPUForceField._load_parameters`` would then raise
  ``FileNotFoundError`` on it.
* A release packer omitted ``constants/bbind02.May.lib``, so every archive
  it produced was rejected by ``pymcpu.params._ensure_files`` as incomplete.
* The two test stubs in this directory each mirrored a different subset.

All of them now read ``pymcpu.params.required_files()``. These tests pin that
for every registered set, and pin the two specific regressions by name.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path, PurePosixPath

import pytest

from pymcpu.params import (
    ParamsError,
    available_sets,
    constants_files,
    ensure_params,
    optional_files,
    required_files,
    shipped_constants,
    table_layout,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
SETS = available_sets()


def _table_folder(set_name: str) -> str:
    """The one folder that holds a set's ``.bin`` tables."""
    (folder,) = {PurePosixPath(rel).parts[0] for rel in table_layout(set_name).values()}
    return folder


@pytest.mark.parametrize("set_name", SETS)
def test_layout_is_present_and_ordered(set_name: str) -> None:
    required = required_files(set_name)
    assert required, "layout.required is empty"
    # Insertion order is load-bearing: it is the order _load_parameters
    # reports a missing file in.
    assert list(required)[0] == "amino acids template"
    assert set(optional_files(set_name)) == {"rama mixture"}
    assert len(table_layout(set_name)) == 6
    assert len(constants_files(set_name)) == 4


@pytest.mark.parametrize("set_name", SETS)
@pytest.mark.parametrize(
    "role, table",
    [
        # The registry used to omit this one -> ensure_params passed, the
        # force field then raised FileNotFoundError.
        ("hbond seq_dep", "hbond_seq_dep"),
        # A release packer used to omit this one -> its archives were rejected.
        ("rotamer library", None),
    ],
)
def test_previously_dropped_files_are_required(set_name: str, role: str, table: str | None) -> None:
    relpath = table_layout(set_name)[table] if table else "constants/bbind02.May.lib"
    assert required_files(set_name)[role] == relpath


@pytest.mark.parametrize("set_name", SETS)
def test_rama_mixture_is_optional_not_required(set_name: str) -> None:
    """The rama-mixture library must stay optional: the move it serves defaults
    to pivot_rama_probability = 0.0, so sets without it are valid."""
    rel = optional_files(set_name)["rama mixture"]
    assert rel not in required_files(set_name).values()
    # ...but it must still be declared as a constant so the archive ships it.
    assert rel in constants_files(set_name)


@pytest.mark.parametrize("set_name", SETS)
def test_a_wheel_can_satisfy_every_required_file(set_name: str) -> None:
    """Each required file is a table the archive carries or a file shipped
    next to it, so a set unpacked from the wheel is complete."""
    shippable = set(table_layout(set_name).values()) | set(shipped_constants(set_name))
    assert set(required_files(set_name).values()) <= shippable


def test_each_set_keeps_its_tables_in_a_folder_of_its_own() -> None:
    """The constants keep their names, but no two sets share a table folder,
    so a folder of one set (``MCPU_PARAMS_DIR``, ``param_dir``) given for
    another lacks every table the other requires and fails loudly instead of
    loading the wrong tables."""
    assert _table_folder("mcpu08") == "mcpu_params"
    assert _table_folder("mcpu26") == "mcpu26_params"
    for a, b in combinations(SETS, 2):
        assert _table_folder(a) != _table_folder(b), (a, b)
    for set_name in SETS:
        tables = set(table_layout(set_name).values())
        assert tables <= set(required_files(set_name).values()), set_name


def test_mcpu26_requires_its_manifest() -> None:
    """mcpu26's sheet factor and defaults come from its
    ``constants/paramset.json``: a folder of its tables without it would load
    with mcpu08's defaults, so the layout requires it. mcpu08 has none."""
    assert required_files("mcpu26")["parameter set manifest"] == "constants/paramset.json"
    assert "constants/paramset.json" not in required_files("mcpu08").values()


@pytest.mark.parametrize("set_name", SETS)
def test_the_shipped_constants_are_the_raw_trees(set_name: str) -> None:
    """``scripts/encode_params.py`` copies the raw tree's constants and
    manifest next to the archive; ``--check`` compares them. The raw tree's
    constants are tracked, so a checkout has both copies."""
    raw = _REPO_ROOT / "src" / "pymcpu" / "parameters" / "pretrained" / set_name
    shipped = _REPO_ROOT / "pymcpu" / "data" / "params" / set_name
    if not raw.is_dir() or not shipped.is_dir():
        pytest.skip("not a source tree")
    for rel in shipped_constants(set_name):
        assert (raw / rel).is_file() == (shipped / rel).is_file(), rel
        if (raw / rel).is_file():
            assert (raw / rel).read_bytes() == (shipped / rel).read_bytes(), rel


@pytest.mark.parametrize("set_name", SETS)
def test_tree_built_from_the_layout_satisfies_ensure_params(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, set_name: str
) -> None:
    """A directory containing exactly ``required_files()`` is accepted.

    This is the assertion that closes the "ensure_params succeeds, then the
    force field raises" hole: both sides now derive from the same dict, so a
    tree that satisfies one satisfies the other by construction.
    """
    for rel in required_files(set_name).values():
        dst = tmp_path / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(b"x")
    monkeypatch.setenv("MCPU_PARAMS_DIR", str(tmp_path))
    assert ensure_params(set_name) == tmp_path.resolve()


@pytest.mark.parametrize("set_name", SETS)
def test_dropping_any_required_file_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, set_name: str
) -> None:
    """Removing any single required file must fail the completeness check --
    otherwise the list is not actually enforced."""
    required = list(required_files(set_name).values())
    victim = table_layout(set_name)["hbond_seq_dep"]
    assert victim in required
    for rel in required:
        if rel == victim:
            continue
        dst = tmp_path / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(b"x")
    monkeypatch.setenv("MCPU_PARAMS_DIR", str(tmp_path))
    with pytest.raises(ParamsError) as excinfo:
        ensure_params(set_name)
    assert victim in str(excinfo.value)
