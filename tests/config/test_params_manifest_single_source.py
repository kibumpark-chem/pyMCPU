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

All of them now read ``pymcpu.params.required_files()``. These tests pin that,
and pin the two specific regressions by name.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pymcpu.params import (
    ParamsError,
    constants_files,
    ensure_params,
    optional_files,
    required_files,
    table_layout,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_layout_is_present_and_ordered() -> None:
    required = required_files("mcpu08")
    assert required, "layout.required is empty"
    # Insertion order is load-bearing: it is the order _load_parameters
    # reports a missing file in.
    assert list(required)[0] == "amino acids template"
    assert set(optional_files("mcpu08")) == {"rama mixture"}
    assert len(table_layout("mcpu08")) == 6
    assert len(constants_files("mcpu08")) == 4


@pytest.mark.parametrize(
    "relpath",
    [
        # The registry used to omit this one -> ensure_params passed, the
        # force field then raised FileNotFoundError.
        "mcpu_params/hbond_seq_dep.bin",
        # A release packer used to omit this one -> its archives were rejected.
        "constants/bbind02.May.lib",
    ],
)
def test_previously_dropped_files_are_required(relpath: str) -> None:
    assert relpath in required_files("mcpu08").values()


def test_rama_mixture_is_optional_not_required() -> None:
    """The rama-mixture library must stay optional: the move it serves defaults
    to pivot_rama_probability = 0.0, so sets without it are valid."""
    rel = optional_files("mcpu08")["rama mixture"]
    assert rel not in required_files("mcpu08").values()
    # ...but it must still be declared as a constant so the archive ships it.
    assert rel in constants_files("mcpu08")


def test_tree_built_from_the_layout_satisfies_ensure_params(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A directory containing exactly ``required_files()`` is accepted.

    This is the assertion that closes the "ensure_params succeeds, then the
    force field raises" hole: both sides now derive from the same dict, so a
    tree that satisfies one satisfies the other by construction.
    """
    for rel in required_files("mcpu08").values():
        dst = tmp_path / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(b"x")
    monkeypatch.setenv("MCPU_PARAMS_DIR", str(tmp_path))
    assert ensure_params("mcpu08") == tmp_path.resolve()


def test_dropping_any_required_file_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Removing any single required file must fail the completeness check --
    otherwise the list is not actually enforced."""
    required = list(required_files("mcpu08").values())
    victim = "mcpu_params/hbond_seq_dep.bin"
    assert victim in required
    for rel in required:
        if rel == victim:
            continue
        dst = tmp_path / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(b"x")
    monkeypatch.setenv("MCPU_PARAMS_DIR", str(tmp_path))
    with pytest.raises(ParamsError) as excinfo:
        ensure_params("mcpu08")
    assert victim in str(excinfo.value)
