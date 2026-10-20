"""The shipped compact parameters must be physically indistinguishable from
the raw ``.bin`` tables, for every registered set.

This is the assertion that makes the whole packaging change safe: the wheel
ships a few-MiB archive instead of 678 MiB of float32, and the engine must not
be able to tell. Energies are compared with ``==``, not ``pytest.approx`` --
anything less would let a real divergence through, and the suite elsewhere
pins exact accept-bit streams that a 1-ulp energy change would decorrelate.

The raw tables are gitignored, so this runs only in a checkout that has a
set's raw tree (``src/pymcpu/parameters/pretrained/<set>``); it skips that
set elsewhere.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from pymcpu import mcpu_core
from pymcpu.params import (
    available_sets,
    bundled_tables_path,
    materialize_from_wheel,
    required_files,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
SETS = available_sets()
#: Each set's raw tree, the reference the archive must reproduce.
_RAW_ROOTS = {s: _REPO_ROOT / "src" / "pymcpu" / "parameters" / "pretrained" / s for s in SETS}


def _require_raw_tree(set_name: str) -> Path:
    raw = _RAW_ROOTS[set_name]
    missing = [r for r in required_files(set_name).values() if not (raw / r).is_file()]
    if missing:
        pytest.skip(f"raw reference tree incomplete: {missing[:2]}")
    return raw


@pytest.fixture(scope="module", params=SETS)
def materialized(request, tmp_path_factory) -> tuple[str, Path]:
    set_name = request.param
    if bundled_tables_path(set_name) is None:
        pytest.skip("no in-wheel parameter archive (run scripts/encode_params.py)")
    cache = tmp_path_factory.mktemp(f"matcache_{set_name}")
    old = os.environ.get("MCPU_CACHE_DIR")
    os.environ["MCPU_CACHE_DIR"] = str(cache)
    try:
        yield set_name, materialize_from_wheel(set_name, verify=True)
    finally:
        if old is None:
            os.environ.pop("MCPU_CACHE_DIR", None)
        else:
            os.environ["MCPU_CACHE_DIR"] = old


def test_every_required_file_is_bit_exact(materialized: tuple[str, Path]) -> None:
    """Decoded tables and copied constants must match the raw tree byte for byte."""
    set_name, root = materialized
    raw = _require_raw_tree(set_name)
    for relpath in required_files(set_name).values():
        produced, reference = root / relpath, raw / relpath
        assert produced.is_file(), f"materialization omitted {relpath}"
        if relpath.endswith(".bin"):
            a = np.fromfile(produced, dtype=np.float32)
            b = np.fromfile(reference, dtype=np.float32)
            assert a.shape == b.shape, f"{relpath}: shape {a.shape} != {b.shape}"
            # bitwise: -0.0 == 0.0 under float comparison
            assert np.array_equal(a.view(np.uint32), b.view(np.uint32)), relpath
        else:
            assert produced.read_bytes() == reference.read_bytes(), relpath


def test_materialization_is_idempotent(materialized: tuple[str, Path]) -> None:
    """A second call must return the same content-addressed directory."""
    set_name, root = materialized
    again = materialize_from_wheel(set_name, verify=False)
    assert again == root
    # name encodes the archive digest, so a table change yields a new directory
    assert again.name.startswith(f"{set_name}-")


def _build(set_name: str, param_dir: Path, steps: int):
    md = pytest.importorskip("mdtraj")
    import pymcpu as mc
    from pymcpu.forcefields.mcpu import MCPUForceField

    pdb = _REPO_ROOT / "examples" / "data" / "1uao.pdb"
    if not pdb.is_file():
        pytest.skip("examples/data/1uao.pdb not present")
    traj = md.load(str(pdb))
    heavy = traj.atom_slice(traj.topology.select("not element H"))
    forcefield = MCPUForceField(heavy, param_set=set_name, param_dir=str(param_dir))
    system = forcefield.create_system(heavy.topology)
    integrator = mcpu_core.Integrator(0.6, 0.1)
    integrator.set_seed(42)
    simulation = mc.Simulation(heavy.topology, system, integrator)
    simulation.context.set_positions(
        (forcefield.coords[0] * 10.0).T.astype(np.float32)
    )
    if steps:
        simulation.step(steps)
    breakdown = simulation.context.energy_breakdown(weighted=False)
    bits = list(integrator.last_accept_bits()) if steps else []
    return breakdown, bits


def test_initial_energies_are_identical(materialized: tuple[str, Path]) -> None:
    set_name, root = materialized
    raw_root = _require_raw_tree(set_name)
    raw, _ = _build(set_name, raw_root, 0)
    mat, _ = _build(set_name, root, 0)
    assert mat["raw_total"] == raw["raw_total"]
    for group in raw["by_group"]:
        assert mat["by_group"][group] == raw["by_group"][group], f"energy group {group}"


@pytest.mark.slow
def test_trajectory_and_accept_bits_are_identical(materialized: tuple[str, Path]) -> None:
    """The strongest form: 2000 MC steps must produce the same accept-bit
    stream. A single differing bit would decorrelate the trajectory."""
    set_name, root = materialized
    raw_root = _require_raw_tree(set_name)
    raw, raw_bits = _build(set_name, raw_root, 2000)
    mat, mat_bits = _build(set_name, root, 2000)
    assert len(raw_bits) == 2000
    assert mat_bits == raw_bits, "accept-bit streams diverged"
    assert mat["raw_total"] == raw["raw_total"]
    for group in raw["by_group"]:
        assert mat["by_group"][group] == raw["by_group"][group], f"energy group {group}"
