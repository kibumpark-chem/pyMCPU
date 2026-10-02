"""The native-contacts bias under the init_only atom reorder.

The reorder renumbers atoms into storage order and asks every energy term to
remap the atom ids it holds. The bias did not, so after the reorder it
measured distances between unrelated atoms (a native bias of 289560 instead
of 8). It now remaps its pairs, a term added after the reorder is remapped
when it is added, and a System that one Context has reordered cannot be
reordered again by another (REMD replicas share one System).

Actin, because init_only cannot place 1UAO's terminal OXT.
"""

from __future__ import annotations

import os

import mdtraj as md
import numpy as np
import pytest

from pymcpu import mcpu_core
from pymcpu.forcefields.mcpu import MCPUForceField
from pymcpu.sampling.collective_variables import (
    NativeContactsCV,
    attach_native_contacts_bias_potential,
    build_ca_index,
    reference_ca_from_pdb,
)
from tests.fixtures.context_builders import resolve_test_pdb

K, OFFSET, BIAS = 1.0, 4, 6  # native bias = 0.5 * K * OFFSET**2 = 8.0


@pytest.fixture(scope="module")
def heavy() -> md.Trajectory:
    traj = md.load(str(resolve_test_pdb()))
    return traj.atom_slice(traj.topology.select("not element H"))


@pytest.fixture(scope="module")
def ff(heavy) -> MCPUForceField:
    return MCPUForceField(heavy)


def _context(heavy, ff, mode: str, attach: str = "before"):
    """A Context with the bias attached before or after the reorder."""
    system = ff.create_system(heavy.topology)
    cv = NativeContactsCV(build_ca_index(ff), reference_ca_from_pdb(str(resolve_test_pdb())),
                          contact_cutoff=8.0, min_seq_sep=3, mode="hard")
    if attach == "before":
        attach_native_contacts_bias_potential(system, cv)
    ctx = mcpu_core.Context(system)
    ctx.set_atom_reorder_mode(mode)
    ctx.set_positions((ff.coords[0] * 10.0).T.astype(np.float32))
    if attach == "after":
        attach_native_contacts_bias_potential(system, cv)
    n0 = float(cv.n_contacts - OFFSET)
    ctx.set_native_contacts_bias(K, n0)
    return ctx, cv, n0


def _run(ctx, steps: int = 300, seed: int = 7) -> None:
    integ = mcpu_core.Integrator(temperature=0.6, step_size_rad=0.1)
    integ.set_seed(seed)
    integ.run(ctx, steps, 0)


@pytest.mark.parametrize("attach", ["before", "after"])
def test_the_bias_counts_native_contacts_after_the_reorder(heavy, ff, attach: str) -> None:
    ctx, cv, n0 = _context(heavy, ff, "init_only", attach)
    assert ctx.atom_permutation_info()["enabled"]
    assert ctx.calculate_total_energy(BIAS) == 0.5 * K * OFFSET**2
    _run(ctx)
    # The running bias against an independent count on the build-order coordinates.
    expected = 0.5 * K * (cv.compute_N(np.asarray(ctx.coords)) - n0) ** 2
    assert ctx.calculate_total_energy(BIAS) == pytest.approx(expected)


@pytest.mark.slow
def test_a_second_reorder_of_one_system_is_refused(heavy, ff) -> None:
    system = ff.create_system(heavy.topology)
    start = (ff.coords[0] * 10.0).T.astype(np.float32)
    first = mcpu_core.Context(system)
    first.set_atom_reorder_mode("init_only")
    first.set_positions(start)
    energy = first.calculate_total_energy(-1)

    second = mcpu_core.Context(system)
    with pytest.raises(RuntimeError, match="already reordered"):
        second.set_atom_reorder_mode("init_only")
    assert first.calculate_total_energy(-1) == energy


@pytest.mark.skipif(not os.environ.get("KORP_MAP_PATH"), reason="set KORP_MAP_PATH")
def test_the_bias_follows_the_reorder_under_korp(heavy) -> None:
    from pymcpu.forcefields.korp import KORPForceField

    korp = KORPForceField(heavy, map_path=os.environ["KORP_MAP_PATH"])
    energies = {}
    for mode in ("off", "init_only"):
        ctx, _, _ = _context(heavy, korp, mode)
        energies[mode] = (ctx.calculate_total_energy(BIAS), ctx.calculate_total_energy(7))
    assert energies["init_only"][0] == energies["off"][0] == 0.5 * K * OFFSET**2
    assert energies["init_only"][1] == energies["off"][1]
