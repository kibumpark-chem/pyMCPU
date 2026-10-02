"""Writing Context.coords after the init_only atom reorder.

With a reorder applied, the coords setter takes its own path instead of
set_positions. It used to skip invalidating Mu's live contact list, so a run
after a reset carried contacts of the old conformation, and under
set_output_internal_order(True) it treated the storage-order array it had
just handed out as build order and scrambled the atoms.
"""

from __future__ import annotations

import mdtraj as md
import numpy as np
import pytest

import pymcpu as mc
from pymcpu import mcpu_core
from pymcpu.forcefields.mcpu import MCPUForceField
from tests.fixtures.context_builders import resolve_test_pdb


@pytest.fixture(scope="module")
def heavy() -> md.Trajectory:
    # Actin: large enough for Mu's live contact list, and it has no terminal
    # OXT, which init_only cannot place.
    traj = md.load(str(resolve_test_pdb()))
    return traj.atom_slice(traj.topology.select("not element H"))


def _reordered(heavy):
    ff = MCPUForceField(heavy)
    ctx = mc.Simulation(heavy.topology, ff.create_system(heavy.topology), mc.Integrator(0.6)).context
    ctx.set_atom_reorder_mode("init_only")
    start = (ff.coords[0] * 10.0).T.astype(np.float32)
    ctx.set_positions(start)
    ctx.calculate_total_energy(-1)
    return ctx, start


def _run(ctx, seed: int, steps: int = 300) -> None:
    integ = mcpu_core.Integrator(temperature=0.6, step_size_rad=0.1)
    integ.set_seed(seed)
    integ.run(ctx, steps, 0)


def test_a_reset_through_the_coords_setter_matches_a_fresh_start(heavy) -> None:
    reset, start = _reordered(heavy)
    _run(reset, seed=99)
    reset.coords = start
    reset.calculate_total_energy(-1)
    _run(reset, seed=5)

    fresh, _ = _reordered(heavy)
    _run(fresh, seed=5)
    assert np.array_equal(reset.coords, fresh.coords)


def test_storage_order_coords_round_trip(heavy) -> None:
    ctx, _ = _reordered(heavy)
    ctx.set_output_internal_order(True)
    stored = np.array(ctx.coords, copy=True)
    energy = ctx.calculate_total_energy(-1)
    ctx.coords = stored
    assert np.array_equal(ctx.coords, stored)
    assert ctx.calculate_total_energy(-1) == energy
