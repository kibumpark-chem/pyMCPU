"""Every debug_force_* hook records the proposal it made.

The hooks build a move without committing it, so a test can inspect it
through last_move_kind(), last_moved_indices(), last_delta_energy() and
last_log_jacobian_weight(). Only debug_force_pivot and
debug_force_rama_pivot_to used to record anything; after the sidechain,
rotamer and rama-pivot hooks the getters still described the previous move.
"""

from __future__ import annotations

import math

import mdtraj as md
import numpy as np
import pytest

import pymcpu as mc
from pymcpu.forcefields.mcpu import MCPUForceField
from pymcpu.runners import default_example_pdb

PIVOT_RESIDUE = 5  # THR
SIDECHAIN_RESIDUE = 8  # TRP: two chi angles, and in the rotamer library


@pytest.fixture()
def sim():
    traj = md.load(str(default_example_pdb()))
    heavy = traj.atom_slice(traj.topology.select("not element H"))
    ff = MCPUForceField(heavy)
    simulation = mc.Simulation(heavy.topology, ff.create_system(heavy.topology), mc.Integrator(0.6))
    simulation.context.set_positions((ff.coords[0] * 10.0).T.astype(np.float32))
    simulation.context.calculate_total_energy(-1)
    simulation.integrator.set_seed(11)
    return simulation, ff


def _sidechain(ff: MCPUForceField, residue: int) -> set[int]:
    block = ff.blocks[residue]
    return set(range(block.sc_start, block.sc_start + block.sc_count))


HOOKS = {
    "sidechain": (lambda i, c: i.debug_force_sc(c, SIDECHAIN_RESIDUE), "Sidechain"),
    "rotamer": (lambda i, c: i.debug_force_rotamer(c, SIDECHAIN_RESIDUE), "Sidechain"),
    "rama_pivot": (lambda i, c: i.debug_force_rama_pivot(c, PIVOT_RESIDUE), "Pivot"),
    "rama_pivot_to": (lambda i, c: i.debug_force_rama_pivot_to(c, PIVOT_RESIDUE, -1.2, 2.3), "Pivot"),
    "pivot": (lambda i, c: i.debug_force_pivot(c, PIVOT_RESIDUE, True), "Pivot"),
}


@pytest.mark.parametrize("hook", sorted(HOOKS))
def test_a_forced_move_replaces_the_previous_record(sim, hook: str) -> None:
    simulation, ff = sim
    integ, ctx = simulation.integrator, simulation.context
    # Something else to overwrite: a psi pivot at a different residue.
    assert integ.debug_force_pivot(ctx, 2, False)
    before = list(integ.last_moved_indices())

    force, kind = HOOKS[hook]
    assert force(integ, ctx)
    moved = list(integ.last_moved_indices())
    assert integ.last_move_kind() == kind
    assert moved and moved != before
    assert math.isfinite(integ.last_delta_energy())
    assert math.isfinite(integ.last_log_jacobian_weight())
    if kind == "Sidechain":
        assert set(moved) <= _sidechain(ff, SIDECHAIN_RESIDUE)


def test_a_forced_move_is_not_committed(sim) -> None:
    simulation, _ = sim
    integ, ctx = simulation.integrator, simulation.context
    coords = np.array(ctx.coords, copy=True)
    energy = ctx.calculate_total_energy(-1)
    for force, _ in HOOKS.values():
        assert force(integ, ctx)
    assert np.array_equal(ctx.coords, coords)
    assert ctx.calculate_total_energy(-1) == energy


def test_the_same_seed_gives_the_same_forced_sidechain_move(sim) -> None:
    simulation, _ = sim
    integ, ctx = simulation.integrator, simulation.context
    records = []
    for _ in range(2):
        integ.set_seed(5)
        assert integ.debug_force_rotamer(ctx, SIDECHAIN_RESIDUE)
        records.append((integ.last_delta_energy(), list(integ.last_moved_indices())))
    assert records[0] == records[1]
