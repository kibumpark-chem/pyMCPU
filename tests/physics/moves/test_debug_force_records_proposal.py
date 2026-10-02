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


def test_a_forced_move_leaves_no_trace_in_a_later_run() -> None:
    """The energy terms queue the changes a move would commit; a forced move
    must not leave them for the next accepted step of run() to commit. With
    the native-contacts bias, whose queued pair flips used to survive, a run
    after forced moves must match a run without them."""
    from pymcpu.sampling.collective_variables import (
        NativeContactsCV,
        attach_native_contacts_bias_potential,
        build_contact_atom_index,
        reference_contact_from_pdb,
    )

    pdb = str(default_example_pdb())
    traj = md.load(pdb)
    heavy = traj.atom_slice(traj.topology.select("not element H"))

    def biased_context():
        ff = MCPUForceField(heavy)
        system = ff.create_system(heavy.topology)
        cv = NativeContactsCV(build_contact_atom_index(ff, "ca"), reference_contact_from_pdb(pdb, "ca"),
                              contact_cutoff=8.0, min_seq_sep=3)
        attach_native_contacts_bias_potential(system, cv)
        ctx = mc.Simulation(heavy.topology, system, mc.Integrator(0.6)).context
        ctx.set_native_contacts_bias(10.0, 6.5)
        ctx.set_positions((ff.coords[0] * 10.0).T.astype(np.float32))
        ctx.calculate_total_energy(-1)
        return ctx

    forced, plain = biased_context(), biased_context()
    hooks = mc.Integrator(0.6)
    hooks.set_seed(2)
    for force, _ in HOOKS.values():
        assert force(hooks, forced)

    results = []
    for ctx in (forced, plain):
        integ = mc.Integrator(0.6, step_size_rad=0.05)
        integ.set_seed(5)
        integ.run(ctx, 300, 0)
        results.append((np.array(ctx.coords, copy=True), ctx.calculate_total_energy(6)))
    assert np.array_equal(results[0][0], results[1][0])
    assert results[0][1] == results[1][1]


@pytest.mark.parametrize("hook", ["pivot", "sidechain", "rotamer"])
def test_a_fixed_residue_is_refused_as_in_run(sim, hook: str) -> None:
    simulation, ff = sim
    integ, ctx = simulation.integrator, simulation.context
    residue = PIVOT_RESIDUE if hook == "pivot" else SIDECHAIN_RESIDUE
    integ.set_fixed_residues([residue], ff.n_res)
    before = integ.get_fixed_rejected()
    force, _ = HOOKS[hook]
    assert not force(integ, ctx)
    assert integ.get_fixed_rejected() == before + 1


def test_the_same_seed_gives_the_same_forced_sidechain_move(sim) -> None:
    simulation, _ = sim
    integ, ctx = simulation.integrator, simulation.context
    records = []
    for _ in range(2):
        integ.set_seed(5)
        assert integ.debug_force_rotamer(ctx, SIDECHAIN_RESIDUE)
        records.append((integ.last_delta_energy(), list(integ.last_moved_indices())))
    assert records[0] == records[1]
