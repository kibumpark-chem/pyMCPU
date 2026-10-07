"""A Mu trial far outside the neighbour grid's box, under a residue energy mask.

The grid wraps, so such a trial takes the contact list like any other: its
clash test and contact walks run on wrapped cells, next to the atoms filed
in the same cells a grid period away. These tests pin down that, under a
mask, the delta of such a trial still equals the change in the full energy,
and that it is rejected for an overlap exactly when it overlaps something
the mask in force at that move leaves on (ignore_all switches a masked pair
off, clash_only keeps its clash), not the mask of an earlier move.

The trial: the atoms a psi pivot at residue 5 of chignolin moves, carried
100 A past the grid's box, except one sidechain atom of residue 8 left 0.6 A
from the CA of residue 3 (where it also overlaps residue 3's neighbours).
"""

from __future__ import annotations

import mdtraj as md
import numpy as np
import pytest

import pymcpu as mc
from pymcpu import mcpu_core
from pymcpu.forcefields.mcpu import MCPUForceField
from pymcpu.runners import default_example_pdb

CLASH = 1e4  # the clash sentinel, weighted, is about 4e4
PIVOT = 5
TAIL = [5, 6, 7, 8, 9]  # the pivot moves the carbonyl of residue 5 too
TIP_RES = 8  # its last sidechain atom stays behind, on top of ...
CA_RES = 3  # ... this residue's CA
SHIFT_A = 100.0


@pytest.fixture(scope="module")
def heavy() -> md.Trajectory:
    traj = md.load(str(default_example_pdb()))
    return traj.atom_slice(traj.topology.select("not element H"))


@pytest.fixture(scope="module")
def trial(heavy):
    """Native coordinates, the trial coordinates and the moved atoms."""
    ff = MCPUForceField(heavy)
    coords = np.ascontiguousarray((ff.coords[0] * 10.0).T.astype(np.float32))
    sim = _sim(heavy, coords)
    assert sim.integrator.debug_force_pivot(sim.context, PIVOT, False)
    moved = sorted(int(a) for a in sim.integrator.last_moved_indices())
    tip = ff.blocks[TIP_RES].sc_start + ff.blocks[TIP_RES].sc_count - 1
    ca = ff.blocks[CA_RES].bb_start + 1
    assert tip in moved and ca not in moved

    new_coords = coords.copy()
    new_coords[0, moved] += np.float32(SHIFT_A)
    new_coords[:, tip] = coords[:, ca] + np.array([0.6, 0.0, 0.0], np.float32)
    # Past the grid's box (2 cutoffs of margin around the structure) by far.
    margin = 2 * sim.context.mu_potential.mu_exact_cutoff
    assert new_coords[0].max() > coords[0].max() + 4 * margin
    return coords, new_coords, moved


def _sim(heavy, coords):
    ff = MCPUForceField(heavy)
    system = ff.create_system(heavy.topology)
    integ = mcpu_core.Integrator(temperature=0.6, step_size_rad=0.01)
    integ.set_seed(1234)
    sim = mc.Simulation(heavy.topology, system, integ)
    sim.context.set_positions(coords)
    sim.context.calculate_total_energy(-1)
    return sim


def _set_mask(sim, residues, mode) -> None:
    if residues:
        sim.system.set_energy_ignored_residues(residues, mode)
    else:
        sim.system.clear_energy_ignored_residues()


def _check(sim, new_coords, moved):
    """PhysicsVerifier's verdict on the (non-rigid) trial."""
    ctx = sim.context
    old_state = ctx.get_state()
    new_state = mcpu_core.State(old_state)
    new_state.coords = new_coords
    patch = mcpu_core.ProposalPatch(new_coords.shape[1])
    for atom in moved:
        patch.mark_moved(int(atom))
    patch.is_valid = True
    patch.is_rigid = False
    return mcpu_core.PhysicsVerifier.verify_potential_delta(
        ctx, old_state, new_state, patch, 1, 1e-3)


@pytest.mark.parametrize(
    ("residues", "mode", "clash"),
    [
        ([], None, True),
        ([1], "ignore_all", True),  # a mask elsewhere
        ([TIP_RES], "clash_only", True),
        ([CA_RES], "clash_only", True),
        ([TIP_RES], "ignore_all", False),
        (TAIL, "ignore_all", False),
    ],
    ids=["none", "elsewhere", "tip-clash-only", "ca-clash-only",
         "tip-ignore-all", "tail-ignore-all"],
)
def test_far_trial_under_a_mask(heavy, trial, residues, mode, clash) -> None:
    coords, new_coords, moved = trial
    sim = _sim(heavy, coords)
    _set_mask(sim, residues, mode)
    check = _check(sim, new_coords, moved)
    assert check.passed, check.message
    assert (check.delta_incremental >= CLASH) == clash


@pytest.mark.parametrize("residues", [[TIP_RES], TAIL], ids=["tip", "tail"])
def test_a_clash_only_far_trial_without_an_overlap(heavy, trial, residues) -> None:
    """The same trial with the tip carried along, so nothing overlaps. Under
    clash_only its delta still equals the change in the full energy."""
    coords, _, moved = trial
    new_coords = coords.copy()
    new_coords[0, moved] += np.float32(SHIFT_A)
    sim = _sim(heavy, coords)
    _set_mask(sim, residues, "clash_only")
    check = _check(sim, new_coords, moved)
    assert check.passed, check.message
    assert check.delta_incremental < CLASH


def test_the_clash_test_follows_mask_changes(heavy, trial) -> None:
    """One context, the same trial, the mask changed between the checks."""
    coords, new_coords, moved = trial
    sim = _sim(heavy, coords)
    for residues, mode, clash in [
        ([TIP_RES], "ignore_all", False),
        ([TIP_RES], "clash_only", True),
        ([TIP_RES], "ignore_all", False),
        ([], None, True),
        (TAIL, "ignore_all", False),
    ]:
        _set_mask(sim, residues, mode)
        check = _check(sim, new_coords, moved)
        assert check.passed, (residues, mode, check.message)
        assert (check.delta_incremental >= CLASH) == clash, (residues, mode)
