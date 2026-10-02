"""Residue energy masks are honoured by Mu's rigid-move clash guard.

With ``ignore_all``, a pair involving a masked residue neither clashes nor
makes a contact. The full energy and the ordinary delta paths already did
this, but the guard that re-checks moved-moved pairs of a rigid move
(``skip_rigid_mm``, on by default) did not. A rigid move carrying an
overlap that the mask allows was therefore rejected as a steric clash, so
masked (linker) residues could get stuck.

``clash_only`` is deliberately asymmetric and stays so: moves are still
rejected on a clash, while the full energy carries no clash penalty
(legacy CLASH_WEIGHT=0).
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
PIVOT = 5  # a psi pivot here moves residues 6..9 rigidly
MASKED = 8


@pytest.fixture(scope="module")
def heavy() -> md.Trajectory:
    traj = md.load(str(default_example_pdb()))
    return traj.atom_slice(traj.topology.select("not element H"))


@pytest.fixture(scope="module")
def overlap(heavy):
    """Coordinates with TRP8's last sidechain atom 0.6 A from GLY6's CA."""
    ff = MCPUForceField(heavy)
    coords = np.ascontiguousarray((ff.coords[0] * 10.0).T.astype(np.float32))
    ca = ff.blocks[6].bb_start + 1
    tip = ff.blocks[MASKED].sc_start + ff.blocks[MASKED].sc_count - 1
    coords[:, tip] = coords[:, ca] + np.array([0.6, 0.0, 0.0], np.float32)
    return coords, (ca, tip)


def _sim(heavy, coords, mask_mode=None, *, skip_rigid_mm=True, double_boundary=False):
    ff = MCPUForceField(heavy)
    system = ff.create_system(heavy.topology)
    if mask_mode is not None:
        system.set_energy_ignored_residues([MASKED], mask_mode)
    # Small pivots: the forced move must not rotate far enough to make a real,
    # unmasked clash, whatever angle the seed draws.
    integ = mcpu_core.Integrator(temperature=0.6, step_size_rad=0.01)
    integ.set_seed(1234)
    sim = mc.Simulation(heavy.topology, system, integ)
    sim.context.set_skip_rigid_mm(skip_rigid_mm)
    sim.context.set_mm_double_boundary(double_boundary)
    sim.context.set_positions(coords)
    sim.context.calculate_total_energy(-1)
    return sim


def _forced_rigid_pivot(sim, pair) -> float:
    integ, ctx = sim.integrator, sim.context
    assert integ.debug_force_pivot(ctx, PIVOT, False)
    assert integ.last_move_is_rigid()
    assert set(pair) <= set(integ.last_moved_indices())  # the overlap moves rigidly
    return integ.last_delta_energy()


def test_the_overlap_is_a_real_clash_without_a_mask(heavy, overlap) -> None:
    coords, pair = overlap
    sim = _sim(heavy, coords)
    assert sim.context.has_steric_clash()
    assert _forced_rigid_pivot(sim, pair) > CLASH


@pytest.mark.parametrize("skip_rigid_mm", [True, False])
@pytest.mark.parametrize("double_boundary", [False, True])
def test_ignore_all_lets_a_rigid_move_carry_a_masked_overlap(
    heavy, overlap, skip_rigid_mm: bool, double_boundary: bool
) -> None:
    coords, pair = overlap
    sim = _sim(heavy, coords, "ignore_all", skip_rigid_mm=skip_rigid_mm,
               double_boundary=double_boundary)
    assert not sim.context.has_steric_clash()
    assert abs(_forced_rigid_pivot(sim, pair)) < CLASH


def test_clash_only_still_rejects_the_move(heavy, overlap) -> None:
    coords, pair = overlap
    sim = _sim(heavy, coords, "clash_only")
    assert _forced_rigid_pivot(sim, pair) > CLASH


def test_clearing_a_clash_only_mask_brings_clashes_back(heavy, overlap) -> None:
    """The full energy drops clashes while a clash_only mask is set. Once the
    mask is cleared it must report them again, as the delta path does."""
    coords, pair = overlap
    sim = _sim(heavy, coords, "clash_only")
    assert not sim.context.has_steric_clash()
    sim.system.clear_energy_ignored_residues()
    sim.context.calculate_total_energy(-1)
    assert sim.context.has_steric_clash()


def test_ignore_all_delta_matches_the_full_recompute(heavy, overlap) -> None:
    """The incremental Mu change of a rigid move equals the change in the
    full energy, as PhysicsVerifier measures it."""
    coords, pair = overlap
    sim = _sim(heavy, coords, "ignore_all")
    ctx, integ = sim.context, sim.integrator
    assert integ.debug_force_pivot(ctx, PIVOT, False)
    moved = list(integ.last_moved_indices())

    # Rotate the same moved set rigidly about the psi axis, CA(5) -> C(5).
    ff = MCPUForceField(heavy)
    a = coords[:, ff.blocks[PIVOT].bb_start + 1].astype(np.float64)
    b = coords[:, ff.blocks[PIVOT].c_start].astype(np.float64)
    k = (b - a) / np.linalg.norm(b - a)
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    rot = np.eye(3) + np.sin(0.05) * kx + (1 - np.cos(0.05)) * kx @ kx
    new_coords = coords.copy()
    new_coords[:, moved] = (rot @ (coords[:, moved] - a[:, None]) + a[:, None]).astype(np.float32)

    old_state = ctx.get_state()
    new_state = mcpu_core.State(old_state)
    new_state.coords = new_coords
    patch = mcpu_core.ProposalPatch(new_coords.shape[1])
    for atom in moved:
        patch.mark_moved(int(atom))
    patch.is_valid = True
    patch.is_rigid = True
    check = mcpu_core.PhysicsVerifier.verify_potential_delta(ctx, old_state, new_state, patch, 1, 1e-3)
    assert check.passed, check.message
    assert abs(check.delta_incremental) < CLASH
