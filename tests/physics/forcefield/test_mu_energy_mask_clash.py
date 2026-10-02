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

import os
import subprocess
import sys
import textwrap
from pathlib import Path

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


def _load_heavy() -> md.Trajectory:
    traj = md.load(str(default_example_pdb()))
    return traj.atom_slice(traj.topology.select("not element H"))


def _make_overlap(heavy):
    """Coordinates with TRP8's last sidechain atom 0.6 A from GLY6's CA."""
    ff = MCPUForceField(heavy)
    coords = np.ascontiguousarray((ff.coords[0] * 10.0).T.astype(np.float32))
    ca = ff.blocks[6].bb_start + 1
    tip = ff.blocks[MASKED].sc_start + ff.blocks[MASKED].sc_count - 1
    coords[:, tip] = coords[:, ca] + np.array([0.6, 0.0, 0.0], np.float32)
    return coords, (ca, tip)


@pytest.fixture(scope="module")
def heavy() -> md.Trajectory:
    return _load_heavy()


@pytest.fixture(scope="module")
def overlap(heavy):
    return _make_overlap(heavy)


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


def _check_rigid(ctx, new_coords, moved):
    """PhysicsVerifier's verdict on moving ``moved`` rigidly to ``new_coords``."""
    old_state = ctx.get_state()
    new_state = mcpu_core.State(old_state)
    new_state.coords = new_coords
    patch = mcpu_core.ProposalPatch(new_coords.shape[1])
    for atom in moved:
        patch.mark_moved(int(atom))
    patch.is_valid = True
    patch.is_rigid = True
    return mcpu_core.PhysicsVerifier.verify_potential_delta(ctx, old_state, new_state, patch, 1, 1e-3)


def _moved_set(sim) -> list[int]:
    assert sim.integrator.debug_force_pivot(sim.context, PIVOT, False)
    return list(sim.integrator.last_moved_indices())


def test_ignore_all_delta_matches_the_full_recompute(heavy, overlap) -> None:
    """The incremental Mu change of a rigid move equals the change in the
    full energy, as PhysicsVerifier measures it."""
    coords, pair = overlap
    sim = _sim(heavy, coords, "ignore_all")
    moved = _moved_set(sim)

    # Rotate the same moved set rigidly about the psi axis, CA(5) -> C(5).
    ff = MCPUForceField(heavy)
    a = coords[:, ff.blocks[PIVOT].bb_start + 1].astype(np.float64)
    b = coords[:, ff.blocks[PIVOT].c_start].astype(np.float64)
    k = (b - a) / np.linalg.norm(b - a)
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    rot = np.eye(3) + np.sin(0.05) * kx + (1 - np.cos(0.05)) * kx @ kx
    new_coords = coords.copy()
    new_coords[:, moved] = (rot @ (coords[:, moved] - a[:, None]) + a[:, None]).astype(np.float32)

    check = _check_rigid(sim.context, new_coords, moved)
    assert check.passed, check.message
    assert abs(check.delta_incremental) < CLASH


def _far_move(heavy, overlap, *, use_cell_pair: bool):
    """The clash_only overlap carried 7.5 A by its rigid segment: the
    simulation, the moved coordinates and the moved atoms."""
    coords, _ = overlap
    sim = _sim(heavy, coords, "clash_only")
    sim.context.set_use_cell_pair(use_cell_pair)
    sim.context.set_cell_pair_min_moved(1)
    moved = _moved_set(sim)
    new_coords = coords.copy()
    new_coords[:, moved] += np.array([[0.0], [-7.5], [0.0]], dtype=np.float32)

    # The segment lands clear of every other atom, so the only clash is the
    # carried overlap, and inside the neighbour grid (2 cutoffs of margin
    # around the structure), so the grid path is what is tested.
    fixed = np.setdiff1d(np.arange(coords.shape[1]), moved)
    gaps = np.linalg.norm(new_coords[:, moved][:, :, None] - coords[:, fixed][:, None, :], axis=0)
    assert gaps.min() > 3.2
    margin = 2 * 5.0765 - 0.5
    assert np.all(new_coords.min(axis=1) > coords.min(axis=1) - margin)
    assert np.all(new_coords.max(axis=1) < coords.max(axis=1) + margin)
    return sim, new_coords, moved


@pytest.mark.parametrize("use_cell_pair", [False, True])
def test_a_far_rigid_move_still_finds_its_moved_moved_clash(heavy, overlap, use_cell_pair: bool) -> None:
    """A rigid move's moved-moved pairs keep their distance, so Mu re-checks
    them for a hard-core clash instead of scoring them. That check looked for
    each atom's moved partners around its NEW position, in a grid of OLD
    positions, and so lost them once the segment travelled about a cell: a
    masked run (which always takes this path) could accept a clash."""
    sim, new_coords, moved = _far_move(heavy, overlap, use_cell_pair=use_cell_pair)
    visits_before = sim.context.neighbor_proxy_stats()["neighbor_num_cell_visits"]
    check = _check_rigid(sim.context, new_coords, moved)
    # The per-atom walk counts cell visits and the cell-pair path does not,
    # so this confirms which path ran (MCPU_USE_CELL_PAIR would override it).
    walked = sim.context.neighbor_proxy_stats()["neighbor_num_cell_visits"] > visits_before
    assert walked == (not use_cell_pair)
    assert check.delta_incremental > CLASH


def test_the_pivot_breakdown_diagnostic_finds_it_too() -> None:
    """MCPU_PIVOT_MU_BREAKDOWN takes a path of its own, which had no
    moved-moved clash check at all. It is read once per process, so this
    runs in a fresh one."""
    code = textwrap.dedent("""
        from tests.physics.forcefield import test_mu_energy_mask_clash as t
        heavy = t._load_heavy()
        sim, new_coords, moved = t._far_move(heavy, t._make_overlap(heavy), use_cell_pair=False)
        print("DELTA", t._check_rigid(sim.context, new_coords, moved).delta_incremental)
    """)
    repo = Path(__file__).resolve().parents[3]
    env = dict(os.environ, MCPU_PIVOT_MU_BREAKDOWN="1")
    run = subprocess.run([sys.executable, "-c", code], cwd=repo, env=env,
                         capture_output=True, text=True, check=True)
    delta = [line for line in run.stdout.splitlines() if line.startswith("DELTA ")]
    assert float(delta[-1].split()[1]) > CLASH
