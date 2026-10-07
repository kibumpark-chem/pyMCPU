"""A rigid pivot scores the carried pairs that rounding takes across their contact cutoff.

A rigid pivot does not re-measure the pairs it carries: the rotation keeps
their distances. It rounds each carried coordinate to float, though, so a pair
sitting on its contact cutoff can be carried across it. The contact list
therefore also holds every contact pair less than 0.05 A outside its cutoff,
and a rigid move re-decides each listed pair it carries. A move that leaves the
neighbour grid re-decides them too: the listed ones, or every carried pair if
there is no list.

TYR1 CD2 and GLY6 CA, a contact pair, are placed just inside or just outside
their contact distance; residues 0-6 are turned rigidly by small random angles (in
double and rounded once, as a pivot does), and PhysicsVerifier compares the
incremental energy change with the full one.
"""

from __future__ import annotations

import mdtraj as md
import numpy as np
import pytest

from pymcpu import mcpu_core
from pymcpu.forcefields.mcpu import MCPUForceField
from pymcpu.runners import default_example_pdb
from tests.fixtures.context_builders import resolve_test_pdb
from tests.physics.forcefield.test_mu_contact_list_resync import CONTACT_R, _atom

TRIALS = 200


def _setup(outside: bool, mask: str | None = None):
    traj = md.load(str(default_example_pdb()))
    heavy = traj.atom_slice(traj.topology.select("not element H"))
    ff = MCPUForceField(heavy)
    i, j = _atom(ff, 1, "CD2"), _atom(ff, 6, "CA")
    start = np.ascontiguousarray((ff.coords[0] * 10.0).T.astype(np.float32))
    away = start[:, j].astype(np.float64) - start[:, i]
    away /= np.linalg.norm(away)
    coords = start.copy()
    coords[:, j] = (start[:, i] + CONTACT_R * away).astype(np.float32)
    for step in range(1000):  # just outside: the first float position past the cutoff
        if not outside or not _inside(coords, i, j):
            break
        coords[:, j] = (start[:, i] + (CONTACT_R + step * 2e-8) * away).astype(np.float32)
    assert _inside(coords, i, j) is not outside
    system = ff.create_system(heavy.topology)
    if mask is not None:
        system.set_energy_ignored_residues([8], mask)
    ctx = mcpu_core.Context(system)
    ctx.set_positions(coords)
    ctx.calculate_total_energy(-1)
    moved = [k for k, a in enumerate(ff.ordered_atom_list) if a.residue_index <= 6]
    return ctx, coords, moved, i, j


def _carry(ctx, coords, moved, rotation, shift=(0.0, 0.0, 0.0)):
    """PhysicsVerifier's verdict on moving ``moved`` rigidly: rotated about
    their centroid, then shifted, in double, each coordinate rounded once."""
    x = coords[:, moved].astype(np.float64)
    centre = x.mean(axis=1, keepdims=True)
    new = coords.copy()
    new[:, moved] = (rotation @ (x - centre) + centre + np.asarray(shift)[:, None]).astype(np.float32)
    old = ctx.get_state()
    state = mcpu_core.State(old)
    state.coords = new
    patch = mcpu_core.ProposalPatch(coords.shape[1])
    for atom in moved:
        patch.mark_moved(atom)
    patch.is_valid = True
    patch.is_rigid = True
    return mcpu_core.PhysicsVerifier.verify_potential_delta(ctx, old, state, patch, 1, 1e-3), new


def _rotation(rng: np.random.Generator, max_angle: float) -> np.ndarray:
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    theta = rng.uniform(-max_angle, max_angle)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(theta) * k + (1 - np.cos(theta)) * k @ k


def _inside(coords, i, j) -> bool:
    d = (coords[:, j] - coords[:, i]).astype(np.float32)
    return bool(np.float32(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]) <= np.float32(CONTACT_R) ** 2)


@pytest.mark.parametrize("refilled", [False, True], ids=["rebuilt", "refilled"])
@pytest.mark.parametrize("outside", [False, True], ids=["just-inside", "just-outside"])
def test_every_carried_crossing_is_scored(outside: bool, refilled: bool) -> None:
    """``rebuilt``: each check builds the list from the coordinates.
    ``refilled``: the state's own list is built (by a forced pivot that is
    not committed) and then rewritten by a recompute, which the checks use."""
    ctx, coords, moved, i, j = _setup(outside)
    if refilled:
        assert mcpu_core.Integrator(temperature=0.6).debug_force_pivot(ctx, 8, False)
        ctx.calculate_total_energy(-1)
    rng = np.random.default_rng(0)
    crossings = 0
    for _ in range(TRIALS):
        check, new = _carry(ctx, coords, moved, _rotation(rng, 2e-3))
        assert check.passed, check.message
        crossings += _inside(coords, i, j) != _inside(new, i, j)
    assert crossings >= 10  # rounding really does carry the pair across


@pytest.mark.parametrize(
    ("mask", "listed"),
    [(None, False), (None, True), ("ignore_all", False)],
    ids=["unlisted", "listed", "masked"],
)
def test_a_carry_out_of_the_grid_is_scored_too(mask, listed: bool) -> None:
    """Shifted 30 A, the segment leaves the neighbour grid, and the delta
    takes the moved-vs-all path, which re-decides every carried pair, or with
    a contact list (``listed``) the carried pairs it lists. Under a mask (on a
    residue the pair is not in) no list is built, so ``masked`` re-decides
    every carried pair too."""
    ctx, coords, moved, i, j = _setup(False, mask)
    if listed:
        assert mcpu_core.Integrator(temperature=0.6).debug_force_pivot(ctx, 8, False)
    rng = np.random.default_rng(1)
    crossings = 0
    for _ in range(50):
        check, new = _carry(ctx, coords, moved, _rotation(rng, 2e-3), shift=(30.0, 0.0, 0.0))
        assert check.passed, check.message
        crossings += _inside(coords, i, j) != _inside(new, i, j)
    assert crossings >= 3


def test_a_rejected_trial_out_of_the_grid_keeps_the_list() -> None:
    """A move out of the neighbour grid cannot use the contact list, so the
    list is dropped if the move is accepted. A rejected trial leaves it, and
    the next move does not pay an O(N^2) rebuild."""
    ctx, coords, moved, _, _ = _setup(False)
    mu = ctx.mu_potential
    old = ctx.get_state()

    def trial(shift):
        x = coords[:, moved].astype(np.float64)
        new = mcpu_core.State(old)
        moved_coords = coords.copy()
        moved_coords[:, moved] = (x + np.asarray(shift)[:, None]).astype(np.float32)
        new.coords = moved_coords
        patch = mcpu_core.ProposalPatch(coords.shape[1])
        for atom in moved:
            patch.mark_moved(atom)
        patch.is_valid = True
        patch.is_rigid = True
        return mu.calculate_energy_change(ctx, old, new, patch)

    trial((0.0, 0.0, 0.5))  # builds the list
    built = mu.contact_list_rebuilds
    trial((30.0, 0.0, 0.0))  # out of the grid, never committed
    trial((0.0, 0.0, 0.5))
    assert mu.contact_list_rebuilds == built


def test_an_accepted_move_out_of_the_grid_drops_the_list() -> None:
    """Chignolin, hot and with large steps, so that moves out of the
    neighbour grid are accepted (two in these 10k steps). Each must drop the
    list, which the next move rebuilds: kept, it would be stale."""
    traj = md.load(str(default_example_pdb()))
    heavy = traj.atom_slice(traj.topology.select("not element H"))
    ff = MCPUForceField(heavy)
    ctx = mcpu_core.Context(ff.create_system(heavy.topology))
    ctx.set_positions((ff.coords[0] * 10.0).T.astype(np.float32))
    ctx.calculate_total_energy(-1)
    integ = mcpu_core.Integrator(temperature=5.0, step_size_rad=0.5)
    integ.set_seed(4)
    mu = ctx.mu_potential
    rebuilds = mu.contact_list_rebuilds
    for chunk in range(20):
        integ.run(ctx, 500, chunk * 500)
        running = float(ctx.get_state().current_energy)
        full = float(ctx.energy_breakdown(True)["weighted_total"])
        assert abs(running - full) < 1e-3, (chunk, running, full)
    assert mu.contact_list_rebuilds - rebuilds >= 3  # the first, then one per drop


@pytest.mark.parametrize("mask", [None, "ignore_all"], ids=["list", "masked"])
def test_a_pair_under_its_hard_core_keeps_its_contact_energy(mask) -> None:
    """Carried pairs are not re-checked for overlap, and far from the origin,
    where a float step is large, rounding can carry one under its hard-core
    cutoff. The running energy then still holds the pair's contact energy,
    so the old side of a move must take that back, and a list rebuilt there
    must hold it; a rebuild that dropped the pair left the running energy one
    contact energy off for good.

    GLY6 CA is put 0.0005 A under the state cutoff of its pair with TYR1 CD2
    (2.7275 A; the full energy calls that a clash), and then moved alone to
    2.8 A. The Mu change must be the full energy's change from the same pair
    just outside the cutoff. The delta reads a list rebuilt from the
    overlapping state; ``masked``: the same under a mask on another residue,
    which the list is built for."""
    traj = md.load(str(default_example_pdb()))
    heavy = traj.atom_slice(traj.topology.select("not element H"))
    ff = MCPUForceField(heavy)
    i, j = _atom(ff, 1, "CD2"), _atom(ff, 6, "CA")
    start = np.ascontiguousarray((ff.coords[0] * 10.0).T.astype(np.float32))
    system = ff.create_system(heavy.topology)
    for pot in system.get_potentials():
        if pot.get_name() != "mu":
            pot.set_enabled(False)
    if mask is not None:
        system.set_energy_ignored_residues([8], mask)
    ctx = mcpu_core.Context(system)
    mu = ctx.mu_potential

    def at(u, r):
        c = start.copy()
        c[:, j] = (start[:, i] + r * u).astype(np.float32)
        return c

    def mu_energy(c):
        ctx.set_positions(c)
        ctx.calculate_total_energy(-1)
        return float(ctx.energy_breakdown(True)["raw_total"]), ctx.has_steric_clash()

    rng = np.random.default_rng(0)
    for _ in range(100):  # a direction in which GLY6 CA overlaps nothing else
        u = rng.normal(size=3)
        u /= np.linalg.norm(u)
        if not any(mu_energy(at(u, r))[1] for r in (2.728, 2.75, 2.8)):
            break
    else:
        pytest.fail("no free direction")
    e_outside, _ = mu_energy(at(u, 2.728))
    e_new, _ = mu_energy(at(u, 2.8))
    _, overlap = mu_energy(at(u, 2.727))
    assert overlap

    old = ctx.get_state()
    new = mcpu_core.State(old)
    new.coords = at(u, 2.8)
    patch = mcpu_core.ProposalPatch(start.shape[1])
    patch.mark_moved(j)
    patch.is_valid = True
    rebuilds = mu.contact_list_rebuilds
    delta = mu.calculate_energy_change(ctx, old, new, patch)
    assert mu.contact_list_rebuilds == rebuilds + 1
    assert delta == pytest.approx(e_new - e_outside, abs=1e-4)


def test_the_drift_budget_rebuilds_the_list_in_time(capfd) -> None:
    """Actin run 1000 A from the origin (frame_offset 0 keeps the engine
    there), where a float step is up to 1.2e-4 A, so a pivot can move a
    carried distance by 2.1e-4 A and the 0.05 A band is used up after about
    230 accepted pivots. Pivot-only, with nothing resetting the running
    energy: it must still equal the full energy, and the list must have been
    rebuilt on schedule. Small pivots keep every move inside the neighbour
    grid, so no rebuild comes from a move that leaves it."""
    traj = md.load(str(resolve_test_pdb()))
    heavy = traj.atom_slice(traj.topology.select("not element H"))
    ff = MCPUForceField(heavy)
    ctx = mcpu_core.Context(ff.create_system(heavy.topology))
    ctx.set_positions((ff.coords[0] * 10.0).T.astype(np.float32) + np.float32(1000.0),
                      frame_offset=(0.0, 0.0, 0.0))
    ctx.calculate_total_energy(-1)
    integ = mcpu_core.Integrator(temperature=0.6, step_size_rad=0.02)
    integ.set_seed(3)
    integ.set_move_weights(1.0, 0.0, 0.0)
    mu = ctx.mu_potential
    rebuilds_before = mu.contact_list_rebuilds
    for chunk in range(100):
        integ.run(ctx, 100, chunk * 100)
        running = float(ctx.get_state().current_energy)
        full = float(ctx.energy_breakdown(True)["weighted_total"])
        assert abs(running - full) < 5e-3, (chunk, running, full)
    assert "leaves the neighbour grid" not in capfd.readouterr().err
    accepted = integ.get_bb_accepted()
    assert accepted > 2000
    assert mu.contact_list_rebuilds - rebuilds_before >= accepted // 240 - 1
