"""The neighbour grids wrap: atoms anywhere have a cell, and walks stay exact.

A grid covers the box it was built for, but its cell index is taken mod the
cell count on each axis, so an atom far outside that box is filed in a cell
of the same grid, next to atoms a whole number of grid periods away. Every
walk measures true distances, so those aliases are dropped. These tests
check that:

* each walk of a grid lists every atom within range of a probe, once, for
  atoms and probes far outside the box, on grids too small for their stencil
  (padded to 2R + 1 cells per axis) and on grids trimmed to a cell cap;
* NaN and huge coordinates get a cell without breaking the walk;
* the Mu and H-bond energy changes of trials far outside the grid, and of a
  whole molecule carried 1000 A away, equal the change in the full energy;
* a run whose atoms leave the box never rebuilds a grid, its running energy
  stays equal to a full recompute, and its H-bond grids match brute force,
  also after set_positions moves the molecule 1000 A away.
"""

from __future__ import annotations

import mdtraj as md
import numpy as np
import pytest

from pymcpu import mcpu_core
from pymcpu.forcefields.mcpu import MCPUForceField
from pymcpu.runners import default_example_pdb

WALKS = ["stencil", "spans", "within"]


def _brute(xyz, probe, r):
    d = np.linalg.norm(xyz.astype(np.float64) - probe[:, None].astype(np.float64), axis=0)
    return set(np.flatnonzero(d < r).tolist())


def _check_walk(xyz, probes, cell, query, lo, hi, max_cells, walk, radius=0.0):
    lists, dims, overflowed = mcpu_core._cell_grid_walk(
        xyz, cell, query, lo, hi, max_cells, probes, walk, radius)
    assert not overflowed
    r = radius if walk == "within" else query
    for k, ids in enumerate(lists):
        assert len(ids) == len(set(ids)), f"probe {k}: an atom was listed twice"
        missed = _brute(xyz, probes[:, k], r) - set(ids)
        assert not missed, f"probe {k}: atoms in range not listed: {sorted(missed)}"
    return dims


def _cloud(rng, n, lo, hi, far):
    """n atoms in [lo, hi) plus clusters around the points in ``far``."""
    pts = [rng.uniform(lo, hi, size=(3, n))]
    for c in far:
        pts.append(np.asarray(c, dtype=np.float64)[:, None] + rng.normal(0.0, 8.0, size=(3, n // 4)))
    return np.ascontiguousarray(np.concatenate(pts, axis=1).astype(np.float32))


FAR = [(1000.0, 0.0, 0.0), (-1000.0, 37.0, 5.0), (12.0, -2500.0, 1e4), (31.0, 31.0, 31.0)]


@pytest.mark.parametrize("walk", WALKS)
def test_walks_list_every_atom_in_range_once_far_outside_the_box(walk) -> None:
    rng = np.random.default_rng(3)
    xyz = _cloud(rng, 400, 0.0, 30.0, FAR)
    probes = np.ascontiguousarray(xyz[:, ::7])
    dims = _check_walk(xyz, probes, 5.1, 5.1, (0, 0, 0), (30, 30, 30), 10**6, walk, 2.83)
    assert dims == (6, 6, 6)


@pytest.mark.parametrize("walk", WALKS)
def test_a_grid_smaller_than_its_stencil_is_padded(walk) -> None:
    """A 2 A box with 2 A cells is one cell wide; a 5 A query needs a stencil
    of 2R + 1 = 7 cells per axis, which would meet some cells twice."""
    rng = np.random.default_rng(5)
    xyz = _cloud(rng, 200, 0.0, 12.0, FAR[:2])
    probes = np.ascontiguousarray(xyz[:, ::5])
    dims = _check_walk(xyz, probes, 2.0, 5.0, (0, 0, 0), (2, 2, 2), 10**6, walk, 1.5)
    assert min(dims) >= 7


@pytest.mark.parametrize("walk", WALKS)
def test_a_capped_grid_is_trimmed_but_exact(walk) -> None:
    rng = np.random.default_rng(7)
    xyz = _cloud(rng, 600, -80.0, 80.0, FAR)
    probes = np.ascontiguousarray(xyz[:, ::9])
    dims = _check_walk(xyz, probes, 5.1, 5.1, (-80, -80, -80), (80, 80, 80), 300, walk, 2.83)
    assert min(dims) >= 3 and dims[0] * dims[1] * dims[2] <= 300


def test_nan_and_huge_coordinates_get_a_cell() -> None:
    rng = np.random.default_rng(11)
    xyz = _cloud(rng, 300, 0.0, 30.0, [])
    xyz[:, 0] = np.nan
    xyz[:, 1] = (3e38, -3e38, 1e30)
    xyz[:, 2] = (1e9, 1e9, 1e9)
    probes = np.ascontiguousarray(xyz[:, 3::11])
    for walk in WALKS:
        _check_walk(xyz, probes, 5.1, 5.1, (0, 0, 0), (30, 30, 30), 10**6, walk, 2.83)


@pytest.mark.parametrize(
    "hi, max_cells",
    [((1e12, 30.0, 30.0), 10**6), ((1e9, 1e9, 1e9), 4096),
     ((np.inf, 30.0, 30.0), 10**6), ((np.nan, 30.0, 30.0), 10**6)],
)
def test_a_huge_or_nan_box_gets_a_bounded_grid(hi, max_cells) -> None:
    """The cell count per axis is capped before the float-to-int conversion,
    so a huge or infinite box is trimmed like any large one, and a NaN
    extent gets 2R + 1 cells."""
    rng = np.random.default_rng(13)
    xyz = _cloud(rng, 300, 0.0, 30.0, FAR[:2])
    probes = np.ascontiguousarray(xyz[:, ::7])
    for walk in WALKS:
        nx, ny, nz = _check_walk(xyz, probes, 5.1, 5.1, (0, 0, 0), hi, max_cells, walk, 2.83)
        assert min(nx, ny, nz) >= 3 and nx * ny * nz <= max_cells
        if np.isnan(hi[0]):
            assert nx == 3
        elif hi[1] == 30.0:
            assert nx > 1000, "the long axis keeps most of the cells"


# --- the engine's grids -------------------------------------------------------


@pytest.fixture(scope="module")
def heavy() -> md.Trajectory:
    traj = md.load(str(default_example_pdb()))
    return traj.atom_slice(traj.topology.select("not element H"))


def _context(heavy):
    ff = MCPUForceField(heavy)
    ctx = mcpu_core.Context(ff.create_system(heavy.topology))
    xyz = np.ascontiguousarray((ff.coords[0] * 10.0).T.astype(np.float32))
    ctx.set_positions(xyz)
    ctx.calculate_total_energy(-1)
    return ctx, xyz


def _trial(ctx, new_xyz, moved, rigid):
    old = ctx.get_state()
    state = mcpu_core.State(old)
    state.coords = np.ascontiguousarray(new_xyz)
    patch = mcpu_core.ProposalPatch(new_xyz.shape[1])
    for a in moved:
        patch.mark_moved(int(a))
    patch.is_valid = True
    patch.is_rigid = rigid
    return old, state, patch


@pytest.mark.parametrize("shift", [(100.0, 0.0, 0.0), (0.0, -1000.0, 0.0), (700.0, 700.0, -700.0)])
def test_trials_far_outside_the_grid_match_the_full_energy(heavy, shift) -> None:
    ctx, xyz = _context(heavy)
    n = xyz.shape[1]
    tail = list(range(n // 2, n))
    new = xyz.copy()
    new[:, tail] += np.asarray(shift, dtype=np.float32)[:, None]
    for check in mcpu_core.PhysicsVerifier.verify_all_potential_deltas(
            ctx, *_trial(ctx, new, tail, rigid=True), 1e-3):
        assert check.passed, (shift, check.energy_group, check.message)


def test_a_molecule_carried_1000_A_away_keeps_its_energy(heavy) -> None:
    ctx, xyz = _context(heavy)
    n = xyz.shape[1]
    new = xyz + np.float32(1000.0)
    for check in mcpu_core.PhysicsVerifier.verify_all_potential_deltas(
            ctx, *_trial(ctx, new, range(n), rigid=True), 1e-3):
        assert check.passed, (check.energy_group, check.message)
        assert abs(check.delta_direct) < 1e-2, (check.energy_group, check.delta_direct)


def _run_and_check(ctx, steps, seed, temperature, after_chunk=None):
    """Run in chunks of 1000 steps; no chunk may rebuild a grid. Then the
    running energy must equal a full recompute and the H-bond grids brute
    force."""
    integ = mcpu_core.Integrator(temperature=temperature, step_size_rad=0.3)
    integ.set_seed(seed)
    for _ in range(steps // 1000):
        ctx.reset_neighbor_proxy_stats()
        integ.run(ctx, 1000)
        assert ctx.neighbor_grid_rebuilds() == 0, "a grid was rebuilt during the run"
        if after_chunk:
            after_chunk(np.asarray(ctx.get_state().coords))
    running = float(ctx.get_state().current_energy)
    total = float(ctx.energy_breakdown(True)["weighted_total"])
    assert abs(running - total) <= max(1e-9, 1e-9 * abs(total)), (running, total)
    assert ctx.hbond_index_ok()
    stats = ctx.neighbor_proxy_stats()
    assert stats["mu_grid_active"] and not ctx.hbond_uses_fallback()
    assert stats["mu_grid_overflows"] == 0 and stats["hbond_grid_overflows"] == 0


def test_a_run_that_leaves_the_box_never_rebuilds_and_stays_exact(heavy) -> None:
    ctx, xyz = _context(heavy)
    margin = 2.0 * float(ctx.mu_potential.mu_exact_cutoff)
    lo = xyz.min(axis=1) - margin
    hi = xyz.max(axis=1) + margin
    left = []
    _run_and_check(ctx, 20000, seed=21, temperature=8.0, after_chunk=lambda c: left.append(
        bool(((c < lo[:, None]) | (c >= hi[:, None])).any())))
    assert any(left), "no atom left the box the grids were built for"


def test_set_positions_far_away_matches_brute_force(heavy) -> None:
    """A resumed or swapped-in state lands anywhere: here, the unfolded state
    of another run, 1000 A away. The grids recentre on it, and its trials,
    its running energy and its H-bond grids stay exact."""
    ctx, _ = _context(heavy)
    _run_and_check(ctx, 20000, seed=4, temperature=8.0)
    far = np.array(ctx.get_state().coords, dtype=np.float32, copy=True) + np.float32(1000.0)
    other, _ = _context(heavy)
    other.set_positions(np.ascontiguousarray(far))
    other.calculate_total_energy(-1)
    assert other.hbond_index_ok()
    n = far.shape[1]
    tail = list(range(n // 2, n))
    new = far.copy()
    new[:, tail] += np.float32(40.0)
    for check in mcpu_core.PhysicsVerifier.verify_all_potential_deltas(
            other, *_trial(other, new, tail, rigid=True), 1e-3):
        assert check.passed, (check.energy_group, check.message)
    _run_and_check(other, 20000, seed=9, temperature=8.0)
