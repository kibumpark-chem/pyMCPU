"""A neighbour-grid cell that fills up switches its grid off, exactly.

Cells hold at most 48 atoms (a compile-time capacity the hot walks are
built around). The hard core keeps real occupancy near half of that, but a
state can still pack more atoms into one cell: here, atoms of residues
masked ``clash_only`` (their overlaps cost nothing in the full energy, so
the state stays finite) collapsed onto one point. The grid that overflows
is switched off and counted, the energy terms take their exact fallbacks
(Mu the all-pairs delta, H-bonds the brute-force search), and the grid comes
back once the atoms spread out again.
"""

from __future__ import annotations

import numpy as np
import pytest

from pymcpu import mcpu_core
from tests.fixtures.context_builders import build_test_context

MU_GROUP = 1


def _collapse(ctx, atoms):
    """Coordinates with ``atoms`` packed within 0.06 A of the first one."""
    xyz = np.array(ctx.get_state().coords, dtype=np.float32, copy=True)
    base = xyz[:, atoms[0]].copy()
    for k, a in enumerate(atoms):
        xyz[:, a] = base + np.float32(0.001 * k)
    return xyz


def _residue_atoms(ctx, residues):
    res = np.asarray(ctx.get_system().atom_to_residue)
    return [int(i) for i in np.flatnonzero(np.isin(res, residues))]


def _o_atoms(ctx, residues):
    blocks = ctx.get_system().get_block_indices()
    return [int(blocks[r].o_start) for r in residues]


def _move_one(ctx, atom, shift=(0.05, -0.04, 0.03)):
    old = ctx.get_state()
    state = mcpu_core.State(old)
    xyz = np.array(old.coords, dtype=np.float32, copy=True)
    xyz[:, atom] += np.asarray(shift, dtype=np.float32)
    state.coords = xyz
    patch = mcpu_core.ProposalPatch(xyz.shape[1])
    patch.mark_moved(atom)
    patch.is_valid = True
    return old, state, patch


@pytest.mark.parametrize("grid", ["mu", "hbond"])
def test_an_overfull_cell_switches_its_grid_off_and_back(grid: str) -> None:
    ctx, _ = build_test_context()
    start = np.array(ctx.get_state().coords, dtype=np.float32, copy=True)
    stats = ctx.neighbor_proxy_stats()
    assert stats["mu_grid_active"] and not ctx.hbond_uses_fallback()
    assert stats["mu_grid_overflows"] == 0 and stats["hbond_grid_overflows"] == 0

    residues = list(range(10, 22)) if grid == "mu" else list(range(10, 70))
    ctx.get_system().set_energy_ignored_residues(residues, "clash_only")
    atoms = _residue_atoms(ctx, residues) if grid == "mu" else _o_atoms(ctx, residues)
    assert len(atoms) >= 60
    ctx.set_positions(_collapse(ctx, atoms))
    ctx.calculate_total_energy(-1)
    stats = ctx.neighbor_proxy_stats()
    if grid == "mu":
        assert stats["mu_grid_overflows"] >= 1
        assert not stats["mu_grid_active"]
    else:
        assert stats["hbond_grid_overflows"] >= 1
        assert ctx.hbond_uses_fallback()

    # Every term's delta for a move elsewhere still matches its full
    # recompute; Mu runs the all-pairs path, H-bonds the brute force.
    far = _residue_atoms(ctx, [200])[0]
    for check in mcpu_core.PhysicsVerifier.verify_all_potential_deltas(
            ctx, *_move_one(ctx, far), 1e-3):
        assert check.passed, (check.energy_group, check.delta_incremental, check.delta_direct)

    # Accepted moves keep the running energy exact while the grid is off
    # (each accept retries the rebuild, which overflows again).
    integ = mcpu_core.Integrator(temperature=1.0)
    integ.set_seed(5)
    integ.run(ctx, 300)
    running = float(ctx.get_state().current_energy)
    total = float(ctx.energy_breakdown(True)["weighted_total"])
    assert abs(running - total) <= max(1e-3, 1e-5 * abs(total)), (running, total)

    # Sane coordinates bring the grid back.
    ctx.set_positions(start)
    stats = ctx.neighbor_proxy_stats()
    assert stats["mu_grid_active"] and not ctx.hbond_uses_fallback()
    assert stats["mu_grid_peak_occupancy"] <= stats["mu_grid_cell_capacity"]
