"""Which atoms the Mu neighbour grid holds.

The grid holds every atom the Mu energy scores (all but the amide H), less
the atoms of residues an ``ignore_all`` energy mask switches off: every pair
with such an atom scores 0 and cannot clash, so leaving them out drops only
zero terms. It also keeps the hard-core bound on how many atoms share a
cell, which masked atoms, free to overlap, do not obey. A mask set or
cleared between runs changes the membership before the next move.
"""

from __future__ import annotations

import mdtraj as md
import numpy as np
import pytest

from pymcpu import mcpu_core
from pymcpu.forcefields.mcpu import MCPUForceField
from pymcpu.runners import default_example_pdb

MASKED = [3, 4, 5]


def _context():
    traj = md.load(str(default_example_pdb()))
    heavy = traj.atom_slice(traj.topology.select("not element H"))
    ff = MCPUForceField(heavy)
    ctx = mcpu_core.Context(ff.create_system(heavy.topology))
    ctx.set_positions((ff.coords[0] * 10.0).T.astype(np.float32))
    ctx.calculate_total_energy(-1)
    return ctx


def _grid_atoms(ctx) -> int:
    return int(ctx.neighbor_proxy_stats()["mu_grid_n_atoms"])


def _masked_atoms(ctx) -> int:
    res = np.asarray(ctx.get_system().atom_to_residue)
    return int(np.isin(res, MASKED).sum())


@pytest.mark.parametrize("mode", ["ignore_all", "clash_only"])
def test_masked_atoms_leave_the_grid_only_under_ignore_all(mode: str) -> None:
    ctx = _context()
    full = _grid_atoms(ctx)
    assert full == ctx.get_system().get_num_atoms()  # heavy atoms only: no amide H
    ctx.get_system().set_energy_ignored_residues(MASKED, mode)
    ctx.calculate_total_energy(-1)
    integ = mcpu_core.Integrator(temperature=1.0)
    integ.set_seed(3)
    integ.run(ctx, 200)
    expected = full - _masked_atoms(ctx) if mode == "ignore_all" else full
    assert _grid_atoms(ctx) == expected


def test_a_mask_changed_between_runs_keeps_the_running_energy_exact() -> None:
    """Set a mask, run, clear it, run, set it again, run. Clearing puts the
    masked atoms back into the grid where they now are, so the moves after
    it must see them. Masked residues move through the others freely, and a
    clear that leaves an overlap behind is a state whose full energy holds a
    clash that the running energy never does (calculate_total_energy does
    not resync a clashing state), so that case is not compared; these
    settings leave none. They include the KIC driver width: at the default
    (pi/6) this seed leaves an overlap after the first masked run."""
    ctx = _context()
    full = _grid_atoms(ctx)
    integ = mcpu_core.Integrator(temperature=0.6, step_size_rad=0.2, kic_step_size_rad=0.1)
    integ.set_seed(7)
    system = ctx.get_system()
    compared = []
    for chunk, mask in enumerate([MASKED, None, MASKED]):
        if mask is None:
            system.clear_energy_ignored_residues()
        else:
            system.set_energy_ignored_residues(mask, "ignore_all")
        start = float(ctx.calculate_total_energy(-1))
        resynced = abs(start - float(ctx.get_state().current_energy)) < 1e-3
        integ.run(ctx, 1000, chunk * 1000)
        expected = full if mask is None else full - _masked_atoms(ctx)
        assert _grid_atoms(ctx) == expected, chunk
        if not resynced:
            continue
        running = float(ctx.get_state().current_energy)
        total = float(ctx.energy_breakdown(True)["weighted_total"])
        assert abs(running - total) <= max(1e-3, 1e-5 * abs(total)), (chunk, running, total)
        compared.append(chunk)
    if 1 not in compared:
        pytest.skip("the masked run left an overlap behind; nothing to compare after the clear")
