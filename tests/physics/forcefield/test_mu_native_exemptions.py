"""Mu exempts a native pair from the clash test only if it is under the move cutoff.

A clash-eligible pair that already overlaps in the structure the force field
is built from is exempt for the whole run, or every move that re-decided it
would be rejected. The test used the exact hard-core radius, which is
0.0015 A or more above the move cutoff, so a pair in between -- no clash
under either cutoff -- lost its protection for good. A structure written out
by a run, and then used as the input of the next, can hold such pairs.

Probe pair: TYR1 CB and TRP8 CH2, hard core 2.73 A, move cutoff 2.7285 A.
"""

from __future__ import annotations

import mdtraj as md
import numpy as np
import pytest

from pymcpu import mcpu_core
from pymcpu.forcefields.mcpu import MCPUForceField
from pymcpu.runners import default_example_pdb

HARD_R = 2.73
MOVE_CUTOFF = 2.7285


def _context_built_with_pair_at(r: float):
    traj = md.load(str(default_example_pdb()))
    heavy = traj.atom_slice(traj.topology.select("not element H"))
    ff = MCPUForceField(heavy)
    cb = ff.blocks[1].sc_start
    tip = ff.blocks[8].sc_start + ff.blocks[8].sc_count - 1
    xyz = (ff.coords[0] * 10.0).T.astype(np.float64)
    near = np.linalg.norm(xyz - xyz[:, [cb]], axis=0) < 6.0
    near[cb] = False
    away = (xyz[:, [cb]] - xyz[:, near]).sum(axis=1)
    away /= np.linalg.norm(away)

    def place(dist: float) -> np.ndarray:
        coords = (ff.coords[0] * 10.0).T.astype(np.float32).copy()
        coords[:, tip] = (xyz[:, cb] + dist * away).astype(np.float32)
        return coords

    built = heavy.slice(0, copy=True)
    built.xyz[0, ff.ordered_atom_list[tip].original_index] = place(r)[:, tip] / 10.0
    ff_built = MCPUForceField(built)
    ctx = mcpu_core.Context(ff_built.create_system(built.topology))
    return ctx, place


@pytest.mark.parametrize(
    ("built_at", "exempt"),
    [(HARD_R - 0.0005, False), (MOVE_CUTOFF - 0.0005, True)],
    ids=["between-cutoff-and-hard-core", "under-the-move-cutoff"],
)
def test_only_a_pair_under_the_move_cutoff_is_exempt(built_at: float, exempt: bool) -> None:
    ctx, place = _context_built_with_pair_at(built_at)
    ctx.set_positions(place(2.0))  # deep overlap
    ctx.calculate_total_energy(-1)
    assert ctx.has_steric_clash() is not exempt
