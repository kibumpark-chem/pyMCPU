"""Replica code reads coordinates in the order it writes them back.

get_coords, which REMD exchanges, checkpoints and the folding CVs use, read
State.coords, which is storage order, and set_positions takes build order.
The two differ after an init_only atom reorder, so a swap or a checkpoint
round trip scrambled the atoms. It now reads Context.coords.
"""

from __future__ import annotations

import mdtraj as md
import numpy as np

from pymcpu import mcpu_core
from pymcpu.forcefields.mcpu import MCPUForceField
from pymcpu.sampling import get_coords
from tests.fixtures.context_builders import resolve_test_pdb


def test_get_coords_is_in_build_order_after_a_reorder() -> None:
    """Actin, because init_only cannot place 1UAO's terminal OXT."""
    traj = md.load(str(resolve_test_pdb()))
    heavy = traj.atom_slice(traj.topology.select("not element H"))
    ff = MCPUForceField(heavy)
    start = np.ascontiguousarray((ff.coords[0] * 10.0).T.astype(np.float32))
    ctx = mcpu_core.Context(ff.create_system(heavy.topology))
    ctx.set_atom_reorder_mode("init_only")
    ctx.set_positions(start)
    assert ctx.atom_permutation_info()["enabled"]
    assert not np.array_equal(np.asarray(ctx.get_state().coords), start)  # storage order
    assert np.array_equal(get_coords(ctx), start)
