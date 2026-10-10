"""The KIC driver draws its angle with kic_step_size_rad, the pivot with
step_size_rad.

The KIC driver used to draw from the pivot's distribution, so step_size_rad
set both moves, and a KIC-only run changed with the pivot width. Each move now
has its own width and its own normal distribution (so they do not share a
cached spare draw either): a KIC-only run is the same for any pivot width, a
pivot-only run the same for any KIC width.
"""
from __future__ import annotations

import mdtraj as md
import numpy as np

from pymcpu import mcpu_core
from pymcpu.forcefields.mcpu import MCPUForceField
from pymcpu.runners import default_example_pdb


KIC_ONLY = (0.0, 1.0, 0.0)
PIVOT_ONLY = (1.0, 0.0, 0.0)


def _run(weights: tuple[float, float, float], steps: int = 1500, **widths: float) -> tuple[np.ndarray, tuple]:
    traj = md.load(str(default_example_pdb()))
    heavy = traj.atom_slice(traj.topology.select("not element H"))
    ff = MCPUForceField(heavy)
    ctx = mcpu_core.Context(ff.create_system(heavy.topology))
    ctx.set_positions((ff.coords[0] * 10.0).T.astype(np.float32))
    ctx.calculate_total_energy(-1)
    integ = mcpu_core.Integrator(temperature=0.6, **widths)
    integ.set_seed(31)
    integ.set_move_weights(*weights)
    integ.run(ctx, steps)
    return np.array(ctx.coords), integ.move_counts()


def test_a_kic_run_does_not_depend_on_the_pivot_width() -> None:
    # At the default driver width (pi/6) about 3% of KIC moves are accepted.
    narrow, counts = _run(KIC_ONLY, steps=3000, step_size_rad=0.05)
    wide, _ = _run(KIC_ONLY, steps=3000, step_size_rad=0.3)
    assert counts["kic"][0] > 50  # enough accepted KIC moves to tell
    np.testing.assert_array_equal(narrow, wide)


def test_a_kic_run_follows_the_kic_width() -> None:
    narrow, counts_narrow = _run(KIC_ONLY, kic_step_size_rad=0.02)
    wide, counts_wide = _run(KIC_ONLY, kic_step_size_rad=0.6)
    assert not np.array_equal(narrow, wide)
    # Smaller driver turns are accepted more often.
    assert counts_narrow["kic"][0] > counts_wide["kic"][0]


def test_a_pivot_run_does_not_depend_on_the_kic_width() -> None:
    narrow, counts = _run(PIVOT_ONLY, kic_step_size_rad=0.02)
    wide, _ = _run(PIVOT_ONLY, kic_step_size_rad=0.6)
    assert counts["pivot"][0] > 50
    np.testing.assert_array_equal(narrow, wide)
