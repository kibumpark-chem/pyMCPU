"""The Mu neighbour-grid cell is never smaller than the Mu cutoff.

The grid code walks a one-cell stencil (27 cells). A cell smaller than the
cutoff needs a wider one, which overflowed fixed-size buffers: with
set_mu_cell_size_scale below 1, or set_mu_cell_size_angstrom below the cutoff,
set_positions crashed the interpreter, and the cell-pair path dropped cells.
Such a request is now raised to the cutoff. Larger cells still work.

The probe runs in a subprocess so a regression fails this test instead of
killing the test run.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

PROBE = textwrap.dedent("""
    import numpy as np, mdtraj as md, pymcpu as mc
    from pymcpu.forcefields.mcpu import MCPUForceField
    from pymcpu.runners import default_example_pdb

    traj = md.load(str(default_example_pdb()))
    heavy = traj.atom_slice(traj.topology.select("not element H"))
    ff = MCPUForceField(heavy)
    start = (ff.coords[0] * 10.0).T.astype(np.float32)

    def build(setter=None, value=None):
        ctx = mc.Simulation(heavy.topology, ff.create_system(heavy.topology), mc.Integrator(0.6)).context
        if setter:
            getattr(ctx, setter)(value)
        ctx.set_positions(start)
        return ctx, ctx.calculate_total_energy(-1)

    ref, energy = build()
    cutoff = float(np.sqrt(ref.mu_potential.mu_cutoff_sq))
    for setter, value in [("set_mu_cell_size_scale", 0.6), ("set_mu_cell_size_angstrom", 3.0),
                          ("set_mu_cell_size_scale", 1.3)]:
        ctx, e = build(setter, value)
        print("CELL", setter, value, ctx.effective_mu_cell_size_A(), cutoff, e == energy)
""")


def test_a_cell_below_the_cutoff_is_raised_to_it() -> None:
    repo = Path(__file__).resolve().parents[2]
    # MCPU_MU_CELL_SCALE and friends would change the default cell.
    env = {k: v for k, v in os.environ.items() if not k.startswith("MCPU_") or k == "MCPU_CACHE_DIR"}
    run = subprocess.run([sys.executable, "-c", PROBE], cwd=repo, env=env, capture_output=True, text=True)
    assert run.returncode == 0, f"probe exited {run.returncode}\n{run.stderr[-2000:]}"
    rows = [line.split()[1:] for line in run.stdout.splitlines() if line.startswith("CELL ")]
    assert len(rows) == 3
    for setter, value, cell, cutoff, same_energy in rows:
        assert float(cell) >= float(cutoff) - 1e-4, (setter, value, cell)
        assert same_energy == "True"
    assert float(rows[2][2]) > float(rows[2][3])  # a larger cell is kept
