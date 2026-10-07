"""Shared harness setup for the scripts/parity_*.py numeric-regression oracles.

Only the byte-identical setup boilerplate is factored here (repo-root
sys.path bootstrap + the no-extra-arg ``build_ctx``: mdtraj load -> strip H ->
MCPUForceField -> create_system -> Context -> set_positions ->
calculate_total_energy(-1)).

The accept-bit / energy comparison blocks in each parity_*.py are
deliberately NOT consolidated here: they differ enough between scripts
(bisection vs. bulk comparison, extra ctx setters, extra stats fields) that
unifying them risks silently changing what each oracle checks. Each
parity_*.py remains independently runnable and independently correct.

Do not import anything physics-adjacent that these scripts don't already
import identically today -- this module's only job is to remove copy-paste,
not to add shared behavior.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mdtraj as md  # noqa: E402
import numpy as np  # noqa: E402

from pymcpu import mcpu_core  # noqa: E402,F401 -- re-exported for callers
from pymcpu.forcefields.mcpu import MCPUForceField  # noqa: E402


def build_ctx(pdb: Path):
    """Build an mcpu_core.Context for ``pdb`` with default settings.

    Callers that need non-default settings (e.g. use_cell_pair, skip_rigid_mm)
    keep their own build_ctx variant -- this only covers the exact sequence
    duplicated byte-for-byte across parity_layered_eval.py and
    parity_sparse_proposal.py.
    """
    traj = md.load(str(pdb))
    if any(a.element.symbol == "H" for a in traj.topology.atoms):
        traj = traj.atom_slice(traj.topology.select("not element H"))
    ff = MCPUForceField(traj)
    system = ff.create_system(traj.topology)
    ctx = mcpu_core.Context(system)
    ctx.set_positions((ff.coords[0] * 10.0).T.astype(np.float32))
    ctx.calculate_total_energy(-1)
    return ctx
