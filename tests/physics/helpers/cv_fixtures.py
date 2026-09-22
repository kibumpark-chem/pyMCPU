"""Chignolin ``MCPUForceField`` + native-coordinate builder shared by the
collective-variable tests in ``tests/physics/cv/`` and the progress-
coordinate factory tests in ``tests/integration/we/`` -- both need the same
"load chignolin, strip hydrogens, build a real C++ forcefield" setup, so it
lives here once instead of being copy-pasted into each test module.
"""

from __future__ import annotations

import mdtraj as md
import numpy as np

from pymcpu.forcefields.mcpu import MCPUForceField


def build_chignolin_forcefield(pdb_path: str, param_set: str = "mcpu_v1") -> MCPUForceField:
    """Heavy-atom-only ``MCPUForceField`` built from ``pdb_path``."""
    traj = md.load(str(pdb_path))
    heavy = traj.atom_slice(traj.topology.select("not element H"))
    return MCPUForceField(heavy, param_set=param_set)


def native_coords_angstrom(forcefield: MCPUForceField) -> np.ndarray:
    """``(3, n_atoms)`` float32 native coordinates in Angstrom.

    ``forcefield.coords`` is mdtraj-loaded and therefore in nanometers;
    CV/pcoord components expect Angstrom to match ``MCPUForceField``'s own
    internal units (see ``MCPUForceField.create_system``'s own ``* 10.0``
    conversion when it hands coordinates to the C++ engine).
    """
    return (forcefield.coords[0] * 10.0).T.astype(np.float32)
