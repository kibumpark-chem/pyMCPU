#!/usr/bin/env python3
"""OpenMM-style example: construct and run a simulation entirely in Python.

No YAML required. All parameters are set explicitly.

Run:
    python examples/openmm_style/run_example.py

This demonstrates the pyMCPU API that mirrors OpenMM's programmatic
Simulation/System/Integrator/Reporter pattern.
"""

from __future__ import annotations

from pathlib import Path

import mdtraj as md
import numpy as np

import pymcpu as mc
from pymcpu.forcefields.mcpu import MCPUForceField

ROOT = Path(__file__).resolve().parents[2]
PDB = ROOT / "examples" / "data" / "1uao.pdb"
OUTPUT_DIR = Path("./out_openmm_example")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load structure (heavy atoms only) ───────────────────────────
    traj = md.load(str(PDB))
    heavy = traj.atom_slice(traj.topology.select("not element H"))
    print(f"Loaded {PDB.name}: {heavy.n_atoms} heavy atoms, {heavy.n_residues} residues")

    # ── 2. Build the force field ───────────────────────────────────────
    forcefield = MCPUForceField(heavy, param_set="mcpu08")

    # ── 3. Create the system (topology → potentials → system) ─────────────
    system = forcefield.create_system(heavy.topology)
    print(f"System: {system.get_num_atoms()} atoms, {system.get_num_residues()} residues")

    # ── 4. Create the integrator ───────────────────────────────────────
    integrator = mc.Integrator(temperature=0.6, step_size_rad=0.1)
    integrator.set_seed(42)

    # ── 5. Assemble the simulation ─────────────────────────────────────
    sim = mc.Simulation(heavy.topology, system, integrator)

    # Set initial coordinates (nm → Angstrom for engine)
    coords_angstroms = forcefield.coords[0] * 10.0
    sim.context.set_positions(coords_angstroms.T.astype(np.float32))
    sim.context.calculate_total_energy(-1)

    # ── 6. Add reporters ───────────────────────────────────────────────
    mapping = forcefield.inverse_mapping
    sim.add_xtc_reporter(str(OUTPUT_DIR / "traj.xtc"), interval=100, inverse_mapping=mapping)
    sim.add_energy_reporter(str(OUTPUT_DIR / "data.csv"), interval=100)

    # ── 7. Run the simulation ──────────────────────────────────────────
    n_steps = 1000
    print(f"Running {n_steps} MC steps at T=0.6...")
    sim.step(n_steps)

    # ── 8. Report final state ──────────────────────────────────────────
    energy = float(sim.context.get_state().current_energy)
    stats = integrator.move_stats()
    print(f"\nFinal energy: {energy:.3f}")
    print(f"Backbone acceptance: {stats['num_accept_pivot']}/{stats['num_propose_pivot']}")
    print(f"Sidechain acceptance: {stats['num_accept_sc']}/{stats['num_propose_sc']}")
    print(f"KIC acceptance: {stats['num_accept_kic']}/{stats['num_propose_kic']}")
    print(f"\nOutputs written to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
