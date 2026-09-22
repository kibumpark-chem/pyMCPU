#!/usr/bin/env python3
"""Minimal actin MC example using shipped acta.pdb (SAFE defaults)."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

import pymcpu as mc
from pymcpu.forcefields.mcpu import MCPUForceField
import mdtraj as md

ROOT = Path(__file__).resolve().parents[2]
PDB_PATH = ROOT / "examples" / "actin" / "input_pdb" / "acta.pdb"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def main() -> None:
    print("pymcpu package root:", mc.PACKAGE_ROOT)
    print("PDB:", PDB_PATH)

    traj = md.load(str(PDB_PATH))
    if any(a.element.symbol == "H" for a in traj.topology.atoms):
        traj = traj.atom_slice(traj.topology.select("not element H"))

    forcefield = MCPUForceField(traj)
    system = forcefield.create_system(traj.topology)
    print(system)

    integrator = mc.mcpu_core.Integrator(
        temperature=0.6,
        step_size_rad=0.1,
    )
    ctx = mc.mcpu_core.Context(system)
    ctx.set_positions((forcefield.coords[0] * 10.0).T.astype(np.float32))
    print(f"Initial energy: {ctx.calculate_total_energy(-1):.3f}")
    print(f"Number of atoms: {system.get_num_atoms()}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    n_steps = 10_000
    report_int = 1_000
    print(f"Running {n_steps} MC steps...")
    t0 = time.time()
    for step_block in range(0, n_steps, report_int):
        integrator.run(ctx, report_int)
        e = float(ctx.get_state().current_energy)
        print(f"Step {step_block + report_int:6d}  E={e:.3f}")
    print(f"MC simulation took {time.time() - t0:.2f} seconds")
    print(
        "Acceptance (bb/sc/kic):",
        integrator.get_bb_accepted(),
        "/",
        integrator.get_sc_accepted(),
        "/",
        integrator.get_kic_accepted(),
    )


if __name__ == "__main__":
    main()
