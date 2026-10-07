#!/usr/bin/env python3
"""Parity: use_cell_pair on vs off — accept bits must be identical."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mdtraj as md
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymcpu import mcpu_core
from pymcpu.forcefields.mcpu import MCPUForceField


def build_ctx(pdb: Path, *, use_cell_pair: bool):
    traj = md.load(str(pdb))
    if any(a.element.symbol == "H" for a in traj.topology.atoms):
        traj = traj.atom_slice(traj.topology.select("not element H"))
    ff = MCPUForceField(traj)
    system = ff.create_system(traj.topology)
    ctx = mcpu_core.Context(system)
    ctx.set_positions((ff.coords[0] * 10.0).T.astype(np.float32))
    ctx.calculate_total_energy(-1)
    # Explicit API (default is false; env MCPU_USE_CELL_PAIR=1 also enables).
    ctx.set_use_cell_pair(bool(use_cell_pair))
    return ctx


def run_once(pdb: Path, *, cell_pair: bool, seed: int, steps: int,
             temperature: float, step_size: float):
    ctx = build_ctx(pdb, use_cell_pair=cell_pair)
    integ = mcpu_core.Integrator(temperature=temperature, step_size_rad=step_size)
    integ.set_seed(seed)
    integ.run(ctx, steps)
    bits = list(integ.last_accept_bits())
    energy = float(ctx.get_state().current_energy)
    return bits, energy


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pdb", type=Path, default=ROOT / "examples/actin/input_pdb/acta.pdb")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--step-size-rad", type=float, default=0.1)
    p.add_argument("--atol", type=float, default=1e-3)
    args = p.parse_args()

    if not args.pdb.exists():
        print(f"ERROR: missing PDB {args.pdb}", file=sys.stderr)
        return 1

    bits0, e0 = run_once(
        args.pdb, cell_pair=False, seed=args.seed, steps=args.steps,
        temperature=args.temperature, step_size=args.step_size_rad,
    )
    bits1, e1 = run_once(
        args.pdb, cell_pair=True, seed=args.seed, steps=args.steps,
        temperature=args.temperature, step_size=args.step_size_rad,
    )

    ok = True
    if bits0 != bits1:
        n_diff = sum(a != b for a, b in zip(bits0, bits1))
        first = next(i for i, (a, b) in enumerate(zip(bits0, bits1)) if a != b)
        print(
            f"FAIL: accept bits differ in {n_diff}/{args.steps} steps "
            f"(first at step {first}: off={bits0[first]} on={bits1[first]})"
        )
        ok = False
    else:
        print(f"OK: accept bits identical ({args.steps} steps)")

    de = abs(e0 - e1)
    if de > args.atol:
        print(f"FAIL: |ΔE|={de} > atol={args.atol} (off={e0} on={e1})")
        ok = False
    else:
        print(f"OK: final energy match (Δ={de})")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
