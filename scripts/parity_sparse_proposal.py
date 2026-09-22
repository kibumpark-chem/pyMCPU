#!/usr/bin/env python3
"""Parity test: sparse proposal on vs off must match accept bits and energies.

Runs the same seed/steps twice and compares per-step accept bits and final energy.
Exit 0 on match within atol; exit 1 on mismatch.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _parity_common import ROOT, build_ctx

from pymcpu import mcpu_core


def run_once(pdb: Path, *, sparse: bool, seed: int, steps: int, temperature: float,
             step_size: float):
    ctx = build_ctx(pdb)
    integ = mcpu_core.Integrator(temperature=temperature, step_size_rad=step_size)
    integ.use_sparse_proposal = sparse
    integ.set_seed(seed)
    integ.run(ctx, steps)
    bits = list(integ.last_accept_bits())
    energy = float(ctx.get_state().current_energy)
    return bits, energy, integ.step_stats()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pdb", type=Path, default=ROOT / "examples/actin/input_pdb/acta.pdb")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--step-size-rad", type=float, default=0.1)
    p.add_argument("--atol", type=float, default=1e-4)
    args = p.parse_args()

    if not args.pdb.exists():
        print(f"ERROR: missing PDB {args.pdb}", file=sys.stderr)
        return 1

    bits_on, e_on, _ = run_once(
        args.pdb, sparse=True, seed=args.seed, steps=args.steps,
        temperature=args.temperature, step_size=args.step_size_rad,
    )
    bits_off, e_off, _ = run_once(
        args.pdb, sparse=False, seed=args.seed, steps=args.steps,
        temperature=args.temperature, step_size=args.step_size_rad,
    )

    ok = True
    if bits_on != bits_off:
        n_diff = sum(a != b for a, b in zip(bits_on, bits_off))
        print(f"FAIL: accept bits differ in {n_diff}/{args.steps} steps")
        ok = False
    else:
        print(f"OK: accept bits identical ({args.steps} steps)")

    de = abs(e_on - e_off)
    if de > args.atol:
        print(f"FAIL: final energy |{e_on} - {e_off}| = {de} > {args.atol}")
        ok = False
    else:
        print(f"OK: final energy match within {args.atol} (Δ={de:g})")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
