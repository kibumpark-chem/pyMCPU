#!/usr/bin/env python3
"""Layered eval smoke parity: two runs with the same seed must match.

Runs the layered Mu path twice with identical seed and checks accept bits
and final energy are reproducible (finite energy sanity check included).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from _parity_common import ROOT, build_ctx

from pymcpu import mcpu_core


def run_once(pdb: Path, *, seed: int, steps: int, temperature: float, step_size: float):
    ctx = build_ctx(pdb)
    integ = mcpu_core.Integrator(
        temperature=temperature, step_size_rad=step_size
    )
    integ.use_sparse_proposal = True
    integ.set_seed(seed)
    integ.run(ctx, steps)
    bits = list(integ.last_accept_bits())
    energy = float(ctx.get_state().current_energy)
    return bits, energy


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--pdb",
        type=Path,
        default=ROOT / "examples/actin/input_pdb/acta.pdb",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--step-size-rad", type=float, default=0.1)
    args = p.parse_args()

    if not args.pdb.exists():
        print(f"ERROR: missing PDB {args.pdb}", file=sys.stderr)
        return 1

    bits_a, e_a = run_once(
        args.pdb,
        seed=args.seed,
        steps=args.steps,
        temperature=args.temperature,
        step_size=args.step_size_rad,
    )
    bits_b, e_b = run_once(
        args.pdb,
        seed=args.seed,
        steps=args.steps,
        temperature=args.temperature,
        step_size=args.step_size_rad,
    )

    ok = True
    if bits_a != bits_b:
        n_diff = sum(a != b for a, b in zip(bits_a, bits_b))
        print(f"FAIL: accept bits differ in {n_diff}/{args.steps} steps")
        for i, (a, b) in enumerate(zip(bits_a, bits_b)):
            if a != b:
                print(f"  first diverge at step {i}: run_a={a} run_b={b}")
                break
        ok = False
    else:
        print(f"OK: accept bits identical ({args.steps} steps)")

    if e_a != e_b:
        print(f"FAIL: final energy differs: {e_a} vs {e_b}")
        ok = False
    else:
        print(f"OK: final energy reproducible ({e_a:.4f})")

    if not np.isfinite(e_a):
        print(f"FAIL: final energy not finite: {e_a}")
        ok = False
    else:
        print("OK: final energy finite")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
