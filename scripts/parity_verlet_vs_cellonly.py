#!/usr/bin/env python3
"""Parity: Mu Verlet (skin>0) vs CellOnly (skin=0) must match energies.

Bisects to the first diverging step (energy or accept bit), reporting move kind
and n_moved when possible. Exit 0 if max |ΔE| <= atol and accept bits match.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _parity_common import ROOT, build_ctx

from pymcpu import mcpu_core


def make_integ(seed: int, temperature: float, step_size: float):
    integ = mcpu_core.Integrator(temperature=temperature, step_size_rad=step_size)
    integ.set_seed(seed)
    return integ


def infer_move_kind(integ_before, integ_after) -> str:
    """Infer which move was attempted from counter deltas after one step."""
    dp = integ_after.get_bb_attempted() - integ_before.get_bb_attempted()
    dk = integ_after.get_kic_attempted() - integ_before.get_kic_attempted()
    ds = integ_after.get_sc_attempted() - integ_before.get_sc_attempted()
    if dp:
        return "Pivot"
    if dk:
        return "KIC"
    if ds:
        return "SC"
    return "Unknown"


def snapshot_counters(integ):
    return (
        integ.get_bb_attempted(),
        integ.get_kic_attempted(),
        integ.get_sc_attempted(),
    )


def run_bisect(pdb: Path, *, skin: float, seed: int, steps: int,
               temperature: float, step_size: float, atol: float,
               skin_only: bool) -> int:
    """Step-by-step bisect; stop at first energy/accept divergence."""
    ctx0 = build_ctx(pdb)
    ctx1 = build_ctx(pdb)
    ctx0.set_mu_skin(0.0)
    ctx1.set_mu_skin(float(skin))
    if skin_only:
        # Keep skin>0 denselist geometry but never use Verlet CSR.
        if hasattr(ctx1, "set_mu_verlet_enabled"):
            ctx1.set_mu_verlet_enabled(False)
        elif hasattr(ctx1, "set_verlet_moved_threshold"):
            ctx1.set_verlet_moved_threshold(0)
        else:
            print("WARN: cannot force CellOnly; rebuild needed", file=sys.stderr)

    integ0 = make_integ(seed, temperature, step_size)
    integ1 = make_integ(seed, temperature, step_size)

    e0_prev = float(ctx0.get_state().current_energy)
    e1_prev = float(ctx1.get_state().current_energy)
    print(f"Initial energies: cell={e0_prev:.6f} test={e1_prev:.6f} "
          f"delta={e1_prev - e0_prev:.6e}")

    for step in range(steps):
        c0 = snapshot_counters(integ0)
        c1 = snapshot_counters(integ1)
        integ0.run(ctx0, 1)
        integ1.run(ctx1, 1)
        bits0 = list(integ0.last_accept_bits())
        bits1 = list(integ1.last_accept_bits())
        accept0 = bool(bits0[0]) if bits0 else False
        accept1 = bool(bits1[0]) if bits1 else False
        e0 = float(ctx0.get_state().current_energy)
        e1 = float(ctx1.get_state().current_energy)
        de = e1 - e0
        dp = integ0.get_bb_attempted() - c0[0]
        dk = integ0.get_kic_attempted() - c0[1]
        ds = integ0.get_sc_attempted() - c0[2]
        if dp:
            move_kind = "Pivot"
        elif dk:
            move_kind = "KIC"
        elif ds:
            move_kind = "SC"
        else:
            move_kind = "Unknown"
        del c1  # second integ counters unused (same RNG path until divergence)

        n_moved = None
        try:
            st = dict(integ0.step_stats())
            if st.get("n_valid_moves", 0):
                n_moved = st.get("moved_atoms_sum", None)
        except Exception:
            pass

        if abs(de) > atol or accept0 != accept1:
            print(f"First divergence at step {step}")
            print(f"  CellOnly  energy: {e0:.6f}")
            print(f"  skin={skin} energy: {e1:.6f}")
            print(f"  delta E: {de:.6f}")
            print(f"  accept CellOnly={int(accept0)} skin={int(accept1)}")
            print(f"  move kind: {move_kind}")
            print(f"  moved_indices count: {n_moved}")
            print(f"  energy before step: cell={e0_prev:.6f} test={e1_prev:.6f}")
            # Neighbor stats from skin run
            try:
                ns = ctx1.neighborStats() if hasattr(ctx1, "neighborStats") else None
                if ns is None and hasattr(ctx1, "neighbor_proxy_report"):
                    print(f"  neighbor_proxy: {ctx1.neighbor_proxy_report()}")
            except Exception as ex:
                print(f"  (neighbor stats unavailable: {ex})")
            return 1

        e0_prev, e1_prev = e0, e1

    print(f"OK: no divergence in {steps} steps "
          f"(skin=0 vs skin={skin}{', CellOnly-forced' if skin_only else ''})")
    print(f"Final energies: cell={e0_prev:.6f} test={e1_prev:.6f} "
          f"delta={e1_prev - e0_prev:.6e}")
    return 0


def run_bulk(pdb: Path, *, skin: float, seed: int, steps: int,
             temperature: float, step_size: float, atol: float) -> int:
    """Original bulk comparison (full run each)."""
    def run_once(sk: float):
        ctx = build_ctx(pdb)
        ctx.set_mu_skin(float(sk))
        integ = make_integ(seed, temperature, step_size)
        integ.run(ctx, steps)
        bits = list(integ.last_accept_bits())
        energy = float(ctx.get_state().current_energy)
        return bits, energy

    bits0, e0 = run_once(0.0)
    bits1, e1 = run_once(skin)
    ok = True
    if bits0 != bits1:
        n_diff = sum(a != b for a, b in zip(bits0, bits1))
        print(f"FAIL: accept bits differ in {n_diff}/{steps} steps "
              f"(skin=0 vs skin={skin})")
        # Find first bit difference for hint
        for i, (a, b) in enumerate(zip(bits0, bits1)):
            if a != b:
                print(f"  first accept-bit divergence at step {i}: "
                      f"cell={int(a)} skin={int(b)}")
                break
        ok = False
    else:
        print(f"OK: accept bits identical ({steps} steps)")

    de = abs(e0 - e1)
    if de > atol:
        print(f"FAIL: final energy |{e0} - {e1}| = {de} > {atol}")
        ok = False
    else:
        print(f"OK: final energy match within {atol} (Δ={de:g})")

    print(f"skin=0 final_E={e0:.6f}  skin={skin} final_E={e1:.6f}")
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pdb", type=Path, default=ROOT / "examples/actin/input_pdb/acta.pdb")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--step-size-rad", type=float, default=0.1)
    p.add_argument("--skin", type=float, default=1.0, help="Verlet skin for second run")
    p.add_argument("--atol", type=float, default=1e-3)
    p.add_argument("--bisect", action="store_true", default=True,
                   help="Stop at first diverging step (default)")
    p.add_argument("--no-bisect", action="store_true",
                   help="Run full trajectories then compare")
    p.add_argument("--skin-only", action="store_true",
                   help="Keep skin>0 but force CellOnly (isolate denselist vs Verlet)")
    args = p.parse_args()

    if not args.pdb.exists():
        print(f"ERROR: missing PDB {args.pdb}", file=sys.stderr)
        return 1

    if args.no_bisect:
        return run_bulk(
            args.pdb, skin=args.skin, seed=args.seed, steps=args.steps,
            temperature=args.temperature, step_size=args.step_size_rad,
            atol=args.atol,
        )

    return run_bisect(
        args.pdb, skin=args.skin, seed=args.seed, steps=args.steps,
        temperature=args.temperature, step_size=args.step_size_rad,
        atol=args.atol, skin_only=args.skin_only,
    )


if __name__ == "__main__":
    sys.exit(main())
