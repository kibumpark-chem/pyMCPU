#!/usr/bin/env python3
"""Simple user-facing actin benchmark (SAFE defaults).

Protocol: steps=3000, warmup=100, seed=42, repeats=3.
Prints median ns/step (min..max) and accept count; optionally writes bench_result.json.
"""

from __future__ import annotations

import json
import platform
import statistics
import time
from pathlib import Path

import mdtraj as md
import numpy as np

from pymcpu import mcpu_core
from pymcpu.forcefields.mcpu import MCPUForceField

ROOT = Path(__file__).resolve().parents[2]
PDB = ROOT / "examples" / "actin" / "input_pdb" / "acta.pdb"

STEPS = 3000
WARMUP = 100
SEED = 42
REPEATS = 3
TEMPERATURE = 0.6
STEP_SIZE_RAD = 0.1


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def build_context():
    traj = md.load(str(PDB))
    if any(a.element.symbol == "H" for a in traj.topology.atoms):
        traj = traj.atom_slice(traj.topology.select("not element H"))
    ff = MCPUForceField(traj)
    system = ff.create_system(traj.topology)
    ctx = mcpu_core.Context(system)
    ctx.set_positions((ff.coords[0] * 10.0).T.astype(np.float32))
    ctx.calculate_total_energy(-1)
    if hasattr(ctx, "set_proxy_print_every"):
        ctx.set_proxy_print_every(-1)
    return ctx


def run_once(repeat: int) -> dict:
    ctx = build_context()
    integ = mcpu_core.Integrator(temperature=TEMPERATURE, step_size_rad=STEP_SIZE_RAD)
    integ.set_seed(SEED)
    if WARMUP:
        integ.run(ctx, WARMUP)
        integ.set_seed(SEED + 1_000_003 + repeat)
    t0 = time.perf_counter()
    integ.run(ctx, STEPS)
    elapsed = time.perf_counter() - t0
    bits = list(integ.last_accept_bits()) if hasattr(integ, "last_accept_bits") else []
    accept = int(sum(bits)) if bits else int(
        integ.get_bb_accepted() + integ.get_sc_accepted() + integ.get_kic_accepted()
    )
    ns_per_step = (elapsed / STEPS) * 1e9
    return {
        "repeat": repeat,
        "elapsed_s": elapsed,
        "ns_per_step": ns_per_step,
        "accept": accept,
        "energy": float(ctx.get_state().current_energy),
    }


def main() -> None:
    assert PDB.exists(), f"missing PDB: {PDB}"
    results = [run_once(i) for i in range(REPEATS)]
    ns = [r["ns_per_step"] for r in results]
    accepts = [r["accept"] for r in results]
    med = statistics.median(ns)
    summary = {
        "hostname": platform.node(),
        "cpu": _cpu_model(),
        "steps": STEPS,
        "warmup": WARMUP,
        "seed": SEED,
        "repeats": REPEATS,
        "ns_per_step_median": med,
        "ns_per_step_min": min(ns),
        "ns_per_step_max": max(ns),
        "accept_counts": accepts,
        "accept_median": int(statistics.median(accepts)),
        "safe_math": True,
        "repeats_detail": results,
    }
    print(
        f"actin bench  steps={STEPS} warmup={WARMUP} seed={SEED} repeats={REPEATS}"
    )
    print(f"host={summary['hostname']}  cpu={summary['cpu']}")
    print(
        f"ns/step median={med:,.0f}  "
        f"(min={min(ns):,.0f} .. max={max(ns):,.0f})"
    )
    print(f"accept counts={accepts}  median={summary['accept_median']}")

    out = Path(__file__).resolve().parent / "bench_result.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
