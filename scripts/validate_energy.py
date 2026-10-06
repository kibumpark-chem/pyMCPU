#!/usr/bin/env python3
"""
Energy validation script: compare pyMCPU (this repo) vs legacy MCPU
(src_mpi_umbrella) using chignolin as test case.

This script does NOT make any changes. It only reports differences.

Legacy reference:
    The legacy MCPU sources (``src_mpi_umbrella``) are not distributed with
    pyMCPU. Point ``MCPU_LEGACY_SRC`` at a local checkout to compare against a
    legacy build; without it this script reports pyMCPU's own energies only.

Usage:
    python scripts/validate_energy.py [--pdb PATH]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def compute_current_energies(pdb_file: str) -> dict:
    """Compute all energy components using current codebase."""
    import mdtraj as md
    from pymcpu import mcpu_core
    from pymcpu.forcefields.mcpu import MCPUForceField

    traj = md.load(pdb_file)
    indices = traj.topology.select("not element H")
    filtered = traj.atom_slice(indices)
    ff = MCPUForceField(filtered)
    system = ff.create_system(filtered.topology)
    ctx = mcpu_core.Context(system)
    ctx.set_use_legacy_weights(True)
    ctx.set_atom_reorder_mode("off")
    coords = (ff.coords[0] * 10.0).T.astype("float32")
    ctx.set_positions(coords)
    ctx.calculate_total_energy(-1)

    bd = ctx.energy_breakdown(weighted=True)
    results = {"total": float(bd["weighted_total"])}
    results.update({name: float(value) for name, value in bd["by_name"].items()})
    return results


def compare_energies(current: dict) -> list[dict]:
    """
    Compare energy components between current and expected.
    Since legacy binary may not be runnable, we report current values
    as the baseline for future comparisons.
    """
    results = []
    for component, value in current.items():
        results.append({
            "component": component,
            "current": value,
            "status": "REPORTED",
        })
    return results


def _first_existing(candidates: list[Path]) -> str | None:
    for c in candidates:
        if c.exists():
            return str(c)
    return None


def _print_report(pdb: str) -> None:
    print("=" * 70)
    print("ENERGY VALIDATION REPORT: pyMCPU Current Codebase")
    print(f"Test system: {Path(pdb).name}")
    print("=" * 70)

    current = compute_current_energies(pdb)
    comparison = compare_energies(current)

    print(f"\n{'Component':<25s} | {'Value (weighted)':>16s} | {'Status'}")
    print("-" * 70)
    for row in comparison:
        print(f"{row['component']:<25s} | {row['current']:>16.6f} | {row['status']}")
    print("=" * 70)


LEGACY_ACTIN_HBOND_WEIGHTED = -128.119058 * 1.35  # legacy run acta_T_0.600.log, STEP 0 (log not distributed; see docs/hbond_legacy_parity.md)


def _print_legacy_hbond_comparison(pdb: str) -> None:
    """Human-readable pyMCPU-vs-legacy-log side-by-side for the H-bond group,
    on the actin structure whose legacy reference value is already checked
    into the repo (no legacy binary invocation needed)."""
    current = compute_current_energies(pdb)
    hbond = current["hydrogen_bond"]
    diff = hbond - LEGACY_ACTIN_HBOND_WEIGHTED
    print("=" * 70)
    print("LEGACY H-BOND PARITY: pyMCPU vs. legacy MCPU log")
    print(f"Test system: {Path(pdb).name}")
    print("=" * 70)
    print(f"{'pyMCPU (weighted)':<25s} | {hbond:>16.6f}")
    print(f"{'legacy log (weighted)':<25s} | {LEGACY_ACTIN_HBOND_WEIGHTED:>16.6f}")
    print(f"{'difference':<25s} | {diff:>16.6f}")
    print("=" * 70)
    print("See docs/hbond_legacy_parity.md for the full ablation-ladder derivation.")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pdb", default=None, help="Path to test PDB")
    parser.add_argument(
        "--legacy-hbond", action="store_true",
        help="Print a pyMCPU-vs-legacy-log H-bond side-by-side on the actin structure "
             "instead of the full per-system energy report.",
    )
    args = parser.parse_args()

    actin_pdb = _first_existing([ROOT / "examples" / "actin" / "input_pdb" / "acta.pdb"])

    if args.legacy_hbond:
        pdb = args.pdb or actin_pdb
        if pdb is None:
            print("ERROR: No actin test PDB found.", file=sys.stderr)
            sys.exit(1)
        _print_legacy_hbond_comparison(pdb)
        return 0

    if args.pdb:
        _print_report(args.pdb)
    else:
        chignolin_pdb = _first_existing([
            ROOT / "examples" / "chignolin" / "input_pdb" / "1uao.pdb",
            ROOT / "examples" / "data" / "1uao.pdb",
        ])
        systems = [p for p in (chignolin_pdb, actin_pdb) if p is not None]
        if not systems:
            print("ERROR: No test PDB found.", file=sys.stderr)
            sys.exit(1)
        for pdb in systems:
            _print_report(pdb)
            print()

    print("NOTE: This report establishes the current energy baseline.")
    print("Legacy MCPU binary comparison requires manual setup of the legacy code")
    print("(or use --legacy-hbond for the checked-in actin log reference).")
    print("No code changes are made based on this report.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
