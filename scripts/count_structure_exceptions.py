#!/usr/bin/env python3
"""Count topology vs structure-specific clash disables for Mu pairs.

Reconstructs ``MuPotentialBuilder.build_topology_masks`` + the Rule 8
structure clash disable from ``MuPotential::cache_necessary_data``
(native_dist² < hard_core_sq). Does not require a C++ get_pair_hot binding.

Gate: if structure_zero > 10% of N², sparse Layer 3 is not valid.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import mdtraj as md
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymcpu.forcefields.mcpu import MCPUForceField  # noqa: E402

_MU_LAMBDA = 1.8
_MU_ALPHA = 0.75


def _topo_disables_clash(atom_i, atom_j, skip_local: int = 4) -> bool:
    """True if build_topology_masks would set check_clash=0 for (i,j).

    Mirrors mu_builder.py Rules 0–6 (pair-level; Rule 0 row mute handled
    by caller checking each atom).
    """
    # Rule 0 (per-atom mute) — checked by caller on each atom
    res_diff = abs(atom_i.residue_index - atom_j.residue_index)
    if res_diff == 0:
        if atom_i.is_sidechain == atom_j.is_sidechain:
            return True
        s_atom = atom_i if atom_i.is_sidechain else atom_j
        b_atom = atom_j if atom_i.is_sidechain else atom_i
        if b_atom.name in ("C", "N", "CA") and s_atom.name == "CB":
            return True
        if atom_i.residue_name == "PRO":
            return True
        if b_atom.name == "CA" and s_atom.name.startswith("G"):
            return True
        return False
    if res_diff == 1:
        first = atom_i if atom_i.residue_index < atom_j.residue_index else atom_j
        second = atom_j if atom_i.residue_index < atom_j.residue_index else atom_i
        is_pro_cd = second.name == "CD" and second.residue_name == "PRO"
        if is_pro_cd and first.name in ("C", "CA"):
            return True
        if (
            first.name == "N"
            or second.name not in ("CA", "N")
            or first.is_sidechain
            or second.is_sidechain
        ):
            return False
        return True
    if res_diff < skip_local:
        return False
    # distant: CYS SG–SG disables clash
    if (
        atom_i.residue_name == "CYS"
        and atom_j.residue_name == "CYS"
        and atom_i.name == "SG"
        and atom_j.name == "SG"
    ):
        return True
    return False


def _is_muted(atom) -> bool:
    return atom.name == "H" or (
        atom.residue_name == "GLY"
        and atom.name == "CA"
        and not atom.is_sidechain
    )


def count_exceptions(pdb: Path, label: str) -> dict:
    traj = md.load(str(pdb))
    if any(a.element.symbol == "H" for a in traj.topology.atoms):
        traj = traj.atom_slice(traj.topology.select("not element H"))
    ff = MCPUForceField(traj)
    atoms = ff.ordered_atom_list
    n = len(atoms)
    coords = (ff.coords[0] * 10.0).astype(np.float64)  # nm → Å, shape (n, 3)

    # Radii / hard_r2 from same lookup as MuPotentialBuilder.build
    radii = np.zeros(n, dtype=np.float64)
    for i, atom in enumerate(atoms):
        a_name = atom.name
        if atom.residue_name == "GLY" and a_name == "CA":
            r_name = "GLY"
        else:
            r_name = (
                "XXX"
                if a_name in ("N", "CA", "C", "O", "OCT", "OXT")
                else atom.residue_name
            )
        if a_name == "H":
            radii[i] = 0.0
        else:
            _, r = ff.atom_type_lookup[(r_name, a_name)]
            radii[i] = float(r)
    hard_r2 = (_MU_ALPHA * (radii[:, None] + radii[None, :])) ** 2

    topology_zero = 0
    structure_zero = 0
    clash_on = 0
    structure_exception_pairs: list[tuple[int, int]] = []
    per_atom_exc = Counter()

    for i in range(n):
        for j in range(i + 1, n):
            muted = _is_muted(atoms[i]) or _is_muted(atoms[j])
            topo_skip = muted or _topo_disables_clash(atoms[i], atoms[j])
            if topo_skip:
                topology_zero += 1
                continue
            # Topology left clash on — Rule 8 may clear it
            d = coords[i] - coords[j]
            dist_sq = float(d @ d)
            if dist_sq < hard_r2[i, j]:
                structure_zero += 1
                structure_exception_pairs.append((i, j))
                per_atom_exc[i] += 1
                per_atom_exc[j] += 1
            else:
                clash_on += 1

    n2 = n * n
    n_pairs = n * (n - 1) // 2
    csr_bytes = (n + 1) * 4 + len(structure_exception_pairs) * 2 * 4
    max_per = max(per_atom_exc.values()) if per_atom_exc else 0

    print(f"System: {label} ({pdb})")
    print(f"N atoms: {n}  (upper-triangle pairs: {n_pairs})")
    print(
        f"Topology clash-zeros:  {topology_zero:>10}  "
        f"({100.0 * topology_zero / n_pairs:.2f}% of pairs, "
        f"{100.0 * topology_zero / n2:.2f}% of N²)"
    )
    print(
        f"Structure clash-zeros: {structure_zero:>10}  "
        f"({100.0 * structure_zero / n_pairs:.4f}% of pairs, "
        f"{100.0 * structure_zero / n2:.4f}% of N²)  ← sparse exceptions"
    )
    print(f"Clash still enabled:   {clash_on:>10}")
    print(
        f"Exception CSR size (symmetric): {csr_bytes} bytes "
        f"({csr_bytes / 1024.0:.2f} KB)"
    )
    print(f"Max exceptions per atom: {max_per}")
    frac_n2 = structure_zero / n2
    if frac_n2 > 0.10:
        print(
            f"GATE FAIL: structure_zero = {100.0 * frac_n2:.2f}% of N² > 10% "
            f"— sparse Layer 3 NOT valid"
        )
    else:
        print(
            f"GATE PASS: structure_zero = {100.0 * frac_n2:.4f}% of N² ≤ 10%"
        )
    print()
    return {
        "label": label,
        "n": n,
        "topology_zero": topology_zero,
        "structure_zero": structure_zero,
        "clash_on": clash_on,
        "csr_bytes": csr_bytes,
        "max_per_atom": max_per,
        "frac_n2": frac_n2,
        "pairs": structure_exception_pairs,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--actin",
        type=Path,
        default=ROOT / "examples/actin/input_pdb/acta.pdb",
    )
    p.add_argument(
        "--chignolin",
        type=Path,
        default=ROOT / "examples/chignolin/input_pdb/1uao.pdb",
    )
    args = p.parse_args()

    results = []
    for path, label in ((args.actin, "actin"), (args.chignolin, "chignolin")):
        if not path.exists():
            print(f"SKIP: missing {path}", file=sys.stderr)
            continue
        results.append(count_exceptions(path, label))

    if not results:
        return 1
    if any(r["frac_n2"] > 0.10 for r in results):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
