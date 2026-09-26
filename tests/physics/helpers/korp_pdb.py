"""Read N/CA/C backbone geometry from a PDB the way upstream ``korpe`` does.

Deliberately a raw-column parser rather than mdtraj: the point of the tests
that use it is to compare against the reference binary on *its* reading of the
file, so anything that silently renumbers, reorders or reinterprets residues
would weaken exactly the comparison being made.
"""

from __future__ import annotations

import numpy as np


def parse_backbone(path):
    """Return ``(coords, res_names, res_seq, chain_ids)``.

    ``coords`` is ``(n_res, 3, 3)`` holding N, CA, C in Angstrom. Residues
    missing any of the three are dropped -- they have no KORP frame. Only the
    blank or "A" altloc is taken.
    """
    residues, order = {}, []
    with open(path) as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            if line[16] not in (" ", "A"):
                continue
            name = line[12:16].strip()
            if name not in ("N", "CA", "C"):
                continue
            key = (line[21], line[22:27])  # chain + resSeq + insertion code
            if key not in residues:
                residues[key] = {
                    "name": line[17:20].strip(),
                    "seq": int(line[22:26]),
                    "chain": line[21],
                }
                order.append(key)
            residues[key][name] = (
                float(line[30:38]), float(line[38:46]), float(line[46:54]),
            )

    coords, names, seqs, chains = [], [], [], []
    for key in order:
        r = residues[key]
        if not all(atom in r for atom in ("N", "CA", "C")):
            continue
        coords.append([r["N"], r["CA"], r["C"]])
        names.append(r["name"])
        seqs.append(r["seq"])
        chains.append(r["chain"])
    return np.asarray(coords, dtype=np.float64), names, seqs, chains
