#!/usr/bin/env python3
"""Convert legacy MCPU's ``seq_dep_hb_mu_low.energy`` into pymcpu's
``mcpu_params/hbond_seq_dep.bin`` (a dense float32 (3,20,20) table).

Developer / one-time build helper (not required at runtime -- the binary
output is what's actually loaded)::

    python scripts/convert_hbond_seq_dep.py \\
        --src /path/to/legacy/config_files/seq_dep_hb_mu_low.energy \\
        --out src/pymcpu/parameters/pretrained/mcpu08/mcpu_params/hbond_seq_dep.bin

Sign convention matches legacy ``hbonds.h``'s ``InitializeHydrogenBonding()``
with ``SEQ_DEP_HB`` enabled: ``seq_hb[type][a][b] = -1.0 * value``. Entries
not present in the source file default to 0.0 (dropping that H-bond's energy
to zero), matching legacy's zero-initialized global array.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

N_SS_TYPES = 3
N_AMINO = 20


def convert(src: Path) -> np.ndarray:
    table = np.zeros((N_SS_TYPES, N_AMINO, N_AMINO), dtype=np.float32)
    for line in src.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        hs, i, j = int(parts[0]), int(parts[1]), int(parts[2])
        value = float(parts[3])
        table[hs, i, j] = -1.0 * value
    return table


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    table = convert(args.src)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.tofile(args.out)
    print(f"wrote {args.out} ({table.size} float32 elements, {table.nbytes} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
