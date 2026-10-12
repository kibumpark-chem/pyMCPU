"""An independent reference for the ``dunbrack2002_sym`` proposal density.

Reads ``bbind02.May.lib`` with its own parser (not the package's builder) and
evaluates the density the move is documented to draw from, in float64:

    q(chi) = sum_k w_k prod_i g_ki(chi_i)

over the rows k of the residue type with a positive probability, w_k the
row's probability renormalised over those rows, and g_ki a normal with the
row's mean and width truncated to one period around the mean and divided by
its mass inside that period. The two-fold symmetric chi (ASP chi2, GLU chi3,
PHE chi2, TYR chi2) have period pi, and their g carries a factor 1/2 for the
random 180 degree image.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from pymcpu import PACKAGE_ROOT

BBIND_FILE = Path(PACKAGE_ROOT) / "data" / "params" / "mcpu08" / "constants" / "bbind02.May.lib"
SYMMETRIC = {"ASP": 1, "GLU": 2, "PHE": 1, "TYR": 1}
N_CHI = {"SER": 1, "CYS": 1, "VAL": 1, "THR": 1, "ILE": 2, "LEU": 2, "ASP": 2, "ASN": 2,
         "HIS": 2, "PHE": 2, "TYR": 2, "TRP": 2, "GLU": 3, "GLN": 3, "MET": 3, "LYS": 4,
         "ARG": 4}


def file_rows(residue: str) -> list[tuple[float, list[float], list[float]]]:
    """(probability in %, means, widths in degrees) of each line of `residue`."""
    rows = []
    for line in BBIND_FILE.read_text().splitlines():
        tok = line.split()
        if tok and tok[0] == residue:
            rows.append((float(tok[7]), [float(tok[11 + 2 * i]) for i in range(4)],
                         [float(tok[12 + 2 * i]) for i in range(4)]))
    return rows


def reference_rows(residue: str):
    """Weights (K,), means and widths (K, n) in radians: the rows the move uses."""
    n = N_CHI[residue]
    rows = [r for r in file_rows(residue) if r[0] > 0.0]
    w = np.array([r[0] for r in rows])
    mu = np.radians([r[1][:n] for r in rows])
    sigma = np.radians([r[2][:n] for r in rows])
    return w / w.sum(), mu, sigma


def reference_log_q(residue: str, chi, *, period_pi: bool = True) -> float:
    """log q(chi) of the ``dunbrack2002_sym`` proposal. ``period_pi=False``
    treats a symmetric chi like the others (a deliberately wrong q)."""
    symmetric = SYMMETRIC.get(residue) if period_pi else None
    weights, means, sigmas = reference_rows(residue)
    total = 0.0
    for w, mu_row, sg_row in zip(weights, means, sigmas):
        density = w
        for c, (x, m, s) in enumerate(zip(chi, mu_row, sg_row)):
            period = math.pi if c == symmetric else 2.0 * math.pi
            d = (x - m + period / 2) % period - period / 2
            mass = math.erf(period / 2 / (math.sqrt(2.0) * s))
            density *= math.exp(-0.5 * (d / s) ** 2) / (math.sqrt(2 * math.pi) * s * mass)
            if c == symmetric:
                density *= 0.5
        total += density
    return math.log(total)


def legacy_log_q(residue: str, chi) -> float:
    """log q(chi) of the ``dunbrack2002`` proposal: every line, untruncated
    normals at the nearest image, period 2 pi throughout (a wrong q for
    ``dunbrack2002_sym``)."""
    n = N_CHI[residue]
    rows = file_rows(residue)
    total_w = sum(r[0] for r in rows)
    total = 0.0
    for p, mu_row, sg_row in rows:
        if p <= 0.0:
            continue
        density = p / total_w
        for x, m, s in zip(chi[:n], np.radians(mu_row[:n]), np.radians(sg_row[:n])):
            d = (x - m + math.pi) % (2 * math.pi) - math.pi
            density *= math.exp(-0.5 * (d / s) ** 2) / (math.sqrt(2 * math.pi) * s)
        total += density
    return math.log(total)


def dihedral(p0, p1, p2, p3) -> float:
    """IUPAC dihedral angle (radians) of four points."""
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1 = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    return math.atan2(np.dot(np.cross(b1, v), w), np.dot(v, w))
