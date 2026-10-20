"""The chi angles a Context caches, against the ones its coordinates give.

``State.sidechain_torsions`` holds four chi angles per residue, with -pi in
the slots past the residue's chi count (all four for glycine and alanine).
``chi_from_coordinates`` measures the same angles in numpy, from the atoms the
force field names for each chi (``MCPUForceField.chi_atom_indices``, build
order, as ``Context.coords`` is). Shared by the chain-end chi tests.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

#: The engine measures chi in float32 from float32 coordinates.
CHI_ATOL_RAD = 1e-4


def dihedral(c: np.ndarray, i: int, j: int, k: int, m: int) -> float:
    """Dihedral i-j-k-m (radians) of coordinates ``c``, shape (3, n_atoms)."""
    p0, p1, p2, p3 = c[:, i], c[:, j], c[:, k], c[:, m]
    b1 = (p2 - p1) / np.linalg.norm(p2 - p1)
    v = (p0 - p1) - np.dot(p0 - p1, b1) * b1
    w = (p3 - p2) - np.dot(p3 - p2, b1) * b1
    return math.atan2(float(np.dot(np.cross(b1, v), w)), float(np.dot(v, w)))


def chi_from_coordinates(forcefield: Any, coords: np.ndarray) -> np.ndarray:
    """(n_res, 4) chi angles of ``coords`` (3, n_atoms, build order); -pi
    where a residue has no such chi, as the engine stores it."""
    coords = np.asarray(coords, dtype=np.float64)
    out = np.full((len(forcefield.chi_atom_indices), 4), -math.pi)
    for r, rows in enumerate(forcefield.chi_atom_indices):
        for k, quad in enumerate(rows):
            if quad[0] >= 0:
                out[r, k] = dihedral(coords, *quad)
    return out


def cached_chi(context: Any) -> np.ndarray:
    """(n_res, 4) chi angles the context's State caches."""
    return np.array(
        [list(t.chi_angles) for t in context.get_state().sidechain_torsions], dtype=np.float64
    )


def angle_gap(a: Any, b: Any) -> np.ndarray:
    """|a - b| on the circle, in [0, pi]."""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return np.abs(np.remainder(d + math.pi, 2.0 * math.pi) - math.pi)


def stale_chi(context: Any, forcefield: Any) -> list[tuple[int, int]]:
    """(residue, chi index) of every cached chi that is not its coordinates'."""
    gap = angle_gap(
        cached_chi(context), chi_from_coordinates(forcefield, np.asarray(context.coords))
    )
    return [(int(r), int(k)) for r, k in np.argwhere(gap > CHI_ATOL_RAD)]
