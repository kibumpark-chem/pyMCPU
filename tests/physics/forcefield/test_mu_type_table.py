"""Mu's per-type-pair parameter table cannot depend on atom order.

The engine compresses the per-atom-pair hard-core distance, contact distance
and energy into one entry per pair of atom types, filled in atom order. That
is only valid if every atom of a type has the same radius. If two did not,
the entry would be whichever pair came last, so Mu would change with atom
order. Both ends now refuse such input: the atom-type file is checked when it
is read, and the engine checks the matrices it is handed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pymcpu import mcpu_core
from pymcpu.forcefields.builders.mu_builder import MuPotentialBuilder
from pymcpu.forcefields.mcpu import MCPUAtom


def test_an_atom_type_file_with_two_radii_for_one_type_is_refused(tmp_path: Path) -> None:
    csv = tmp_path / "atom_types.csv"
    csv.write_text("atom,residue,type,radius\nCB,ALA,0,1.88\nCB,ARG,0,1.90\n")
    with pytest.raises(ValueError, match="atom type 0 has radius 1.88 for ALA CB but 1.9 for ARG CB"):
        MuPotentialBuilder.load_atom_types(str(csv))


def test_one_radius_per_type_loads(tmp_path: Path) -> None:
    csv = tmp_path / "atom_types.csv"
    csv.write_text("atom,residue,type,radius\nCB,ALA,0,1.88\nCB,ARG,0,1.88\nCG,ARG,1,1.88\n")
    assert MuPotentialBuilder.load_atom_types(str(csv))[("ARG", "CB")] == (0, 1.88)


@pytest.mark.parametrize("bad", ["asymmetric", "nan"])
def test_an_unusable_energy_matrix_is_refused(tmp_path: Path, bad: str) -> None:
    from pymcpu.forcefields.builders.mu_builder import N_ATOM_TYPES

    matrix = np.zeros((N_ATOM_TYPES, N_ATOM_TYPES), dtype=np.float32)
    if bad == "asymmetric":
        matrix[0, 1], matrix[1, 0] = -0.5, -0.7
    else:
        matrix[2, 3] = matrix[3, 2] = np.nan
    path = tmp_path / "mu_potentials.bin"
    matrix.tofile(path)
    with pytest.raises(ValueError, match="not symmetric" if bad == "asymmetric" else "non-finite"):
        MuPotentialBuilder.load_parameters(str(path))


def _three_atom_mu(radii: list[float]) -> mcpu_core.MuPotential:
    """Three far-apart CB atoms of types 0, 0, 1 with the given radii."""
    atoms = [MCPUAtom(original_index=i, name="CB", residue_name=res, residue_index=i)
             for i, res in enumerate(("ALA", "ARG", "LEU"))]
    lookup = {("ALA", "CB"): (0, radii[0]), ("ARG", "CB"): (0, radii[1]),
              ("LEU", "CB"): (1, radii[2])}
    return MuPotentialBuilder.build(
        atom_list=atoms, atom_to_residue=[0, 1, 2],
        mu_potential_matrix=np.full((2, 2), -1.0, dtype=np.float32),
        atom_type_lookup=lookup,
    )


def _cache(mu: mcpu_core.MuPotential) -> None:
    n = 3
    ones = [1] * (n * n)
    coords = np.array([[0.0, 0.0, 0.0], [20.0, 0.0, 0.0], [40.0, 0.0, 0.0]], dtype=np.float32).T
    mu.cache_necessary_data(ones, ones, coords)


def test_the_engine_refuses_two_radii_for_one_type() -> None:
    # Pairs (0, 2) and (1, 2) are both type pair (0, 1), with different distances.
    # The constructor refuses it; it keeps only the per-type table.
    with pytest.raises(ValueError, match=r"atoms 1 and 2 \(types 0 and 1\)"):
        _three_atom_mu([1.88, 1.90, 1.88])


def test_the_engine_accepts_one_radius_per_type() -> None:
    _cache(_three_atom_mu([1.88, 1.88, 1.88]))


def test_the_engine_refuses_matrices_of_the_wrong_size() -> None:
    m = np.zeros((3, 3), dtype=np.float32)
    with pytest.raises(ValueError, match=r"\(n, n\) with n = len\(types\)"):
        mcpu_core.MuPotential(m, m, np.zeros((2, 2), dtype=np.float32),
                              [0, 0, 1], [0, 1, 2])
