"""Pivots turn the backbone torsions at both chain ends.

A pivot used to draw its residue from 1 to N-2, so psi of the first residue
never changed (KIC windows start at residue 1 too) and phi of the last residue
changed only through KIC. A pivot now draws one of the chain's 2N-2 backbone
torsions, psi(0), phi(1), psi(1), ..., psi(N-2), phi(N-1), uniformly. Proline
phi and torsions whose rotation would move a fixed residue are redrawn as
before, so the draw does not depend on the state and the proposal stays
symmetric; the moving part is still the shorter end.

psi(0) turns N(0) and the sidechain of residue 0 about CA(0)-C(0); phi(N-1)
turns C, O, the C-terminal OXT and the sidechain of residue N-1 about
N(N-1)-CA(N-1).
"""
from __future__ import annotations

import math

import mdtraj as md
import numpy as np
import pytest

from pymcpu import mcpu_core
from pymcpu.forcefields.mcpu import MCPUForceField
from pymcpu.runners import default_example_pdb
from tests.fixtures.context_builders import TEST_PDB


LAYOUTS = ["off", "init_only"]


def _build(pdb: str, reorder: str, residues: tuple[int, int] | None = None):
    traj = md.load(pdb)
    select = "not element H"
    if residues is not None:
        select += f" and resid {residues[0]} to {residues[1]}"
    heavy = traj.atom_slice(traj.topology.select(select))
    ff = MCPUForceField(heavy)
    ctx = mcpu_core.Context(ff.create_system(heavy.topology))
    ctx.set_atom_reorder_mode(reorder)
    ctx.set_positions((ff.coords[0] * 10.0).T.astype(np.float32))
    ctx.calculate_total_energy(-1)
    res = np.array([a.residue_index for a in ff.ordered_atom_list])
    names = np.array([a.name for a in ff.ordered_atom_list])
    return ctx, res, names


def _atom(res: np.ndarray, names: np.ndarray, r: int, name: str) -> int:
    return int(np.nonzero((res == r) & (names == name))[0][0])


def _dihedral(c: np.ndarray, i: int, j: int, k: int, m: int) -> float:
    p0, p1, p2, p3 = c[:, i], c[:, j], c[:, k], c[:, m]
    b1 = (p2 - p1) / np.linalg.norm(p2 - p1)
    v = (p0 - p1) - np.dot(p0 - p1, b1) * b1
    w = (p3 - p2) - np.dot(p3 - p2, b1) * b1
    return math.atan2(np.dot(np.cross(b1, v), w), np.dot(v, w))


def _psi0(res, names):
    return (_atom(res, names, 0, "N"), _atom(res, names, 0, "CA"),
            _atom(res, names, 0, "C"), _atom(res, names, 1, "N"))


def _phi_last(res, names):
    last = int(res.max())
    return (_atom(res, names, last - 1, "C"), _atom(res, names, last, "N"),
            _atom(res, names, last, "CA"), _atom(res, names, last, "C"))


def _turn(a: float, b: float) -> float:
    return abs(math.remainder(a - b, 2.0 * math.pi))


def test_the_end_torsions_are_pivot_sites() -> None:
    ctx, res, names = _build(str(default_example_pdb()), "off")
    n = int(res.max()) + 1
    integ = mcpu_core.Integrator(temperature=0.6, step_size_rad=0.1)
    integ.set_seed(3)

    assert integ.debug_force_pivot(ctx, 0, False)  # psi(0)
    moved = set(integ.last_moved_indices())
    n0, ca0, c0 = (_atom(res, names, 0, a) for a in ("N", "CA", "C"))
    # Gly 1: only N turns. The identity layout also lists CA and C, the axis
    # atoms, which a turn about their own axis leaves in place.
    assert n0 in moved and moved <= {n0, ca0, c0}

    assert integ.debug_force_pivot(ctx, n - 1, True)  # phi(N-1)
    want = {_atom(res, names, n - 1, a) for a in ("C", "O", "OXT")}  # Gly 10
    assert set(integ.last_moved_indices()) == want

    # Residue 0 has no phi to turn and residue N-1 no psi.
    assert not integ.debug_force_pivot(ctx, 0, True)
    assert not integ.debug_force_pivot(ctx, n - 1, False)


@pytest.mark.parametrize("reorder", LAYOUTS)
def test_a_pivot_run_turns_both_end_torsions(reorder: str) -> None:
    ctx, res, names = _build(str(default_example_pdb()), reorder)
    start = np.array(ctx.coords)
    integ = mcpu_core.Integrator(temperature=0.6, step_size_rad=0.1)
    integ.set_seed(5)
    integ.set_move_weights(1.0, 0.0, 0.0)
    integ.run(ctx, 3000)
    end = np.array(ctx.coords)
    for torsion in (_psi0(res, names), _phi_last(res, names)):
        assert _turn(_dihedral(end, *torsion), _dihedral(start, *torsion)) > 0.01


def _pairwise(c: np.ndarray, atoms: list[int]) -> np.ndarray:
    x = c[:, atoms].T
    return np.linalg.norm(x[:, None, :] - x[None, :, :], axis=2)


# (structure, residue range, which end): actin's ends carry sidechains,
# chignolin's last residue carries the OXT.
END_CASES = [
    ("actin", (0, 4), "psi0"),
    ("actin", (370, 374), "phi_last"),
    ("chignolin", None, "phi_last"),
]


@pytest.mark.parametrize("reorder", LAYOUTS)
@pytest.mark.parametrize("structure,residues,end", END_CASES)
def test_an_end_pivot_turns_one_rigid_piece(structure, residues, end, reorder) -> None:
    """With every other residue fixed, the end torsion is the only pivot
    site. Each accepted move turns the end's moving atoms rigidly about the
    bond, so the distances inside the moving piece (with the two axis atoms)
    and inside the rest of the chain never change, while the torsion does."""
    if structure == "actin":
        if not TEST_PDB.is_file():
            pytest.skip(f"needs the actin example structure ({TEST_PDB})")
        pdb = str(TEST_PDB)
    else:
        pdb = str(default_example_pdb())
    ctx, res, names = _build(pdb, reorder, residues)
    n = int(res.max()) + 1
    if end == "psi0":
        r, torsion = 0, _psi0(res, names)
        axis = [_atom(res, names, 0, "CA"), _atom(res, names, 0, "C")]
        moving = [int(i) for i in np.nonzero(res == 0)[0] if names[i] not in ("CA", "C", "O")]
    else:
        r, torsion = n - 1, _phi_last(res, names)
        axis = [_atom(res, names, r, "N"), _atom(res, names, r, "CA")]
        moving = [int(i) for i in np.nonzero(res == r)[0] if names[i] not in ("N", "CA", "H")]
    if structure == "chignolin":
        assert _atom(res, names, r, "OXT") in moving
    elif end == "psi0":
        assert _atom(res, names, 0, "CB") in moving
    rest = [i for i in range(len(res)) if i not in moving]

    start = np.array(ctx.coords)
    integ = mcpu_core.Integrator(temperature=0.6, step_size_rad=0.5)
    integ.set_seed(9)
    integ.set_move_weights(1.0, 0.0, 0.0)
    integ.set_fixed_residues([i for i in range(n) if i != r], n)
    integ.run(ctx, 400)
    end_coords = np.array(ctx.coords)

    assert integ.get_bb_accepted() > 10
    assert _turn(_dihedral(end_coords, *torsion), _dihedral(start, *torsion)) > 0.01
    piece = moving + axis
    np.testing.assert_allclose(_pairwise(end_coords, piece), _pairwise(start, piece), atol=2e-4)
    np.testing.assert_allclose(end_coords[:, rest], start[:, rest], atol=2e-4)
