"""KIC's Sturm root finder against numpy's eigenvalue roots.

A KIC closure solves a degree-16 polynomial; every real root is one closure,
and the ratio of closure counts before and after a move enters the
acceptance. The Sturm solver counts sign changes with AVX2 in the default
build, so a lane or ordering slip there would change the closure count
without failing any energy check. This compares the count with the real
roots numpy finds, over chignolin's windows with the anchors displaced at
random to get polynomials with 0 to many real roots.
"""

from __future__ import annotations

import mdtraj as md
import numpy as np

from pymcpu import mcpu_core
from pymcpu.forcefields.mcpu import MCPUForceField
from pymcpu.runners import default_example_pdb

SEED = 20261004
TRIALS_PER_WINDOW = 40


def _real_roots(coeffs):
    """Real roots of sum(coeffs[i] x^i), or None when two roots are too
    close for a count to be compared (a near-double root)."""
    roots = np.roots(np.asarray(coeffs, dtype=np.float64)[::-1])
    real = np.sort(roots[np.abs(roots.imag) <= 1e-9 * np.maximum(1.0, np.abs(roots))].real)
    near_axis = roots[(np.abs(roots.imag) > 1e-9 * np.maximum(1.0, np.abs(roots)))
                      & (np.abs(roots.imag) < 1e-4 * np.maximum(1.0, np.abs(roots)))]
    if near_axis.size or (real.size > 1 and np.min(np.diff(real)) < 1e-6):
        return None
    return real


def test_closure_count_matches_numpy_real_roots():
    traj = md.load(str(default_example_pdb()))
    heavy = traj.atom_slice(traj.topology.select("not element H"))
    forcefield = MCPUForceField(heavy)
    system = forcefield.create_system(heavy.topology)
    X = (forcefield.coords[0] * 10.0).astype(np.float32).astype(np.float64)
    ref = system.get_kic_reference()
    blocks = system.get_block_indices()
    rng = np.random.default_rng(SEED)

    compared = 0
    counts = set()
    for r in range(system.get_num_residues() - 2):
        R0, R1, R2 = r, r + 1, r + 2
        solver = mcpu_core.TripeptideSolver()
        solver.initialize(
            [ref["len_ac"][R0], ref["len_cn"][R0], ref["len_na"][R1],
             ref["len_ac"][R1], ref["len_cn"][R1], ref["len_na"][R2]],
            [ref["ang_nac"][R0], ref["ang_acn"][R0], ref["ang_cna"][R0],
             ref["ang_nac"][R1], ref["ang_acn"][R1], ref["ang_cna"][R1],
             ref["ang_nac"][R2]],
            [ref["omega"][R0], ref["omega"][R1]],
        )
        n1 = X[blocks[R0].bb_start]
        a1 = X[blocks[R0].ca_atom()]
        a3 = X[blocks[R2].ca_atom()]
        c3 = X[blocks[R2].c_atom()]
        for _ in range(TRIALS_PER_WINDOW):
            shift = rng.normal(scale=0.4, size=3)
            a3_t = a3 + shift
            c3_t = c3 + shift + rng.normal(scale=0.1, size=3)
            solutions = solver.solve(list(n1), list(a1), list(a3_t), list(c3_t))
            found = len(solutions) + solver.last_rejected()
            if found == 0:
                continue  # may have stopped before building a polynomial
            real = _real_roots(solver.get_polynomial_coefficients())
            if real is None:
                continue
            assert found == real.size, (r, found, real)
            compared += 1
            counts.add(found)

    assert compared >= 100
    assert len(counts) >= 3, counts
