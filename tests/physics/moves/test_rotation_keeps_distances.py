"""A pivot keeps every distance inside the pieces it turns rigidly.

A phi pivot turns C and the sidechain of residue r about N-CA, a psi pivot
turns N and the sidechain about CA-C, so the distances among N, CA, C and the
sidechain atoms of one residue never change. Nor do those inside a peptide
plane, CA(r) C(r) O(r) N(r+1) CA(r+1). In the engine they change only by
rounding.

The rotation used to be applied in float32, rounding twice: relative to the
pivot, then in lab coordinates. That shrank these distances steadily, about
-2.5e-9 A per accepted move on average, so bonds got shorter the longer a run
went (CA-C by 3e-3 A over 5M pivot-only chignolin steps). It is now done in
double and rounded once, which leaves unbiased noise.
"""

from __future__ import annotations

import mdtraj as md
import numpy as np

from pymcpu import mcpu_core
from pymcpu.forcefields.mcpu import MCPUForceField
from pymcpu.runners import default_example_pdb

STEPS = 100_000


def _preserved_pairs(ff: MCPUForceField) -> np.ndarray:
    atoms = ff.ordered_atom_list
    res = np.array([a.residue_index for a in atoms])
    name = np.array([a.name for a in atoms])

    def index(r: int, atom: str) -> int:
        hit = np.nonzero((res == r) & (name == atom))[0]
        return int(hit[0])

    pairs = set()
    for r in range(int(res.max()) + 1):
        unit = [i for i in np.nonzero(res == r)[0] if name[i] not in ("O", "OXT", "OCT")]
        pairs.update((a, b) for k, a in enumerate(unit) for b in unit[k + 1:])
        if r < res.max():
            plane = [index(r, "CA"), index(r, "C"), index(r, "O"), index(r + 1, "N"), index(r + 1, "CA")]
            pairs.update((min(a, b), max(a, b)) for k, a in enumerate(plane) for b in plane[k + 1:])
    return np.array(sorted(pairs))


def _distances(coords: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    c = np.asarray(coords, dtype=np.float64)
    d = c[:, pairs[:, 0]] - c[:, pairs[:, 1]]
    return np.sqrt((d * d).sum(axis=0))


def test_pivots_do_not_shrink_the_rigid_pieces() -> None:
    traj = md.load(str(default_example_pdb()))
    heavy = traj.atom_slice(traj.topology.select("not element H"))
    ff = MCPUForceField(heavy)
    ctx = mcpu_core.Context(ff.create_system(heavy.topology))
    ctx.set_positions((ff.coords[0] * 10.0).T.astype(np.float32))
    ctx.calculate_total_energy(-1)
    integ = mcpu_core.Integrator(temperature=0.6, step_size_rad=0.1)
    integ.set_seed(11)
    integ.set_move_weights(1.0, 0.0, 0.0)

    pairs = _preserved_pairs(ff)
    before = _distances(ctx.coords, pairs)
    integ.run(ctx, STEPS, 0)
    change = _distances(ctx.coords, pairs) - before

    assert integ.get_bb_accepted() > 30_000  # enough moves to see a trend
    # Seeds 11-13, float32 rotation: mean -1.0e-4 A, largest 5e-4 A, 91-92%
    # of the pairs shorter. Double: mean at most 1.3e-6 A, largest 6e-5 A,
    # half of them shorter.
    assert abs(change.mean()) < 1e-5, change.mean()
    assert np.abs(change).max() < 2e-4, np.abs(change).max()
    assert (change < 0).mean() < 0.7, (change < 0).mean()
