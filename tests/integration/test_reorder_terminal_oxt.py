"""The init_only atom reorder on a chain with a C-terminal OXT.

The builder puts OXT in the O segment after the last residue's O, and no
BlockIndices field names it. The reorder emitted only the atoms the blocks
name, so on any chain ending in OXT (most PDB files) it came up one atom short
and set_atom_reorder_mode("init_only") raised "atom count mismatch after tree
emit". It now emits OXT right after its residue's O, inside that residue's
span, so every move that carries the last residue carries OXT too.
"""

from __future__ import annotations

import mdtraj as md
import numpy as np
import pytest

from pymcpu import mcpu_core
from pymcpu.forcefields.mcpu import MCPUForceField
from pymcpu.runners import default_example_pdb


@pytest.fixture(scope="module")
def chignolin():
    # 1UAO ends in OXT.
    traj = md.load(str(default_example_pdb()))
    heavy = traj.atom_slice(traj.topology.select("not element H"))
    ff = MCPUForceField(heavy)
    assert any(a.name == "OXT" for a in ff.ordered_atom_list)
    return heavy, ff


def _context(heavy, ff, mode: str):
    ctx = mcpu_core.Context(ff.create_system(heavy.topology))
    ctx.set_atom_reorder_mode(mode)
    start = np.ascontiguousarray((ff.coords[0] * 10.0).T.astype(np.float32))
    ctx.set_positions(start)
    return ctx, start


def _terms(ctx) -> dict[str, float]:
    total = float(ctx.calculate_total_energy(-1))
    terms = {k: float(v) for k, v in ctx.energy_breakdown(False)["by_name"].items()}
    terms["total"] = total
    return terms


def test_init_only_places_oxt_and_leaves_the_energy_alone(chignolin) -> None:
    heavy, ff = chignolin
    off, start = _context(heavy, ff, "off")
    reordered, _ = _context(heavy, ff, "init_only")

    assert reordered.atom_permutation_info()["enabled"]
    assert np.array_equal(reordered.coords, start)  # build order out
    # OXT sits in the last residue's span, between its O and its end.
    oxt = next(i for i, a in enumerate(ff.ordered_atom_list) if a.name == "OXT")
    internal = reordered.atom_permutation_info()["ext_to_int"][oxt]
    last = reordered.get_system().get_block_indices()[-1]
    assert last.o_start < internal < last.res_end

    # Same terms up to summation order (atoms are visited in another order).
    a, b = _terms(off), _terms(reordered)
    assert a.keys() == b.keys()
    for k in a:
        assert b[k] == pytest.approx(a[k], rel=1e-5, abs=1e-4), k


@pytest.mark.parametrize(
    ("weights", "accepted"),
    [((1.0, 0.0, 0.0), "get_bb_accepted"),
     ((0.0, 1.0, 0.0), "get_kic_accepted"),
     ((0.0, 0.0, 1.0), "get_sc_accepted")],
    ids=["pivot", "kic", "sidechain"],
)
def test_moves_carry_oxt_with_the_carbonyl(chignolin, weights, accepted) -> None:
    """Every move type carries the last residue's span whole, OXT included,
    and the running energy stays equal to the full energy."""
    heavy, ff = chignolin
    ctx, _ = _context(heavy, ff, "init_only")
    ctx.calculate_total_energy(-1)
    names = [(a.residue_index, a.name) for a in ff.ordered_atom_list]
    last = max(r for r, _ in names)
    idx = {n: names.index((last, n)) for n in ("CA", "C", "O", "OXT")}

    def dists(coords):
        c = np.asarray(coords, dtype=np.float64)
        return np.array([np.linalg.norm(c[:, idx[a]] - c[:, idx["OXT"]]) for a in ("CA", "C", "O")])

    before = dists(ctx.coords)
    integ = mcpu_core.Integrator(temperature=0.6, step_size_rad=0.1)
    integ.set_seed(5)
    integ.set_move_weights(*weights)
    integ.run(ctx, 3000, 0)
    assert getattr(integ, accepted)() > 100

    assert np.abs(dists(ctx.coords) - before).max() < 1e-3
    running = float(ctx.get_state().current_energy)
    full = float(ctx.calculate_total_energy(-1))
    assert running == pytest.approx(full, abs=max(1e-3, 1e-5 * abs(full)))
