"""Sidechain moves draw only residues they can move.

A sidechain step used to draw any residue, redrawing prolines, so a step that
landed on a glycine or an alanine, which have no chi angle, proposed nothing
and was counted as a rejected sidechain attempt: 17% of the sidechain steps
on chignolin, 8% on actin. The step now draws uniformly from the residues
with chi angles, except prolines (whose ring the moves leave alone) and, in
rotamer-library mode, residue types without rotamer rows. Fixed residues are
still redrawn. The draw does not depend on the state, so the proposal stays
symmetric. A positive sidechain weight on a chain with no such residue is an
error at setup instead of a run that silently wastes those steps.
"""
from __future__ import annotations

import mdtraj as md
import numpy as np
import pytest

from pymcpu import mcpu_core
from pymcpu.config import check_move_weights
from pymcpu.forcefields.mcpu import MCPUForceField
from pymcpu.runners import default_example_pdb
from tests.fixtures.context_builders import TEST_PDB


# Chignolin (1uao): GYDPETGTWG. Glycines 0, 6, 9 have no chi; proline 3 is
# never moved.
CHIGNOLIN_SITES = [1, 2, 4, 5, 7, 8]


def _build(pdb: str, residues: tuple[int, int] | None = None):
    traj = md.load(pdb)
    select = "not element H"
    if residues is not None:
        select += f" and resid {residues[0]} to {residues[1]}"
    heavy = traj.atom_slice(traj.topology.select(select))
    ff = MCPUForceField(heavy)
    ctx = mcpu_core.Context(ff.create_system(heavy.topology))
    ctx.set_positions((ff.coords[0] * 10.0).T.astype(np.float32))
    ctx.calculate_total_energy(-1)
    return ff, ctx


def _actin(residues: tuple[int, int] | None = None):
    if not TEST_PDB.is_file():
        pytest.skip(f"needs the actin example structure ({TEST_PDB})")
    return _build(str(TEST_PDB), residues)


def test_the_sites_are_the_residues_with_chi_angles() -> None:
    ff, _ = _build(str(default_example_pdb()))
    assert ff.sidechain_move_residues == CHIGNOLIN_SITES


@pytest.mark.slow  # full actin
def test_the_python_sites_match_the_engine() -> None:
    ff, ctx = _actin()
    system = ctx.get_system()
    ntors = list(system.get_torsions_per_residue())
    engine = [r for r, k in enumerate(ntors) if k > 0 and not system.is_proline(r)]
    assert ff.sidechain_move_residues == engine
    # Every one of them has rotamer rows, so rotamer mode draws the same sites.
    library = system.get_rotamer_library()
    assert all(library.num_rows(system.amino_index(r)) > 0 for r in engine)


def test_every_sidechain_step_moves_a_residue_with_chi_angles() -> None:
    """Tiny continuous chi steps at a huge temperature: every proposal is
    accepted, so a step that proposed nothing would show up as an attempt
    without an acceptance. Each accepted step changes the coordinates of
    exactly one residue, which must be one of the sites, and the sites are
    drawn uniformly."""
    ff, ctx = _build(str(default_example_pdb()))
    res = np.array([a.residue_index for a in ff.ordered_atom_list])
    integ = mcpu_core.Integrator(temperature=1.0e6, step_size_rad=0.1,
                                 sidechain_step_size_rad=1.0e-3)
    integ.set_seed(17)
    integ.set_sidechain_move_mode("continuous")
    integ.set_move_weights(0.0, 0.0, 1.0)

    steps = 1800
    hits = np.zeros(int(res.max()) + 1, dtype=int)
    previous = np.array(ctx.coords)
    for _ in range(steps):
        integ.run(ctx, 1)
        now = np.array(ctx.coords)
        moved = np.unique(res[np.any(now != previous, axis=0)])
        assert len(moved) == 1
        hits[moved[0]] += 1
        previous = now

    assert integ.get_sc_attempted() == steps
    assert integ.get_sc_accepted() == steps
    assert integ.move_counts()["sidechain"] == (steps, steps)
    assert np.nonzero(hits)[0].tolist() == CHIGNOLIN_SITES
    # Uniform over the six sites: 300 expected each; 5 sigma is about 80.
    assert np.all(np.abs(hits[CHIGNOLIN_SITES] - steps / 6) < 80), hits


def test_rotamer_steps_are_never_empty() -> None:
    """In rotamer-library mode every sidechain step proposes a move too (a
    step that lands on a residue it cannot change is not a valid move)."""
    _, ctx = _build(str(default_example_pdb()))
    integ = mcpu_core.Integrator(temperature=0.6, step_size_rad=0.1)
    integ.set_seed(23)
    integ.set_sidechain_move_mode("rotamer_library")
    integ.set_move_weights(0.0, 0.0, 1.0)
    integ.run(ctx, 2000)
    assert integ.move_counts()["rotamer"][1] == 2000
    assert integ.step_stats()["n_valid_moves"] == 2000


def test_no_movable_sidechain_is_a_setup_error() -> None:
    """Actin's Ala-Pro-Pro (332-334): the only chi angles are prolines'."""
    ff, ctx = _actin((332, 334))
    assert ff.sidechain_move_residues == []
    with pytest.raises(ValueError, match="no residue of this chain has a chi angle"):
        check_move_weights(ff, [0.5, 0.25, 0.25])
    check_move_weights(ff, [1.0, 0.0, 0.0])

    integ = mcpu_core.Integrator(temperature=0.6, step_size_rad=0.1)
    integ.set_seed(1)
    integ.set_move_weights(0.5, 0.0, 0.5)
    with pytest.raises(ValueError, match="no residue in this system has a chi angle"):
        integ.run(ctx, 10)
    integ.set_move_weights(1.0, 0.0, 0.0)
    integ.run(ctx, 10)
