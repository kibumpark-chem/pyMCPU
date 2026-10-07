"""KORP honours the residue energy mask (System.set_energy_ignored_residues).

The orientational pair term is all energy, so both modes drop every pair with
a masked residue. The CA excluded-volume guard is all clash, so ignore_all
stops testing a masked residue and clash_only keeps every test. The delta and
the full energy follow the same rules, and an unmasked run is unchanged.
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path

import mdtraj as md
import numpy as np
import pytest

import pymcpu as mc
from pymcpu import mcpu_core
from pymcpu.forcefields.korp import KORPForceField

KORP_GROUP, GUARD_GROUP = 7, 8
CLASH = 1e4
MASKED = 20           # residue the tests mask
PARTNER = 60          # a residue far from MASKED in sequence


@pytest.fixture(scope="module")
def built():
    path = os.environ.get("KORP_MAP_PATH")
    if not path:
        pytest.skip("set KORP_MAP_PATH to the korp6Dv1.bin energy map")
    pdb = Path(path).parent / "CASP12DCsel20/T0860D1.pdb"
    if not pdb.is_file():
        pytest.skip("T0860D1.pdb not found next to the map (needs the KORP bundle)")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        traj = md.load(str(pdb))
    ff = KORPForceField(traj)
    coords = np.ascontiguousarray((ff.coords[0] * 10.0).T.astype(np.float32))
    return traj, ff, coords


def _sim(built, coords=None, mask=None, *, system=None, seed=7):
    traj, ff, native = built
    system = system if system is not None else ff.create_system(traj.topology)
    if mask is not None:
        system.set_energy_ignored_residues(*mask)
    integrator = mcpu_core.Integrator(temperature=0.6, step_size_rad=0.1)
    integrator.set_seed(seed)
    integrator.set_move_weights(0.5, 0.5, 0.0)
    sim = mc.Simulation(ff.output_topology, system, integrator)
    sim.full_energy_every_steps = 10**9     # only the delta path updates the total
    sim.context.set_positions(native if coords is None else coords)
    ff.apply_energy_weights(sim.context)
    return sim


def _group(sim, group) -> float:
    sim.context.calculate_total_energy(-1)
    return float(sim.context.energy_breakdown(weighted=False)["by_group"][group])


def _clashes(sim) -> bool:
    sim.context.calculate_total_energy(-1)
    return sim.context.has_steric_clash()


def _frame_atoms(ff, r) -> list[int]:
    return [ff.n_atom_index[r], ff.ca_atom_index[r], ff.c_atom_index[r]]


def _shifted(built, r, shift):
    """Native coordinates with residue r's N, CA and C translated by shift."""
    _, ff, native = built
    coords = native.copy()
    coords[:, _frame_atoms(ff, r)] += np.asarray(shift, dtype=np.float32)[:, None]
    return coords


def _onto(built, r, target):
    """Native coordinates with residue r moved so its CA sits 1 A from target's."""
    _, ff, native = built
    shift = (native[:, ff.ca_atom_index[target]] + np.float32(1.0)
             - native[:, ff.ca_atom_index[r]])
    return _shifted(built, r, shift)


def _offset(sim) -> float:
    running = float(sim.context.get_state().current_energy)
    return abs(running - float(sim.context.calculate_total_energy(-1)))


def _tol(sim) -> float:
    return 1e-9 * max(1.0, abs(float(sim.context.calculate_total_energy(-1))))


@pytest.mark.parametrize("mode", ["ignore_all", "clash_only"])
def test_a_masked_residue_scores_as_if_it_were_far_away(built, mode):
    """Masking a residue removes exactly its pairs: the energy equals that of
    the unmasked chain with the residue moved out of everyone's range."""
    masked = _group(_sim(built, mask=([MASKED], mode)), KORP_GROUP)
    away = _group(_sim(built, _shifted(built, MASKED, [1000.0, 0.0, 0.0])), KORP_GROUP)
    native = _group(_sim(built), KORP_GROUP)
    assert masked == pytest.approx(away, abs=1e-9)
    assert masked != pytest.approx(native, abs=1e-3)


@pytest.mark.parametrize("mode", ["ignore_all", "clash_only"])
def test_moving_a_masked_residue_changes_nothing(built, mode):
    """Full energy and delta agree that a masked residue contributes 0
    wherever it goes (well clear of a clash)."""
    sim = _sim(built, mask=([MASKED], mode))
    _, ff, native = built
    e0 = float(sim.context.calculate_total_energy(-1))
    far = _shifted(built, MASKED, [1000.0, 0.0, 0.0])
    checks = _verify(sim, far, _frame_atoms(ff, MASKED))
    for c in checks:
        assert c.passed, c.message
        assert c.delta_incremental == 0.0
    sim.context.set_positions(far)
    assert float(sim.context.calculate_total_energy(-1)) == pytest.approx(e0, abs=1e-9)


def _verify(sim, new_coords, moved):
    ctx = sim.context
    old_state = ctx.get_state()
    new_state = mcpu_core.State(old_state)
    # A State holds engine-frame coordinates (see Context.frame_offset).
    offset = np.asarray(ctx.frame_offset, dtype=np.float32)[:, None]
    new_state.coords = np.ascontiguousarray(new_coords - offset)
    patch = mcpu_core.ProposalPatch(new_coords.shape[1])
    for atom in moved:
        patch.mark_moved(int(atom))
    patch.is_valid = True
    patch.is_rigid = False
    return mcpu_core.PhysicsVerifier.verify_all_potential_deltas(
        ctx, old_state, new_state, patch, 1e-3)


def _guard_delta(sim, new_coords, moved) -> float:
    checks = _verify(sim, new_coords, moved)
    for c in checks:
        assert c.passed, c.message
    return next(c.delta_incremental for c in checks if c.energy_group == GUARD_GROUP)


@pytest.mark.parametrize("who", ["masked", "partner"])
def test_a_masked_clash(built, who):
    """A CA overlap with a masked residue is no clash under ignore_all and
    still one under clash_only, in the full energy and in the delta, whichever
    of the two residues moves."""
    _, ff, _ = built
    mover, target = (MASKED, PARTNER) if who == "masked" else (PARTNER, MASKED)
    clashing = _onto(built, mover, target)
    moved = _frame_atoms(ff, mover)

    assert _clashes(_sim(built, clashing))
    assert _guard_delta(_sim(built), clashing, moved) > CLASH

    ignore_all = _sim(built, clashing, mask=([MASKED], "ignore_all"))
    assert not _clashes(ignore_all)
    assert _group(ignore_all, GUARD_GROUP) == 0.0
    assert _guard_delta(_sim(built, mask=([MASKED], "ignore_all")), clashing, moved) == 0.0

    clash_only = _sim(built, clashing, mask=([MASKED], "clash_only"))
    assert _clashes(clash_only)
    assert _guard_delta(_sim(built, mask=([MASKED], "clash_only")), clashing, moved) > CLASH


@pytest.mark.parametrize("mode", ["ignore_all", "clash_only"])
def test_running_total_matches_the_full_energy(built, mode):
    """Over a run, the running total built from deltas equals a full
    recompute, through a mask being set and (for clash_only) cleared."""
    residues = list(range(15, 26))
    sim = _sim(built, mask=(residues, mode))
    sim.step(400)
    assert _offset(sim) < _tol(sim)
    sim.system.clear_energy_ignored_residues()
    if mode == "clash_only":    # an ignore_all run may leave overlaps behind
        sim.step(300)
        assert _offset(sim) < _tol(sim)
    sim.system.set_energy_ignored_residues(residues, mode)
    sim.step(300)
    assert _offset(sim) < _tol(sim)


def test_two_contexts_on_one_system(built):
    """A mask set between runs applies to every context on the system."""
    traj, ff, _ = built
    system = ff.create_system(traj.topology)
    a = _sim(built, system=system, seed=1)
    b = _sim(built, system=system, seed=2)
    a.step(200)
    b.step(200)
    for mask in (([MASKED], "clash_only"), ([MASKED, PARTNER], "ignore_all")):
        system.set_energy_ignored_residues(*mask)
        a.step(200)
        b.step(200)
        assert _offset(a) < _tol(a)
        assert _offset(b) < _tol(b)
