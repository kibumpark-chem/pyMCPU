"""A rigid move cannot carry a pair under the hard-core cutoff a state is judged by.

A rigid pivot does not re-measure the pairs it carries: the rotation keeps
their distances. It rounds each carried coordinate to float, though, and those
rounding errors add up over carries as a random walk. A pair a move left just
outside its move cutoff can therefore be walked, carry by carry, past the
0.001 A the full energy allows under that cutoff (STATE_CLASH_BUFFER_A). The
1igd replica-exchange run at T = 1.6 did exactly that, after ~1e5 carries ~120
A from the origin, and its per-cycle full recompute aborted on the overlap.

So the contact list also holds every clash pair less than 0.05 A outside its
move cutoff, contact pair or not, and a rigid move that carries a listed pair
under the state cutoff is rejected; without a list, every carried pair is
tested. The rounding is modelled here by a carry that also nudges one atom of
the pair (the engine trusts ``is_rigid``), and an end-to-end run carries a
pair sitting just above the state cutoff with real pivots.

Three probe pairs, each placed on the side of TYR1 CB facing away from its
neighbours, so no other pair comes near a clash: CB with TRP8 CH2 (a contact
pair), with GLU4's last side-chain atom and with its own OH (clash-only pairs:
residues under four apart make no contact). A pivot never changes the
distance of the last pair, so it is the one the end-to-end run walks.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import mdtraj as md
import numpy as np
import pytest

from pymcpu import mcpu_core
from pymcpu.forcefields.mcpu import MCPUForceField
from pymcpu.runners import default_example_pdb

CLASH = 1e4  # the clash sentinel, weighted, is about 4e4
BUFFER = float(mcpu_core.STATE_CLASH_BUFFER_A)
PAIRS = {"contact": 8, "clash-only": 4, "same-residue": 1}  # partner residue of TYR1 CB


def _build(partner_residue: int):
    traj = md.load(str(default_example_pdb()))
    heavy = traj.atom_slice(traj.topology.select("not element H"))
    ff = MCPUForceField(heavy)
    start = np.ascontiguousarray((ff.coords[0] * 10.0).T.astype(np.float32))
    cb = ff.blocks[1].sc_start  # TYR1 CB
    block = ff.blocks[partner_residue]
    tip = block.sc_start + block.sc_count - 1
    assert ff.ordered_atom_list[cb].name == "CB"
    xyz = start.astype(np.float64)
    near = np.linalg.norm(xyz - xyz[:, [cb]], axis=0) < 6.0
    near[cb] = False
    away = (xyz[:, [cb]] - xyz[:, near]).sum(axis=1)
    away /= np.linalg.norm(away)

    def place(r: float) -> np.ndarray:
        coords = start.copy()
        coords[:, tip] = (xyz[:, cb] + r * away).astype(np.float32)
        return coords

    ctx = mcpu_core.Context(ff.create_system(heavy.topology))
    return ctx, place, cb, tip, ff


def _clashes(ctx, coords) -> bool:
    ctx.set_positions(coords)
    ctx.calculate_total_energy(-1)
    return bool(ctx.has_steric_clash())


def _state_cutoff(ctx, place) -> float:
    """The distance under which the full energy calls the pair a clash."""
    lo, hi = 1.0, 4.0
    assert _clashes(ctx, place(lo)) and not _clashes(ctx, place(hi))
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if _clashes(ctx, place(mid)):
            lo = mid
        else:
            hi = mid
    return hi


@pytest.fixture(scope="module", params=list(PAIRS), ids=list(PAIRS))
def probe(request):
    ctx, place, cb, tip, ff = _build(PAIRS[request.param])
    cut = _state_cutoff(ctx, place)
    if request.param != "same-residue":  # its OH, moved, meets other atoms
        # The contact pair scores a contact; the clash-only pair never does.
        e_near = (ctx.set_positions(place(cut + 0.3)), ctx.calculate_total_energy(-1))[1]
        e_far = (ctx.set_positions(place(6.0)), ctx.calculate_total_energy(-1))[1]
        assert (abs(e_near - e_far) > 0.1) is (request.param == "contact")
    return request.param, ctx, place, cb, tip, ff, cut


def _carry(ctx, coords, ff, cb, tip, to: float):
    """PhysicsVerifier's verdict on turning residues 0..8 rigidly about
    N(9) -> CA(9) by a small angle, with the pair's distance set to ``to``
    afterwards: what many carries' rounding could have made of it."""
    moved = [k for k, a in enumerate(ff.ordered_atom_list) if a.residue_index <= 8]
    origin = coords[:, ff.blocks[9].bb_start].astype(np.float64)
    axis = coords[:, ff.blocks[9].bb_start + 1] - origin
    axis /= np.linalg.norm(axis)
    kx = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    rot = np.eye(3) + np.sin(0.002) * kx + (1 - np.cos(0.002)) * kx @ kx
    x = (rot @ (coords[:, moved] - origin[:, None]) + origin[:, None])
    new_coords = coords.copy()
    new_coords[:, moved] = x.astype(np.float32)
    u = new_coords[:, tip].astype(np.float64) - new_coords[:, cb]
    new_coords[:, tip] = (new_coords[:, cb] + to * u / np.linalg.norm(u)).astype(np.float32)
    old_state = ctx.get_state()
    new_state = mcpu_core.State(old_state)
    new_state.coords = new_coords
    patch = mcpu_core.ProposalPatch(coords.shape[1])
    for atom in moved:
        patch.mark_moved(atom)
    patch.is_valid = True
    patch.is_rigid = True
    return mcpu_core.PhysicsVerifier.verify_potential_delta(ctx, old_state, new_state, patch, 1, 1e-3)


def _check_carries(probe, refilled: bool) -> None:
    _, ctx, place, cb, tip, ff, cut = probe
    coords = place(cut + 0.5 * BUFFER)  # under the move cutoff, a state that is allowed
    ctx.set_positions(coords)
    ctx.calculate_total_energy(-1)
    assert not ctx.has_steric_clash()
    if refilled:
        # The state's own list, built by a forced pivot that is not
        # committed and then rewritten by a recompute.
        assert mcpu_core.Integrator(temperature=0.6).debug_force_pivot(ctx, 8, False)
        ctx.calculate_total_energy(-1)
    # Still inside the allowance: carried and scored as any other pair.
    kept = _carry(ctx, coords, ff, cb, tip, cut + 0.2 * BUFFER)
    assert kept.passed, kept.message
    assert abs(kept.delta_incremental) < CLASH
    # Past it: the move is rejected, as the full energy of its state says.
    under = _carry(ctx, coords, ff, cb, tip, cut - 0.5 * BUFFER)
    assert under.passed, under.message
    assert under.delta_incremental > CLASH


@pytest.mark.parametrize("refilled", [False, True], ids=["rebuilt", "refilled"])
def test_a_carry_under_the_state_cutoff_is_rejected(probe, refilled: bool) -> None:
    """``rebuilt``: the move builds the contact list from the coordinates.
    ``refilled``: it uses the list a full recompute rewrote."""
    _check_carries(probe, refilled)


@pytest.mark.parametrize("pair", list(PAIRS))
def test_without_the_contact_list_a_carry_under_the_state_cutoff_is_rejected(pair) -> None:
    """The same carries on the moved-vs-all delta, which tests every carried
    pair. MCPU_CONTACT_LIST is read once per process, so this runs in a fresh
    one."""
    code = textwrap.dedent(f"""
        from tests.physics.forcefield import test_mu_carried_clash as t
        ctx, place, cb, tip, ff = t._build(t.PAIRS[{pair!r}])
        cut = t._state_cutoff(ctx, place)
        t._check_carries(({pair!r}, ctx, place, cb, tip, ff, cut), False)
        print("CHECK ok")
    """)
    repo = Path(__file__).resolve().parents[3]
    env = dict(os.environ, MCPU_CONTACT_LIST="0")
    run = subprocess.run([sys.executable, "-c", code], cwd=repo, env=env,
                         capture_output=True, text=True)
    assert run.returncode == 0, run.stderr[-2000:]
    assert "CHECK ok" in run.stdout


def test_real_pivots_never_carry_a_pair_under_the_state_cutoff() -> None:
    """TYR1 CB and OH start 1e-6 A above the state cutoff, where the rounding
    of a few carries can take them under. Hot pivots then run one step at a
    time, the full recompute judging every state they carry the pair to. (On
    a build that does not test carried pairs this seed clashes at step 194.)"""
    ctx, place, cb, tip, ff = _build(PAIRS["same-residue"])
    cut = _state_cutoff(ctx, place)
    assert not _clashes(ctx, place(cut + 1e-6))
    integrator = mcpu_core.Integrator(temperature=1.6)
    integrator.set_move_weights(1.0, 0.0, 0.0)
    integrator.set_seed(3)
    def pair_distance() -> float:
        x = np.asarray(ctx.coords, dtype=np.float64)
        return float(np.linalg.norm(x[:, tip] - x[:, cb]))

    carries = 0
    d = pair_distance()
    for step in range(1, 1501):
        integrator.run(ctx, 1, step)
        d_new = pair_distance()
        if d_new != d:  # accepted, and only a carry moves this pair
            assert integrator.last_move_is_rigid()
            carries += 1
            d = d_new
            ctx.calculate_total_energy(-1)
            assert not ctx.has_steric_clash(), f"step {step}, after {carries} carries"
    assert carries >= 200  # the pair really was carried, many times
