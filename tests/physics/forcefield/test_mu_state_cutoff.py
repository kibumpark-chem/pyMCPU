"""Mu tests a move against its hard-core cutoff and a state against one 0.001 A looser.

A rigid pivot carries the pairs inside its segment without measuring them
again, since the rotation keeps their distances. It rounds each carried
coordinate to float, though, so a pair a move left exactly on the cutoff can
end up a few 1e-6 A under it. The full energy (and so has_steric_clash and
Simulation's clash check) therefore allows a pair up to
``mcpu_core.STATE_CLASH_BUFFER_A`` under the cutoff and scores it as any pair
at its distance. A move that re-decides a pair is still tested against the
cutoff itself.

The probe pair is TYR1's CB and the last ring atom of TRP8, placed on the
side of CB facing away from its neighbours, so no other pair comes near a
clash. It is a contact pair, so its energy shows whether it is counted.
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
CUTOFF = 2.7285  # (round(1000 * hard_r) - 1.5) / 1000 for CB-CH2, hard_r = 2.73


def _build(mask=None):
    traj = md.load(str(default_example_pdb()))
    heavy = traj.atom_slice(traj.topology.select("not element H"))
    ff = MCPUForceField(heavy)
    start = np.ascontiguousarray((ff.coords[0] * 10.0).T.astype(np.float32))
    cb = ff.blocks[1].sc_start  # TYR1 CB
    tip = ff.blocks[8].sc_start + ff.blocks[8].sc_count - 1  # TRP8 CH2
    names = [(a.residue_name, a.name) for a in ff.ordered_atom_list]
    assert names[cb] == ("TYR", "CB") and names[tip] == ("TRP", "CH2")
    xyz = start.astype(np.float64)
    near = np.linalg.norm(xyz - xyz[:, [cb]], axis=0) < 6.0
    near[cb] = False
    away = (xyz[:, [cb]] - xyz[:, near]).sum(axis=1)
    away /= np.linalg.norm(away)

    def place(r: float) -> np.ndarray:
        coords = start.copy()
        coords[:, tip] = (xyz[:, cb] + r * away).astype(np.float32)
        return coords

    system = ff.create_system(heavy.topology)
    if mask is not None:
        system.set_energy_ignored_residues([4], mask)  # a residue the pair is not in
    ctx = mcpu_core.Context(system)
    return ctx, place, cb, tip, ff


@pytest.fixture(scope="module")
def setup():
    return _build()


def _judge(ctx, coords):
    ctx.set_positions(coords)
    energy = ctx.calculate_total_energy(-1)
    return energy, ctx.has_steric_clash()


def test_a_pair_inside_the_buffer_is_no_clash_and_still_a_contact(setup) -> None:
    ctx, place, *_ = setup
    energy, clash = _judge(ctx, place(CUTOFF + 0.01))
    assert not clash
    for under in (0.0005, 0.0009):
        assert _judge(ctx, place(CUTOFF - under)) == (energy, False), under


def test_a_pair_beyond_the_buffer_is_a_clash(setup) -> None:
    ctx, place, *_ = setup
    for under in (0.0011, 0.0015, 0.01):
        energy, clash = _judge(ctx, place(CUTOFF - under))
        assert clash and energy > CLASH, under


def _move_tip(ctx, coords, tip):
    """PhysicsVerifier's verdict on moving only ``tip`` to ``coords``."""
    old_state = ctx.get_state()
    new_state = mcpu_core.State(old_state)
    new_state.coords = coords
    patch = mcpu_core.ProposalPatch(coords.shape[1])
    patch.mark_moved(int(tip))
    patch.is_valid = True
    return mcpu_core.PhysicsVerifier.verify_potential_delta(ctx, old_state, new_state, patch, 1, 1e-3)


@pytest.mark.parametrize("under", [0.0005, 0.0015])
def test_a_move_into_the_buffer_is_still_rejected(setup, under: float) -> None:
    """The move cutoff is unchanged. Rejecting a move whose state the full
    energy would allow is expected, and the verifier says so."""
    ctx, place, _, tip, _ = setup
    ctx.set_positions(place(CUTOFF + 0.01))
    ctx.calculate_total_energy(-1)
    check = _move_tip(ctx, place(CUTOFF - under), tip)
    assert check.delta_incremental > CLASH
    assert check.passed, check.message
    if under < mcpu_core.STATE_CLASH_BUFFER_A:
        assert "move cutoff" in check.message


def _check_out_of_buffer(mask, to: float) -> None:
    ctx, place, _, tip, _ = _build(mask)
    ctx.set_positions(place(CUTOFF - 0.0005))
    ctx.calculate_total_energy(-1)
    assert not ctx.has_steric_clash()
    check = _move_tip(ctx, place(to), tip)
    assert check.passed, check.message
    assert abs(check.delta_incremental) < CLASH
    assert (abs(check.delta_direct) > 0.1) is (to == 5.5)


@pytest.mark.parametrize("mask", [None, "ignore_all"], ids=["unmasked", "ignore-all"])
@pytest.mark.parametrize("to", [5.5, CUTOFF + 0.3], ids=["breaks-contact", "keeps-contact"])
def test_a_move_out_of_the_buffer_scores_the_pair_as_the_full_energy_does(mask, to) -> None:
    """The old side of a move judges the state that exists, so it counts a
    pair inside the buffer as the full energy does: here the contact it holds
    is broken (to 5.5 A) or kept (to just outside the cutoff). A mask on an
    unrelated residue must not change that."""
    _check_out_of_buffer(mask, to)


@pytest.mark.parametrize("to", [5.5, CUTOFF + 0.3], ids=["breaks-contact", "keeps-contact"])
def test_without_the_contact_list_a_move_out_of_the_buffer_agrees(to) -> None:
    """The same move on the moved-vs-all delta, which re-measures the old
    side instead of reading the contact list. MCPU_CONTACT_LIST is read once
    per process, so this runs in a fresh one."""
    code = textwrap.dedent(f"""
        from tests.physics.forcefield import test_mu_state_cutoff as t
        t._check_out_of_buffer(None, {to!r})
        print("CHECK ok")
    """)
    repo = Path(__file__).resolve().parents[3]
    env = dict(os.environ, MCPU_CONTACT_LIST="0")
    run = subprocess.run([sys.executable, "-c", code], cwd=repo, env=env,
                         capture_output=True, text=True)
    assert run.returncode == 0, run.stderr[-2000:]
    assert "CHECK ok" in run.stdout


def test_a_rigid_move_carries_a_pair_inside_the_buffer(setup) -> None:
    """Residues 0..8 turned rigidly about N(9) -> CA(9): the pair inside the
    buffer goes along, unchecked, and the move is scored as any other."""
    ctx, place, _, _, ff = setup
    coords = place(CUTOFF - 0.0005)
    ctx.set_positions(coords)
    ctx.calculate_total_energy(-1)
    moved = [k for k, a in enumerate(ff.ordered_atom_list) if a.residue_index <= 8]
    origin = coords[:, ff.blocks[9].bb_start].astype(np.float64)
    axis = coords[:, ff.blocks[9].bb_start + 1] - origin
    axis /= np.linalg.norm(axis)
    kx = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    rot = np.eye(3) + np.sin(0.002) * kx + (1 - np.cos(0.002)) * kx @ kx
    new_coords = coords.copy()
    new_coords[:, moved] = (rot @ (coords[:, moved] - origin[:, None]) + origin[:, None]).astype(np.float32)
    old_state = ctx.get_state()
    new_state = mcpu_core.State(old_state)
    new_state.coords = new_coords
    patch = mcpu_core.ProposalPatch(coords.shape[1])
    for atom in moved:
        patch.mark_moved(atom)
    patch.is_valid = True
    patch.is_rigid = True
    check = mcpu_core.PhysicsVerifier.verify_potential_delta(ctx, old_state, new_state, patch, 1, 1e-3)
    assert check.passed, check.message
    assert abs(check.delta_incremental) < CLASH
