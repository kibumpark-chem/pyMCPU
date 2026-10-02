"""The CA-CA steric guard that makes KORP safe to sample with.

KORP has no hard-core repulsion -- nothing in a potential fitted to real
structures says a 1 A CA-CA contact is impossible, because no such contact
appears in the training set. Driving MC with KORP alone therefore lets a chain
collapse through itself. This term is the floor that prevents it.

It is a filter, not an energy: zero in every accepted state, the clash sentinel
otherwise. So the tests are about *what it refuses*, and just as importantly
about what it must not refuse -- a guard that vetoed real secondary structure
would quietly strangle sampling rather than failing outright.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

HELPERS = Path(__file__).resolve().parents[1] / "helpers"
if str(HELPERS) not in sys.path:
    sys.path.insert(0, str(HELPERS))

from korp_system import build_backbone_system, residue_atom_indices  # noqa: E402

from pymcpu import mcpu_core  # noqa: E402
from pymcpu.forcefields.builders.korp_builder import KorpPotentialBuilder  # noqa: E402

GUARD_GROUP = 8
CLASH_SENTINEL = 99999.0


def _chain(n_res, rise=3.8):
    """A straight chain of CAs at the real 3.8 A spacing."""
    coords = np.zeros((n_res, 3, 3), dtype=np.float64)
    for r in range(n_res):
        ca = np.array([rise * r, 0.0, 0.0])
        coords[r, 0] = ca + (-1.0, 0.9, 0.0)   # N
        coords[r, 1] = ca                       # CA
        coords[r, 2] = ca + (1.0, 0.9, 0.0)     # C
    return coords


def _ideal_helix(n_res):
    """An alpha helix: 1.5 A rise, 100 degrees per residue, 2.3 A radius."""
    coords = np.zeros((n_res, 3, 3), dtype=np.float64)
    for r in range(n_res):
        ang = np.deg2rad(100.0 * r)
        ca = np.array([2.3 * np.cos(ang), 2.3 * np.sin(ang), 1.5 * r])
        coords[r, 0] = ca + (-0.9, 0.6, -0.5)
        coords[r, 1] = ca
        coords[r, 2] = ca + (0.9, 0.6, 0.5)
    return coords


def _energy(coords, res_seq=None, chain_ids=None, **kwargs):
    n_res = coords.shape[0]
    system, context = build_backbone_system(coords)
    _, ca_atom, _ = residue_atom_indices(n_res)
    guard = KorpPotentialBuilder.build_steric_guard(
        ca_atom=ca_atom,
        res_seq=res_seq if res_seq is not None else list(range(n_res)),
        chain_ids=chain_ids if chain_ids is not None else ["A"] * n_res,
        **kwargs,
    )
    guard.set_energy_group(GUARD_GROUP)
    system.add_potential(guard)
    return context.energy_breakdown(weighted=False)["by_group"][GUARD_GROUP]


def test_an_extended_chain_is_clean():
    assert _energy(_chain(12)) == 0.0


def test_an_ideal_helix_is_clean():
    """The guard must not veto real secondary structure.

    An alpha helix puts CA(i) and CA(i+3) about 5 A apart, which is the
    tightest arrangement a real protein makes -- comfortably above the 4 A
    floor, but close enough that a careless threshold would reject it.
    """
    assert _energy(_ideal_helix(20)) == 0.0


def test_overlapping_residues_are_rejected():
    coords = _chain(12)
    coords[9, :] = coords[1, :] + 0.5   # drop residue 9 on top of residue 1
    assert _energy(coords) == pytest.approx(CLASH_SENTINEL)


@pytest.mark.parametrize("floor", [3.0, 3.2, 4.0])
def test_the_threshold_is_where_it_says_it_is(floor):
    """Probed either side of the configured floor, whatever it is set to.

    Parameterised rather than written against the default on purpose: this is
    testing the comparison, not the constant, and an earlier version of this
    file hard-coded 4.0 and then silently disagreed with the code when the
    default was recalibrated against real structures.
    """
    coords = _chain(12, rise=6.0)      # wide enough that only the probe is close
    base = coords[1, 1].copy()
    for offset, expect_clash in ((floor - 0.1, True), (floor + 0.1, False)):
        coords[9, 1] = base + np.array([0.0, offset, 0.0])
        coords[9, 0] = coords[9, 1] + (-1.0, 0.9, 0.0)
        coords[9, 2] = coords[9, 1] + (1.0, 0.9, 0.0)
        energy = _energy(coords, min_distance=floor)
        assert (energy > 0.0) is expect_clash, f"floor {floor}, offset {offset}"


def test_the_default_floor_clears_real_backbone_geometry():
    """The default is measured, and this is the measurement in test form.

    Across 1CEO, 1DOS, T0860D1, actin and chignolin the closest CA-CA contact
    at >= 3 apart in sequence is 3.53 A -- real packing, not numbering
    artefacts. A floor at or above that rejects native structures, which is
    what the first version of this term did at 4.0 A.
    """
    guard = KorpPotentialBuilder.build_steric_guard(
        ca_atom=[1, 5, 9], res_seq=[1, 2, 3], chain_ids=["A"] * 3)
    assert guard.min_distance == pytest.approx(3.2)
    assert guard.min_distance < 3.53
    assert guard.min_separation == 3


def test_near_neighbours_in_sequence_are_exempt():
    """Bonded CAs sit at 3.8 A and must be excused from any floor above that.

    Uses an explicit 4.0 A floor rather than the default: the point is the
    exemption, and it is only observable when the bonded distance is actually
    below the floor being applied.
    """
    coords = _chain(8, rise=3.8)
    assert _energy(coords, min_distance=4.0) == 0.0
    # With the exemption removed, every bonded pair is now a clash.
    assert _energy(coords, min_separation=1,
                   min_distance=4.0) == pytest.approx(CLASH_SENTINEL)


def test_different_chains_are_never_exempt():
    """Two chains can overlap whatever their residues happen to be numbered."""
    coords = np.concatenate([_chain(4), _chain(4)], axis=0)
    coords[4:] += 0.0   # the two copies sit exactly on top of each other
    res_seq = list(range(4)) + list(range(4))   # deliberately colliding numbers
    same_chain = ["A"] * 8
    two_chains = ["A"] * 4 + ["B"] * 4

    # Read as one chain, the duplicate numbering makes every cross pair look
    # adjacent in sequence and so exempt.
    assert _energy(coords, res_seq=res_seq, chain_ids=same_chain) == 0.0
    # Read as two chains, the overlap is seen.
    assert _energy(coords, res_seq=res_seq,
                   chain_ids=two_chains) == pytest.approx(CLASH_SENTINEL)


def test_the_guard_hard_rejects_a_clashing_move():
    """The sentinel has to reach the integrator as a rejection, not a number."""
    coords = _chain(14)
    n_res = coords.shape[0]
    system, context = build_backbone_system(coords)
    _, ca_atom, _ = residue_atom_indices(n_res)
    guard = KorpPotentialBuilder.build_steric_guard(
        ca_atom=ca_atom, res_seq=list(range(n_res)), chain_ids=["A"] * n_res)
    guard.set_energy_group(GUARD_GROUP)
    system.add_potential(guard)

    integrator = mcpu_core.Integrator(temperature=0.6, step_size_rad=0.9)
    integrator.set_seed(5)
    integrator.set_move_weights(1.0, 0.0, 0.0)   # large pivots, backbone only
    integrator.run(context, 400)

    stats = integrator.move_stats()
    assert stats["steric_rejected"] > 0, (
        "no move was steric-rejected at this amplitude; the guard is not "
        "reaching the integrator's rejection path"
    )
    # Whatever survived must still be clash-free.
    assert context.energy_breakdown(weighted=False)["by_group"][GUARD_GROUP] == 0.0


def _rotation(axis, theta):
    k = axis / np.linalg.norm(axis)
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * kx + (1 - np.cos(theta)) * kx @ kx


def test_a_rigid_move_cannot_round_a_pair_under_the_floor():
    """A rigid pivot keeps CA-CA distances only in real arithmetic. It is
    applied in float32, so a pair it carries that sits exactly on the floor
    can be rounded under it. The guard used to skip such moved-moved pairs for
    a rigid move, and so accepted a state its own full energy calls a clash."""
    n_res = 12
    floor = float(np.float32(3.2))
    coords = _chain(n_res, rise=6.0)
    coords[10] += coords[6, 1] + (0.0, floor, 0.0) - coords[10, 1]  # CA(10) on the floor from CA(6)
    system, context = build_backbone_system(coords)
    _, ca_atom, _ = residue_atom_indices(n_res)
    guard = KorpPotentialBuilder.build_steric_guard(
        ca_atom=ca_atom, res_seq=list(range(n_res)), chain_ids=["A"] * n_res, min_distance=floor,
    )
    guard.set_energy_group(GUARD_GROUP)
    system.add_potential(guard)
    assert context.energy_breakdown(weighted=False)["by_group"][GUARD_GROUP] == 0.0

    # Residues 6..11 (their N, CA, C and O) turn rigidly about N(6) -> CA(6).
    moved = [3 * r + k for r in range(6, n_res) for k in range(3)] + [3 * n_res + r for r in range(6, n_res)]
    pos = np.asarray(context.get_state().coords, dtype=np.float32)
    origin = pos[:, 3 * 6]
    axis = (pos[:, 3 * 6 + 1] - origin).astype(np.float64)
    old_state = context.get_state()
    crossings = 0
    for theta in np.linspace(1e-3, 1.0, 200):
        rot = _rotation(axis, theta).astype(np.float32)
        new = pos.copy()
        new[:, moved] = (rot @ (pos[:, moved] - origin[:, None])) + origin[:, None]
        new_state = mcpu_core.State(old_state)
        new_state.coords = new
        patch = mcpu_core.ProposalPatch(pos.shape[1])
        for atom in moved:
            patch.mark_moved(int(atom))
        patch.is_valid = True
        patch.is_rigid = True
        check = mcpu_core.PhysicsVerifier.verify_potential_delta(
            context, old_state, new_state, patch, GUARD_GROUP, 1e-3)
        crossings += check.delta_direct >= CLASH_SENTINEL / 2
        assert check.passed, (theta, check.message)
    assert crossings > 0  # the construction really rounds the pair under the floor
