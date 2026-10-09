"""The H-bond angle bins stop at the last bin.

The H-bond term bins each of its angles in 20 deg steps from 0 to 180 deg,
nine bins. A cosine of exactly -1 (two plane normals antiparallel to within
about 0.02 deg) used to give bin 9, because acos(-1) / 20 deg is 9.000001 in
float, and bin 9 read the first bin of the next table entry. The bin is now
clamped to 8, as TripletPotential::get_bin_30 clamps its own, and a NaN cosine
gives bin 0.
"""
from __future__ import annotations

import math

import mdtraj as md
import numpy as np
import pytest

from pymcpu import mcpu_core
from pymcpu.forcefields.mcpu import MCPUForceField
from tests.fixtures.context_builders import TEST_PDB

HBOND_GROUP = 4  # calculate_total_energy group of the H-bond term


def test_a_cosine_of_minus_one_is_the_last_bin() -> None:
    assert mcpu_core._hbond_angle_bin(-1.0) == 8
    # Rounding can push a cosine just past -1; acos would return NaN there.
    below = float(np.nextafter(np.float32(-1.0), np.float32(-2.0)))
    assert below < -1.0
    assert mcpu_core._hbond_angle_bin(below) == 8


def test_a_nan_cosine_is_bin_0() -> None:
    assert mcpu_core._hbond_angle_bin(float("nan")) == 0


def test_the_bin_is_the_angle_in_20_degree_steps() -> None:
    for deg in np.arange(0.5, 180.0, 1.0):
        c = math.cos(math.radians(float(deg)))
        assert mcpu_core._hbond_angle_bin(c) == min(int(deg // 20), 8), deg
    assert mcpu_core._hbond_angle_bin(1.0) == 0
    assert mcpu_core._hbond_angle_bin(1.5) == 0


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def _rotation_to_z(v: np.ndarray) -> np.ndarray:
    v = _unit(v)
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(v, z)
    s = np.linalg.norm(axis)
    if s < 1e-12:
        return np.eye(3)
    axis /= s
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + s * k + (1.0 - v @ z) * (k @ k)


@pytest.mark.slow
def test_the_hbond_energy_does_not_jump_at_180_degrees() -> None:
    """Actin's i -> i+4 H-bond from the NH of residue 83 to the CO of residue
    79, with both N-CA-C planes made exactly parallel (normals at 180 deg).
    Tilting the acceptor plane by about 0.1 deg must not change the energy;
    before the clamp it changed by 0.75 (-172.21 flat, -172.96 tilted)."""
    if not TEST_PDB.is_file():
        pytest.skip(f"needs the actin example structure ({TEST_PDB})")
    traj = md.load(str(TEST_PDB))
    heavy = traj.atom_slice(traj.topology.select("not element H"))
    ff = MCPUForceField(heavy)
    blocks = ff.blocks
    donor, acceptor = 83, 79
    plane_d = [blocks[donor].bb_start, blocks[donor].bb_start + 1, blocks[donor].c_start]
    plane_a = [blocks[acceptor].bb_start, blocks[acceptor].bb_start + 1, blocks[acceptor].c_start]

    p0 = (ff.coords[0] * 10.0).astype(np.float64)
    nd = np.cross(p0[plane_d[0]] - p0[plane_d[1]], p0[plane_d[2]] - p0[plane_d[1]])
    na = np.cross(p0[plane_a[0]] - p0[plane_a[1]], p0[plane_a[2]] - p0[plane_a[1]])
    assert _unit(nd) @ _unit(na) < -0.99  # nearly antiparallel to start with

    q = (_rotation_to_z(nd) @ (p0 - p0[plane_d[0]]).T).T
    flat = q.copy()
    flat[plane_d, 2] = 0.0
    flat[plane_a, 2] = q[plane_a, 2].mean()
    flat = flat.astype(np.float32).astype(np.float64)
    tilted = flat.copy()
    tilted[plane_a, 2] += 2e-3 * (flat[plane_a, 0] - flat[plane_a[1], 0])

    def hbond(coords: np.ndarray) -> float:
        ctx = mcpu_core.Context(ff.create_system(heavy.topology))
        ctx.set_positions(np.ascontiguousarray(coords.T.astype(np.float32)))
        ctx.calculate_total_energy(-1)
        return float(ctx.calculate_total_energy(HBOND_GROUP))

    e_flat, e_tilted, e_start = hbond(flat), hbond(tilted), hbond(q)
    assert e_flat == pytest.approx(e_tilted, abs=1e-4)
    assert e_flat == pytest.approx(e_start, abs=1e-4)
