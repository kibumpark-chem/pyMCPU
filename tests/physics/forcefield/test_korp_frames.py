"""KORP's residue frame and its six pair coordinates.

Two things are being pinned here, and both are the kind that fail silently.

The first is the frame convention. The KORP paper's Eq. (2) and the code that
actually built the released map disagree about one cross product, and the two
frames differ by a 180-degree rotation about ``vz``. Both are right-handed, so
no invariant catches it -- but every psi angle shifts by pi and lands in a
different bin, which shows up only as an energy that is wrong by an amount
nobody can eyeball. ``test_paper_convention_would_shift_psi_by_pi`` makes that
difference explicit rather than leaving it as a comment.

The second is SE(3) invariance. pyMCPU's incremental KORP energy skips every
residue pair that moved rigidly *together*, on the grounds that all six
coordinates are then unchanged. That is the single largest optimisation in the
delta path and it is only sound if the invariance is exact, so it is asserted
here directly rather than inferred from the delta agreeing with itself.
"""

from __future__ import annotations

import numpy as np
import pytest

from pymcpu.forcefields.korp_map import pair_coordinates, residue_frame

TWO_PI = 2.0 * np.pi


def _random_rotation(rng):
    """A proper rotation (det +1) via QR, with the sign fixed up."""
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q @ np.diag(np.sign(np.diag(r)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1.0
    return q


def _random_residue(rng):
    """N, CA, C with roughly real backbone geometry."""
    ca = rng.normal(scale=5.0, size=3)
    n = ca + _random_rotation(rng) @ np.array([1.458, 0.0, 0.0])
    c = ca + _random_rotation(rng) @ np.array([1.525, 0.0, 0.0])
    return n, ca, c


def _frames(rng, separation=8.0):
    n1, ca1, c1 = _random_residue(rng)
    n2, ca2, c2 = _random_residue(rng)
    shift = ca1 - ca2 + _random_rotation(rng) @ np.array([separation, 0.0, 0.0])
    return (residue_frame(n1, ca1, c1),
            residue_frame(n2, ca2 + shift, c2 + shift))


def test_frame_is_orthonormal_and_right_handed():
    rng = np.random.default_rng(0)
    for _ in range(200):
        n, ca, c = _random_residue(rng)
        origin, R = residue_frame(n, ca, c)
        assert np.allclose(origin, ca)          # the origin IS CA
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-10)
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-10)
        vx, vy, vz = R[:, 0], R[:, 1], R[:, 2]
        assert np.allclose(np.cross(vx, vy), vz, atol=1e-10)


def test_frame_axes_match_the_released_code_not_the_paper():
    """``vy ~ vz x (C - CA)``. The paper prints ``vz x (N - CA)``."""
    rng = np.random.default_rng(1)
    for _ in range(50):
        n, ca, c = _random_residue(rng)
        _, R = residue_frame(n, ca, c)
        r12, r13 = n - ca, c - ca
        vz = (r12 + r13) / np.linalg.norm(r12 + r13)
        vy = np.cross(vz, r13)
        vy /= np.linalg.norm(vy)
        assert np.allclose(R[:, 2], vz, atol=1e-10)
        assert np.allclose(R[:, 1], vy, atol=1e-10)
        assert np.allclose(R[:, 0], np.cross(vy, vz), atol=1e-10)


def test_paper_convention_would_shift_psi_by_pi():
    """The discrepancy is a real bin change, not a sign nobody notices."""
    rng = np.random.default_rng(2)
    shifted = 0
    for _ in range(100):
        fa, fb = _frames(rng)
        _, _, psi_a, _, psi_b, _ = pair_coordinates(fa, fb)

        # Rebuild both frames the way the paper's Eq. (2) reads.
        def paper(frame):
            origin, R = frame
            vz = R[:, 2]
            # vz x (N - CA) == -(vz x (C - CA)), so the paper's vy is the
            # negation of the code's, and vx = vy x vz flips with it.
            vy = -R[:, 1]
            vx = np.cross(vy, vz)
            return origin, np.stack([vx, vy, vz], axis=1)

        _, _, p_psi_a, _, p_psi_b, _ = pair_coordinates(paper(fa), paper(fb))
        for ours, theirs in ((psi_a, p_psi_a), (psi_b, p_psi_b)):
            assert abs(((ours - theirs) % TWO_PI) - np.pi) < 1e-8
            shifted += 1
    assert shifted == 200


def test_all_six_coordinates_are_invariant_under_rigid_motion():
    """The precondition for skipping moved-moved pairs in the delta path."""
    rng = np.random.default_rng(3)
    for _ in range(200):
        n1, ca1, c1 = _random_residue(rng)
        n2, ca2, c2 = _random_residue(rng)
        ca2 = ca2 + np.array([9.0, 0.0, 0.0])
        n2, c2 = n2 + np.array([9.0, 0.0, 0.0]), c2 + np.array([9.0, 0.0, 0.0])

        before = pair_coordinates(residue_frame(n1, ca1, c1),
                                  residue_frame(n2, ca2, c2))

        R0, t = _random_rotation(rng), rng.normal(scale=20.0, size=3)
        mv = lambda v: R0 @ v + t  # noqa: E731
        after = pair_coordinates(
            residue_frame(mv(n1), mv(ca1), mv(c1)),
            residue_frame(mv(n2), mv(ca2), mv(c2)),
        )
        assert np.allclose(before, after, atol=1e-9)


def test_reflection_flips_psi_and_chi_but_not_theta():
    """KORP is chirality sensitive; a reflection-blind implementation is wrong."""
    rng = np.random.default_rng(4)
    M = np.diag([1.0, 1.0, -1.0])
    for _ in range(100):
        n1, ca1, c1 = _random_residue(rng)
        n2, ca2, c2 = _random_residue(rng)
        off = np.array([7.5, 0.0, 0.0])
        n2, ca2, c2 = n2 + off, ca2 + off, c2 + off

        d, ta, pa, tb, pb, chi = pair_coordinates(
            residue_frame(n1, ca1, c1), residue_frame(n2, ca2, c2))
        d_m, ta_m, pa_m, tb_m, pb_m, chi_m = pair_coordinates(
            residue_frame(M @ n1, M @ ca1, M @ c1),
            residue_frame(M @ n2, M @ ca2, M @ c2))

        assert d_m == pytest.approx(d, abs=1e-9)
        assert ta_m == pytest.approx(ta, abs=1e-9)
        assert tb_m == pytest.approx(tb, abs=1e-9)
        assert (TWO_PI - pa_m) == pytest.approx(pa, abs=1e-7)
        assert (TWO_PI - pb_m) == pytest.approx(pb, abs=1e-7)
        assert (TWO_PI - chi_m) == pytest.approx(chi, abs=1e-7)


def test_coordinates_stay_in_range():
    rng = np.random.default_rng(5)
    for _ in range(300):
        d, ta, pa, tb, pb, chi = pair_coordinates(*_frames(rng))
        assert d > 0.0
        assert 0.0 <= ta <= np.pi and 0.0 <= tb <= np.pi
        for ang in (pa, pb, chi):
            assert -1e-9 <= ang <= TWO_PI + 1e-9


def test_pair_coordinates_are_not_symmetric_in_the_partners():
    """A/B order is load-bearing: the map is not symmetric under a swap."""
    rng = np.random.default_rng(6)
    fa, fb = _frames(rng)
    d1, ta1, pa1, tb1, pb1, chi1 = pair_coordinates(fa, fb)
    d2, ta2, pa2, tb2, pb2, chi2 = pair_coordinates(fb, fa)
    assert d1 == pytest.approx(d2)
    # Swapping exchanges the two residues' roles.
    assert ta2 == pytest.approx(tb1, abs=1e-9)
    assert pa2 == pytest.approx(pb1, abs=1e-9)
