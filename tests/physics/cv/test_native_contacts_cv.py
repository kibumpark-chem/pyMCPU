"""NativeContactsCV pure-Python geometry: counting N and fraction Q.

Pure first-principles unit tests of the CV's own counting/conversion math on
a hand-constructed toy structure -- no legacy reference values anywhere.
"""

from __future__ import annotations

import numpy as np
import pytest

from pymcpu.sampling.collective_variables import NativeContactsCV, attach_native_contacts_bias_potential


def _make_cv() -> NativeContactsCV:
    # 4 residues on a line; cutoff 11 + min_seq_sep 2 -> pairs (0,2) d=10,
    # (1,3) d=10; (0,3) d=15 is outside cutoff -> n_contacts == 2.
    ca_idx = np.array([0, 1, 2, 3], dtype=np.int64)
    ref = np.array(
        [
            [0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [15.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    return NativeContactsCV(
        ca_internal_idx=ca_idx,
        ref_ca_xyz=ref,
        contact_cutoff=11.0,
        min_seq_sep=2,
        mode="hard",
        q_cutoff=11.0,
    )


def test_compute_n_and_q_at_reference_structure() -> None:
    cv = _make_cv()
    assert cv.n_contacts == 2  # see derivation in _make_cv's comment
    coords = np.zeros((3, 4), dtype=np.float64)
    coords[0, :] = [0.0, 5.0, 10.0, 15.0]
    n = cv.compute_N(coords)
    q = cv.compute_Q(coords)
    assert n == pytest.approx(2.0)
    assert q == pytest.approx(1.0)  # at the reference structure itself, Q == 1
    assert cv.compute(coords) == pytest.approx(q)


def test_compute_n_breaks_contacts_when_stretched() -> None:
    cv = _make_cv()
    coords = np.zeros((3, 4), dtype=np.float64)
    coords[0, :] = [0.0, 50.0, 100.0, 150.0]  # all pairwise distances >> cutoff
    assert cv.compute_N(coords) == pytest.approx(0.0)
    assert cv.compute_Q(coords) == pytest.approx(0.0)


def test_fraction_count_conversion() -> None:
    cv = _make_cv()
    # n_contacts == 2 (from _make_cv): fraction*2 and count/2 round-trip to 1.
    assert cv.fraction_to_count(0.5) == pytest.approx(1.0)
    assert cv.count_to_fraction(2.0) == pytest.approx(1.0)


def test_soft_mode_rejected_for_cpp_attach() -> None:
    ca_idx = np.array([0, 1, 2], dtype=np.int64)
    ref = np.array([[0.0, 0, 0], [5.0, 0, 0], [10.0, 0, 0]], dtype=np.float64)
    cv = NativeContactsCV(ca_idx, ref, contact_cutoff=8.0, min_seq_sep=1, mode="soft")
    # The C++ bias force only implements hard-cutoff N; soft mode is
    # analysis-only and must be rejected at attach time, not silently misused.
    with pytest.raises(ValueError, match="hard-cutoff"):
        attach_native_contacts_bias_potential(object(), cv)


def _toy_geometry() -> tuple[np.ndarray, np.ndarray]:
    # Same 4-residues-on-a-line geometry as _make_cv: pairwise distances are
    # 5 * |i - j| Angstrom.
    ca_idx = np.array([0, 1, 2, 3], dtype=np.int64)
    ref = np.array(
        [
            [0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [15.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    return ca_idx, ref


def test_explicit_pairs_override_cutoff_and_seq_sep_derivation() -> None:
    # With contact_cutoff=11.0 and min_seq_sep=2 (same as _make_cv), the
    # cutoff/seq-sep rule would derive (0,2) and (1,3) -- see _make_cv's
    # comment. Pass explicit pairs the derivation rule would NOT select on
    # its own: (0,1) fails min_seq_sep (sequence separation 1 < 2), and (0,3)
    # fails the cutoff (distance 15 >= 11). Assert the explicit pairs are
    # used verbatim -- not unioned, intersected, or otherwise combined with
    # what cutoff/seq-sep derivation would have produced.
    ca_idx, ref = _toy_geometry()
    explicit_pairs = np.array([[0, 1], [0, 3]], dtype=np.int64)
    cv = NativeContactsCV(
        ca_internal_idx=ca_idx,
        ref_ca_xyz=ref,
        contact_cutoff=11.0,
        min_seq_sep=2,
        mode="hard",
        q_cutoff=11.0,
        native_contact_pairs=explicit_pairs,
    )
    assert cv.n_contacts == 2
    np.testing.assert_array_equal(cv.pairs_i, explicit_pairs[:, 0])
    np.testing.assert_array_equal(cv.pairs_j, explicit_pairs[:, 1])
    np.testing.assert_array_equal(cv.native_contact_pairs, explicit_pairs)
    # r0 comes from the reference distance of the explicit pairs, not the
    # cutoff/seq-sep-derived set.
    assert cv.r0[0] == pytest.approx(5.0)  # dist(0, 1)
    assert cv.r0[1] == pytest.approx(15.0)  # dist(0, 3)


def test_explicit_pairs_out_of_range_index_raises() -> None:
    ca_idx, ref = _toy_geometry()  # n_res == 4, valid indices are 0..3
    with pytest.raises(ValueError, match="outside"):
        NativeContactsCV(
            ca_internal_idx=ca_idx,
            ref_ca_xyz=ref,
            native_contact_pairs=[[0, 4]],
        )


def test_explicit_pairs_self_pair_raises() -> None:
    ca_idx, ref = _toy_geometry()
    with pytest.raises(ValueError, match="self-pair"):
        NativeContactsCV(
            ca_internal_idx=ca_idx,
            ref_ca_xyz=ref,
            native_contact_pairs=[[1, 1]],
        )


def test_explicit_pairs_duplicate_reversed_pair_raises() -> None:
    ca_idx, ref = _toy_geometry()
    # (0, 2) and its reverse (2, 0) must be treated as the same pair.
    with pytest.raises(ValueError, match="duplicate"):
        NativeContactsCV(
            ca_internal_idx=ca_idx,
            ref_ca_xyz=ref,
            native_contact_pairs=[[0, 2], [2, 0]],
        )


def test_explicit_pairs_empty_list_raises() -> None:
    ca_idx, ref = _toy_geometry()
    # An explicitly-empty list must raise, distinguishing "explicitly no
    # contacts" from "omit the argument to fall back to auto-derivation."
    with pytest.raises(ValueError, match="native_contact_pairs"):
        NativeContactsCV(
            ca_internal_idx=ca_idx,
            ref_ca_xyz=ref,
            native_contact_pairs=[],
        )


def test_soft_mode_with_explicit_pairs_computes_sane_r0_and_n_eff() -> None:
    ca_idx, ref = _toy_geometry()
    explicit_pairs = np.array([[0, 2], [1, 3]], dtype=np.int64)
    cv = NativeContactsCV(
        ca_internal_idx=ca_idx,
        ref_ca_xyz=ref,
        mode="soft",
        native_contact_pairs=explicit_pairs,
    )
    assert cv.n_contacts == 2
    # r0 taken directly from the reference distances of the explicit pairs:
    # dist(0,2) == dist(1,3) == 10 A on this toy geometry.
    np.testing.assert_allclose(cv.r0, [10.0, 10.0])

    coords = np.zeros((3, 4), dtype=np.float64)
    coords[0, :] = [0.0, 5.0, 10.0, 15.0]  # == reference structure
    n_eff = cv.compute_N(coords)
    q = cv.compute_Q(coords)
    # At the reference structure itself (r == r0 for both explicit pairs),
    # the soft Best-Hummer-Eaton formula should read as essentially fully
    # formed: N_eff close to n_contacts (not 0, not >> n_contacts), and Q
    # close to 1.
    assert 0.0 < n_eff <= cv.n_contacts + 1e-6
    assert n_eff == pytest.approx(cv.n_contacts, abs=1e-3)
    assert q == pytest.approx(1.0, abs=1e-3)
