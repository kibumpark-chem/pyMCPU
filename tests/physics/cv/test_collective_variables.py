"""``pymcpu.sampling.cv_factory``'s collective-variable components -- pure geometry/
physics self-consistency checks against a real chignolin forcefield.

No legacy MCPU comparison lives here: every assertion checks an internal
mathematical identity (native Q == 1, native RMSD == 0, N == Q * n_contacts,
RMSD is rigid-body invariant, two identical two-state references agree),
never a dbfold_actin reference value, so this belongs in ``tests/physics/``
rather than ``tests/legacy_parity/``. The config-dispatch/error-handling half
of the original combined file (``build_cv`` spec parsing, unknown-type
and mismatched-reference errors) is pure software plumbing with no physics
content and lives separately in
``tests/integration/sampling/test_cv_factory.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from pymcpu.sampling.cv_factory import (
    CaRmsd,
    CompositeCV,
    NativeContactsN,
    NativeContactsQ,
    TwoStateDelta,
    TwoStateRmsd,
    build_cv,
)
from tests.physics.helpers.cv_fixtures import build_chignolin_forcefield, native_coords_angstrom

# Freshly measured on this engine build (see accompanying report): a
# float32 native-vs-native RMSD residual of ~1e-15 A and a rotation+
# translation round-trip residual of ~2e-7 A. 1e-4 leaves ~3 orders of
# magnitude of headroom for BLAS/platform variance in the Kabsch SVD while
# still being far tighter than a blanket 1e-3.
_RMSD_ATOL = 1e-4

# Shared native-contacts spec fragment -- every native_contacts_q/n spec in
# this file agrees on cutoff/seq-sep, only "reference_pdb" varies per call.
_NATIVE_CONTACTS_PARAMS = {"contact_cutoff": 10.0, "min_seq_sep": 3}


@pytest.fixture(scope="module")
def forcefield(chignolin_pdb_path: str):
    # Module-scoped: building the C++ forcefield is the expensive part of
    # each test here, and it's read-only after construction.
    return build_chignolin_forcefield(chignolin_pdb_path)


@pytest.fixture
def native_coords(forcefield) -> np.ndarray:
    return native_coords_angstrom(forcefield)


def test_native_contacts_q_at_native_is_one(chignolin_pdb_path: str, forcefield, native_coords: np.ndarray) -> None:
    pc = build_cv(
        [{"type": "native_contacts_q", "reference_pdb": str(chignolin_pdb_path), **_NATIVE_CONTACTS_PARAMS}],
        forcefield,
    )
    assert isinstance(pc, NativeContactsQ)
    assert pc.ndim == 1
    assert pc.labels == ("Q",)
    # The native contact set is defined FROM this same reference structure,
    # so Q evaluated at the reference itself is exactly 1 by construction.
    np.testing.assert_allclose(pc(native_coords), [1.0])


def test_native_contacts_n_matches_q_times_n_contacts(
    chignolin_pdb_path: str, forcefield, native_coords: np.ndarray
) -> None:
    spec = {"reference_pdb": str(chignolin_pdb_path), **_NATIVE_CONTACTS_PARAMS}
    pc_q = build_cv([{"type": "native_contacts_q", **spec}], forcefield)
    pc_n = build_cv([{"type": "native_contacts_n", **spec}], forcefield)
    assert isinstance(pc_n, NativeContactsN)
    n_contacts = pc_n.cv.n_contacts
    # Q := N / n_contacts by definition (NativeContactsCV.compute_Q) -- this
    # checks the two build_cv dispatch paths stay consistent with that
    # identity rather than drifting apart under maintenance.
    np.testing.assert_allclose(pc_n(native_coords), pc_q(native_coords) * n_contacts)


def test_native_contact_pairs_spec_key_sets_explicit_pairs(
    chignolin_pdb_path: str, forcefield, native_coords: np.ndarray
) -> None:
    """A ``native_contact_pairs`` spec key bypasses cutoff/min_seq_sep
    derivation entirely: n_contacts must equal exactly the given pair count
    (not whatever cutoff+seq-sep would have derived), and since every pair's
    live distance at the native structure equals its own r0, Q/N there must
    be exactly 1.0 / len(pairs) given a cutoff generous enough to count
    every pair as formed."""
    explicit_pairs = [[0, 5], [1, 6], [2, 8]]
    spec = {
        "reference_pdb": str(chignolin_pdb_path),
        "contact_cutoff": 1000.0,
        "native_contact_pairs": explicit_pairs,
    }
    pc_q = build_cv([{"type": "native_contacts_q", **spec}], forcefield)
    pc_n = build_cv([{"type": "native_contacts_n", **spec}], forcefield)
    assert isinstance(pc_q, NativeContactsQ)
    assert isinstance(pc_n, NativeContactsN)
    assert pc_q.cv.n_contacts == len(explicit_pairs)
    assert pc_q.cv.native_contact_pairs is not None
    assert pc_q.cv.native_contact_pairs.tolist() == explicit_pairs
    np.testing.assert_allclose(pc_q(native_coords), [1.0])
    np.testing.assert_allclose(pc_n(native_coords), [float(len(explicit_pairs))])


def test_native_contact_pairs_invalid_pair_raises(chignolin_pdb_path: str, forcefield) -> None:
    """An invalid pair in the spec (a self-pair here) must propagate
    NativeContactsCV's ValueError up through build_cv's dispatch rather
    than being swallowed or silently ignored."""
    spec = {
        "type": "native_contacts_q",
        "reference_pdb": str(chignolin_pdb_path),
        "native_contact_pairs": [[3, 3]],
    }
    with pytest.raises(ValueError, match="self-pair"):
        build_cv([spec], forcefield)


def test_ca_rmsd_at_native_is_zero(chignolin_pdb_path: str, forcefield, native_coords: np.ndarray) -> None:
    pc = build_cv([{"type": "ca_rmsd", "reference_pdb": str(chignolin_pdb_path)}], forcefield)
    assert isinstance(pc, CaRmsd)
    np.testing.assert_allclose(pc(native_coords), [0.0], atol=_RMSD_ATOL)


def test_ca_rmsd_is_rigid_body_invariant(chignolin_pdb_path: str, forcefield, native_coords: np.ndarray) -> None:
    """Regression test for the Kabsch-superposition bug found and fixed
    while building this pcoord (a spurious extra transpose that inflated
    RMSD for any non-aligned mobile structure -- see
    ``pymcpu.sampling.collective_variables.kabsch_rmsd``): a pure rotation
    + translation of the mobile structure must still report ~0 RMSD against
    an identical reference."""
    pc = build_cv([{"type": "ca_rmsd", "reference_pdb": str(chignolin_pdb_path)}], forcefield)

    theta = 0.7
    rotation = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    translation = np.array([[5.0], [-3.0], [2.0]], dtype=np.float32)
    rotated = rotation @ native_coords + translation

    np.testing.assert_allclose(pc(rotated), [0.0], atol=_RMSD_ATOL)


def test_composite_concatenates_ndim_and_labels(
    chignolin_pdb_path: str, forcefield, native_coords: np.ndarray
) -> None:
    pc = build_cv(
        [
            {"type": "native_contacts_q", "reference_pdb": str(chignolin_pdb_path), **_NATIVE_CONTACTS_PARAMS},
            {"type": "ca_rmsd", "reference_pdb": str(chignolin_pdb_path)},
        ],
        forcefield,
    )
    assert isinstance(pc, CompositeCV)
    assert pc.ndim == 2
    assert pc.labels == ("Q", "RMSD")
    result = pc(native_coords)
    assert result.shape == (2,)
    np.testing.assert_allclose(result, [1.0, 0.0], atol=_RMSD_ATOL)


def test_two_state_rmsd_and_delta(chignolin_pdb_path: str, forcefield, native_coords: np.ndarray) -> None:
    # Same reference for both states -> both dims must agree, delta ~ 0.
    spec_two_state = {
        "reference_a": str(chignolin_pdb_path),
        "reference_b": str(chignolin_pdb_path),
    }
    pc_pair = build_cv([{"type": "two_state_rmsd", **spec_two_state}], forcefield)
    assert isinstance(pc_pair, TwoStateRmsd)
    assert pc_pair.ndim == 2
    a, b = pc_pair(native_coords)
    assert a == pytest.approx(b, abs=_RMSD_ATOL)

    pc_delta = build_cv([{"type": "two_state_delta", **spec_two_state}], forcefield)
    assert isinstance(pc_delta, TwoStateDelta)
    assert pc_delta.ndim == 1
    np.testing.assert_allclose(pc_delta(native_coords), [0.0], atol=_RMSD_ATOL)
