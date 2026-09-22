"""``pymcpu.sampling.cv_factory.build_cv`` -- the spec-driven CV-spec
dispatch factory: unknown-type errors, a custom user factory loaded by
dotted import path, and a malformed-reference error path.

Pure software-plumbing/config-dispatch, not physics -- this is WESTPA-facing
configuration parsing with no legacy MCPU equivalent, which is why it lives
here rather than in ``tests/physics/``. The CV math itself (Q/RMSD/
CompositeCV/TwoState numeric correctness) is covered by
``tests/physics/cv/test_collective_variables.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from pymcpu.sampling.collective_variables import CARMSDCV
from pymcpu.sampling.cv_factory import build_cv
from tests.physics.helpers.cv_fixtures import build_chignolin_forcefield, native_coords_angstrom


@pytest.fixture(scope="module")
def forcefield(chignolin_pdb_path: str):
    return build_chignolin_forcefield(chignolin_pdb_path)


@pytest.fixture
def native_coords(forcefield) -> np.ndarray:
    return native_coords_angstrom(forcefield)


class _ConstantCV:
    """Minimal custom CV: exercises ``build_cv``'s ``type: custom``
    dispatch, which only requires ``ndim``/``labels``/``__call__``."""

    ndim = 1
    labels = ("const",)

    def __init__(self, value: float) -> None:
        self.value = value

    def __call__(self, coords_3xn: np.ndarray) -> np.ndarray:
        return np.array([self.value], dtype=np.float64)


def _make_constant_cv(value: float) -> _ConstantCV:
    return _ConstantCV(value)


def test_custom_pcoord_factory(forcefield, native_coords: np.ndarray) -> None:
    spec = {
        "type": "custom",
        # Dotted path to this module's own factory function above --
        # exercises build_cv's importlib-based dynamic dispatch.
        #
        # Derived from __name__ rather than written out. A literal path here
        # is invisible to every tool: moving or renaming this file leaves a
        # string that still parses, still lints, and fails only at runtime as
        # a confusing ModuleNotFoundError from inside the factory loader.
        "factory": f"{__name__}._make_constant_cv",
        "kwargs": {"value": 3.5},
    }
    pc = build_cv([spec], forcefield)
    np.testing.assert_allclose(pc(native_coords), [3.5])


def test_unknown_pcoord_type_raises(forcefield) -> None:
    with pytest.raises(ValueError, match="unknown CV type"):
        build_cv([{"type": "not_a_real_cv"}], forcefield)


def test_mismatched_reference_raises_rather_than_returning_zero() -> None:
    # A reference with the wrong residue count must raise, not silently
    # produce a meaningless-but-plausible-looking Q/RMSD value.
    bad_ref = np.zeros((3, 3))  # 3 residues, nowhere near chignolin's 10
    with pytest.raises(ValueError):
        CARMSDCV(ca_internal_idx=np.arange(10), ref_ca_xyz=bad_ref)
