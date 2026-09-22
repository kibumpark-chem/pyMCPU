"""Finite/sanity checks for calculateEnergy on the real chignolin/actin
forcefield, one per potential type not already covered by an exact-formula test.

These only assert that each term evaluates to a finite, physically
plausible number on a real structure -- not a pinned value (see
``test_triplet_potential_formula.py`` for the exact-formula tests on
synthetic minimal systems).
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.slow

import numpy as np

from tests.physics.helpers.constants import MU_CLASH_SENTINEL


def test_hbond_energy_is_finite(chignolin_context) -> None:
    energy = chignolin_context.calculate_total_energy(4)
    assert np.isfinite(energy)


def test_mu_energy_is_finite_and_below_clash_sentinel(chignolin_context) -> None:
    energy = chignolin_context.calculate_total_energy(1)
    assert np.isfinite(energy)
    # The folded reference structure should not itself be in a steric clash.
    assert energy < MU_CLASH_SENTINEL


def test_aromatic_energy_is_finite(chignolin_context) -> None:
    energy = chignolin_context.calculate_total_energy(5)
    assert np.isfinite(energy)


def test_qbias_energy_formula_is_finite_and_nonnegative(chignolin_with_qbias) -> None:
    """Q-bias is a harmonic umbrella (0.5 * k * (N - n_target)^2), so its
    energy must be finite and non-negative for any N."""
    context, _forcefield = chignolin_with_qbias
    n_pairs = None
    for potential in context.get_system().get_potentials():
        if hasattr(potential, "num_pairs"):
            n_pairs = int(potential.num_pairs())
            break
    assert n_pairs is not None and n_pairs > 0

    k_bias = 10.0
    n_target = 0.5 * n_pairs  # half of native contacts formed
    context.set_native_contacts_bias(k_bias, n_target)

    energy = context.calculate_total_energy(6)
    assert np.isfinite(energy)
    assert energy >= 0.0
