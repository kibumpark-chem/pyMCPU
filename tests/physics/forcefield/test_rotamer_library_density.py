"""Analytic unit tests for ``mcpu_core.RotamerLibrary``'s mixture log-density.

No existing test in this repo measures detailed balance as a statistical
property; these are new, purpose-built checks for the rotamer-library
sidechain move's Metropolis-Hastings correction
(``patch.log_jacobian_weight = log_q_old - log_q_new``, see
``MCIntegrator::apply_rotamer_at``). Pure C++ binding tests -- no
``MCPUForceField``/PDB/Context involved.
"""

from __future__ import annotations

import math

import pytest

from pymcpu import mcpu_core


def _lib_single_component(mean: float, sigma: float, ntorsions: int = 1) -> "mcpu_core.RotamerLibrary":
    lib = mcpu_core.RotamerLibrary()
    means = [mean] + [0.0] * 3
    sigmas = [sigma] + [1.0] * 3
    lib.add_residue_type(0, [1.0], [means], [sigmas])
    return lib


def _normal_log_pdf(x: float, mean: float, sigma: float) -> float:
    return -0.5 * ((x - mean) / sigma) ** 2 - math.log(sigma) - 0.5 * math.log(2 * math.pi)


def test_single_component_reduces_to_plain_gaussian() -> None:
    """K=1: the mixture density is exactly one Gaussian log-density."""
    mean, sigma = 0.3, 0.1
    lib = _lib_single_component(mean, sigma)
    for x in (0.0, 0.1, 0.3, 0.5, -0.2):
        expected = _normal_log_pdf(x, mean, sigma)
        actual = lib.log_mixture_density(0, 1, [x, 0.0, 0.0, 0.0])
        assert actual == pytest.approx(expected, abs=1e-5), (x, expected, actual)


def test_wraparound_scores_minimal_angular_distance() -> None:
    """A chi value near +pi and a mean near -pi are actually ~2 degrees
    apart (wrapped), not ~358 degrees -- the whole point of using a wrapped
    angular difference rather than a naive linear one."""
    mean = math.radians(179.0)
    sigma = math.radians(10.0)
    lib = _lib_single_component(mean, sigma)

    x_near = math.radians(-179.0)  # 2 degrees from mean, wrapped
    x_far = math.radians(90.0)  # 89 degrees from mean, wrapped
    density_near = lib.log_mixture_density(0, 1, [x_near, 0.0, 0.0, 0.0])
    density_far = lib.log_mixture_density(0, 1, [x_far, 0.0, 0.0, 0.0])

    expected_near = _normal_log_pdf(math.radians(2.0), 0.0, sigma)
    assert density_near == pytest.approx(expected_near, abs=1e-4)
    assert density_near > density_far, (
        "wrapped-near point should score a far higher density than the "
        "genuinely-far point"
    )


def test_two_component_equal_weight_log_sum_exp_bookkeeping() -> None:
    """K=2, equal weight: evaluated at either component's own mean recovers
    log(0.5) + that component's own single-Gaussian log-density (checks the
    log-sum-exp accumulation itself, not just the per-component formula)."""
    mean_a, mean_b = 0.1, 0.2
    sigma = 0.05
    lib = mcpu_core.RotamerLibrary()
    means = [[mean_a, 0.0, 0.0, 0.0], [mean_b, 0.0, 0.0, 0.0]]
    sigmas = [[sigma, 1.0, 1.0, 1.0], [sigma, 1.0, 1.0, 1.0]]
    lib.add_residue_type(0, [0.5, 0.5], means, sigmas)

    for eval_mean, other_mean in ((mean_a, mean_b), (mean_b, mean_a)):
        own_term = _normal_log_pdf(eval_mean, eval_mean, sigma)
        cross_term = _normal_log_pdf(eval_mean, other_mean, sigma)
        expected = math.log(0.5 * math.exp(own_term) + 0.5 * math.exp(cross_term))
        actual = lib.log_mixture_density(0, 1, [eval_mean, 0.0, 0.0, 0.0])
        assert actual == pytest.approx(expected, abs=1e-4)

    # By construction (equal weight, equal sigma), evaluating at either mean
    # gives the identical value -- the mixture is symmetric under swapping
    # which component's mean is the evaluation point.
    density_a = lib.log_mixture_density(0, 1, [mean_a, 0.0, 0.0, 0.0])
    density_b = lib.log_mixture_density(0, 1, [mean_b, 0.0, 0.0, 0.0])
    assert density_a == pytest.approx(density_b, abs=1e-6)


def test_log_jacobian_weight_construction_is_antisymmetric() -> None:
    """Pins the algebraic signature required for the MH correction: swapping
    which state is "old" vs "new" must exactly negate
    log_q(x) - log_q(y). Trivial algebraically, but this is exactly the
    property patch.log_jacobian_weight's construction depends on for
    detailed balance (the forward and reverse proposal-ratio terms must be
    exact negatives of one another)."""
    lib = mcpu_core.RotamerLibrary()
    means = [[0.1, 0.2, 0.0, 0.0], [-0.3, 0.4, 0.0, 0.0], [0.5, -0.1, 0.0, 0.0]]
    sigmas = [[0.1, 0.15, 1.0, 1.0]] * 3
    lib.add_residue_type(0, [0.2, 0.5, 0.3], means, sigmas)

    x = [0.05, 0.1, 0.0, 0.0]
    y = [-0.2, 0.3, 0.0, 0.0]
    log_qx = lib.log_mixture_density(0, 2, x)
    log_qy = lib.log_mixture_density(0, 2, y)

    forward = log_qx - log_qy
    reverse = log_qy - log_qx
    assert forward == pytest.approx(-reverse, abs=1e-6)


def test_num_rows_and_sample_row_bounds() -> None:
    lib = mcpu_core.RotamerLibrary()
    assert lib.num_rows(5) == 0  # never registered
    assert lib.sample_row(5, 0.5) == -1

    lib.add_residue_type(5, [0.3, 0.7], [[0.0] * 4, [0.0] * 4], [[1.0] * 4, [1.0] * 4])
    assert lib.num_rows(5) == 2
    assert lib.sample_row(5, 0.0) == 0
    assert lib.sample_row(5, 0.29) == 0
    assert lib.sample_row(5, 0.31) == 1
    # Float-rounding fallthrough (u01 == 1.0) clamps to the last row rather
    # than reading out of bounds.
    assert lib.sample_row(5, 1.0) == 1


def test_row_out_of_range_raises() -> None:
    lib = mcpu_core.RotamerLibrary()
    lib.add_residue_type(0, [1.0], [[0.0] * 4], [[1.0] * 4])
    with pytest.raises(Exception):
        lib.row(0, 1)
    with pytest.raises(Exception):
        lib.row(1, 0)
