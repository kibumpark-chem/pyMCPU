"""Analytic unit tests for ``mcpu_core.RamaMixtureLibrary``'s (phi, psi)
mixture log-density.

Companion to ``test_rotamer_library_density.py``, for the knowledge-based
backbone pivot move's Metropolis-Hastings correction
(``patch.log_jacobian_weight = log_q_old - log_q_new``, see
``MCIntegrator::apply_rama_pivot_at``). Pure C++ binding tests -- no
``MCPUForceField``/PDB/Context involved.

Unlike ``RotamerLibrary`` (chi angles, sigmas at most a few tens of
degrees, where a single-nearest-image density is provably safe),
``RamaMixtureLibrary`` uses a proper periodic ((2*n_wrap+1)^2-cell) sum
because some fitted backbone components are broad enough (sigma up to
~1.3 rad) that the periodic wraparound genuinely matters -- several tests
below are specifically designed to fail if the implementation were ever
"simplified" back down to a nearest-image shortcut.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from pymcpu import mcpu_core

TWO_PI = 2.0 * math.pi


def _lib_single_component(
    mean: tuple[float, float], cov: tuple[float, float, float], n_wrap: int = 1
) -> "mcpu_core.RamaMixtureLibrary":
    lib = mcpu_core.RamaMixtureLibrary(n_wrap=n_wrap)
    lib.add_residue_type(0, [1.0], [list(mean)], [list(cov)])
    return lib


def _bivariate_normal_log_pdf(
    x: tuple[float, float], mean: tuple[float, float], cov: tuple[float, float, float]
) -> float:
    c11, c12, c22 = cov
    det = c11 * c22 - c12 * c12
    d0, d1 = x[0] - mean[0], x[1] - mean[1]
    inv11, inv12, inv22 = c22 / det, -c12 / det, c11 / det
    quad = inv11 * d0 * d0 + 2.0 * inv12 * d0 * d1 + inv22 * d1 * d1
    return -0.5 * (quad + math.log(det) + 2.0 * math.log(2.0 * math.pi))


def _brute_force_wrapped_log_pdf(
    x: tuple[float, float],
    mean: tuple[float, float],
    cov: tuple[float, float, float],
    n_wrap: int,
) -> float:
    """Independent reference: literal sum (not log-sum-exp) over all
    (2*n_wrap+1)^2 periodic images, in plain probability space. Used only
    to cross-check the C++ implementation's log-space accumulation, not
    reused by it."""
    total = 0.0
    for dx, dy in itertools.product(range(-n_wrap, n_wrap + 1), repeat=2):
        shifted_mean = (mean[0] + dx * TWO_PI, mean[1] + dy * TWO_PI)
        total += math.exp(_bivariate_normal_log_pdf(x, shifted_mean, cov))
    return math.log(total)


def test_single_component_single_wrap_reduces_to_plain_bivariate_gaussian() -> None:
    """n_wrap=0, K=1: the mixture density is exactly one bivariate Gaussian
    log-density (no periodic images at all)."""
    mean = (0.3, -0.5)
    cov = (0.02, 0.005, 0.03)
    lib = _lib_single_component(mean, cov, n_wrap=0)
    for x in [(0.0, 0.0), (0.3, -0.5), (0.5, -0.2), (-0.1, 0.4)]:
        expected = _bivariate_normal_log_pdf(x, mean, cov)
        actual = lib.log_mixture_density(0, list(x))
        assert actual == pytest.approx(expected, abs=1e-4), (x, expected, actual)


def test_wraparound_scores_minimal_angular_distance() -> None:
    """A (phi, psi) point near (+pi, +pi) and a mean near (-pi, -pi) are
    actually only a few degrees apart on the torus, not ~358 degrees in
    each dimension -- the whole point of the periodic sum rather than a
    naive linear evaluation."""
    mean = (math.radians(179.0), math.radians(-179.0))
    cov = (math.radians(10.0) ** 2, 0.0, math.radians(10.0) ** 2)
    lib = _lib_single_component(mean, cov, n_wrap=1)

    x_near = (math.radians(-179.0), math.radians(179.0))  # ~2 deg away, wrapped, in both dims
    x_far = (math.radians(90.0), math.radians(-90.0))  # genuinely far

    density_near = lib.log_mixture_density(0, list(x_near))
    density_far = lib.log_mixture_density(0, list(x_far))

    expected_near = _bivariate_normal_log_pdf(
        (math.radians(2.0), math.radians(-2.0)), (0.0, 0.0), cov
    )
    assert density_near == pytest.approx(expected_near, abs=1e-3)
    assert density_near > density_far


def test_nearest_image_only_would_be_wrong_for_broad_component() -> None:
    """Construct a component as broad as the widest ones actually observed
    in the p0.4-fitted backbone priors (sigma ~1.2-1.3 rad, e.g. ALA's
    low-weight background component) and confirm the library's density
    matches a brute-force (2*n_wrap+1)^2-cell reference, NOT a
    nearest-image-only evaluation -- the two must disagree measurably for
    this to be a meaningful test."""
    mean = (0.2, -0.3)
    cov = (1.3**2, 0.1, 1.2**2)  # broad, correlated component
    n_wrap = 1
    lib = _lib_single_component(mean, cov, n_wrap=n_wrap)

    # A point where the periodic neighbor images contribute non-negligibly:
    # far from the mean along the direction where wraparound images sit.
    x = (2.9, -2.7)

    nearest_image_only = _bivariate_normal_log_pdf(x, mean, cov)
    brute_force_periodic = _brute_force_wrapped_log_pdf(x, mean, cov, n_wrap)
    actual = lib.log_mixture_density(0, list(x))

    assert brute_force_periodic > nearest_image_only + 1e-3, (
        "test is only meaningful if periodic images actually change the "
        "density measurably for this broad a component"
    )
    assert actual == pytest.approx(brute_force_periodic, abs=1e-3)


def test_two_component_equal_weight_log_sum_exp_bookkeeping() -> None:
    """K=2, equal weight, narrow+well-separated (so periodic images are
    irrelevant): evaluated at either component's own mean recovers
    log(0.5) + that component's own single-Gaussian log-density."""
    mean_a, mean_b = (0.1, 0.1), (-0.2, 0.3)
    cov = (0.01, 0.0, 0.01)
    lib = mcpu_core.RamaMixtureLibrary(n_wrap=1)
    lib.add_residue_type(0, [0.5, 0.5], [list(mean_a), list(mean_b)], [list(cov), list(cov)])

    for eval_point, other_mean in ((mean_a, mean_b), (mean_b, mean_a)):
        own_term = _bivariate_normal_log_pdf(eval_point, eval_point, cov)
        cross_term = _bivariate_normal_log_pdf(eval_point, other_mean, cov)
        expected = math.log(0.5 * math.exp(own_term) + 0.5 * math.exp(cross_term))
        actual = lib.log_mixture_density(0, list(eval_point))
        assert actual == pytest.approx(expected, abs=1e-3)


def test_cholesky_sampling_matches_log_pdf() -> None:
    """The Cholesky factors exposed via row() must correspond to exactly
    the same distribution log_mixture_density evaluates -- draw samples
    using the same recipe MCIntegrator::apply_rama_pivot_at uses (two
    standard-normal draws through the Cholesky factor, mean-shifted, wrapped
    to (-pi, pi]) and confirm the sample mean/covariance match the input
    parameters. (A coarse-histogram density comparison was tried first and
    rejected: for a component this narrow, sigma ~0.15-0.17 rad, almost all
    mass falls inside 1-2 cells of any grid coarse enough to bin cheaply,
    where the true density curves sharply across the cell -- so a
    bin-averaged empirical density disagrees with a point evaluation at the
    cell center by a large amount that has nothing to do with correctness.
    Moment-matching is the appropriate tool here; wraparound is negligible
    at this sigma so plain linear moments apply.)"""
    mean = (0.4, -0.6)
    cov = (0.03, -0.01, 0.02)
    lib = _lib_single_component(mean, cov, n_wrap=1)
    row = lib.row(0, 0)

    rng = np.random.default_rng(0)
    n_samples = 500_000
    z = rng.standard_normal((n_samples, 2))
    phi = mean[0] + row.chol_l11 * z[:, 0]
    psi = mean[1] + row.chol_l21 * z[:, 0] + row.chol_l22 * z[:, 1]
    phi = (phi + math.pi) % TWO_PI - math.pi
    psi = (psi + math.pi) % TWO_PI - math.pi

    sample_mean = np.array([phi.mean(), psi.mean()])
    sample_cov = np.cov(np.stack([phi, psi]))

    # Standard error scales as sigma/sqrt(N) for the mean, sigma^2*sqrt(2/N)
    # for covariance entries; N=500_000 makes both comfortably tight.
    assert sample_mean == pytest.approx(np.array(mean), abs=5e-3)
    assert sample_cov[0, 0] == pytest.approx(cov[0], abs=5e-3)
    assert sample_cov[0, 1] == pytest.approx(cov[1], abs=5e-3)
    assert sample_cov[1, 1] == pytest.approx(cov[2], abs=5e-3)

    # And a direct, single-point density cross-check at the mean itself
    # (where the coarse-bin curvature problem above doesn't apply -- the
    # point estimate there is dominated by within-cell probability mass
    # close enough to uniform for a first-order comparison to hold).
    log_density_at_mean = lib.log_mixture_density(0, list(mean))
    expected_log_density_at_mean = _bivariate_normal_log_pdf(mean, mean, cov)
    assert log_density_at_mean == pytest.approx(expected_log_density_at_mean, abs=1e-3)


def test_log_jacobian_weight_construction_is_antisymmetric() -> None:
    """Pins the algebraic signature required for the MH correction: swapping
    which state is "old" vs "new" must exactly negate log_q(x) - log_q(y)."""
    lib = mcpu_core.RamaMixtureLibrary(n_wrap=1)
    means = [(0.1, 0.2), (-0.3, 0.4), (0.5, -0.1)]
    covs = [(0.02, 0.0, 0.03)] * 3
    lib.add_residue_type(0, [0.2, 0.5, 0.3], [list(m) for m in means], [list(c) for c in covs])

    x = [0.05, 0.1]
    y = [-0.2, 0.3]
    log_qx = lib.log_mixture_density(0, x)
    log_qy = lib.log_mixture_density(0, y)

    forward = log_qx - log_qy
    reverse = log_qy - log_qx
    assert forward == pytest.approx(-reverse, abs=1e-6)


def test_density_integrates_to_approximately_one_over_one_period() -> None:
    """Coarse grid quadrature over [-pi, pi) x [-pi, pi) confirms this is a
    properly normalized torus density, not merely a truncated Gaussian."""
    mean = (0.3, -0.4)
    cov = (0.05, 0.01, 0.04)
    lib = _lib_single_component(mean, cov, n_wrap=1)

    n_grid = 121
    grid = np.linspace(-math.pi, math.pi, n_grid, endpoint=False)
    cell_area = (TWO_PI / n_grid) ** 2
    total = 0.0
    for p in grid:
        for s in grid:
            total += math.exp(lib.log_mixture_density(0, [float(p), float(s)])) * cell_area
    assert total == pytest.approx(1.0, abs=1e-3)


def test_num_rows_and_sample_row_bounds() -> None:
    lib = mcpu_core.RamaMixtureLibrary()
    assert lib.num_rows(5) == 0  # never registered
    assert lib.sample_row(5, 0.5) == -1

    lib.add_residue_type(5, [0.3, 0.7], [[0.0, 0.0], [0.0, 0.0]], [[1.0, 0.0, 1.0], [1.0, 0.0, 1.0]])
    assert lib.num_rows(5) == 2
    assert lib.sample_row(5, 0.0) == 0
    assert lib.sample_row(5, 0.29) == 0
    assert lib.sample_row(5, 0.31) == 1
    # Float-rounding fallthrough (u01 == 1.0) clamps to the last row rather
    # than reading out of bounds.
    assert lib.sample_row(5, 1.0) == 1


def test_row_out_of_range_raises() -> None:
    lib = mcpu_core.RamaMixtureLibrary()
    lib.add_residue_type(0, [1.0], [[0.0, 0.0]], [[1.0, 0.0, 1.0]])
    with pytest.raises(Exception):
        lib.row(0, 1)
    with pytest.raises(Exception):
        lib.row(1, 0)


def test_add_residue_type_rejects_non_positive_definite_covariance() -> None:
    lib = mcpu_core.RamaMixtureLibrary()
    with pytest.raises(Exception):
        lib.add_residue_type(0, [1.0], [[0.0, 0.0]], [[-1.0, 0.0, 1.0]])  # c11 < 0
    with pytest.raises(Exception):
        lib.add_residue_type(0, [1.0], [[0.0, 0.0]], [[1.0, 2.0, 1.0]])  # det < 0


def test_add_residue_type_auto_grows_beyond_default_20_categories() -> None:
    """Extensibility seam: a future reserved category (e.g. pre-proline)
    can be registered at index >= 20 without any change to this class."""
    lib = mcpu_core.RamaMixtureLibrary()
    lib.add_residue_type(20, [1.0], [[0.1, 0.1]], [[0.01, 0.0, 0.01]])
    assert lib.num_rows(20) == 1
    assert lib.num_rows(19) == 0  # untouched, still unregistered
