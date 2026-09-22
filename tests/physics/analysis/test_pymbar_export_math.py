"""Unit tests for the pure-math MBAR/pymbar reduced-potential helper.

``reduced_potentials_from_arrays`` (pymcpu.analysis.pymbar_export) computes
the standard umbrella-sampling reduced potential u_kn = (E + bias) / T for
every (temperature x N0-window) state against every sample, where
bias = 0.5 * k * (N - N0)^2. These tests check that formula against
hand-derived arithmetic -- no simulation, no legacy reference, and no h5py
dependency -- covering: the formula itself, the zero-bias edge case
(reduces to plain E/T), and state/sample ordering with multiple temps.
"""

from __future__ import annotations

import numpy as np

from pymcpu.analysis.pymbar_export import reduced_potentials_from_arrays


def test_reduced_potential_formula() -> None:
    # One sample, two temps x two N0 windows -> 4 states.
    e = np.array([10.0])
    N = np.array([4.0])
    temps = np.array([0.5, 1.0])
    n0 = np.array([0.0, 5.0])
    k = 2.0
    kB = 1.0

    u = reduced_potentials_from_arrays(e, N, temps, n0, k, kB=kB)
    assert u.shape == (4, 1)

    # Manual: bias to N0=0 is 0.5*2*(4-0)^2=16; bias to N0=5 is 0.5*2*(4-5)^2=1.
    expected = np.array(
        [
            [(10.0 + 16.0) / 0.5],  # T=0.5, N0=0
            [(10.0 + 1.0) / 0.5],  # T=0.5, N0=5
            [(10.0 + 16.0) / 1.0],  # T=1.0, N0=0
            [(10.0 + 1.0) / 1.0],  # T=1.0, N0=5
        ]
    )
    np.testing.assert_allclose(u, expected)


def test_zero_k_bias_is_just_e_over_t() -> None:
    e = np.array([3.0, 6.0])
    N = np.array([1.0, 2.0])
    temps = np.array([1.5])
    n0 = np.array([0.0, 10.0])
    u = reduced_potentials_from_arrays(e, N, temps, n0, k_bias=0.0)
    assert u.shape == (2, 2)
    # k_bias=0 must zero out the harmonic term regardless of N/N0, leaving E/T.
    np.testing.assert_allclose(u[0], e / 1.5)
    np.testing.assert_allclose(u[1], e / 1.5)


def test_multi_sample_state_ordering() -> None:
    e = np.array([0.0, 1.0])
    N = np.array([0.0, 0.0])
    temps = np.array([2.0, 4.0])
    n0 = np.array([0.0])
    u = reduced_potentials_from_arrays(e, N, temps, n0, k_bias=1.0)
    assert u.shape == (2, 2)
    # States must be ordered by temperature index (rows), samples by column,
    # so row 0 is T=2.0's per-sample energies and row 1 is T=4.0's.
    np.testing.assert_allclose(u[0], [0.0 / 2.0, 1.0 / 2.0])
    np.testing.assert_allclose(u[1], [0.0 / 4.0, 1.0 / 4.0])
