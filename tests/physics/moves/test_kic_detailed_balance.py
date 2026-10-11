"""KIC alone samples the exact Boltzmann distribution.

A chain where one KIC window is the only thing that can move has states on
closed curves: the driver angle theta = psi of a proline, and a closure of
the window for each theta (see tests/helpers/kic_closure.py, which finds the
closures independently of the engine's solver). The exact distribution has
density J exp(-E / T) per unit of theta, zero where the engine's hard-core
check finds a clash; E is the engine's own energy. It is enumerated densely
along the curves, and each MC run starts from a state drawn from it, so
every sample of a correct kernel is distributed exactly, however slowly the
run mixes. Over many seeds, the mean Legendre moments P1-P4 of three window
torsions (theta, phi of the middle residue and psi of the last) must match
the exact ones: Hotelling's T^2 over the seeds.

Each test also checks its own power: the same samples must reject the
distribution a kernel without the Jacobian factor would sample (chignolin),
or without the closure-count factor n_new / n_old (actin).

Chignolin's window (0-based residues: PRO 3; GLU 4, THR 5, GLY 6) has one
closure curve with two closures per theta and no clash, so it tests the
Jacobian, the driver and the choice among closures. The actin window (PRO
28; ARG 29, ALA 30, VAL 31, in residues 26-33 of the example structure) has
four curves, mostly with six or eight closures per theta, and clashes on
three quarters of them, so it also tests n_new / n_old. It runs the wide
driver only: with a narrow one each run stays on the island of clash-free
states it starts on, and a hundred seeds cannot weigh the small islands.

Only the psi driver can be tested this way. The phi-driver window at r needs
residues r to r+3 free, and so does the psi-driver window at r+1, so the
phi driver never runs alone and the two together leave a two-dimensional
state space.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.polynomial import legendre
from scipy import stats

from pymcpu import mcpu_core
from pymcpu.runners import default_example_pdb
from tests.fixtures.context_builders import TEST_PDB
from tests.helpers.kic_closure import KicWindow, closures_per_theta, evaluate, exact_curve

TORSIONS = (0, 3, 6)   # psi(r-1) = theta, phi(r+1), psi(r+2)
MOMENTS = 4


def _reference(window: KicWindow, T: float) -> dict:
    x, dtheta, curve = exact_curve(window)
    E, clash, J, tors = evaluate(window, x)
    ok = ~clash
    boltzmann = np.where(ok, np.exp(-(E - E[ok].min()) / T), 0.0)
    return {"window": window, "T": T, "x": x, "tors": tors, "dtheta": dtheta, "J": J,
            "boltzmann": boltzmann, "curve": curve}


def _normalized(w: np.ndarray) -> np.ndarray:
    return w / w.sum()


def _run_kic(window: KicWindow, ref: dict, weights: np.ndarray, width: float, n_seeds: int,
             n_samples: int, stride: int) -> np.ndarray:
    """KIC-only runs from exact starts; the seven window torsions every ``stride`` steps."""
    ctx = window.new_context()
    integ = mcpu_core.Integrator(temperature=ref["T"], kic_step_size_rad=width)
    integ.set_move_weights(0.0, 1.0, 0.0)
    integ.set_fixed_residues(window.fixed, window.n_res)
    rng = np.random.default_rng(20261009)
    samples = np.empty((n_seeds, n_samples, 7))
    coords = np.empty((n_samples, 3, len(window.res)))
    for i in range(n_seeds):
        integ.set_seed(1 + i)
        # A start within about 1e-6 rad of a point where theta turns back
        # along a curve can have its two merging closures unresolved by the
        # engine's float solver; every KIC step from it is then refused
        # ("presolve zero") and the run would never move. Draw again.
        for _ in range(20):
            k = int(rng.choice(len(weights), p=weights))
            ctx.coords = window.coords(ref["x"][k:k + 1])[0]
            ctx.calculate_total_energy(-1)
            refused = integ.get_kic_presolve_zero()
            integ.run(ctx, 50)
            if integ.get_kic_presolve_zero() == refused:
                break
        else:
            raise AssertionError("no start the engine can move from")
        for s in range(n_samples):
            integ.run(ctx, stride)
            coords[s] = ctx.coords
        samples[i] = window.torsions(coords)
        # The run stays on the closure curves (to float32 precision).
        _, G = window.closure(window.locate(ctx.coords)[None])
        assert np.abs(G).max() < 1e-4
    accepted, _ = integ.move_counts(include_unused=True)["kic"]
    assert accepted >= 1000
    return samples


def _features(ref: dict, weights: np.ndarray, samples: np.ndarray):
    """Legendre moments of the window torsions: per reference point and per seed (mean)."""
    at_ref, per_seed = [], []
    for j in TORSIONS:
        x = ref["tors"][:, j]
        center = math.atan2((weights * np.sin(x)).sum(), (weights * np.cos(x)).sum())
        xr = np.remainder(x - center + math.pi, 2 * math.pi) - math.pi
        lo, hi = xr[weights > 0].min(), xr[weights > 0].max()
        mid, half = 0.5 * (lo + hi), 0.51 * (hi - lo)
        u = (xr - mid) / half
        us = (np.remainder(samples[:, :, j] - center + math.pi, 2 * math.pi) - math.pi - mid) / half
        for k in range(1, MOMENTS + 1):
            c = np.eye(MOMENTS + 1)[k]
            at_ref.append(legendre.legval(u, c))
            per_seed.append(legendre.legval(us, c).mean(axis=1))
    return np.stack(at_ref, 1), np.stack(per_seed, 1)


def _hotelling_p(per_seed: np.ndarray, expected: np.ndarray) -> float:
    n, p = per_seed.shape
    d = per_seed.mean(axis=0) - expected
    t2 = n * d @ np.linalg.solve(np.cov(per_seed, rowvar=False), d)
    return float(stats.f.sf((n - p) / (p * (n - 1)) * t2, p, n - p))


@pytest.fixture(scope="module")
def chignolin() -> dict:
    return _reference(KicWindow(default_example_pdb(), proline=3), T=0.6)


def test_the_closure_curve_holds_the_native_window(chignolin) -> None:
    window = chignolin["window"]
    x = window.locate(window.base)
    _, G = window.closure(x[None])
    assert np.abs(G).max() < 1e-9
    assert np.isclose(np.ptp(np.degrees(chignolin["x"][:, 0])), 27.3, atol=0.5)


@pytest.mark.parametrize("width,layout", [
    pytest.param(0.1, "off", id="0.1rad"),
    pytest.param(1.0, "off", id="1.0rad"),
    pytest.param(0.1, "init_only", id="0.1rad-init_only", marks=pytest.mark.slow),
])
def test_kic_samples_the_exact_distribution(chignolin, width: float, layout: str) -> None:
    ref = chignolin
    window = ref["window"] if layout == "off" else KicWindow(default_example_pdb(), proline=3,
                                                             reorder=layout)
    exact = _normalized(ref["J"] * ref["dtheta"] * ref["boltzmann"])
    samples = _run_kic(window, ref, exact, width, n_seeds=64, n_samples=500, stride=20)
    at_ref, per_seed = _features(ref, exact, samples)
    assert _hotelling_p(per_seed, exact @ at_ref) > 1e-3
    no_jacobian = _normalized(ref["dtheta"] * ref["boltzmann"])
    assert _hotelling_p(per_seed, no_jacobian @ at_ref) < 1e-10


@pytest.mark.slow
def test_kic_weighs_closure_counts_exactly() -> None:
    ref = _reference(KicWindow(TEST_PDB, proline=2, residues=(26, 33)), T=1.0)
    n_closures = closures_per_theta(ref["x"], ref["curve"])
    exact = _normalized(ref["J"] * ref["dtheta"] * ref["boltzmann"])
    assert exact[np.round(n_closures) >= 6].sum() > 0.5
    samples = _run_kic(ref["window"], ref, exact, 1.0, n_seeds=128, n_samples=1000, stride=25)
    at_ref, per_seed = _features(ref, exact, samples)
    assert _hotelling_p(per_seed, exact @ at_ref) > 1e-3
    no_count_ratio = _normalized(exact / np.maximum(n_closures, 1.0))
    assert _hotelling_p(per_seed, no_count_ratio @ at_ref) < 1e-10
