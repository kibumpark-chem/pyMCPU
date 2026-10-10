"""KIC (kinematic-closure) rejection counters.

The Integrator tracks five ways a KIC move can end without being proposed:

* ``kic_presolve_zero`` -- the solver found no closure at all for the current
  window's own endpoints, so the reverse move's solution count would be zero.
* ``kic_reverse_missing`` -- the solver found closures for the current window,
  but none of them is the current window itself (within 1e-3 A), so the move
  could not be reversed. It is refused rather than accepted one-way.
* ``kic_jacobian_invalid`` -- a near-singular Jacobian at either end.
* ``kic_proline_skipped`` -- the window (or the phi driver's residue r+3)
  contains a proline, whose phi KIC must not change.
* ``kic_geometry_invalid`` -- closures the solver DROPPED because one of the
  three N-CA-C angles missed its target by more than 1e-6 rad. Unlike the
  others this counts closures, not moves: one move can drop several.

Before the loop-closure fix ``kic_geometry_invalid`` existed but nothing ever
incremented it, so the old ceiling on it here passed trivially while the
solver was writing closures whose N-CA-C was off by up to ~40 degrees.

These tests check the counters are wired up (exist, start at zero, appear in
``move_stats()``) and, over a real trajectory, stay under engineering-judgment
rate ceilings. The ceilings are expressed as fractions of KIC attempts so the
test is stable regardless of step count or move-mix ratio -- they are not
physics baselines, just regression tripwires for a KIC solver in good health.
The geometry itself is checked directly in ``test_kic_closure_fixes.py``.
"""

from __future__ import annotations

import pytest

from pymcpu import mcpu_core
from tests.fixtures.context_builders import build_raw_context

pytestmark = pytest.mark.slow

_COUNTERS = (
    "kic_presolve_zero",
    "kic_reverse_missing",
    "kic_jacobian_invalid",
    "kic_geometry_invalid",
    "kic_proline_skipped",
)


def test_counters_exist_and_start_at_zero() -> None:
    """Every counter is accessible and zero before any run."""
    integ = mcpu_core.Integrator(temperature=0.6, step_size_rad=0.1)
    assert integ.get_kic_presolve_zero() == 0
    assert integ.get_kic_reverse_missing() == 0
    assert integ.get_kic_jacobian_invalid() == 0
    assert integ.get_kic_geometry_invalid() == 0
    assert integ.get_kic_proline_skipped() == 0


def test_move_stats_includes_kic_counters() -> None:
    """move_stats() dict contains every KIC counter key, all zero."""
    integ = mcpu_core.Integrator(temperature=0.6, step_size_rad=0.1)
    stats = integ.move_stats()
    for key in _COUNTERS:
        assert key in stats, key
        assert stats[key] == 0, key


def test_kic_counters_stay_low_during_normal_run() -> None:
    """Run a reference trajectory and verify rejection counters are near zero.

    Thresholds are expressed as fractions of KIC attempts so the test stays
    stable regardless of step count, move-type mix ratio, or how many KIC
    draws fall out-of-bounds.
    """
    ctx, _ = build_raw_context()
    n_steps = 10_000
    integ = mcpu_core.Integrator(temperature=0.6, step_size_rad=0.1)
    integ.set_seed(42)
    integ.run(ctx, n_steps)

    stats = integ.move_stats()
    kic_attempted = stats["num_propose_kic"]
    kic_accepted = stats["num_accept_kic"]
    rates = {
        key: (stats[key] / kic_attempted if kic_attempted else float("nan"))
        for key in _COUNTERS
    }

    print(f"\n--- KIC counter smoke test ({n_steps} steps) ---")
    print(f"  KIC attempted:        {kic_attempted}")
    print(f"  KIC accepted:         {kic_accepted}")
    for key in _COUNTERS:
        print(f"  {key + ':':22s}{stats[key]}  (rate: {rates[key]:.2e})")
    if kic_attempted > 0:
        print(f"  KIC acceptance rate:  {kic_accepted / kic_attempted:.3f}")
    print("---")

    assert kic_attempted > 0, (
        f"No KIC moves were attempted in {n_steps} steps -- "
        f"test cannot validate counters; increase n_steps or check move mix"
    )

    # Pre-solve returning 0 means the solver cannot reconstruct the current
    # state from its own endpoints. A tiny rate can appear near points where
    # two closure roots merge; a large rate indicates real corruption.
    assert rates["kic_presolve_zero"] < 1e-2, rates

    # Reverse check: at the default driver width (pi/6), 0.49 % of KIC
    # proposals in a 200k-step run from the start structure; over 20 seeds of
    # this 10k-step run, 0.34 % on average and at most 0.99 % (this seed:
    # 0.97 %). A wider driver raises the rate: 0.26 % at 0.1 rad in the same
    # 200k steps. A large rate means the current windows no longer match the
    # start-structure targets, i.e. the geometry has drifted (or the run was
    # started from a drifted structure).
    assert rates["kic_reverse_missing"] < 2e-2, rates

    # Near-singular Jacobians can occur at rare backbone geometries. A rate
    # above 0.1% would indicate a systematic problem.
    assert rates["kic_jacobian_invalid"] < 1e-3, rates

    # Closures the solver dropped for a wrong N-CA-C, per KIC attempt. The
    # pole-free back-substitution makes these rare (0 to 0.05 % in long runs);
    # 1 % would mean the solver has regressed.
    assert rates["kic_geometry_invalid"] < 1e-2, rates

    # Actin has prolines, so some windows must be skipped -- and a skip is a
    # KIC draw that proposes nothing, so it is also a KIC attempt at most.
    assert 0 < stats["kic_proline_skipped"] <= kic_attempted, stats
