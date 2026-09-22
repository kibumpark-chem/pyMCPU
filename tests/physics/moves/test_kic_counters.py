"""KIC (kinematic-closure) detailed-balance rejection counters.

The Integrator tracks three failure modes for its KIC pivot moves:

* ``kic_presolve_zero`` -- the solver couldn't reconstruct the current state
  from its own endpoints (a detailed-balance sanity re-solve).
* ``kic_jacobian_invalid`` -- a near-singular Jacobian at the proposed
  geometry.
* ``kic_geometry_invalid`` -- extraneous polynomial roots that would break
  frozen bond lengths/angles.

These tests check the counters are wired up (exist, start at zero, appear in
``move_stats()``) and, over a real trajectory, stay under engineering-judgment
rate ceilings. The ceilings are expressed as fractions of KIC attempts so the
test is stable regardless of step count or move-mix ratio -- they are not
physics baselines, just regression tripwires for a KIC solver in good health.
This is purely an internal check of pyMCPU's own move-proposal machinery;
there is no legacy MCPU equivalent to compare against (legacy uses a
different, non-KIC pivot scheme).
"""

from __future__ import annotations

import pytest

from pymcpu import mcpu_core
from tests.fixtures.context_builders import build_raw_context

pytestmark = pytest.mark.slow


def test_counters_exist_and_start_at_zero() -> None:
    """New counters are accessible and zero before any run."""
    integ = mcpu_core.Integrator(temperature=0.6, step_size_rad=0.1)
    assert integ.get_kic_presolve_zero() == 0
    assert integ.get_kic_jacobian_invalid() == 0
    assert integ.get_kic_geometry_invalid() == 0


def test_move_stats_includes_kic_counters() -> None:
    """move_stats() dict contains the KIC rejection counter keys."""
    integ = mcpu_core.Integrator(temperature=0.6, step_size_rad=0.1)
    stats = integ.move_stats()
    assert "kic_presolve_zero" in stats
    assert "kic_jacobian_invalid" in stats
    assert "kic_geometry_invalid" in stats
    assert stats["kic_presolve_zero"] == 0
    assert stats["kic_jacobian_invalid"] == 0
    assert stats["kic_geometry_invalid"] == 0


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
    presolve_zero = stats["kic_presolve_zero"]
    jacobian_invalid = stats["kic_jacobian_invalid"]
    geometry_invalid = stats["kic_geometry_invalid"]

    presolve_rate = presolve_zero / kic_attempted if kic_attempted else float("nan")
    jacobian_rate = jacobian_invalid / kic_attempted if kic_attempted else float("nan")
    geometry_rate = geometry_invalid / kic_attempted if kic_attempted else float("nan")

    print(f"\n--- KIC counter smoke test ({n_steps} steps) ---")
    print(f"  KIC attempted:        {kic_attempted}")
    print(f"  KIC accepted:         {kic_accepted}")
    print(f"  kic_presolve_zero:    {presolve_zero}  (rate: {presolve_rate:.2e})")
    print(f"  kic_jacobian_invalid: {jacobian_invalid}  (rate: {jacobian_rate:.2e})")
    print(f"  kic_geometry_invalid: {geometry_invalid}  (rate: {geometry_rate:.2e})")
    if kic_attempted > 0:
        print(f"  KIC acceptance rate:  {kic_accepted / kic_attempted:.3f}")
    print("---")

    assert kic_attempted > 0, (
        f"No KIC moves were attempted in {n_steps} steps -- "
        f"test cannot validate counters; increase n_steps or check move mix"
    )

    # Pre-solve returning 0 means the solver cannot reconstruct the current
    # state from its own endpoints. A tiny rate can appear from float
    # geometry; a large rate indicates real drift/corruption.
    assert presolve_rate < 1e-2, (
        f"kic_presolve_zero={presolve_zero} (rate={presolve_rate:.2e}) -- "
        f"expected < 1%"
    )

    # Near-singular Jacobians can occur at rare backbone geometries. A rate
    # above 0.1% would indicate a systematic problem.
    assert jacobian_rate < 1e-3, (
        f"kic_jacobian_invalid={jacobian_invalid}/{kic_attempted} "
        f"(rate={jacobian_rate:.2e}) exceeds 0.1% threshold"
    )

    # Extraneous polynomial roots that break frozen bond lengths/angles. Some
    # rejections are expected; a majority of attempts failing the gate would
    # indicate the solver is systematically broken.
    assert geometry_rate < 0.2, (
        f"kic_geometry_invalid={geometry_invalid}/{kic_attempted} "
        f"(rate={geometry_rate:.2e}) exceeds 20% threshold"
    )
