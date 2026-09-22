"""Deterministic hbond delta-energy regression.

Guards the hbond neighbor/candidate bookkeeping hotpath (added when the
dedup scheme switched from an ``unordered_set`` to a stamp-based one):
same-seed runs must reproduce bit-identical accept sequences, energies, and
candidate/geometry-check counters. All comparisons are self-vs-self or
delegate to the internal ``verify_physics_consistency`` checker -- no legacy
MCPU comparison.
"""

from __future__ import annotations

import pytest

from pymcpu import mcpu_core
from tests.fixtures.context_builders import (
    ATOL,
    build_test_context,
    require_safe_math_for_accept_determinism,
)
from tests.physics.helpers.hotpath_runs import run_hotpath

pytestmark = pytest.mark.slow

# Repeatability allowance for two back-to-back identical-seed runs -- not a
# physics constant. Chosen empirically: tight enough that a real
# nondeterminism regression (e.g. an uninitialized read, an iteration-order
# dependency) still fails, loose enough to absorb this hardware's benign
# floating-point associativity noise across repeated runs.
DETERMINISM_ATOL = 1e-5


def test_hbond_delta_accept_bits_deterministic() -> None:
    require_safe_math_for_accept_determinism()
    a = run_hotpath(seed=7777, steps=150)
    b = run_hotpath(seed=7777, steps=150)
    assert a.bits == b.bits
    assert a.accept == b.accept
    assert a.energy == pytest.approx(b.energy, abs=DETERMINISM_ATOL)
    assert a.e_hbond == pytest.approx(b.e_hbond, abs=DETERMINISM_ATOL)
    assert a.proxy_stat("hbond_num_candidates_iterated") == b.proxy_stat(
        "hbond_num_candidates_iterated"
    )
    assert a.proxy_stat("hbond_num_geom_checks") == b.proxy_stat("hbond_num_geom_checks")


def test_hbond_actin_verify_skin0_and_skin1() -> None:
    """``verify_physics_consistency`` (delta-vs-recompute self-check, see
    ``tests/physics/test_energy_consistency.py``) must hold at both
    Verlet-skin extremes -- skin=0 (cell-list only, rebuilt every step) and
    skin=1.0 (Verlet list reused across steps) -- since the hbond hotpath
    takes a different neighbor-source code path in each regime."""
    for skin in (0.0, 1.0):
        ctx, _ = build_test_context(with_qbias=False)
        ctx.set_mu_skin(skin)
        integ = mcpu_core.Integrator(temperature=300.0, step_size_rad=0.05)
        integ.verify_physics_consistency(ctx, num_steps=30, atol=ATOL)
