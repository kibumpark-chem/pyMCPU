"""Deterministic mu (contact/pair-potential) delta-energy regression.

Same shape as ``test_hbond_delta_hotpath.py`` but for the mu-potential
neighbor hotpath: same-seed runs must reproduce bit-identical accept
sequences, energies, and pair-distance-check counters at both Verlet-skin
extremes. All comparisons are self-vs-self or delegate to the internal
``verify_physics_consistency`` checker -- no legacy MCPU comparison.
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

# See tests/physics/test_hbond_delta_hotpath.py's DETERMINISM_ATOL for why
# this is an empirical repeatability allowance, not a physics constant.
DETERMINISM_ATOL = 1e-5


def test_mu_delta_accept_bits_deterministic() -> None:
    require_safe_math_for_accept_determinism()
    a = run_hotpath(seed=4242, steps=200)
    b = run_hotpath(seed=4242, steps=200)
    assert a.bits == b.bits
    assert a.accept == b.accept
    assert a.energy == pytest.approx(b.energy, abs=DETERMINISM_ATOL)
    assert a.proxy_stat("mu_num_pairs_within_rcut") == b.proxy_stat(
        "mu_num_pairs_within_rcut"
    )
    assert a.proxy_stat("mu_num_pair_distance_checks") == b.proxy_stat(
        "mu_num_pair_distance_checks"
    )


def test_mu_delta_skin1_accept_bits_deterministic() -> None:
    """Same determinism check as above, at skin=1.0 (Verlet-list reuse
    across steps) -- the mu hotpath sources candidate pairs from a
    differently-maintained structure there than at skin=0."""
    require_safe_math_for_accept_determinism()
    a = run_hotpath(seed=4242, steps=200, skin=1.0)
    b = run_hotpath(seed=4242, steps=200, skin=1.0)
    assert a.bits == b.bits
    assert a.proxy_stat("mu_num_pairs_within_rcut") == b.proxy_stat(
        "mu_num_pairs_within_rcut"
    )


def test_actin_verify_after_mu_hotpath() -> None:
    for skin in (0.0, 1.0):
        ctx, _ = build_test_context(with_qbias=False)
        ctx.set_mu_skin(skin)
        integ = mcpu_core.Integrator(temperature=300.0, step_size_rad=0.05)
        integ.verify_physics_consistency(ctx, num_steps=20, atol=ATOL)
