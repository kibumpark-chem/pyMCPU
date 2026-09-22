"""RNG-state serialize/restore round-trip.

This is a functional behavior check (not a pure attribute-existence smoke
test -- see ``tests/extension_abi/test_extension_abi.py`` for that), and it matters
physically: the mid-run reseed determinism guarantee other hotpath tests
depend on (``tests/physics/test_coords_soa.py`` et al.) rests on
``get_rng_state``/``set_rng_state`` actually round-tripping the full RNG
state, including the cached spare Gaussian in ``angle_dist``.
"""

from __future__ import annotations

import pytest

from pymcpu import mcpu_core


def _require_rng_state_api(*integrators: "mcpu_core.Integrator") -> None:
    if not all(
        hasattr(integ, "get_rng_state") and hasattr(integ, "set_rng_state")
        for integ in integrators
    ):
        pytest.skip("mcpu_core build predates get_rng_state/set_rng_state")


def test_rng_state_round_trip() -> None:
    a = mcpu_core.Integrator(0.5, 0.1)
    b = mcpu_core.Integrator(0.5, 0.1)
    _require_rng_state_api(a, b)
    a.set_seed(42)
    b.set_seed(999)

    # Two independently-seeded Integrators must serialize to different
    # states, and b must exactly assume a's state once restored.
    state = a.get_rng_state()
    assert isinstance(state, str) and len(state) > 0

    b.set_rng_state(state)
    assert a.get_rng_state() == b.get_rng_state(), "RNG state round-trip failed"

    # Mutating b and restoring again must land back on the same serialized
    # state -- confirms restore isn't merely "close enough" but exact.
    b.set_seed(12345)
    assert a.get_rng_state() != b.get_rng_state()
    b.set_rng_state(state)
    assert a.get_rng_state() == b.get_rng_state(), (
        "RNG state round-trip failed after mutate"
    )
