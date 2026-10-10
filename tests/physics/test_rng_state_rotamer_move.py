"""RNG checkpoint/restore round trip through a real rotamer-mode run.

A plain RNG-state round trip never constructs a second
cache-bearing ``std::normal_distribution`` (it only ever touches ``rng``
directly), so it can't catch a forgotten ``unit_normal_dist_.reset()`` in
``get_rng_state()``/``set_rng_state()`` -- the rotamer-library sidechain
move's per-chi Gaussian noise is the only thing that ever calls
``unit_normal_dist_``. This test exercises that path directly.
"""

from __future__ import annotations

import pytest

from pymcpu import mcpu_core
from tests.fixtures.context_builders import build_raw_context

pytestmark = pytest.mark.slow


def test_rotamer_mode_checkpoint_restore_round_trip() -> None:
    """A restored integrator must reproduce the exact continuation of the
    original -- including cases where the checkpoint moment happens to fall
    right after unit_normal_dist_ cached a spare Gaussian.

    integ2 is deliberately given an UNRELATED rng/cache history (via
    ctx_dirty) before restoring, so this can't pass merely because two
    identical-seed replicas stay in lockstep by construction -- that would
    hide a "both get_rng_state and set_rng_state forgot to reset" bug (the
    two replicas would trivially agree with each other while still being
    wrong relative to a genuinely fresh restore).
    """
    seed = 4242
    n_warmup = 300
    n_continue = 300

    ctx1, _ = build_raw_context(virtual_amide_h=True)
    integ1 = mcpu_core.Integrator(temperature=0.6, step_size_rad=0.1)
    integ1.set_sidechain_move_mode("rotamer_library")
    integ1.set_seed(seed)
    integ1.run(ctx1, n_warmup)
    checkpoint = integ1.get_rng_state()
    integ1.run(ctx1, n_continue)
    accept_bits_ref = list(integ1.last_accept_bits())
    coords_ref = ctx1.get_state().coords.copy()

    # Reproduce ctx1's state-at-checkpoint on ctx2 via an independent,
    # identically-seeded replay.
    ctx2, _ = build_raw_context(virtual_amide_h=True)
    integ2 = mcpu_core.Integrator(temperature=0.6, step_size_rad=0.1)
    integ2.set_sidechain_move_mode("rotamer_library")
    integ2.set_seed(seed)
    integ2.run(ctx2, n_warmup)

    # Dirty integ2's rng/cache with unrelated draws, then force it back to
    # the checkpoint -- if reset() is missing, stale cache contamination
    # from this unrelated history would leak into the continuation below.
    ctx_dirty, _ = build_raw_context(virtual_amide_h=True)
    integ2.set_seed(seed + 1)
    integ2.run(ctx_dirty, 137)
    integ2.set_rng_state(checkpoint)

    integ2.run(ctx2, n_continue)
    assert list(integ2.last_accept_bits()) == accept_bits_ref, (
        "restored rotamer-mode integrator's accept-bit stream diverged from "
        "the original continuation -- possible RNG cache leak across the "
        "checkpoint boundary"
    )
    assert (ctx2.get_state().coords == coords_ref).all()
