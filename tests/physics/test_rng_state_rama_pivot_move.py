"""RNG checkpoint/restore round trip through a real rama-pivot-mode run.

Structural mirror of ``test_rng_state_rotamer_move.py`` for the
knowledge-based backbone pivot move. The rama-pivot move draws one
``coin_flip`` (mixture-component row) and TWO ``unit_normal_dist_`` draws
(the bivariate-normal target, via the component's Cholesky factor) per
attempt -- reusing the same cache-bearing distribution member the
rotamer-library move already exercises, so this pins that a future change
adding a THIRD consumer of ``unit_normal_dist_`` doesn't reopen the
spare-Gaussian-cache-across-checkpoint hazard that member's docs describe.
"""

from __future__ import annotations

import pytest

from pymcpu import mcpu_core
from tests.fixtures.context_builders import build_raw_context

pytestmark = pytest.mark.slow


def _inject_synthetic_rama_mixture(system) -> None:
    lib = mcpu_core.RamaMixtureLibrary(n_wrap=1)
    for amino_idx in range(20):
        lib.add_residue_type(amino_idx, [1.0], [[0.3, -0.5]], [[0.02, 0.0, 0.02]])
    system.set_rama_mixture_library(lib)


def test_rama_pivot_mode_checkpoint_restore_round_trip() -> None:
    seed = 4242
    n_warmup = 300
    n_continue = 300

    ctx1, _ = build_raw_context(virtual_amide_h=True)
    _inject_synthetic_rama_mixture(ctx1.get_system())
    integ1 = mcpu_core.Integrator(temperature=0.6, step_size_rad=0.1)
    integ1.set_pivot_rama_probability(1.0)
    integ1.set_seed(seed)
    integ1.run(ctx1, n_warmup)
    checkpoint = integ1.get_rng_state()
    integ1.run(ctx1, n_continue)
    accept_bits_ref = list(integ1.last_accept_bits())
    coords_ref = ctx1.get_state().coords.copy()

    # Reproduce ctx1's state-at-checkpoint on ctx2 via an independent,
    # identically-seeded replay.
    ctx2, _ = build_raw_context(virtual_amide_h=True)
    _inject_synthetic_rama_mixture(ctx2.get_system())
    integ2 = mcpu_core.Integrator(temperature=0.6, step_size_rad=0.1)
    integ2.set_pivot_rama_probability(1.0)
    integ2.set_seed(seed)
    integ2.run(ctx2, n_warmup)

    # Dirty integ2's rng/cache with unrelated draws, then force it back to
    # the checkpoint -- if reset() is missing, stale cache contamination
    # from this unrelated history would leak into the continuation below.
    ctx_dirty, _ = build_raw_context(virtual_amide_h=True)
    _inject_synthetic_rama_mixture(ctx_dirty.get_system())
    integ2.set_seed(seed + 1)
    integ2.run(ctx_dirty, 137)
    integ2.set_rng_state(checkpoint)

    integ2.run(ctx2, n_continue)
    assert list(integ2.last_accept_bits()) == accept_bits_ref, (
        "restored rama-pivot-mode integrator's accept-bit stream diverged "
        "from the original continuation -- possible RNG cache leak across "
        "the checkpoint boundary"
    )
    assert (ctx2.get_state().coords == coords_ref).all()


def test_rama_pivot_schedule_checkpoint_restore_round_trip() -> None:
    """Same round trip, but pivot_rama_probability set via a non-degenerate
    schedule (0 < p < 1) so the extra kernel-selection coin_flip is also
    exercised across the checkpoint boundary."""
    seed = 777
    n_warmup = 300
    n_continue = 300

    ctx1, _ = build_raw_context(virtual_amide_h=True)
    _inject_synthetic_rama_mixture(ctx1.get_system())
    integ1 = mcpu_core.Integrator(temperature=0.6, step_size_rad=0.1)
    integ1.set_pivot_rama_schedule(t_low=0.4, t_high=1.0, p_min=0.05, p_max=0.35)
    integ1.set_seed(seed)
    integ1.run(ctx1, n_warmup)
    checkpoint = integ1.get_rng_state()
    integ1.run(ctx1, n_continue)
    accept_bits_ref = list(integ1.last_accept_bits())

    ctx2, _ = build_raw_context(virtual_amide_h=True)
    _inject_synthetic_rama_mixture(ctx2.get_system())
    integ2 = mcpu_core.Integrator(temperature=0.6, step_size_rad=0.1)
    integ2.set_pivot_rama_schedule(t_low=0.4, t_high=1.0, p_min=0.05, p_max=0.35)
    integ2.set_seed(seed)
    integ2.run(ctx2, n_warmup)

    ctx_dirty, _ = build_raw_context(virtual_amide_h=True)
    _inject_synthetic_rama_mixture(ctx_dirty.get_system())
    integ2.set_seed(seed + 1)
    integ2.run(ctx_dirty, 137)
    integ2.set_rng_state(checkpoint)

    integ2.run(ctx2, n_continue)
    assert list(integ2.last_accept_bits()) == accept_bits_ref
