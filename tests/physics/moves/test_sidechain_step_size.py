"""Independent sidechain chi amplitude.

MCPU's Monte Carlo has two natural step sizes, and legacy keeps them as two
separate config knobs:

    MC_STEP_SIZE     2 deg   -> backbone torsions  (init.h:1487, STEP_SIZE)
    SIDECHAIN_NOISE 10 deg   -> sidechain chi      (init.h:1489)

Sidechain steps are 5x larger on purpose: rotating a chi displaces only a few
sidechain atoms, while rotating a backbone torsion swings a whole chain
segment, so the same angular step is a far bigger structural move on the
backbone.

pyMCPU used a single ``step_size_rad`` for the pivot move, the KIC driver angle
AND the continuous sidechain move, so it could match one of legacy's two
amplitudes but never both -- which made a like-for-like benchmark against
legacy impossible (and silently mismatched: the published 4.3 runs used
0.1 rad = 5.7 deg for everything, 2.9x too large on the backbone and 1.75x too
small on chi).

``sidechain_step_size_rad`` splits them. It defaults to "same as backbone", so
omitting it reproduces the previous single-amplitude behavior. The KIC driver
has its own width, ``kic_step_size_rad``, so ``step_size_rad`` sets the pivot
only.

Internal-consistency test: every assertion compares pyMCPU against its own
behavior at a different setting. No legacy binary, log, or hardcoded legacy
number is read -- only legacy's *design* (two knobs) motivates the feature.
"""

from __future__ import annotations

import math

import pytest

from pymcpu import mcpu_core
from tests.fixtures.context_builders import build_test_context

DEG = math.pi / 180.0


def _integrator(backbone_rad: float, sidechain_rad: float | None = None):
    if sidechain_rad is None:
        return mcpu_core.Integrator(temperature=0.6, step_size_rad=backbone_rad)
    return mcpu_core.Integrator(temperature=0.6, step_size_rad=backbone_rad,
                                sidechain_step_size_rad=sidechain_rad)


def test_defaults_to_the_backbone_amplitude() -> None:
    """Omitting the argument must reproduce the old single-amplitude behavior."""
    integ = _integrator(0.1)
    assert integ.backbone_step_size_rad() == pytest.approx(0.1)
    assert integ.sidechain_step_size_rad() == pytest.approx(0.1)


def test_explicit_amplitude_is_kept_independently() -> None:
    integ = _integrator(2.0 * DEG, 10.0 * DEG)
    assert integ.backbone_step_size_rad() == pytest.approx(2.0 * DEG)
    assert integ.sidechain_step_size_rad() == pytest.approx(10.0 * DEG)
    # The legacy ratio, which a single shared amplitude cannot express.
    assert integ.sidechain_step_size_rad() / integ.backbone_step_size_rad() == pytest.approx(5.0)


def test_negative_restores_same_as_backbone() -> None:
    integ = _integrator(0.05, 0.3)
    assert integ.sidechain_step_size_rad() == pytest.approx(0.3)
    integ.set_sidechain_step_size_rad(-1.0)
    assert integ.sidechain_step_size_rad() == pytest.approx(0.05)


def _accepts(sidechain_rad: float, mode: str, steps: int = 4000) -> int:
    ctx, _ff = build_test_context()
    integ = _integrator(0.1, sidechain_rad)
    integ.set_sidechain_move_mode(mode)
    integ.set_seed(4242)
    integ.run(ctx, steps)
    return int(sum(integ.last_accept_bits()))


def test_larger_sidechain_steps_are_accepted_less_often() -> None:
    """The knob must actually reach the proposal, not just be stored.

    Bigger chi perturbations land further up the energy surface, so acceptance
    has to fall. This is the cheapest end-to-end proof that
    ``sc_angle_dist_`` is wired into ``apply_sidechain_at``.
    """
    small = _accepts(2.0 * DEG, "continuous")
    large = _accepts(30.0 * DEG, "continuous")
    assert small > large, (
        f"acceptance did not fall when the sidechain amplitude grew 15x "
        f"(2 deg -> {small} accepts, 30 deg -> {large} accepts); the amplitude "
        f"is probably not reaching the proposal"
    )


def test_rotamer_mode_ignores_the_sidechain_amplitude() -> None:
    """The rotamer-library move takes its per-chi widths from the library rows.

    So in ``rotamer_library`` mode the knob must be completely inert -- two runs
    differing only in ``sidechain_step_size_rad`` must produce bit-identical
    accept sequences. If this ever fails, the amplitude has leaked into the
    rotamer path, where it would corrupt the Metropolis-Hastings correction
    (``log_mixture_density`` scores against the library's sigmas, so sampling
    with any other width makes the proposal density wrong).
    """
    ctx_a, _ff_a = build_test_context()
    integ_a = _integrator(0.1, 2.0 * DEG)
    integ_a.set_sidechain_move_mode("rotamer_library")
    integ_a.set_seed(99)
    integ_a.run(ctx_a, 3000)

    ctx_b, _ff_b = build_test_context()
    integ_b = _integrator(0.1, 30.0 * DEG)
    integ_b.set_sidechain_move_mode("rotamer_library")
    integ_b.set_seed(99)
    integ_b.run(ctx_b, 3000)

    assert list(integ_a.last_accept_bits()) == list(integ_b.last_accept_bits()), (
        "sidechain_step_size_rad changed a rotamer_library run; it must only "
        "affect the continuous sidechain proposal"
    )


def test_backbone_amplitude_still_drives_the_pivot() -> None:
    """Splitting the amplitudes must not have detached the backbone one.

    Two runs differing only in ``step_size_rad`` (sidechain amplitude pinned to
    the same value in both) must diverge, which they can only do through the
    pivot / KIC driver.
    """
    def accepts(backbone_rad: float) -> int:
        ctx, _ff = build_test_context()
        integ = _integrator(backbone_rad, 5.0 * DEG)
        integ.set_sidechain_move_mode("continuous")
        integ.set_seed(7)
        integ.run(ctx, 4000)
        return int(sum(integ.last_accept_bits()))

    assert accepts(1.0 * DEG) > accepts(20.0 * DEG), (
        "acceptance did not fall when only the BACKBONE amplitude grew; "
        "step_size_rad may no longer reach the pivot / KIC driver"
    )


def test_physics_consistency_with_split_amplitudes() -> None:
    """Incremental and full-recompute energies must still agree.

    A new distribution object in the proposal path is exactly the kind of change
    that can desync the incremental energy bookkeeping, so run the engine's own
    consistency verifier at legacy's amplitudes.
    """
    ctx, _ff = build_test_context()
    integ = _integrator(2.0 * DEG, 10.0 * DEG)
    integ.set_sidechain_move_mode("continuous")
    integ.set_seed(11)
    integ.verify_physics_consistency(ctx, 300, 1e-3)
