"""Mu's pair walk on pivot-heavy runs: the shortcuts must not change a result.

A move's Mu energy change walks the moved atoms' new neighbours twice: the
clash-first pass (does the move overlap a fixed atom?) and the contact walk
(which pairs does it make?). Both skip work that cannot matter -- cells that
hold only moved atoms, slots a vector distance test rules out -- and the
clash-first pass visits the atoms most likely to overlap first. None of that
may change which moves are accepted or the energy change of any of them.

Actin pivots carry hundreds of atoms, so these runs go through every path
the default move mix uses on a large protein, including many rejected moves.
"""

from __future__ import annotations

from pymcpu import mcpu_core
from tests.fixtures.context_builders import ATOL, build_test_context

PIVOT_ONLY = (1.0, 0.0, 0.0)
NEVER = 10**9


def _run(context, steps: int, seed: int = 7):
    integrator = mcpu_core.Integrator(temperature=0.6, step_size_rad=0.1)
    integrator.set_move_weights(*PIVOT_ONLY)
    integrator.set_seed(seed)
    integrator.run(context, steps)
    return bytes(integrator.last_accept_bits()), context.get_state().current_energy


def test_clash_first_pass_agrees_with_contact_walk() -> None:
    # The clash-first pass is an early exit: every overlap it reports, the
    # contact walk would report too. Running it on every move and on none
    # must give the same trajectory, bit for bit.
    with_pass, _ = build_test_context()
    with_pass.set_clash_first_min_moved(0)
    without_pass, _ = build_test_context()
    without_pass.set_clash_first_min_moved(NEVER)

    bits_a, energy_a = _run(with_pass, 400)
    bits_b, energy_b = _run(without_pass, 400)

    assert 0 < sum(bits_a) < len(bits_a), "need accepted and rejected moves"
    assert bits_a == bits_b
    assert energy_a == energy_b


def test_pivot_delta_matches_full_recompute() -> None:
    context, _ = build_test_context()
    context.set_clash_first_min_moved(0)
    integrator = mcpu_core.Integrator(temperature=0.6, step_size_rad=0.1)
    integrator.set_move_weights(*PIVOT_ONLY)
    integrator.set_seed(11)
    integrator.verify_physics_consistency(context, num_steps=60, atol=ATOL)
