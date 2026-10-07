"""H-bond ledger debug check.

HBondPotential takes the old side of each delta from the accepted state's
ledger of nonzero pair energies (State.hbond_cache) and scores only the new
side. ``Context.set_hbond_ledger_check(True)`` makes every delta also score
both states over all pairs that touch an affected residue, the way the code
did before the ledger, and count any pair where the two paths disagree on
whether it changed, on its new energy, or on its old energy. The count must
stay 0 through pivot, KIC and side-chain moves, and the virtual-H grid must
still match brute force after the accepted moves have updated it.
"""
from __future__ import annotations

import pytest

from pymcpu import mcpu_core
from tests.fixtures.context_builders import build_test_context


@pytest.mark.parametrize("weights", [(0.25, 0.25, 0.5), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)])
def test_hbond_ledger_matches_rescoring(weights) -> None:
    ctx, _ = build_test_context(with_qbias=False)
    ctx.set_hbond_ledger_check(True)
    integ = mcpu_core.Integrator(temperature=300.0, step_size_rad=0.05)
    integ.set_move_weights(*weights)
    integ.set_seed(2024)
    integ.run(ctx, 200)
    checks, mismatches = ctx.hbond_ledger_check_counts()
    assert checks > 0
    assert mismatches == 0
    assert ctx.hbond_index_ok()


def test_hbond_ledger_long_pivot_run() -> None:
    """A long pivot-only run, where rigid sites carry listed pairs across many
    accepted moves within their slack (HydrogenBondPotential.cpp, Slack): every
    carried energy must still match a fresh score."""
    ctx, _ = build_test_context(with_qbias=False)
    ctx.set_hbond_ledger_check(True)
    integ = mcpu_core.Integrator(temperature=300.0, step_size_rad=0.05)
    integ.set_move_weights(1.0, 0.0, 0.0)
    integ.set_seed(7)
    integ.run(ctx, 2000)
    checks, mismatches = ctx.hbond_ledger_check_counts()
    assert checks > 0
    assert mismatches == 0
