"""Proline move-proposal policy: resample instead of reject.

Proline's ring locks its own phi torsion and lacks a rotatable chi at the
first sidechain position, so pyMCPU's move proposer treats PRO specially:
rather than proposing a phi-pivot or sidechain move and then rejecting it
post-hoc (which would count as a wasted/degenerate proposal), it *resamples*
a different move up front. This file checks that policy end-to-end: which
residues get flagged proline, which forced move types resample vs. propose
normally, and that a real run redraws proline phi pivots but never draws a
proline sidechain at all (the sidechain move draws only residues it can move). Purely an internal move-selection policy check -- no legacy MCPU
reference value is involved.
"""

from __future__ import annotations

import pytest

from pymcpu import mcpu_core
from tests.fixtures.context_builders import build_raw_context

pytestmark = pytest.mark.slow


def _pro_residues(topology) -> list[int]:
    return [r.index for r in topology.residues if r.name == "PRO"]


def test_proline_flags_set() -> None:
    ctx, top = build_raw_context(virtual_amide_h=True)
    pros = _pro_residues(top)
    assert pros, "test PDB should contain proline"
    for r in pros:
        assert ctx.get_system().is_proline(r)
    for r in range(ctx.get_system().get_num_residues()):
        if r not in pros:
            assert not ctx.get_system().is_proline(r)


def test_force_phi_at_pro_resamples() -> None:
    ctx, top = build_raw_context(virtual_amide_h=True)
    pros = _pro_residues(top)
    # Prefer an interior PRO so the pivot residue range is valid.
    pro = next(r for r in pros if 1 <= r <= ctx.get_system().get_num_residues() - 2)
    integ = mcpu_core.Integrator(temperature=300.0, step_size_rad=0.05)
    integ.set_seed(7)
    before = integ.num_pivot_resample_pro_phi()
    proposed = integ.debug_force_pivot(ctx, pro, True)
    assert proposed is False
    assert integ.num_pivot_resample_pro_phi() == before + 1


def test_force_psi_at_pro_allowed() -> None:
    ctx, top = build_raw_context(virtual_amide_h=True)
    pros = _pro_residues(top)
    pro = next(r for r in pros if 1 <= r <= ctx.get_system().get_num_residues() - 2)
    integ = mcpu_core.Integrator(temperature=300.0, step_size_rad=0.05)
    integ.set_seed(7)
    before = integ.num_pivot_resample_pro_phi()
    proposed = integ.debug_force_pivot(ctx, pro, False)
    assert proposed is True
    assert integ.num_pivot_resample_pro_phi() == before


def test_force_sc_at_pro_resamples() -> None:
    ctx, top = build_raw_context(virtual_amide_h=True)
    pro = _pro_residues(top)[0]
    integ = mcpu_core.Integrator(temperature=300.0, step_size_rad=0.05)
    integ.set_seed(7)
    before = integ.num_sc_resample_pro()
    proposed = integ.debug_force_sc(ctx, pro)
    assert proposed is False
    assert integ.num_sc_resample_pro() == before + 1


def test_non_pro_sc_unchanged_policy() -> None:
    ctx, top = build_raw_context(virtual_amide_h=True)
    n_res = ctx.get_system().get_num_residues()
    non_pro = next(
        r
        for r in range(n_res)
        if not ctx.get_system().is_proline(r)
        and ctx.get_system().get_block_indices()[r].sc_count >= 2
    )
    integ = mcpu_core.Integrator(temperature=300.0, step_size_rad=0.05)
    integ.set_seed(11)
    before = integ.num_sc_resample_pro()
    proposed = integ.debug_force_sc(ctx, non_pro)
    assert proposed is True
    assert integ.num_sc_resample_pro() == before


@pytest.mark.parametrize("reorder", ["off", "init_only"])
def test_run_redraws_pro_phi_and_never_draws_a_pro_sidechain(reorder: str) -> None:
    ctx, top = build_raw_context(virtual_amide_h=True, reorder=reorder)
    assert _pro_residues(top)
    integ = mcpu_core.Integrator(temperature=0.6, step_size_rad=0.1)
    integ.set_seed(42)
    integ.run(ctx, 400)
    # With actin-sized PRO content the phi counter fires over 400 steps
    # regardless of whether atom reordering is applied. The sidechain move
    # draws only residues it can move, so a run never lands on a proline
    # sidechain; that counter counts refused debug_force_* calls only.
    assert integ.num_pivot_resample_pro_phi() > 0
    assert integ.num_sc_resample_pro() == 0
