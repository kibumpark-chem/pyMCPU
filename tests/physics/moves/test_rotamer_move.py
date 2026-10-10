"""Rotamer-library sidechain move: proposal policy, counters, and physics.

Mirrors ``test_kic_counters.py``/``test_proline_skip_policy.py``'s structure
for the new discrete rotamer-library sidechain move
(``sidechain_move_mode="rotamer_library"``, ``MCIntegrator::apply_rotamer_at``).
Purely an internal check of pyMCPU's own move-proposal machinery and its
detailed-balance-relevant invariants -- legacy MCPU's own rotamer path has a
confirmed, uncorrected detailed-balance gap (see the design plan), so there
is no legacy reference behavior to compare against here.
"""

from __future__ import annotations

import pytest

from pymcpu import mcpu_core
from tests.fixtures.context_builders import build_raw_context

pytestmark = pytest.mark.slow


def _pro_residues(topology) -> list[int]:
    return [r.index for r in topology.residues if r.name == "PRO"]


def _gly_ala_residues(ctx) -> list[int]:
    system = ctx.get_system()
    return [
        r
        for r in range(system.get_num_residues())
        if system.get_torsions_per_residue()[r] == 0
    ]


def _multi_chi_residue(ctx) -> int:
    """First residue with >= 2 chi angles (a real cascading-rotation case)."""
    system = ctx.get_system()
    for r in range(system.get_num_residues()):
        if system.get_torsions_per_residue()[r] >= 2 and not system.is_proline(r):
            return r
    pytest.skip("test PDB has no multi-chi residue")


def test_move_stats_includes_rotamer_counters() -> None:
    integ = mcpu_core.Integrator(temperature=0.6, step_size_rad=0.1)
    stats = integ.move_stats()
    assert "num_propose_rotamer" in stats
    assert "num_accept_rotamer" in stats
    assert stats["num_propose_rotamer"] == 0
    assert stats["num_accept_rotamer"] == 0


def test_debug_force_rotamer_on_multi_chi_residue_succeeds() -> None:
    ctx, top = build_raw_context(virtual_amide_h=True)
    residue = _multi_chi_residue(ctx)
    integ = mcpu_core.Integrator(temperature=300.0, step_size_rad=0.1)
    integ.set_seed(7)
    proposed = integ.debug_force_rotamer(ctx, residue)
    assert proposed is True


def test_debug_force_rotamer_at_pro_resamples() -> None:
    ctx, top = build_raw_context(virtual_amide_h=True)
    pros = _pro_residues(top)
    assert pros, "test PDB should contain proline"
    pro = pros[0]
    integ = mcpu_core.Integrator(temperature=300.0, step_size_rad=0.1)
    integ.set_seed(7)
    before = integ.num_sc_resample_pro()
    proposed = integ.debug_force_rotamer(ctx, pro)
    assert proposed is False
    assert integ.num_sc_resample_pro() == before + 1


def test_debug_force_rotamer_at_gly_ala_rejects_without_counting() -> None:
    """Gly/Ala (ntorsions=0) have nothing to propose -- rejected exactly
    like the continuous move's num_sc_atoms<2 fast-reject, without
    incrementing the (proline-specific) resample counter."""
    ctx, top = build_raw_context(virtual_amide_h=True)
    no_chi = _gly_ala_residues(ctx)
    assert no_chi, "test PDB should contain a Gly or Ala residue"
    residue = no_chi[0]
    integ = mcpu_core.Integrator(temperature=300.0, step_size_rad=0.1)
    integ.set_seed(7)
    before = integ.num_sc_resample_pro()
    proposed = integ.debug_force_rotamer(ctx, residue)
    assert proposed is False
    assert integ.num_sc_resample_pro() == before


def test_rotamer_run_only_moves_sidechain_atoms() -> None:
    """Over a real run entirely in rotamer_library mode, every residue's
    cached backbone torsions (phi/psi/pCA/bCA) must stay bit-identical --
    this move never touches backbone atoms nor calls
    recompute_backbone_torsion.

    Note: sidechain_move_mode only changes what runs inside the "Sidechain"
    slot of the pivot(25%)/KIC(25%)/sidechain(50%) move-mix -- pivot/KIC
    moves still occur and DO change backbone torsions. So this steps one MC
    move at a time and only checks the invariant across steps where
    ``last_move_kind() == "Sidechain"`` (set correctly whenever the patch
    was valid, i.e. proposed-and-either-accepted-or-rejected -- an invalid/
    resampled-away step never touches ``ctx``'s committed state at all,
    regardless of what kind it reports)."""
    ctx, _ = build_raw_context(virtual_amide_h=True)
    integ = mcpu_core.Integrator(temperature=0.6, step_size_rad=0.1)
    integ.set_sidechain_move_mode("rotamer_library")
    integ.set_seed(42)

    n_sidechain_steps = 0
    for _ in range(400):
        before = [
            (t.phi, t.psi, t.p_ca, t.b_ca) for t in ctx.get_state().backbone_torsions
        ]
        integ.run(ctx, 1)
        if integ.last_move_kind() != "Sidechain":
            continue
        n_sidechain_steps += 1
        after = [
            (t.phi, t.psi, t.p_ca, t.b_ca) for t in ctx.get_state().backbone_torsions
        ]
        assert after == before, (
            "a Sidechain-slot (rotamer-library) step changed backbone torsions"
        )

    assert n_sidechain_steps > 0, "no Sidechain-slot steps occurred"
    assert integ.move_stats()["num_propose_rotamer"] > 0


def test_default_mode_matches_explicit_rotamer_library() -> None:
    """A run that never calls set_sidechain_move_mode must be byte-identical
    to one that explicitly selects "rotamer_library" -- pins that
    rotamer_library really is the default (MCIntegrator's C++ default and
    IntegratorConfig/WEConfig's Python defaults)."""
    ctx_a, _ = build_raw_context(virtual_amide_h=True)
    ctx_b, _ = build_raw_context(virtual_amide_h=True)

    integ_default = mcpu_core.Integrator(temperature=0.6, step_size_rad=0.1)
    integ_default.set_seed(123)
    integ_default.run(ctx_a, 500)

    integ_explicit = mcpu_core.Integrator(temperature=0.6, step_size_rad=0.1)
    integ_explicit.set_sidechain_move_mode("rotamer_library")
    integ_explicit.set_seed(123)
    integ_explicit.run(ctx_b, 500)

    assert list(integ_default.last_accept_bits()) == list(
        integ_explicit.last_accept_bits()
    )
    coords_a = ctx_a.get_state().coords
    coords_b = ctx_b.get_state().coords
    assert (coords_a == coords_b).all()


def test_continuous_mode_is_still_available_and_differs_from_default() -> None:
    """Explicitly selecting "continuous" must still work (it's the
    alternative, non-default sidechain-move algorithm) and must diverge from
    the rotamer_library default given the same seed -- confirms the two
    modes are genuinely different code paths, not just differently-labeled
    aliases of the same behavior."""
    ctx_default, _ = build_raw_context(virtual_amide_h=True)
    ctx_continuous, _ = build_raw_context(virtual_amide_h=True)

    integ_default = mcpu_core.Integrator(temperature=0.6, step_size_rad=0.1)
    integ_default.set_seed(123)
    integ_default.run(ctx_default, 200)

    integ_continuous = mcpu_core.Integrator(temperature=0.6, step_size_rad=0.1)
    integ_continuous.set_sidechain_move_mode("continuous")
    integ_continuous.set_seed(123)
    integ_continuous.run(ctx_continuous, 200)

    assert integ_continuous.move_stats()["num_propose_sc"] > 0
    assert integ_continuous.move_stats()["num_propose_rotamer"] == 0
    assert not (ctx_default.get_state().coords == ctx_continuous.get_state().coords).all()
