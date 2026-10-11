"""Knowledge-based backbone (phi,psi) pivot move: proposal policy, counters,
and physics.

Mirrors ``test_rotamer_move.py``'s structure for the joint (phi,psi)
independence-sampler backbone move (``MCIntegrator::apply_rama_pivot_at``,
mixing probability ``pivot_rama_probability``). No real ``RamaMixtureLibrary``
parameter file is packaged yet (that's the Part 2 data-pipeline work) --
these tests inject a small synthetic mixture directly via
``System.set_rama_mixture_library`` so the move mechanics can be exercised and
verified independently of that pipeline.
"""

from __future__ import annotations

import numpy as np
import pytest

from pymcpu import mcpu_core
from tests.fixtures.context_builders import ATOL, build_raw_context

pytestmark = pytest.mark.slow


def _inject_synthetic_rama_mixture(system) -> None:
    """Registers a single-component (phi,psi) mixture for every one of the
    20 amino-acid categories -- enough to exercise the move mechanics
    without depending on the real fitted-parameter pipeline."""
    lib = mcpu_core.RamaMixtureLibrary(n_wrap=1)
    for amino_idx in range(20):
        lib.add_residue_type(amino_idx, [1.0], [[0.3, -0.5]], [[0.02, 0.0, 0.02]])
    system.set_rama_mixture_library(lib)


def _pro_residues(topology) -> list[int]:
    return [r.index for r in topology.residues if r.name == "PRO"]


def _interior_non_pro_residue(ctx) -> int:
    system = ctx.get_system()
    n = system.get_num_residues()
    for r in range(1, n - 1):
        if not system.is_proline(r):
            return r
    pytest.skip("test PDB has no interior non-proline residue")


def test_move_stats_includes_rama_pivot_counters() -> None:
    integ = mcpu_core.Integrator(temperature=0.6, step_size_rad=0.1)
    stats = integ.move_stats()
    assert "num_propose_rama_pivot" in stats
    assert "num_accept_rama_pivot" in stats
    assert stats["num_propose_rama_pivot"] == 0
    assert stats["num_accept_rama_pivot"] == 0


def test_default_pivot_rama_probability_is_zero() -> None:
    """The knowledge-based pivot ships OPT-IN.

    Measured acceptance is 0.003-0.013 (folded and expanded alike), so a
    nonzero default would spend p * 0.25 of every run's step budget at a
    few-per-thousand accept rate without saying so. See
    set_pivot_rama_probability's docs for the mechanism."""
    integ = mcpu_core.Integrator(temperature=0.6, step_size_rad=0.1)
    assert integ.pivot_rama_probability() == pytest.approx(0.0)


def test_pivot_rama_probability_rejects_out_of_range() -> None:
    integ = mcpu_core.Integrator(temperature=0.6, step_size_rad=0.1)
    with pytest.raises(Exception):
        integ.set_pivot_rama_probability(-0.1)
    with pytest.raises(Exception):
        integ.set_pivot_rama_probability(1.1)


def test_pivot_rama_schedule_evaluates_piecewise_linear() -> None:
    integ = mcpu_core.Integrator(temperature=0.7, step_size_rad=0.1)
    integ.set_pivot_rama_schedule(t_low=0.4, t_high=1.0, p_min=0.05, p_max=0.35)
    # temperature=0.7 is the midpoint of [0.4, 1.0] -> midpoint of [0.05, 0.35]
    assert integ.pivot_rama_probability() == pytest.approx(0.2, abs=1e-6)


def test_pivot_rama_schedule_clamps_outside_breakpoints() -> None:
    integ_cold = mcpu_core.Integrator(temperature=0.2, step_size_rad=0.1)
    integ_cold.set_pivot_rama_schedule(t_low=0.4, t_high=1.0, p_min=0.05, p_max=0.35)
    assert integ_cold.pivot_rama_probability() == pytest.approx(0.05)

    integ_hot = mcpu_core.Integrator(temperature=1.5, step_size_rad=0.1)
    integ_hot.set_pivot_rama_schedule(t_low=0.4, t_high=1.0, p_min=0.05, p_max=0.35)
    assert integ_hot.pivot_rama_probability() == pytest.approx(0.35)


def test_debug_force_rama_pivot_succeeds_on_non_proline() -> None:
    ctx, top = build_raw_context(virtual_amide_h=True)
    _inject_synthetic_rama_mixture(ctx.get_system())
    residue = _interior_non_pro_residue(ctx)
    integ = mcpu_core.Integrator(temperature=300.0, step_size_rad=0.1)
    integ.set_seed(7)
    proposed = integ.debug_force_rama_pivot(ctx, residue)
    assert proposed is True


def test_debug_force_rama_pivot_at_pro_resamples() -> None:
    ctx, top = build_raw_context(virtual_amide_h=True)
    _inject_synthetic_rama_mixture(ctx.get_system())
    pros = _pro_residues(top)
    assert pros, "test PDB should contain proline"
    pro = pros[0]
    integ = mcpu_core.Integrator(temperature=300.0, step_size_rad=0.1)
    integ.set_seed(7)
    before = integ.num_pivot_resample_pro_phi()
    proposed = integ.debug_force_rama_pivot(ctx, pro)
    assert proposed is False
    assert integ.num_pivot_resample_pro_phi() == before + 1


def test_rama_pivot_only_moves_targeted_residue_and_downstream() -> None:
    """Residue r's own chi angles must be numerically unchanged (proved:
    the phi/psi rotation axes both pass through backbone atoms, never
    touching sidechain-internal geometry beyond a common rigid rotation),
    and every residue strictly upstream of the chosen target must have
    bit-identical backbone torsions before/after.

    NOTE: this must inspect ``ctx.get_state()`` only after a genuinely
    COMMITTED move -- ``debug_force_rama_pivot`` only ever mutates the
    Integrator's private (uncommitted) proposal buffer, never
    ``context.state`` itself, so a before/after check straddling a
    debug-forced call alone would pass vacuously regardless of whether the
    move logic is correct (ctx.get_state() never changes either way). This
    mirrors test_rotamer_move.py::test_rotamer_run_only_moves_sidechain_atoms's
    established fix for the identical trap: step via real integ.run(ctx, 1)
    calls and only check the invariant on steps that were genuinely
    accepted (checked via last_accept_bits()).

    Rather than needing to know exactly which residue r was picked each
    step (not exposed to Python), this exploits a structural fact instead:
    `pivot_residue_dist` is always constructed as [1, N-2] (see run()'s
    init), so residue 0 can NEVER be a pivot target and is always upstream
    of whatever target was chosen -- its own backbone phi must therefore
    stay bit-identical across every accepted Pivot-slot step, regardless
    of which residue >= 1 was actually moved."""
    ctx, top = build_raw_context(virtual_amide_h=True)
    # Deliberately uses the REAL p0.4-fitted priors (already wired by
    # default via build_raw_context's MCPUForceField pipeline), not the
    # synthetic single-mode override used elsewhere in this file: the
    # synthetic mixture proposes the same aggressive target for every
    # residue type regardless of local structure, which induces a hard
    # steric clash (rejected unconditionally, at ANY temperature) on
    # nearly every attempt -- confirmed empirically (0/300 accepts even at
    # T=1e6). Real fitted priors propose much more plausible local
    # conformations, giving this test actual accepted steps to check.

    integ = mcpu_core.Integrator(temperature=1.0e6, step_size_rad=0.1)  # near-always-accept
    integ.set_pivot_rama_probability(1.0)
    integ.set_seed(11)

    n_checked = 0
    for _ in range(2000):
        before_phi0 = ctx.get_state().backbone_torsions[0].phi

        integ.run(ctx, 1)
        if integ.last_move_kind() != "Pivot" or integ.last_accept_bits()[-1] != 1:
            continue
        n_checked += 1

        after_phi0 = ctx.get_state().backbone_torsions[0].phi
        assert after_phi0 == pytest.approx(before_phi0, abs=1e-5)

    assert n_checked > 0, "no accepted rama-pivot steps occurred to check"


def test_fixed_residue_in_downstream_segment_blocks_rama_pivot() -> None:
    """The move only rotates the C-term (downstream) side,
    so a fixed residue anywhere in [r+1, N) must block it, with no N-term
    fallback."""
    ctx, top = build_raw_context(virtual_amide_h=True)
    system = ctx.get_system()
    _inject_synthetic_rama_mixture(system)
    n = system.get_num_residues()
    residue = _interior_non_pro_residue(ctx)

    integ = mcpu_core.Integrator(temperature=300.0, step_size_rad=0.1)
    integ.set_seed(11)
    integ.set_fixed_residues([n - 1], n)  # fix the last residue (downstream of `residue`)

    before = integ.get_fixed_rejected()
    proposed = integ.debug_force_rama_pivot(ctx, residue)
    assert proposed is False
    assert integ.get_fixed_rejected() == before + 1


def test_fixed_last_residue_ends_every_rama_pivot_step(chignolin_context) -> None:
    """With the last residue fixed no rama pivot is allowed, so each step
    ends with no move, is counted once, and moves nothing."""
    system = chignolin_context.get_system()
    _inject_synthetic_rama_mixture(system)
    n = system.get_num_residues()

    integ = mcpu_core.Integrator(temperature=300.0, step_size_rad=0.1)
    integ.set_seed(5)
    integ.set_move_weights(1.0, 0.0, 0.0)
    integ.set_pivot_rama_probability(1.0)
    integ.set_fixed_residues([n - 1], n)

    before = np.array(chignolin_context.get_state().coords, dtype=np.float32).copy()
    integ.run(chignolin_context, 50)
    after = np.array(chignolin_context.get_state().coords, dtype=np.float32)
    assert integ.get_fixed_rejected() == 50
    np.testing.assert_array_equal(before, after)


def test_rama_pivot_physics_consistency(chignolin_context) -> None:
    """The critical regression test for the union-marking/is_rigid bug
    class: incremental vs. full delta-energy must match for every energy
    group, for every proposed rama-pivot move."""
    system = chignolin_context.get_system()
    _inject_synthetic_rama_mixture(system)

    integ = mcpu_core.Integrator(temperature=300.0, step_size_rad=0.1)
    integ.set_pivot_rama_probability(1.0)
    integ.set_seed(3)
    mcpu_core.PhysicsVerifier.verify_mc_energy_consistency(
        integ, chignolin_context, num_steps=200, atol=ATOL
    )


def test_p_zero_is_bit_identical_to_genuine_legacy_code() -> None:
    """pivot_rama_probability=0.0 must reproduce EXACTLY the same
    accept/reject trajectory as the genuinely pre-feature engine --
    including RNG-draw count (no silent extra coin_flip at this extreme).

    NOTE: the default is now 0.0, so a freshly-constructed Integrator IS
    this configuration; the explicit set_pivot_rama_probability(0.0) below
    is kept so the test still states its premise and keeps passing if the
    default ever moves again. The reference numbers below were captured by
    running this exact scenario (same seed, same steps, same PDB) against
    the genuinely unmodified engine in the untouched `mcpu_dev` conda env
    (pkg/pyMCPU, pre-dating this feature's RamaMixtureLibrary/
    apply_rama_pivot_at/dispatch_pivot_move changes entirely) -- see the
    git history of this test for the exact capture command.

    RE-CAPTURED 2026-09-28 after the KIC loop-closure fix (CHANGELOG
    [Unreleased] -> Fixed). That fix changes KIC and the plain pivot's
    N-terminal branch, so every trajectory that runs them changes; it does
    not touch dispatch_pivot_move or the rama-mixture move. The unfixed
    engine still gives the numbers captured above (sum 92, same head; checked
    on a GCC 14.2 build of the unfixed tree). The fixed engine gives the same
    head and sum 79: replayed one step at a time, the two engines are
    bit-identical through step 9, and step 10 is an accepted KIC move that
    moves the same 15 atoms in both, 12 of which land up to 7.6e-6 A apart
    (float32 rounding of the new closure arithmetic); the runs then separate.
    So the no-extra-draw property was established against the pre-feature
    engine and is now pinned on the fixed one.

    RE-CAPTURED 2026-10-02 after the rotation fix (CHANGELOG [Unreleased] ->
    Fixed, "Pivots no longer shrink the protein"): rigid rotations are now
    done in double and rounded to float once. Same head, sum 81 instead of
    79. Replayed one step at a time, the coordinates first differ at step 7,
    an accepted sidechain move whose chi rotation lands one float step
    (4.8e-7 A) away, and the accept bits first differ at step 106.

    RE-CAPTURED 2026-10-09 after pivots gained the chain-end torsions and
    sidechain steps lost the residues without chi angles: a pivot now draws
    one of the chain's 2N-2 backbone torsions with one draw instead of two,
    and a sidechain step draws only residues it can move. Neither touches
    dispatch_pivot_move or the rama-mixture move, but both change which
    random numbers a run reads. The parent engine still gives sum 81 and the
    head above. Replayed one step at a time, steps 1 and 2 (a sidechain step
    and a pivot, both rejected) leave the coordinates identical in the two
    engines, and from step 3 on the runs draw different moves. Since the
    pinned numbers no longer tie this test to the pre-feature engine, the
    no-extra-draw property is now also checked directly: at p = 1e-30 the
    coin is drawn on every pivot step but never picks the Ramachandran move,
    so that run must differ from the p = 0 run.

    RE-CAPTURED 2026-10-10 after the KIC driver's default width went from
    0.1 rad to pi/6. That changes KIC proposals only, and the parent engine
    still gives sum 77 and the head [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0,
    0, 0, 0, 1, 0, 0, 1]. Replayed one step at a time, the coordinates stay
    identical through step 15: the KIC proposals at steps 5, 6 and 12 are
    rejected in both engines, at steps 5 and 12 for different reasons (a
    clash in one engine and not in the other), and from step 13 on the runs
    draw different moves. Step 16, a KIC move in the parent engine, is an
    accepted rotamer move here."""
    ctx, _ = build_raw_context(virtual_amide_h=True)
    integ = mcpu_core.Integrator(temperature=300.0, step_size_rad=0.1)
    integ.set_pivot_rama_probability(0.0)
    integ.set_seed(99)
    integ.run(ctx, 300)

    legacy_accept_bits_head = [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    # 92 before the KIC fix, 79 before the rotation fix, 81 before the
    # chain-end pivots and sidechain sites, 77 before the KIC driver's
    # default width became pi/6
    legacy_accept_bits_sum = 82

    bits = list(integ.last_accept_bits())
    assert bits[:20] == legacy_accept_bits_head
    assert sum(bits) == legacy_accept_bits_sum
    assert integ.get_rama_pivot_attempted() == 0

    # Any p above 0 draws the coin on every pivot step. At 1e-30 the coin
    # never picks the Ramachandran move, so the extra draw is the only thing
    # that can separate this run from the p = 0 one.
    ctx_tiny, _ = build_raw_context(virtual_amide_h=True)
    tiny = mcpu_core.Integrator(temperature=300.0, step_size_rad=0.1)
    tiny.set_pivot_rama_probability(1e-30)
    tiny.set_seed(99)
    tiny.run(ctx_tiny, 300)
    assert tiny.get_rama_pivot_attempted() == 0
    assert list(tiny.last_accept_bits()) != bits
