"""Tests for residue energy masking (linker masking feature).

This is a pyMCPU-only feature with no legacy MCPU equivalent -- it exists to
let callers (e.g. linker-residue scaffolding) exclude specific residues from
energy evaluation without removing them from the system. Covers:

- API wiring: set/get/clear energy-ignored residues and modes.
- Mode ``ignore_all``: all energy contributions (including clash) are masked.
- Mode ``clash_only``: delta steric clash remains active; contact/other
  terms are masked. Full Mu energy is contact-only (legacy CLASH_WEIGHT=0)
  -- no 99999 penalty.

That the running energy stays exact while a mask is set, changed and
cleared between runs is checked in ``test_energy_definition_resync.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from pymcpu import mcpu_core
from tests.fixtures.context_builders import ATOL, build_test_context
from tests.physics.helpers.constants import MU_CLASH_SAFETY_MARGIN

#: Tolerance for "this term should be unchanged" comparisons -- not a
#: physics baseline, just tight enough to sit below expected float noise.
_UNCHANGED_ATOL = 1e-6


# ---------------------------------------------------------------------------
# 1. API wiring tests
# ---------------------------------------------------------------------------
class TestEnergyMaskAPI:
    def test_set_ignore_all(self) -> None:
        ctx, _ = build_test_context(with_qbias=False)
        sys = ctx.get_system()
        sys.set_energy_ignored_residues([0, 1, 2], "ignore_all")
        assert sys.is_residue_energy_ignored(0)
        assert sys.is_residue_energy_ignored(1)
        assert sys.is_residue_energy_ignored(2)
        assert not sys.is_residue_energy_ignored(3)
        assert sys.energy_mask_mode() == "ignore_all"

    def test_set_clash_only(self) -> None:
        ctx, _ = build_test_context(with_qbias=False)
        sys = ctx.get_system()
        sys.set_energy_ignored_residues([5], "clash_only")
        assert sys.is_residue_energy_ignored(5)
        assert not sys.is_residue_energy_ignored(4)
        assert sys.energy_mask_mode() == "clash_only"

    def test_clear(self) -> None:
        ctx, _ = build_test_context(with_qbias=False)
        sys = ctx.get_system()
        sys.set_energy_ignored_residues([0, 1], "ignore_all")
        sys.clear_energy_ignored_residues()
        assert not sys.is_residue_energy_ignored(0)
        assert not sys.is_residue_energy_ignored(1)

    def test_invalid_mode_raises(self) -> None:
        ctx, _ = build_test_context(with_qbias=False)
        sys = ctx.get_system()
        with pytest.raises(Exception):
            sys.set_energy_ignored_residues([0], "bogus_mode")

    def test_out_of_range_raises(self) -> None:
        ctx, _ = build_test_context(with_qbias=False)
        sys = ctx.get_system()
        n_res = sys.get_num_residues()
        with pytest.raises(Exception):
            sys.set_energy_ignored_residues([n_res + 10], "ignore_all")


# ---------------------------------------------------------------------------
# 2. ignore_all mode: clash is masked
# ---------------------------------------------------------------------------
class TestIgnoreAllMode:
    def test_masked_residue_clash_ignored(self) -> None:
        """When a masked residue overlaps another, no clash penalty in ignore_all."""
        ctx, _ = build_test_context(with_qbias=False)
        sys = ctx.get_system()
        n_res = sys.get_num_residues()

        # Pick residues far apart in sequence (clash mask is only enabled for
        # pairs with sufficient sequence separation, typically > 3-4 residues).
        target_res = n_res // 4
        partner_res = 3 * n_res // 4
        sys.set_energy_ignored_residues([target_res], "ignore_all")

        # Baseline energy (no clash, native structure); use RAW to avoid weight factors.
        e_baseline = float(ctx.calculate_total_energy_raw(1))
        assert e_baseline < MU_CLASH_SAFETY_MARGIN, "Native should not have clash"

        # Create overlap: move ALL atoms of target_res onto partner_res atom.
        a2r = list(sys.atom_to_residue)
        target_atoms = [i for i, r in enumerate(a2r) if r == target_res]
        partner_atoms = [i for i, r in enumerate(a2r) if r == partner_res]
        assert len(target_atoms) > 0 and len(partner_atoms) > 0

        coords = np.array(ctx.coords, dtype=np.float32).copy()
        for ai in target_atoms:
            coords[:, ai] = coords[:, partner_atoms[0]]

        ctx.set_positions(coords)
        ctx.calculate_total_energy(-1)

        e_after = float(ctx.calculate_total_energy_raw(1))

        # In ignore_all mode, clash involving the masked residue is ignored.
        assert e_after < MU_CLASH_SAFETY_MARGIN, (
            f"ignore_all should suppress clash: got raw Mu energy {e_after}"
        )

    def test_masked_non_clash_terms_also_masked(self) -> None:
        """Non-clash terms (triplet, hbond) are also masked for ignored residues."""
        ctx, _ = build_test_context(with_qbias=False)
        sys = ctx.get_system()
        n_res = sys.get_num_residues()

        # Compute baseline energy per group without mask.
        e_bb = float(ctx.calculate_total_energy(2))  # BB triplet
        e_sc = float(ctx.calculate_total_energy(3))  # SC triplet

        # Mask a middle residue that contributes to triplet terms.
        target_res = n_res // 2
        sys.set_energy_ignored_residues([target_res], "ignore_all")

        e_bb_masked = float(ctx.calculate_total_energy(2))
        e_sc_masked = float(ctx.calculate_total_energy(3))

        # With a residue masked, at least one triplet energy should differ
        # (the masked residue's contribution is removed).
        assert (
            e_bb_masked != pytest.approx(e_bb, abs=_UNCHANGED_ATOL)
            or e_sc_masked != pytest.approx(e_sc, abs=_UNCHANGED_ATOL)
        ), "At least one triplet term should change when a residue is masked"


# ---------------------------------------------------------------------------
# 3. clash_only mode: clash remains, other terms masked
# ---------------------------------------------------------------------------
class TestClashOnlyMode:
    def test_clash_still_detected(self) -> None:
        """clash_only: full energy stays contact-only; delta still StericClash-rejects."""
        ctx, _ = build_test_context(with_qbias=False)
        sys = ctx.get_system()
        n_res = sys.get_num_residues()

        # Use residues far apart (clash mask only active for large seq separation).
        target_res = n_res // 4
        partner_res = 3 * n_res // 4
        sys.set_energy_ignored_residues([target_res], "clash_only")

        # Baseline: no clash (use RAW to avoid weight factors).
        e_baseline = float(ctx.calculate_total_energy_raw(1))
        assert e_baseline < MU_CLASH_SAFETY_MARGIN

        a2r = list(sys.atom_to_residue)
        target_atoms = [i for i, r in enumerate(a2r) if r == target_res]
        partner_atoms = [i for i, r in enumerate(a2r) if r == partner_res]
        assert len(target_atoms) > 0 and len(partner_atoms) > 0

        old_state = mcpu_core.State(ctx.get_state())
        proposed_state = mcpu_core.State(ctx.get_state())
        coords = np.asarray(proposed_state.coords, dtype=np.float32).copy()
        atom_i = target_atoms[0]
        atom_j = partner_atoms[0]
        coords[:, atom_i] = coords[:, atom_j]
        proposed_state.coords = coords

        patch = mcpu_core.ProposalPatch(sys.get_num_atoms())
        patch.mark_moved(atom_i)
        patch.is_rigid = False

        # Was previously re-hardcoded here as a bare 1e-3 literal, coincidentally
        # equal to ATOL -- import the shared constant instead of duplicating it.
        check = mcpu_core.PhysicsVerifier.verify_potential_delta(
            ctx, old_state, proposed_state, patch, 1, ATOL
        )
        assert check.passed, check.message
        assert check.delta_incremental >= MU_CLASH_SAFETY_MARGIN

        # Full energy on the overlapping coords: contact-only, no 99999 penalty.
        ctx.set_positions(coords)
        e_after = float(ctx.calculate_total_energy_raw(1))
        assert e_after < MU_CLASH_SAFETY_MARGIN, (
            f"full energy should remain contact-only: got raw Mu energy {e_after}"
        )
        assert np.isfinite(e_after)
        ctx.calculate_total_energy(-1)
        assert not ctx.has_steric_clash()

    def test_non_clash_terms_masked(self) -> None:
        """Non-clash terms are masked even in clash_only mode."""
        ctx, _ = build_test_context(with_qbias=False)
        sys = ctx.get_system()
        n_res = sys.get_num_residues()

        # Baseline.
        e_bb = float(ctx.calculate_total_energy(2))
        e_hbond = float(ctx.calculate_total_energy(4))

        # Mask a middle residue.
        target_res = n_res // 2
        sys.set_energy_ignored_residues([target_res], "clash_only")

        e_bb_masked = float(ctx.calculate_total_energy(2))
        e_hbond_masked = float(ctx.calculate_total_energy(4))

        # At least one non-clash term should differ.
        terms_changed = (
            e_bb_masked != pytest.approx(e_bb, abs=_UNCHANGED_ATOL)
            or e_hbond_masked != pytest.approx(e_hbond, abs=_UNCHANGED_ATOL)
        )
        assert terms_changed, (
            "clash_only mode should still mask non-clash terms for ignored residues"
        )

    def test_contact_energy_masked_but_clash_kept(self) -> None:
        """Contact (Mu) energy is zeroed for masked pairs; delta clash still active."""
        ctx, _ = build_test_context(with_qbias=False)
        sys = ctx.get_system()
        n_res = sys.get_num_residues()

        # Baseline Mu energy (no mask, no clash).
        e_mu_baseline = float(ctx.calculate_total_energy(1))
        assert e_mu_baseline < MU_CLASH_SAFETY_MARGIN

        # Mask several residues -> their contact contributions should vanish.
        residues_to_mask = list(range(n_res // 4, n_res // 2))
        sys.set_energy_ignored_residues(residues_to_mask, "clash_only")

        e_mu_masked = float(ctx.calculate_total_energy(1))

        # Contact energy should change (some contacts zeroed): the native
        # structure has many contacts, so masking a chunk should change E.
        assert e_mu_masked != pytest.approx(e_mu_baseline, abs=_UNCHANGED_ATOL), (
            "Masking contacts should change Mu energy in native structure"
        )
        # Should still be finite (full energy is contact-only).
        assert np.isfinite(e_mu_masked)
        assert e_mu_masked < MU_CLASH_SAFETY_MARGIN
