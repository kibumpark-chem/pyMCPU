"""Linker ("ghost") residue energy masking: physics/engine self-consistency.

Linker residues are movable-but-energy-masked: unlike fixed residues (see
``tests/physics/test_fixed_residues.py``), the integrator is free to move
them, but their contribution to one or more energy terms is dropped. This
file covers, purely as internal consistency (no legacy MCPU reference
values are involved anywhere here):

* config-layer validation/normalization of ``linker_energy_mode`` and the
  fixed/linker disjointness constraint, including through the YAML loader
* applying the mask to a live ``System`` (``ignore_all`` vs ``clash_only``)
* that a linker residue is never added to the integrator's *fixed* set --
  it must remain movable
* that ``NativeContactsCV`` drops any contact pair touching an
  energy-ignored residue
* that ``PhysicsVerifier.verify_potential_delta`` (incremental vs. full-energy
  recompute) stays self-consistent once a mask is active
* that masking a block of residues still yields finite energies for the
  aromatic energy group

The wall-clock perf smoke test for masking overhead lives separately in
``tests/perf/test_linker_perf_smoke.py`` -- timing is a non-functional
concern, not a physics-correctness one.
"""

from __future__ import annotations

import numpy as np
import pytest

from pymcpu import mcpu_core
from pymcpu.config import (
    ConstraintsConfig,
    apply_linker_energy_mask,
    normalize_linker_energy_mode,
    validate_fixed_linker_disjoint,
    yaml_dict_to_config,
)
from pymcpu.sampling.collective_variables import NativeContactsCV
from tests.fixtures.context_builders import build_test_context

#: Tolerance for PhysicsVerifier's incremental-vs-recomputed energy delta
#: check. Matches the literal used by the other potential-delta parity test in
#: this suite (tests/test_energy_mask.py) for the same
#: verify_potential_delta(..., energy_group, atol) call shape -- this is the
#: project's de facto standard atol for a single float32 finite-difference
#: perturbation, not a legacy-comparison tolerance (see
#: tests/legacy_parity/framework.py for that separate concept).
POTENTIAL_DELTA_ATOL = 1e-2


class TestLinkerConfigValidation:
    """``linker_energy_mode``/disjointness validation, independent of YAML."""

    def test_normalize_mode_defaults_to_ignore_all(self) -> None:
        assert normalize_linker_energy_mode(None) == "ignore_all"
        assert normalize_linker_energy_mode("") == "ignore_all"

    def test_normalize_mode_is_case_and_space_insensitive(self) -> None:
        assert normalize_linker_energy_mode("Clash_Only") == "clash_only"

    def test_normalize_mode_rejects_unknown_value(self) -> None:
        with pytest.raises(ValueError, match="linker_energy_mode"):
            normalize_linker_energy_mode("soft_only")

    def test_overlapping_fixed_and_linker_rejected(self) -> None:
        with pytest.raises(ValueError, match="disjoint"):
            validate_fixed_linker_disjoint([1, 2, 3], [3, 4])

    def test_disjoint_fixed_and_linker_accepted(self) -> None:
        validate_fixed_linker_disjoint([0, 1], [2, 3])  # must not raise

    def test_constraints_config_defaults_to_ignore_all(self) -> None:
        c = ConstraintsConfig(linker_residues=[1])
        assert c.linker_energy_mode == "ignore_all"


class TestLinkerYamlParsing:
    """``linker_residue_indices``/``linker_energy_mode`` through the flat
    YAML config loader (the CLI-facing entry point), rather than the config
    dataclasses directly."""

    def test_yaml_parses_linker_and_fixed_keys(self) -> None:
        cfg = yaml_dict_to_config(
            {
                "pdb": "examples/data/1uao.pdb",
                "temperatures": [0.5],
                "linker_residue_indices": [2, 3],
                "linker_energy_mode": "clash_only",
                "fixed_residue_indices": [0, 1],
            }
        )
        assert cfg.constraints.linker_residues == [2, 3]
        assert cfg.constraints.linker_energy_mode == "clash_only"
        assert cfg.constraints.fixed_residues == [0, 1]

    def test_yaml_rejects_overlapping_fixed_and_linker(self) -> None:
        with pytest.raises(ValueError, match="disjoint"):
            yaml_dict_to_config(
                {
                    "pdb": "examples/data/1uao.pdb",
                    "temperatures": [0.5],
                    "fixed_residue_indices": [5],
                    "linker_residue_indices": [5],
                }
            )


class TestLinkerMaskApplication:
    """Applying ``apply_linker_energy_mask`` to a live ``System``."""

    def test_ignore_all_mode_marks_residue_ignored(self) -> None:
        ctx, _ = build_test_context(with_qbias=False)
        system = ctx.get_system()
        apply_linker_energy_mask(system, [0, 1], "ignore_all", fixed_residues=[5])
        assert system.is_residue_energy_ignored(0)
        assert system.energy_mask_mode() == "ignore_all"

    def test_clash_only_mode_marks_residue_ignored(self) -> None:
        ctx, _ = build_test_context(with_qbias=False)
        system = ctx.get_system()
        apply_linker_energy_mask(system, [2], "clash_only")
        assert system.is_residue_energy_ignored(2)
        assert system.energy_mask_mode() == "clash_only"

    def test_overlapping_fixed_residue_rejected(self) -> None:
        ctx, _ = build_test_context(with_qbias=False)
        system = ctx.get_system()
        with pytest.raises(ValueError, match="disjoint"):
            apply_linker_energy_mask(system, [1], "ignore_all", fixed_residues=[1])

    def test_linker_residue_remains_movable_on_integrator(self) -> None:
        """A linker residue is energy-masked, not fixed -- the integrator
        must still be free to move it (unlike a truly fixed residue, see
        tests/physics/test_fixed_residues.py)."""
        ctx, _ = build_test_context(with_qbias=False)
        system = ctx.get_system()
        n_res = system.get_num_residues()
        linker = [n_res // 2]
        apply_linker_energy_mask(system, linker, "ignore_all")

        integrator = mcpu_core.Integrator(0.6, 0.1)
        integrator.set_seed(7)  # arbitrary: only finiteness is asserted, not a trajectory
        assert not integrator.has_fixed_residues()
        integrator.run(ctx, 20)
        assert np.isfinite(float(ctx.calculate_total_energy(-1)))


class TestNativeContactsEnergyMask:
    """``NativeContactsCV`` must drop any pair touching an ignored residue."""

    def test_ignored_residue_pairs_excluded_from_contact_set(self) -> None:
        n_res = 20
        ca_idx = np.arange(n_res, dtype=np.int64)
        # Linear chain, 1.0 A CA spacing: pairs with |i-j| >= 4 fall within
        # the 6.0 A cutoff by construction, giving a non-trivial bare
        # contact set to mask against. Purely synthetic geometry -- no
        # external reference structure involved.
        ref_ca = np.zeros((n_res, 3), dtype=np.float64)
        ref_ca[:, 0] = np.arange(n_res) * 1.0

        bare = NativeContactsCV(
            ca_internal_idx=ca_idx,
            ref_ca_xyz=ref_ca,
            contact_cutoff=6.0,
            min_seq_sep=4,
            mode="hard",
        )
        energy_ignored = np.zeros(n_res, dtype=bool)
        energy_ignored[5] = True
        energy_ignored[6] = True
        masked = NativeContactsCV(
            ca_internal_idx=ca_idx,
            ref_ca_xyz=ref_ca,
            contact_cutoff=6.0,
            min_seq_sep=4,
            mode="hard",
            energy_ignored_residue_mask=energy_ignored,
        )

        assert masked.n_contacts < bare.n_contacts
        assert masked.n_excluded_energy_ignored > 0
        pairs = zip(masked.pairs_i.tolist(), masked.pairs_j.tolist())
        assert not any(i in (5, 6) or j in (5, 6) for i, j in pairs)


class TestHBondPotentialDeltaWithMask:
    def test_potential_delta_self_consistent_with_mask_active(self) -> None:
        """Incremental HBond ΔE (energy group 4) must still match a full
        energy recompute once a block of residues is energy-ignored."""
        ctx, _ = build_test_context(with_qbias=False)
        system = ctx.get_system()
        n_res = system.get_num_residues()
        # Mask a large middle block so HBond terms touching it are dropped.
        ignored = list(range(max(1, n_res // 4), min(n_res - 1, 3 * n_res // 4)))
        system.set_energy_ignored_residues(ignored, "ignore_all")
        assert np.isfinite(float(ctx.calculate_total_energy(4)))

        old_state = mcpu_core.State(ctx.get_state())
        proposed = mcpu_core.State(ctx.get_state())
        coords = np.asarray(proposed.coords, dtype=np.float32).copy()

        atom_to_res = list(system.atom_to_residue)
        movable_candidates = [
            i for i, r in enumerate(atom_to_res) if r not in ignored and r > 0
        ]
        # Must find a real non-ignored atom to perturb -- a silent fallback
        # to atom 0 would risk a degenerate (no-op or ignored-residue) move.
        assert movable_candidates, "no non-ignored atom available to perturb"
        movable = movable_candidates[0]
        coords[0, movable] += 0.05  # small finite perturbation, consistent with other potential-delta tests
        proposed.coords = coords

        patch = mcpu_core.ProposalPatch(system.get_num_atoms())
        patch.mark_moved(movable)
        patch.is_rigid = False
        check = mcpu_core.PhysicsVerifier.verify_potential_delta(
            ctx, old_state, proposed, patch, 4, POTENTIAL_DELTA_ATOL
        )
        assert check.passed, check.message


class TestAromaticPotentialWithMask:
    def test_aromatic_energy_finite_with_mask_active(self) -> None:
        ctx, _ = build_test_context(with_qbias=False)
        system = ctx.get_system()
        n_res = system.get_num_residues()
        e_before = float(ctx.calculate_total_energy(5))
        system.set_energy_ignored_residues(list(range(0, min(5, n_res))), "ignore_all")
        e_after = float(ctx.calculate_total_energy(5))
        assert np.isfinite(e_before)
        assert np.isfinite(e_after)
