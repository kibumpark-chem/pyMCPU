"""Fixed-residue engine mechanics: pure physics/engine self-consistency.

Fixed residues are a stronger constraint than linker energy-masking (see
``tests/physics/test_linker_energy_mask.py``): a fixed residue's coordinates
must never change under MC, not merely have its energy contribution dropped.
Covers:

* the low-level ``Integrator`` set/get/clear/mask API and its input
  validation
* that fixed-residue atoms are literally frozen across hundreds of MC
  steps while non-fixed atoms are free to move
* that fixing every residue halts all movement without crashing
* that ``NativeContactsCV`` excludes fixed-fixed residue pairs (but keeps
  fixed-nonfixed pairs)

None of this compares against legacy MCPU reference values -- it is all
internal consistency of this engine's own fixed-residue implementation.
Config/CLI-facing plumbing for the same feature (YAML schema,
``parse_int_list``, the ``Simulation`` convenience wrapper) lives in
``tests/config/test_fixed_residues_config.py`` instead, since that is
software correctness rather than a physics question.
"""

from __future__ import annotations

import numpy as np
import pytest

from pymcpu import mcpu_core
from pymcpu.sampling.collective_variables import NativeContactsCV
from tests.fixtures.context_builders import build_test_context


def _build_context_with_fixed(fixed_residues: list[int], seed: int = 42):
    """Build a context + integrator with ``fixed_residues`` applied.

    ``seed`` is an arbitrary deterministic value -- these tests only assert
    exact-equality of untouched coordinates and whether *something* moved,
    never a specific numeric trajectory, so any fixed seed suffices.
    """
    context, forcefield = build_test_context(with_qbias=False)
    n_res = context.get_system().get_num_residues()
    integrator = mcpu_core.Integrator(0.6, 0.1)
    integrator.set_seed(seed)
    integrator.set_fixed_residues(fixed_residues, n_res)
    return context, integrator, forcefield


class TestFixedResidueIntegratorAPI:
    """Set/get/clear/mask/validation on the ``Integrator`` directly (no MC
    steps run in this class -- see TestFixedResiduesEnforcement for that)."""

    def test_set_and_get_round_trips(self) -> None:
        integrator = mcpu_core.Integrator(0.6)
        integrator.set_fixed_residues([0, 2, 4], 10)
        assert integrator.get_fixed_residues() == [0, 2, 4]
        assert integrator.has_fixed_residues()

    def test_clear_empties_the_fixed_set(self) -> None:
        integrator = mcpu_core.Integrator(0.6)
        integrator.set_fixed_residues([0, 1], 5)
        integrator.clear_fixed_residues()
        assert integrator.get_fixed_residues() == []
        assert not integrator.has_fixed_residues()

    def test_mask_marks_exactly_the_fixed_indices(self) -> None:
        integrator = mcpu_core.Integrator(0.6)
        integrator.set_fixed_residues([1, 3], 5)
        # mask[i] == 1 iff i is in the fixed list -- direct API contract, no external baseline.
        assert list(integrator.get_fixed_residue_mask()) == [0, 1, 0, 1, 0]

    def test_out_of_range_index_raises(self) -> None:
        integrator = mcpu_core.Integrator(0.6)
        with pytest.raises(Exception):
            integrator.set_fixed_residues([10], 5)

    def test_negative_index_raises(self) -> None:
        integrator = mcpu_core.Integrator(0.6)
        with pytest.raises(Exception):
            integrator.set_fixed_residues([-1], 5)

    def test_empty_list_is_a_no_op(self) -> None:
        integrator = mcpu_core.Integrator(0.6)
        integrator.set_fixed_residues([], 5)
        assert integrator.get_fixed_residues() == []


class TestFixedResiduesEnforcement:
    """MC-step-level enforcement: fixed atoms don't move, others do."""

    def test_fixed_residues_do_not_move(self) -> None:
        """Coordinates of fixed residues must remain unchanged after MC steps."""
        context, integrator, _ = _build_context_with_fixed([0, 1, 2])
        atom_to_res = list(context.get_system().atom_to_residue)
        fixed_atoms = sorted(i for i, r in enumerate(atom_to_res) if r in (0, 1, 2))

        coords_before = np.array(context.get_state().coords, dtype=np.float32).copy()
        integrator.run(context, 300)  # empirically enough steps to guarantee moves elsewhere
        coords_after = np.array(context.get_state().coords, dtype=np.float32)

        for atom_idx in fixed_atoms:
            np.testing.assert_array_equal(
                coords_before[:, atom_idx],
                coords_after[:, atom_idx],
                err_msg=f"Fixed atom {atom_idx} (residue {atom_to_res[atom_idx]}) moved!",
            )

    def test_nonfixed_residues_do_move(self) -> None:
        """Non-fixed atoms must change after enough MC steps (simulation progresses)."""
        context, integrator, _ = _build_context_with_fixed([0, 1])
        atom_to_res = list(context.get_system().atom_to_residue)
        nonfixed_atoms = [i for i, r in enumerate(atom_to_res) if r not in (0, 1)]

        coords_before = np.array(context.get_state().coords, dtype=np.float32).copy()
        integrator.run(context, 500)  # empirically enough steps for >=1 accepted move at T=0.6
        coords_after = np.array(context.get_state().coords, dtype=np.float32)

        changed = any(
            not np.array_equal(coords_before[:, i], coords_after[:, i])
            for i in nonfixed_atoms
        )
        assert changed, "No non-fixed atoms moved after 500 MC steps"

    def test_fixing_every_residue_halts_movement_without_crash(self) -> None:
        """Fixing all residues should not crash; just all moves rejected."""
        context, integrator, _ = _build_context_with_fixed([])
        n_res = context.get_system().get_num_residues()
        integrator.set_fixed_residues(list(range(n_res)), n_res)

        coords_before = np.array(context.get_state().coords, dtype=np.float32).copy()
        integrator.run(context, 50)
        coords_after = np.array(context.get_state().coords, dtype=np.float32)
        np.testing.assert_array_equal(coords_before, coords_after)


class TestNativeContactsFixedExclusion:
    """``NativeContactsCV`` must drop fixed-fixed pairs but keep
    fixed-nonfixed pairs (synthetic linear-chain geometry, not tied to any
    real structure). This auto-exclusion is a derived-mode-only policy
    choice, though: an explicitly-specified ``native_contact_pairs`` fixed-
    fixed pair is kept, while ``energy_ignored_residue_mask`` still hard-
    rejects any explicit pair touching a masked residue (a correctness
    guard, not a policy choice)."""

    def test_fixed_fixed_pairs_excluded(self) -> None:
        ca_idx = np.array([0, 1, 2, 3, 4], dtype=np.int64)
        ref_ca = np.array(
            [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [8.0, 0.0, 0.0], [12.0, 0.0, 0.0], [16.0, 0.0, 0.0]],
            dtype=np.float64,
        )
        cv_all = NativeContactsCV(ca_idx, ref_ca, contact_cutoff=10.0, min_seq_sep=1, mode="hard")
        n_all = cv_all.n_contacts

        fixed_mask = np.array([True, True, False, False, False])
        cv_fixed = NativeContactsCV(
            ca_idx, ref_ca, contact_cutoff=10.0, min_seq_sep=1, mode="hard",
            fixed_residue_mask=fixed_mask,
        )
        assert cv_fixed.n_contacts == n_all - 1  # exactly the (0,1) fixed-fixed pair dropped
        assert cv_fixed.n_excluded_fixed_fixed == 1

        pairs = set(zip(cv_fixed.pairs_i.tolist(), cv_fixed.pairs_j.tolist()))
        assert (0, 1) not in pairs

    def test_fixed_nonfixed_pairs_kept(self) -> None:
        ca_idx = np.array([0, 1, 2], dtype=np.int64)
        ref_ca = np.array([[0, 0, 0], [4, 0, 0], [8, 0, 0]], dtype=np.float64)
        fixed_mask = np.array([True, False, False])
        cv = NativeContactsCV(
            ca_idx, ref_ca, contact_cutoff=10.0, min_seq_sep=1, mode="hard",
            fixed_residue_mask=fixed_mask,
        )
        pairs = set(zip(cv.pairs_i.tolist(), cv.pairs_j.tolist()))
        assert (0, 1) in pairs  # one fixed, one not -> not excluded

    def test_omitting_fixed_mask_excludes_nothing(self) -> None:
        """Backwards compat: no ``fixed_residue_mask`` means no exclusions."""
        ca_idx = np.array([0, 1, 2], dtype=np.int64)
        ref_ca = np.array([[0, 0, 0], [4, 0, 0], [8, 0, 0]], dtype=np.float64)
        cv = NativeContactsCV(ca_idx, ref_ca, contact_cutoff=10.0, min_seq_sep=1, mode="hard")
        assert cv.n_excluded_fixed_fixed == 0

    def test_compute_n_uses_the_filtered_pair_list(self) -> None:
        ca_idx = np.array([0, 1, 2, 3], dtype=np.int64)
        ref_ca = np.array([[0, 0, 0], [4, 0, 0], [8, 0, 0], [12, 0, 0]], dtype=np.float64)
        fixed_mask = np.array([True, True, False, False])
        cv = NativeContactsCV(
            ca_idx, ref_ca, contact_cutoff=10.0, min_seq_sep=1, mode="hard",
            q_cutoff=10.0, fixed_residue_mask=fixed_mask,
        )
        # All non-excluded pairs are within cutoff at these coords, so
        # compute_N should recover exactly the (pre-filtered) n_contacts.
        coords = np.zeros((3, 4), dtype=np.float64)
        coords[0, :] = [0, 4, 8, 12]
        assert cv.compute_N(coords) == cv.n_contacts

    def test_explicit_fixed_fixed_pair_is_kept_not_excluded(self) -> None:
        """Unlike the derived branch (see ``test_fixed_fixed_pairs_excluded``
        above), an explicit ``native_contact_pairs`` entry is a deliberate
        user choice -- ``fixed_residue_mask`` must not silently drop a
        fixed-fixed pair the caller typed by hand."""
        ca_idx = np.array([0, 1, 2, 3, 4], dtype=np.int64)
        ref_ca = np.array(
            [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [8.0, 0.0, 0.0], [12.0, 0.0, 0.0], [16.0, 0.0, 0.0]],
            dtype=np.float64,
        )
        fixed_mask = np.array([True, True, False, False, False])
        cv = NativeContactsCV(
            ca_idx, ref_ca, contact_cutoff=10.0, min_seq_sep=1, mode="hard",
            fixed_residue_mask=fixed_mask, native_contact_pairs=[(0, 1)],
        )
        pairs = set(zip(cv.pairs_i.tolist(), cv.pairs_j.tolist()))
        assert (0, 1) in pairs  # both residues fixed, but explicit -> kept
        assert cv.n_contacts == 1
        assert cv.n_excluded_fixed_fixed == 0

    def test_explicit_pair_touching_energy_ignored_residue_raises(self) -> None:
        """``energy_ignored_residue_mask`` is a correctness guard, not a
        policy choice, so it still hard-rejects an explicit pair touching a
        masked (linker/ghost) residue -- unlike ``fixed_residue_mask``
        above."""
        ca_idx = np.array([0, 1, 2], dtype=np.int64)
        ref_ca = np.array([[0, 0, 0], [4, 0, 0], [8, 0, 0]], dtype=np.float64)
        energy_ignored_mask = np.array([False, True, False])
        with pytest.raises(ValueError):
            NativeContactsCV(
                ca_idx, ref_ca, contact_cutoff=10.0, min_seq_sep=1, mode="hard",
                energy_ignored_residue_mask=energy_ignored_mask,
                native_contact_pairs=[(0, 1)],
            )
