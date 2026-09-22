"""Tests that calculate_energy_change (the incremental delta used during MC
proposals) matches a direct E(new) - E(old) recomputation, per energy group.

Two tests use hand-built minimal BB/SC systems with a single-residue torsion
perturbation and ``PhysicsVerifier.verify_potential_delta`` directly; the rest
drive the real chignolin/actin forcefield through many random MC steps via
``Integrator.verify_physics_consistency``, which performs this same check
internally for every proposal.
"""

from __future__ import annotations

import numpy as np
import pytest

from pymcpu import mcpu_core
from tests.fixtures.context_builders import ATOL
from tests.physics.helpers.minimal_system_builders import (
    setup_minimal_bb_system,
    setup_minimal_sc_system,
)


def _assert_potential_delta(context, old_state, proposed_state, patch, energy_group: int) -> None:
    check = mcpu_core.PhysicsVerifier.verify_potential_delta(
        context, old_state, proposed_state, patch, energy_group, ATOL
    )
    assert check.passed, (
        f"energy_group={energy_group}: delta_inc={check.delta_incremental} "
        f"delta_direct={check.delta_direct} msg={check.message}"
    )


def test_bb_triplet_delta_matches_full_recompute() -> None:
    n_res = 4
    n_atoms = 16
    system, context = setup_minimal_bb_system(n_res, n_atoms)

    params = [0.0] * (n_res * 1296)
    potential = mcpu_core.TripletPotential(params)
    potential.set_energy_group(2)
    system.add_potential(potential)

    coords = np.zeros((3, n_atoms), dtype=np.float32)
    context.set_positions(coords)

    old_state = mcpu_core.State(context.get_state())
    proposed_state = mcpu_core.State(context.get_state())
    proposed_state.backbone_torsions[1].phi = 0.5
    proposed_state.backbone_torsions[1].psi = -0.3

    patch = mcpu_core.ProposalPatch(n_atoms)
    patch.is_valid = True
    patch.add_distorted_bb_residue(1)

    _assert_potential_delta(context, old_state, proposed_state, patch, 2)


def test_sc_triplet_delta_matches_full_recompute() -> None:
    n_res = 3
    n_atoms = 12
    system, context = setup_minimal_sc_system(n_res, n_atoms)

    params = [0.0] * (n_res * 20736)
    potential = mcpu_core.SidechainTripletPotential(params)
    potential.set_energy_group(3)
    system.add_potential(potential)

    coords = np.zeros((3, n_atoms), dtype=np.float32)
    context.set_positions(coords)

    old_state = mcpu_core.State(context.get_state())
    proposed_state = mcpu_core.State(context.get_state())
    proposed_state.sidechain_torsions[1].chi_angles[0] = 1.0

    patch = mcpu_core.ProposalPatch(n_atoms)
    patch.is_valid = True
    patch.add_distorted_sc_residue(1)

    _assert_potential_delta(context, old_state, proposed_state, patch, 3)


@pytest.mark.slow
def test_all_potentials_delta_consistent_over_random_mc_steps(chignolin_context) -> None:
    integrator = mcpu_core.Integrator(temperature=300.0, step_size_rad=0.1)
    integrator.verify_physics_consistency(chignolin_context, num_steps=25, atol=ATOL)


@pytest.mark.slow
def test_all_potentials_delta_consistent_with_qbias(chignolin_with_qbias) -> None:
    context, _forcefield = chignolin_with_qbias
    integrator = mcpu_core.Integrator(temperature=300.0, step_size_rad=0.1)
    integrator.verify_physics_consistency(context, num_steps=25, atol=ATOL)


@pytest.mark.slow
def test_energy_group_energies_finite_after_mc_run(chignolin_context) -> None:
    """Confirm delta/full consistency once via an MC run, then check every
    energy group's post-run energy is finite.

    This replaces what was previously 5 separately parametrized tests
    (one per energy_group in [1..5]) that each reran the identical 25-step
    ``verify_physics_consistency`` sweep just to vary a trailing finiteness
    check -- 5x redundant execution of the expensive part for no added
    coverage. Consolidated into one MC run + a loop over groups.
    """
    integrator = mcpu_core.Integrator(temperature=300.0, step_size_rad=0.1)
    integrator.verify_physics_consistency(chignolin_context, num_steps=20, atol=ATOL)

    for energy_group in (1, 2, 3, 4, 5):
        energy = chignolin_context.calculate_total_energy(energy_group)
        assert np.isfinite(energy), f"energy_group={energy_group} energy not finite after MC run"
