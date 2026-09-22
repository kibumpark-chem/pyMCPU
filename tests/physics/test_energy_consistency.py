"""Full-forcefield internal energy-consistency checks.

Every assertion here compares the engine against itself -- cached vs.
recomputed total energy, per-energy-group sums vs. the whole, and the C++
``PhysicsVerifier`` self-check -- there is no legacy MCPU comparison in this
file. Legacy-reference comparisons live under ``tests/legacy_parity/``.
"""

from __future__ import annotations

from enum import IntEnum

import pytest

from pymcpu import mcpu_core
from tests.fixtures.context_builders import ATOL

pytestmark = pytest.mark.slow


class EnergyGroup(IntEnum):
    """Force-group indices assigned in ``MCPUForceField.create_system()``
    (see the ``set_energy_group`` calls in ``pymcpu/forcefields/mcpu.py``) --
    named here so a group-sum test reads as physics terms rather than a bare
    ``range(1, 6)``."""

    MU_CONTACT = 1  # pairwise knowledge-based contact potential (mu_potential)
    BACKBONE_TORSION = 2  # backbone phi/psi triplet potential (bb_potential)
    SIDECHAIN_TORSION = 3  # sidechain triplet potential (sc_potential)
    HBOND = 4  # hydrogen-bond potential (hbond_potential)
    AROMATIC = 5  # aromatic ring-stacking potential (aromatic_potential)


def test_mc_energy_consistency(chignolin_context) -> None:
    """``verify_mc_energy_consistency`` drives 20 random MC move proposals
    and, for every energy group present, asserts the incremental delta-energy
    (``Potential::calculateEnergyChange``) matches a direct recomputation
    (``e_new - e_old`` from ``Potential::calculateEnergy``) within ``atol`` --
    i.e. the delta-energy hotpaths agree with a from-scratch recompute for
    both accepted and rejected proposals, not just the committed trajectory
    (see ``PhysicsVerifier::verify_potential_delta`` /
    ``MCIntegrator::verify_physics_consistency``).
    """
    integrator = mcpu_core.Integrator(temperature=300.0, step_size_rad=0.1)
    mcpu_core.PhysicsVerifier.verify_mc_energy_consistency(
        integrator, chignolin_context, num_steps=20, atol=ATOL
    )


def test_accepted_move_energy_drift(chignolin_context) -> None:
    """After accepted moves, the ``current_energy`` cached incrementally on
    ``State`` must not drift from a from-scratch total-energy recompute."""
    integrator = mcpu_core.Integrator(temperature=300.0, step_size_rad=0.1)
    integrator.run(chignolin_context, num_steps=10)
    e_cached = chignolin_context.get_state().current_energy
    e_recalc = chignolin_context.calculate_total_energy(-1)
    # ATOL: internal self-consistency tolerance (float32 accumulation noise
    # between the incremental-update path and a full recompute), not a
    # legacy-comparison tolerance -- see tests/fixtures/context_builders.py.
    assert e_cached == pytest.approx(e_recalc, abs=ATOL)


def test_per_group_sum_equals_total(chignolin_context) -> None:
    """The five per-energy-group energies must sum to the same total the
    engine reports for group -1 (all potentials) -- i.e. the total-energy path
    isn't silently double-counting or dropping a energy group."""
    group_energies = [
        chignolin_context.calculate_total_energy(group) for group in EnergyGroup
    ]
    total = chignolin_context.calculate_total_energy(-1)
    assert sum(group_energies) == pytest.approx(total, abs=ATOL)
