"""Full-forcefield internal energy-consistency checks.

Incremental energy changes vs. a from-scratch recompute (the C++
``PhysicsVerifier`` self-check), and per-energy-group sums vs. the whole,
with sanity bounds on the folded reference structure (every group finite,
Mu below the clash sentinel, total negative). There is no legacy MCPU
comparison in this file; legacy-reference comparisons live under
``tests/legacy_parity/``.
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np
import pytest

from pymcpu import mcpu_core
from tests.fixtures.context_builders import ATOL
from tests.physics.helpers.constants import MU_CLASH_SENTINEL


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
    """``verify_mc_energy_consistency`` draws MC move proposals from the
    native state and, for every energy group present, asserts the
    incremental delta-energy (``Potential::calculateEnergyChange``) matches
    a direct recomputation (``e_new - e_old`` from
    ``Potential::calculateEnergy``) within ``atol`` (see
    ``PhysicsVerifier::verify_potential_delta`` /
    ``MCIntegrator::verify_physics_consistency``).

    This checks the default move mix, the rotamer-library sidechain move
    included, and it is the default suite's full-forcefield delta check, so
    it is not marked slow (about 4 s plus the context build). The proposals
    are seeded and many: against a delta bug that fires only on some
    residues, 10, 20 and 30 random proposals missed it in 8, 5 and 3 of 20
    seeds, and 100 or more caught it in all 20.
    """
    integrator = mcpu_core.Integrator(temperature=300.0, step_size_rad=0.1)
    integrator.set_seed(2026)
    mcpu_core.PhysicsVerifier.verify_mc_energy_consistency(
        integrator, chignolin_context, num_steps=500, atol=ATOL
    )


@pytest.mark.slow
def test_mc_energy_consistency_with_qbias(chignolin_with_qbias) -> None:
    """The same check with the native-contacts bias (group 6) attached."""
    context, _forcefield = chignolin_with_qbias
    integrator = mcpu_core.Integrator(temperature=300.0, step_size_rad=0.1)
    integrator.set_seed(2026)
    integrator.verify_physics_consistency(context, num_steps=100, atol=ATOL)


@pytest.mark.slow
def test_per_group_sum_equals_total(chignolin_context) -> None:
    """The five per-energy-group energies are finite and sum to the total
    the engine reports for group -1 (all potentials) -- the total-energy
    path isn't silently double-counting or dropping an energy group. The
    folded reference structure is not in a steric clash and has a
    net-favourable (negative) total energy."""
    group_energies = {
        group: chignolin_context.calculate_total_energy(group) for group in EnergyGroup
    }
    total = chignolin_context.calculate_total_energy(-1)
    for group, energy in group_energies.items():
        assert np.isfinite(energy), f"{group.name} energy is not finite"
    assert group_energies[EnergyGroup.MU_CONTACT] < MU_CLASH_SENTINEL
    assert total < 0.0
    assert sum(group_energies.values()) == pytest.approx(total, abs=ATOL)
