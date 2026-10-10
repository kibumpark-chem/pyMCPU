"""Coverage for ``PhysicsVerifier.verify_all_potential_deltas``, the vector
form of the per-group delta check.

It verifies an invariant the incremental path can genuinely violate, which is
why it is worth pinning. ``verify_potential_delta`` (the scalar form) is
called directly by many tests (the ``forcefield/test_mu_*`` files among
them); this pins that the "all" form agrees with it for every added energy
group, so the two cannot drift.
"""

from __future__ import annotations

import numpy as np
import pytest

from pymcpu import mcpu_core
from tests.fixtures.context_builders import ATOL
from tests.physics.helpers.minimal_system_builders import setup_minimal_bb_system


def _bb_proposal(context, n_res: int, n_atoms: int):
    """A single-residue backbone torsion perturbation."""
    coords = np.zeros((3, n_atoms), dtype=np.float32)
    context.set_positions(coords)

    old_state = mcpu_core.State(context.get_state())
    proposed_state = mcpu_core.State(context.get_state())
    proposed_state.backbone_torsions[1].phi = 0.5
    proposed_state.backbone_torsions[1].psi = -0.3

    patch = mcpu_core.ProposalPatch(n_atoms)
    patch.is_valid = True
    patch.add_distorted_bb_residue(1)
    return old_state, proposed_state, patch


def test_verify_all_potential_deltas_covers_every_added_group() -> None:
    """The vector form must return one check per added energy group, all
    passing, and each must agree with the scalar ``verify_potential_delta``."""
    n_res, n_atoms = 4, 16
    system, context = setup_minimal_bb_system(n_res, n_atoms)

    # One potential (group 2) so the "all" form has something to iterate over.
    bb_triplet = mcpu_core.TripletPotential([0.0] * (n_res * 1296))
    bb_triplet.set_energy_group(2)
    system.add_potential(bb_triplet)

    old_state, proposed_state, patch = _bb_proposal(context, n_res, n_atoms)

    checks = mcpu_core.PhysicsVerifier.verify_all_potential_deltas(
        context, old_state, proposed_state, patch, ATOL
    )

    # Non-vacuous: the verifier must actually have looked at something.
    assert len(checks) >= 1, "verify_all_potential_deltas returned no checks"

    groups = {int(c.energy_group) for c in checks}
    assert 2 in groups, f"energy group 2 not checked; got {sorted(groups)}"

    for check in checks:
        assert check.passed, (
            f"energy_group={check.energy_group}: "
            f"delta_incremental={check.delta_incremental} "
            f"delta_direct={check.delta_direct} message={check.message}"
        )
        # The two deltas are what the check compares; pin the tolerance too so
        # a future change to `passed` cannot mask a real divergence.
        assert abs(check.delta_incremental - check.delta_direct) <= ATOL

    # Agreement with the already-trusted scalar form, group by group.
    for group in sorted(groups):
        scalar = mcpu_core.PhysicsVerifier.verify_potential_delta(
            context, old_state, proposed_state, patch, group, ATOL
        )
        vector = next(c for c in checks if int(c.energy_group) == group)
        assert scalar.passed == vector.passed
        assert scalar.delta_incremental == pytest.approx(
            vector.delta_incremental, abs=1e-6
        )
        assert scalar.delta_direct == pytest.approx(vector.delta_direct, abs=1e-6)
