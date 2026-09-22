"""Delta-vs-full energy self-consistency across the full MCPU forcefield.

pyMCPU maintains two independent code paths for the same physical quantity:
an incremental "delta energy" applied during MC proposals, and a full
recomputation from scratch. These must agree for every energy group, or the
Metropolis acceptance criterion silently uses the wrong energy. This is a
pure internal consistency property -- no legacy MCPU reference value is
involved.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.slow

from pymcpu import mcpu_core
from tests.fixtures.context_builders import ATOL


def test_incremental_delta_matches_full_recompute_for_all_potentials(chignolin_context) -> None:
    """Across 10 random MC proposals, calculate_energy_change must match
    E(new) - E(new) recomputed from scratch, per energy group."""
    integrator = mcpu_core.Integrator(temperature=300.0, step_size_rad=0.1)
    mcpu_core.PhysicsVerifier.verify_mc_energy_consistency(
        integrator, chignolin_context, num_steps=10, atol=ATOL
    )
