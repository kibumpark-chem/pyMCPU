"""Delta-vs-full energy consistency holds regardless of whether legacy
outer weights are applied.

Pure internal consistency (same property as ``test_energy_conservation.py``)
-- these only check that incremental and full-recompute energies still
agree with each other once weighting is toggled, not that any weight value
itself is correct (see ``tests/legacy_parity/test_legacy_weight_constants.py``
for that).
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.slow

from pymcpu import mcpu_core
from tests.fixtures.context_builders import ATOL


def test_delta_consistency_with_legacy_weights_enabled(chignolin_context) -> None:
    ctx = chignolin_context
    ctx.set_use_legacy_weights(True)
    ctx.calculate_total_energy(-1)
    integrator = mcpu_core.Integrator(temperature=300.0, step_size_rad=0.1)
    integrator.verify_physics_consistency(ctx, num_steps=25, atol=ATOL)
    mcpu_core.PhysicsVerifier.verify_mc_energy_consistency(
        integrator, ctx, num_steps=20, atol=ATOL
    )


def test_delta_consistency_with_legacy_weights_disabled(chignolin_context) -> None:
    ctx = chignolin_context
    ctx.set_use_legacy_weights(False)
    ctx.calculate_total_energy(-1)
    integrator = mcpu_core.Integrator(temperature=300.0, step_size_rad=0.1)
    integrator.verify_physics_consistency(ctx, num_steps=25, atol=ATOL)
