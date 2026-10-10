"""Guards that turn two silent failure modes into errors.

Both are things the engine used to do quietly, and quietly is the problem:

* A system whose residues have no chi angles (a backbone-only force field)
  still drew sidechain moves, which returned immediately without proposing
  anything. The run completed, the step counter advanced, and half the budget
  produced nothing. Nothing in the output said so.
* ``uniform_int_distribution(1, N - 2)`` is undefined for fewer than three
  residues. In a release build that does not crash; it returns whatever the
  implementation happens to produce.
"""

from __future__ import annotations

import numpy as np
import pytest

from pymcpu import mcpu_core
from tests.physics.helpers.minimal_system_builders import setup_minimal_bb_system


def _backbone_only(n_res=8):
    """A chi-free system: exactly the shape a backbone-only force field builds."""
    n_atoms = 4 * n_res  # [N,CA,C] * n_res, then [O] * n_res
    system, context = setup_minimal_bb_system(n_res, n_atoms)
    coords = np.zeros((3, n_atoms), dtype=np.float32)
    for r in range(n_res):
        base = 3.8 * r
        coords[0, 3 * r:3 * r + 3] = [base, base + 1.2, base + 2.4]
        coords[0, 3 * n_res + r] = base + 2.9
    # A hand-built System must supply KIC's start-structure closure targets
    # itself; MCPUForceField.create_system does it for you.
    system.set_kic_reference(coords)
    context.set_positions(coords)
    return system, context


def _integrator(weights=None):
    integrator = mcpu_core.Integrator(temperature=0.6, step_size_rad=0.1)
    integrator.set_seed(7)
    if weights is not None:
        integrator.set_move_weights(*weights)
    return integrator


def test_sidechain_weight_on_a_chi_free_system_is_rejected():
    _, context = _backbone_only()
    integrator = _integrator()  # default mix still has 0.5 on sidechain
    with pytest.raises(ValueError, match="no residue"):
        integrator.run(context, 10)


def test_the_error_says_how_to_fix_it():
    _, context = _backbone_only()
    integrator = _integrator()
    with pytest.raises(ValueError, match=r"set_move_weights\(pivot, kic, 0\.0\)"):
        integrator.run(context, 10)


def test_zero_sidechain_weight_runs_on_a_chi_free_system():
    _, context = _backbone_only()
    integrator = _integrator((0.5, 0.5, 0.0))
    integrator.run(context, 20)  # must not raise
    stats = integrator.move_stats()
    assert stats["num_propose_sc"] == 0
    assert stats["num_propose_pivot"] + stats["num_propose_kic"] == 20


def test_verify_physics_consistency_applies_the_same_guard():
    _, context = _backbone_only()
    integrator = _integrator()
    with pytest.raises(ValueError, match="no residue"):
        integrator.verify_physics_consistency(context, 5, 1e-3)


@pytest.mark.parametrize("n_res", [1, 2])
def test_too_few_residues_is_rejected_rather_than_undefined(n_res):
    _, context = _backbone_only(n_res)
    integrator = _integrator((1.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="at least 3 residues"):
        integrator.run(context, 5)


def test_three_residues_is_allowed():
    _, context = _backbone_only(3)
    integrator = _integrator((1.0, 0.0, 0.0))
    integrator.run(context, 5)  # the boundary case must still work
