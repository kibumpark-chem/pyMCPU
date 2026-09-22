"""Determinism + proposal-pool invariants after the State/ProposalPatch
pooling refactor.

All three tests are self-consistency checks on pyMCPU's own engine
(same-seed reproducibility, the internal ``verify_physics_consistency``
checker, and a finite-energy sanity check on the pooled-proposal code path)
-- none references dbfold_actin or a legacy numeric value.
"""

from __future__ import annotations

import math

import mdtraj as md
import pytest

from pymcpu import mcpu_core
from pymcpu.forcefields.mcpu import MCPUForceField
from tests.fixtures.context_builders import (
    ATOL,
    build_test_context,
    require_safe_math_for_accept_determinism,
)
from tests.physics.helpers.hotpath_runs import run_hotpath

pytestmark = pytest.mark.slow


def _build_small_context(pdb_path: str) -> "mcpu_core.Context":
    """Build a minimal-overhead context from ``minimal_pdb_path`` for a fast
    pooled-proposal smoke run (doesn't need the full actin default -- just
    a real, valid small structure)."""
    traj = md.load(pdb_path)
    if any(atom.element.symbol == "H" for atom in traj.topology.atoms):
        traj = traj.atom_slice(traj.topology.select("not element H"))
    forcefield = MCPUForceField(traj)
    system = forcefield.create_system(traj.topology)
    ctx = mcpu_core.Context(system)
    ctx.set_positions((forcefield.coords[0] * 10.0).T.astype("float32"))
    ctx.calculate_total_energy(-1)
    return ctx


def test_accept_bits_deterministic_fixed_seed() -> None:
    """Same seed -> identical accept/reject sequence and totals."""
    require_safe_math_for_accept_determinism()
    steps = 200
    a = run_hotpath(seed=12345, steps=steps, pooled_proposal=False)
    b = run_hotpath(seed=12345, steps=steps, pooled_proposal=False)
    # Assert against the actual `steps` value passed in, not a separately
    # hardcoded literal -- so the two can't silently drift apart if `steps`
    # is ever changed (the old test re-hardcoded 200 here independently).
    assert len(a.bits) == steps
    assert a.bits == b.bits
    assert a.accept == b.accept
    assert a.energy == pytest.approx(b.energy, abs=1e-5)


def test_actin_verify_skin0_and_skin1() -> None:
    for skin in (0.0, 1.0):
        ctx, _ = build_test_context(with_qbias=False)
        ctx.set_mu_skin(skin)
        integ = mcpu_core.Integrator(temperature=300.0, step_size_rad=0.05)
        integ.verify_physics_consistency(ctx, num_steps=30, atol=ATOL)


def test_pooled_proposal_runs_finite_energy(minimal_pdb_path: str) -> None:
    """Pooled proposal path completes with finite energy (proxy timers were
    removed alongside the pooling refactor, so this only asserts the run
    doesn't produce NaN/inf, not any specific value)."""
    ctx = _build_small_context(minimal_pdb_path)
    ctx.set_mu_skin(0.0)
    integ = mcpu_core.Integrator(temperature=0.6, step_size_rad=0.05)
    if hasattr(integ, "set_use_pooled_proposal"):
        integ.set_use_pooled_proposal(True)
    integ.set_seed(9)
    integ.run(ctx, 100)
    e = float(ctx.get_state().current_energy)
    assert math.isfinite(e)
