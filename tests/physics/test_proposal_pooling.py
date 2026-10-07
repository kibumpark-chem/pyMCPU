"""Determinism and exactness of the reused proposal buffer.

The integrator syncs one proposal State per run and, after a step that does
not commit, restores only the moved atoms. These are self-consistency checks
on that path: same-seed reproducibility, ``verify_physics_consistency``
(every delta against a full recompute), and a finite-energy run.
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
    smoke run (doesn't need the full actin default -- just
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
    a = run_hotpath(seed=12345, steps=steps)
    b = run_hotpath(seed=12345, steps=steps)
    # Assert against the actual `steps` value passed in, not a separately
    # hardcoded literal -- so the two can't silently drift apart if `steps`
    # is ever changed (the old test re-hardcoded 200 here independently).
    assert len(a.bits) == steps
    assert a.bits == b.bits
    assert a.accept == b.accept
    assert a.energy == pytest.approx(b.energy, abs=1e-5)


def test_actin_verify() -> None:
    ctx, _ = build_test_context(with_qbias=False)
    integ = mcpu_core.Integrator(temperature=300.0, step_size_rad=0.05)
    integ.verify_physics_consistency(ctx, num_steps=30, atol=ATOL)


def test_reused_proposal_runs_finite_energy(minimal_pdb_path: str) -> None:
    """A run on the reused proposal buffer ends with a finite energy."""
    ctx = _build_small_context(minimal_pdb_path)
    integ = mcpu_core.Integrator(temperature=0.6, step_size_rad=0.05)
    integ.set_seed(9)
    integ.run(ctx, 100)
    e = float(ctx.get_state().current_energy)
    assert math.isfinite(e)
