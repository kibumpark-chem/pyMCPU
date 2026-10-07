"""Long-run precision of the running energy total and of the geometry.

Energy sums, deltas and ``State.current_energy`` are double, so the running
total stays within rounding of a full recompute however many moves are
accepted.

The tolerance 1e-7*|E| + 1e-6 leaves room for the delta and full paths
contracting per-pair arithmetic into FMAs differently, and is still three
orders of magnitude below a single missed contact.
"""

from __future__ import annotations

from pathlib import Path

import mdtraj as md
import numpy as np
import pytest

from pymcpu import mcpu_core

REPO = Path(__file__).resolve().parents[2]
ACTIN_PDB = REPO / "examples" / "actin" / "input_pdb" / "acta.pdb"


def _drift_tol(energy: float) -> float:
    return 1e-7 * abs(energy) + 1e-6


def test_running_energy_tracks_full_recompute_over_1e5_steps(chignolin_context) -> None:
    """1e5 steps with no recompute: the running total matches a full one."""
    ctx = chignolin_context
    ctx.calculate_total_energy(-1)
    integrator = mcpu_core.Integrator(temperature=0.6, step_size_rad=0.1)
    integrator.set_seed(7)
    integrator.run(ctx, 100_000)
    accepted = integrator.get_bb_accepted() + integrator.get_sc_accepted()
    assert accepted > 5_000
    running = ctx.get_state().current_energy
    full = ctx.calculate_total_energy(-1)
    assert abs(running - full) <= _drift_tol(full), (running, full, running - full)


def _bonds(X: np.ndarray, window: int = 32) -> np.ndarray:
    """Pairs closer than 1.95 A among atoms at most ``window`` apart in the
    atom list. Covalent partners sit in the same or the next residue, and a
    residue has at most 14 heavy atoms, so the window finds every bond."""
    pairs = []
    for k in range(1, window + 1):
        i = np.nonzero(np.linalg.norm(X[k:] - X[:-k], axis=1) < 1.95)[0]
        pairs.append(np.stack([i, i + k], 1))
    return np.concatenate(pairs)


def _lengths(X: np.ndarray, bonds: np.ndarray) -> np.ndarray:
    return np.linalg.norm(X[bonds[:, 0]] - X[bonds[:, 1]], axis=1)


@pytest.mark.slow
def test_bond_lengths_do_not_drift_over_1e6_steps() -> None:
    """Coordinates are float32 storage of double arithmetic, so bonds only
    random-walk at rounding size. A move that rounds with a bias would show
    up here as rms > 1e-3 A."""
    if not ACTIN_PDB.exists():
        pytest.skip(f"{ACTIN_PDB} not found")
    from pymcpu.forcefields.mcpu import MCPUForceField

    traj = md.load(str(ACTIN_PDB))
    traj = traj.atom_slice(traj.topology.select("not element H"))
    ff = MCPUForceField(traj)
    ctx = mcpu_core.Context(ff.create_system(traj.topology))
    ctx.set_positions((ff.coords[0] * 10.0).T.astype(np.float32))
    ctx.calculate_total_energy(-1)

    X0 = np.asarray(ctx.coords, dtype=np.float64).T
    bonds = _bonds(X0)
    integrator = mcpu_core.Integrator(temperature=0.6, step_size_rad=0.1)
    integrator.set_seed(1)
    integrator.run(ctx, 10_000)
    # Close non-bonded pairs of the input (clashes) are not kept rigid; keep
    # only the distances the first moves left unchanged.
    X1 = np.asarray(ctx.coords, dtype=np.float64).T
    bonds = bonds[np.abs(_lengths(X1, bonds) - _lengths(X0, bonds)) < 0.01]
    assert len(bonds) > 1000

    integrator.run(ctx, 990_000)
    X = np.asarray(ctx.coords, dtype=np.float64).T
    db = _lengths(X, bonds) - _lengths(X0, bonds)
    assert float(np.sqrt((db**2).mean())) < 1e-3
    assert float(np.abs(db).max()) < 5e-3

    running = ctx.get_state().current_energy
    full = ctx.calculate_total_energy(-1)
    assert abs(running - full) <= _drift_tol(full), (running, full)
