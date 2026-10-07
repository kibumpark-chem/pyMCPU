"""KORP's per-state cache of frames and pair energies must track the coordinates.

OrientationalPairPotential keeps the accepted state's residue frames and pair
energies on the State and folds each accepted move into them, so the old side
of a move's delta is a lookup. If a commit were missed or applied to the wrong
move, the running energy would walk away from a full recompute; these runs
check that it does not, with no full recompute inside the run.
"""

from __future__ import annotations

import pytest

from pymcpu.forcefields.korp import KORPForceField

from .test_korp_forcefield import _simulate, chain_traj  # noqa: F401


@pytest.mark.parametrize("weights,seed", [((1.0, 0.0, 0.0), 3), ((0.5, 0.5, 0.0), 5)])
def test_running_energy_tracks_a_full_recompute(chain_traj, weights, seed):  # noqa: F811
    ff = KORPForceField(chain_traj)
    sim = _simulate(ff, chain_traj)
    sim.full_energy_every_steps = 10**9
    sim.integrator.set_seed(seed)
    sim.integrator.set_move_weights(*weights)
    sim.step(3000)
    stats = sim.integrator.move_stats()
    accepted = stats["num_accept_pivot"] + stats["num_accept_kic"]
    assert accepted > 50, "too few accepted moves to exercise the cache"

    ctx = sim.context
    running = float(ctx.get_state().current_energy)
    full = float(ctx.calculate_total_energy(-1))
    assert abs(running - full) <= max(1e-3, 1e-5 * abs(full)), (running, full)


def test_the_cache_follows_new_positions(chain_traj):  # noqa: F811
    import numpy as np

    native = (KORPForceField(chain_traj).coords[0] * 10.0).T.astype(np.float32)

    ff = KORPForceField(chain_traj)
    sim = _simulate(ff, chain_traj)
    sim.full_energy_every_steps = 10**9
    sim.integrator.set_seed(11)
    sim.integrator.set_move_weights(1.0, 0.0, 0.0)
    sim.step(1000)

    # Jump back to the start. The cache built for the moved state must be
    # dropped, or the next moves score against it; the run recomputes the
    # energy for the new coordinates on entry.
    ctx = sim.context
    ctx.set_positions(native)
    sim.step(1000)
    running = float(ctx.get_state().current_energy)
    full = float(ctx.calculate_total_energy(-1))
    assert abs(running - full) <= max(1e-3, 1e-5 * abs(full)), (running, full)
