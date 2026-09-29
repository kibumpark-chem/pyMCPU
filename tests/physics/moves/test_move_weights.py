"""The Pivot / KIC / Sidechain slot weights.

These exist because the mix used to be three function-local floats inside
``MCIntegrator::run`` with no way to reach them, which made a backbone-only
force field impossible: its residues have no chi angles, so half of every run
was spent generating sidechain proposals that returned immediately and were
discarded without comment.

The assertion that matters most here is the *negative* one --
``test_passing_the_defaults_explicitly_changes_nothing``. Making the mix
configurable meant rewriting the dispatch that every single MC step goes
through, so the thing worth proving is that an untouched integrator still does
exactly what it did before, down to the accepted-move bits.
"""

from __future__ import annotations

import math

import mdtraj as md
import numpy as np
import pytest

import pymcpu as mc
from pymcpu import mcpu_core
from pymcpu.forcefields.mcpu import MCPUForceField
from pymcpu.runners import default_example_pdb

STEPS = 400
SEED = 20260924


def _simulation(weights=None):
    traj = md.load(str(default_example_pdb()))
    heavy = traj.atom_slice(traj.topology.select("not element H"))
    forcefield = MCPUForceField(heavy, param_set="mcpu08")
    system = forcefield.create_system(heavy.topology)

    integrator = mcpu_core.Integrator(temperature=0.6, step_size_rad=0.1)
    integrator.set_seed(SEED)
    if weights is not None:
        integrator.set_move_weights(*weights)

    sim = mc.Simulation(heavy.topology, system, integrator)
    sim.context.set_positions(
        (forcefield.coords[0] * 10.0).T.astype(np.float32))
    return sim, integrator


def _run(weights=None, steps=STEPS):
    sim, integrator = _simulation(weights)
    sim.step(steps)
    return integrator.move_stats(), list(integrator.last_accept_bits())


def test_default_is_the_historical_mix():
    integrator = mcpu_core.Integrator(temperature=0.6, step_size_rad=0.1)
    assert integrator.move_weights() == pytest.approx((0.25, 0.25, 0.50))


def test_passing_the_defaults_explicitly_changes_nothing():
    """The regression guard for the dispatch rewrite.

    0.25f + 0.25f is exactly 0.5f, and dividing by a sum of exactly 1.0f is
    exact, so this is required to hold bit-for-bit rather than approximately.
    """
    untouched_stats, untouched_bits = _run(None)
    explicit_stats, explicit_bits = _run((0.25, 0.25, 0.50))
    assert explicit_bits == untouched_bits
    assert explicit_stats == untouched_stats


def test_weights_are_normalized():
    integrator = mcpu_core.Integrator(temperature=0.6, step_size_rad=0.1)
    integrator.set_move_weights(2.0, 2.0, 4.0)
    assert integrator.move_weights() == pytest.approx((0.25, 0.25, 0.50))
    integrator.set_move_weights(1.0, 1.0, 0.0)
    assert integrator.move_weights() == pytest.approx((0.5, 0.5, 0.0))


def test_zero_sidechain_weight_proposes_no_sidechain_moves():
    """What a backbone-only force field actually needs."""
    stats, _ = _run((0.5, 0.5, 0.0))
    assert stats["num_propose_sc"] == 0
    assert stats["num_propose_rotamer"] == 0
    assert stats["num_propose_pivot"] + stats["num_propose_kic"] == STEPS


def test_zero_kic_weight_proposes_no_kic_moves():
    stats, _ = _run((1.0, 0.0, 0.0))
    assert stats["num_propose_kic"] == 0
    assert stats["num_propose_pivot"] == STEPS


def test_slot_frequencies_follow_the_weights():
    stats, _ = _run((0.8, 0.2, 0.0), steps=4000)
    pivot, kic = stats["num_propose_pivot"], stats["num_propose_kic"]
    assert pivot + kic == 4000
    # 4000 draws, p=0.8: sigma ~ 25 proposals, so 6 sigma is ~150.
    assert abs(pivot - 3200) < 150


@pytest.mark.parametrize(
    "weights",
    [(-1.0, 1.0, 1.0), (1.0, -0.5, 1.0), (0.0, 0.0, 0.0),
     (math.nan, 1.0, 1.0), (math.inf, 1.0, 1.0)],
)
def test_invalid_weights_are_rejected(weights):
    integrator = mcpu_core.Integrator(temperature=0.6, step_size_rad=0.1)
    with pytest.raises(ValueError):
        integrator.set_move_weights(*weights)
    # And the integrator is left on its previous, valid setting.
    assert integrator.move_weights() == pytest.approx((0.25, 0.25, 0.50))


def test_same_weights_and_seed_reproduce_the_same_run():
    first_stats, first_bits = _run((0.7, 0.3, 0.0))
    second_stats, second_bits = _run((0.7, 0.3, 0.0))
    assert first_bits == second_bits
    assert first_stats == second_stats
