"""The running energy is exact after a run, whatever changed before it.

The running total ``current_energy`` is updated from each accepted move's
delta, so it is only right if it started from a full recompute under the
current energy definition. The Integrator recomputes on entry when the
coordinates were replaced or anything that defines the energy changed: the
energy mask, a group weight, the legacy-weights switch, the native-contacts
bias, or which potentials are enabled. Before that, each of these left a
constant offset (e.g. -123.1 on T4L after an ignore_all mask) until the next
periodic recompute.

Every test here runs with the periodic recompute off
(``full_energy_every_steps = 10**9``), so only the resync on entry can make
the running total match a full recompute.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import pymcpu as mc
from pymcpu.forcefields.mcpu import MCPUForceField
from pymcpu.simulation import StericClashError

PDB = Path(__file__).resolve().parents[2] / "examples" / "data" / "1uao.pdb"
EXACT = 1e-9


@pytest.fixture(scope="module")
def built():
    import mdtraj as md

    if not PDB.exists():
        pytest.skip(f"missing fixture {PDB}")
    traj = md.load(str(PDB))
    heavy = traj.atom_slice(traj.topology.select("not element H"))
    ff = MCPUForceField(heavy, param_set="mcpu08")
    return heavy, ff


def _simulation(built, seed=7, system=None, temperature=0.6):
    heavy, ff = built
    system = system if system is not None else ff.create_system(heavy.topology)
    integrator = mc.Integrator(temperature=temperature, step_size_rad=0.1)
    integrator.set_seed(seed)
    sim = mc.Simulation(heavy.topology, system, integrator)
    sim.full_energy_every_steps = 10**9
    sim.context.set_positions((ff.coords[0] * 10.0).T.astype(np.float32))
    return sim


def _offset(sim) -> float:
    running = float(sim.context.get_state().current_energy)
    return abs(running - float(sim.context.calculate_total_energy(-1)))


def test_set_positions_then_run_is_exact(built):
    sim = _simulation(built)
    assert float(sim.context.get_state().current_energy) == 0.0
    sim.step(300)
    assert sim.context.energy_resyncs == 1
    assert _offset(sim) < EXACT


def test_resync_does_not_change_the_trajectory(built):
    def bits(seed_by_hand):
        sim = _simulation(built)
        if seed_by_hand:
            sim.context.calculate_total_energy(-1)
        sim.step(300)
        return "".join(str(int(b)) for b in sim.integrator.last_accept_bits())

    assert bits(False) == bits(True)


def test_no_resync_when_nothing_changed(built):
    sim = _simulation(built)
    sim.step(100)
    before = sim.context.energy_resyncs
    for _ in range(3):
        sim.step(100)
    assert sim.context.energy_resyncs == before


@pytest.mark.parametrize("mode", ["ignore_all", "clash_only"])
def test_mask_set_and_clear(built, mode):
    sim = _simulation(built)
    sim.step(200)
    sim.system.set_energy_ignored_residues([3, 4], mode)
    sim.step(200)
    assert _offset(sim) < EXACT
    if mode == "clash_only":  # an ignore_all run may leave overlaps behind
        sim.system.clear_energy_ignored_residues()
        sim.step(200)
        assert _offset(sim) < EXACT


def test_clearing_a_mask_over_overlapping_atoms_raises(built):
    # Residues masked with ignore_all are never clash-tested, so a hot masked
    # run lets them pass through each other and the rest of the chain.
    sim = _simulation(built, temperature=3.0)
    sim.system.set_energy_ignored_residues(list(range(2, 8)), "ignore_all")
    sim.step(5000)
    unmasked = _simulation(built)  # its own System, with no mask
    unmasked.context.set_positions(np.array(sim.context.coords, dtype=np.float32))
    unmasked.context.calculate_total_energy(-1)
    assert unmasked.context.has_steric_clash(), "the masked run left no overlap"

    sim.system.clear_energy_ignored_residues()
    with pytest.raises(StericClashError, match="clash before the first move"):
        sim.step(1)
    with pytest.raises(StericClashError):  # still not in step: raises again
        sim.step(1)


@pytest.mark.parametrize(
    "change",
    [
        lambda sim: sim.context.set_energy_weight(1, 0.9),
        lambda sim: sim.context.set_use_legacy_weights(False),
        lambda sim: sim.context.set_native_contacts_bias(0.5, 3.0),
        lambda sim: sim.system.get_potentials()[0].set_enabled(False),
    ],
    ids=["weight", "legacy_weights", "bias", "enabled"],
)
def test_energy_definition_change_is_exact_after_run(built, change):
    sim = _simulation(built)
    sim.step(200)
    before = sim.context.energy_resyncs
    change(sim)
    sim.step(200)
    assert sim.context.energy_resyncs == before + 1
    assert _offset(sim) < EXACT


def test_two_contexts_on_one_system(built):
    heavy, ff = built
    system = ff.create_system(heavy.topology)
    a = _simulation(built, seed=1, system=system)
    b = _simulation(built, seed=2, system=system)
    a.step(200)
    b.step(200)
    system.set_energy_ignored_residues([2], "clash_only")
    a.step(200)
    b.step(200)
    assert _offset(a) < EXACT
    assert _offset(b) < EXACT


def test_full_energy_every_steps_counts_steps(built):
    sim = _simulation(built)
    sim.full_energy_every_steps = 250
    calls = []
    real = sim.recompute_energy
    sim.recompute_energy = lambda: calls.append(sim.current_step) or real()
    for _ in range(6):
        sim.step(100)
    assert calls == [300, 600]


def test_old_cadence_name_raises(built):
    sim = _simulation(built)
    with pytest.raises(AttributeError, match="full_energy_every_steps"):
        sim.full_energy_every = 10**9
    with pytest.raises(AttributeError, match="full_energy_every_steps"):
        sim.full_energy_every  # noqa: B018


@pytest.mark.parametrize("bad", [None, 0, -5, 2.5, "10"])
def test_cadence_must_be_a_positive_whole_number(bad):
    from pymcpu.config import IntegratorConfig

    with pytest.raises(ValueError, match="full_energy_every_steps"):
        IntegratorConfig(full_energy_every_steps=bad)
