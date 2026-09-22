"""``Simulation`` must seed the running total energy on its first ``step()``.

The bug this pins, and why it was invisible:

``Context.set_positions()`` does not compute an energy, so ``current_energy``
is 0.0 on a fresh ``Context``. ``Simulation.step()`` deliberately dropped its
unconditional pre-run O(N^2) recompute as redundant -- ``current_energy`` is
exact on entry via the post-run recompute of the previous cycle, via
``swap_context_coordinates()`` after an accepted REMD exchange, and via
checkpoint restore. All three require a PREVIOUS cycle, so none of them covers
the first ``step()`` call, and the incremental accumulator then stayed off by
exactly the starting energy for the whole run.

Measured on the README quickstart (1UAO, seed 42, T=0.6): a constant
14.706589 offset, *independent of step count* -- 100 steps and 20 000 steps
showed the same value to 5 decimals, which is what distinguishes a missing
initial term from float32 accumulation. Once seeded, 10 000 steps drift by
2.9e-5, three orders of magnitude under ``energy_drift_warn_atol`` (1e-3).

Accept bits are identical either way, because Metropolis uses dE and never the
running total -- so this corrupted only REPORTED energies. That still mattered:
the documented quickstart printed an energy-drift warning, and
``attempt_exchange()`` reads ``current_energy`` for REMD acceptance.

``full_energy_every`` defaults to 1 (recompute after every ``step()`` call),
which resynced the value and hid the bug from every short-call test. It is
raised here on purpose -- that is the only way to observe the accumulator.
"""

from __future__ import annotations

import numpy as np
import pytest

import pymcpu as mc
from pymcpu.forcefields.mcpu import MCPUForceField


def _fresh_simulation(chignolin_pdb):
    import mdtraj as md

    traj = md.load(str(chignolin_pdb))
    heavy = traj.atom_slice(traj.topology.select("not element H"))
    forcefield = MCPUForceField(heavy, param_set="mcpu_v1")
    system = forcefield.create_system(heavy.topology)
    integrator = mc.Integrator(temperature=0.6, step_size_rad=0.1)
    integrator.set_seed(42)
    sim = mc.Simulation(heavy.topology, system, integrator)
    sim.context.set_positions((forcefield.coords[0] * 10.0).T.astype(np.float32))
    return sim


@pytest.fixture()
def chignolin_pdb():
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "examples" / "data" / "1uao.pdb"
    if not path.exists():
        pytest.skip(f"missing fixture {path}")
    return path


def test_current_energy_is_zero_before_any_step(chignolin_pdb) -> None:
    """Documents the precondition: set_positions() does NOT seed the total.

    If this ever starts failing because ``set_positions`` learned to compute an
    energy, the seeding in ``Simulation.step()`` becomes redundant and should be
    removed rather than left as a second source of truth.
    """
    sim = _fresh_simulation(chignolin_pdb)
    assert float(sim.context.get_state().current_energy) == 0.0


def test_first_step_seeds_running_energy(chignolin_pdb) -> None:
    """After one step with recompute disabled, the accumulator must be right."""
    sim = _fresh_simulation(chignolin_pdb)
    sim.full_energy_every = 10**9  # disable the resync that would mask the bug
    sim.step(200)

    incremental = float(sim.context.get_state().current_energy)
    recomputed = float(sim.context.calculate_total_energy(-1))
    assert incremental == pytest.approx(recomputed, abs=sim.energy_drift_warn_atol), (
        f"running energy {incremental} disagrees with a full recompute "
        f"{recomputed} by {abs(recomputed - incremental)}; the first step() did "
        f"not seed current_energy (see this module's docstring)"
    )


def test_offset_does_not_grow_with_step_count(chignolin_pdb) -> None:
    """A *constant* disagreement means a missing initial term, not drift.

    This is the discriminating measurement, so it is asserted rather than left
    as a note: whatever residual exists must not scale with the step count.
    """
    residuals = []
    for steps in (100, 1000):
        sim = _fresh_simulation(chignolin_pdb)
        sim.full_energy_every = 10**9
        sim.step(steps)
        inc = float(sim.context.get_state().current_energy)
        rec = float(sim.context.calculate_total_energy(-1))
        residuals.append(abs(rec - inc))

    for steps, residual in zip((100, 1000), residuals):
        assert residual < sim.energy_drift_warn_atol, (
            f"{steps} steps left a residual of {residual}, above "
            f"{sim.energy_drift_warn_atol}"
        )


def test_seeding_does_not_change_the_trajectory(chignolin_pdb) -> None:
    """Seeding must be observationally neutral on the Markov chain itself.

    Metropolis consumes dE, never the running total, so seeding may correct the
    reported energy but must not move a single accept bit. That invariant is
    what made this safe to change after the C3 rename was verified bit-exact.
    """
    sim = _fresh_simulation(chignolin_pdb)
    sim.full_energy_every = 10**9
    sim.step(300)
    bits_seeded = "".join(str(int(b)) for b in sim.integrator.last_accept_bits())

    # Same run, but with the running total deliberately pre-seeded by hand --
    # the idiom every scripts/parity_*.py uses. Must be the same chain.
    sim2 = _fresh_simulation(chignolin_pdb)
    sim2.context.calculate_total_energy(-1)
    sim2.full_energy_every = 10**9
    sim2.step(300)
    bits_manual = "".join(str(int(b)) for b in sim2.integrator.last_accept_bits())

    assert bits_seeded == bits_manual
    assert len(bits_seeded) == 300
