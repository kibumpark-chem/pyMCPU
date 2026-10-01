"""A move slot with zero weight is never proposed, even at the edge of the roll.

``set_move_weights`` normalizes in float32, so the pivot and KIC weights can
sum to just under 1 (0.4/0.2/0 gives 0.99999994f). ``uniform_real_distribution
<float>`` can return exactly that value -- about once per 1e7 steps -- and the
slot selector used to fall through to the zero-weight sidechain slot. That
wasted a step, and with per-kind move columns it also made the next ``run()``
fail because a kind with no CSV column had been attempted.

The test forces that roll by loading a Mersenne Twister state whose next
output is 0xFFFFFFFF, a value the generator produces naturally.
"""

from __future__ import annotations

import numpy as np
import pytest

md = pytest.importorskip("mdtraj")

import pymcpu as mc  # noqa: E402
from pymcpu.runners import default_example_pdb  # noqa: E402

_MASK = 0xFFFFFFFF


def _temper(y: int) -> int:
    """std::mt19937's output tempering."""
    y ^= y >> 11
    y ^= (y << 7) & 0x9D2C5680
    y ^= (y << 15) & 0xEFC60000
    y ^= y >> 18
    return y & _MASK


def _untemper(y: int) -> int:
    """Invert :func:`_temper`."""

    def undo_right(v: int, shift: int) -> int:
        result = v
        for _ in range(32 // shift + 1):
            result = v ^ (result >> shift)
        return result & _MASK

    def undo_left(v: int, shift: int, mask: int) -> int:
        result = v
        for _ in range(32 // shift + 1):
            result = v ^ ((result << shift) & mask)
        return result & _MASK

    y = undo_right(y, 18)
    y = undo_left(y, 15, 0xEFC60000)
    y = undo_left(y, 7, 0x9D2C5680)
    return undo_right(y, 11)


def _force_next_output(integrator, value: int) -> None:
    words = integrator.get_rng_state().split()
    assert len(words) == 625  # 624 state words, then the position
    words[623] = str(_untemper(value))
    words[624] = "623"  # the next draw returns words[623] tempered, with no twist
    integrator.set_rng_state(" ".join(words))


def test_untemper_inverts_temper() -> None:
    for value in (0, 1, 0xFFFFFFFF, 0x12345678, 0xDEADBEEF):
        assert _temper(_untemper(value)) == value


def test_zero_weight_sidechain_slot_is_unreachable(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    traj = md.load(str(default_example_pdb()))
    heavy = traj.atom_slice(traj.topology.select("not element H"))
    ff = mc.MCPUForceField(heavy)
    system = ff.create_system(heavy.topology)
    integrator = mc.Integrator(temperature=0.6, step_size_rad=0.1)
    integrator.set_seed(11)
    integrator.set_move_weights(0.4, 0.2, 0.0)
    pivot, kic, _ = integrator.move_weights()
    assert np.float32(pivot) + np.float32(kic) < np.float32(1.0)  # the float gap exists

    sim = mc.Simulation(heavy.topology, system, integrator)
    sim.context.set_positions((ff.coords[0] * 10.0).T.astype(np.float32))
    sim.add_energy_reporter("e.csv", interval=1)

    _force_next_output(integrator, 0xFFFFFFFF)  # this step's slot roll -> 0.99999994f
    sim.step(1)

    assert integrator.get_sc_attempted() == 0
    assert integrator.get_rotamer_attempted() == 0
    assert integrator.get_kic_attempted() == 1  # the roll lands in the last weighted slot
    sim.step(5)  # and the energy CSV keeps its columns: no error
