"""Coordinates set from outside the moves are checked for hard-core overlaps.

No move can leave a pair more than 0.001 A under its hard-core cutoff, so an
overlap in an accepted state came in with its coordinates: a checkpoint
restore, a replica swap, EngineSession.set_coords, a driver's start
structure. Each of those now checks
the state it sets, the way Simulation.step does: StericClashError by
default, a warning under MCPU_CLASH_FATAL=0. The MPI checks used to ignore
MCPU_CLASH_FATAL, and folding resume, set_coords and replica swaps did not
check at all, so the recompute silently kept the previous energy.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pytest

from pymcpu.config import EngineSpec, normalize_move_settings
from pymcpu.sampling import EngineSession, swap_context_coordinates
from pymcpu.sampling.replica_exchange_core import build_replica_simulation
from pymcpu.simulation import StericClashError


@pytest.fixture
def spec(engine_spec_factory) -> EngineSpec:
    return engine_spec_factory()


def _overlap(coords: np.ndarray) -> np.ndarray:
    """TRP8's CD2 put 0.5 A from PRO3's CA (atoms 70 and 10 in 1UAO's engine order)."""
    out = np.array(coords, dtype=np.float32, copy=True)
    out[:, 70] = out[:, 10] + np.array([0.5, 0.0, 0.0], dtype=np.float32)
    return out


def test_set_coords_refuses_an_overlap(spec: EngineSpec) -> None:
    eng = EngineSession(spec)
    clean = eng.coords()
    eng.set_coords(clean)
    with pytest.raises(StericClashError, match="EngineSession.set_coords"):
        eng.set_coords(_overlap(clean))


def test_mcpu_clash_fatal_0_warns_instead(spec: EngineSpec, monkeypatch, caplog) -> None:
    monkeypatch.setenv("MCPU_CLASH_FATAL", "0")
    eng = EngineSession(spec)
    with caplog.at_level(logging.WARNING, logger="pymcpu.simulation"):
        eng.set_coords(_overlap(eng.coords()))
    assert any("steric clash at EngineSession.set_coords" in r.getMessage() for r in caplog.records)


def test_a_replica_swap_checks_both_states(spec: EngineSpec) -> None:
    eng_a, eng_b = EngineSession(spec), EngineSession(spec)
    clean = eng_a.coords()
    ctx_a, ctx_b = eng_a._ensure_sim().context, eng_b._ensure_sim().context
    swap_context_coordinates(ctx_a, ctx_b, clean, clean)
    with pytest.raises(StericClashError, match="replica swap"):
        swap_context_coordinates(ctx_a, ctx_b, clean, _overlap(clean))


def test_a_clashing_start_structure_raises(spec: EngineSpec) -> None:
    """The drivers seed the running energy with a full recompute. That
    recompute is the resync the first run would otherwise do, so the driver
    checks its verdict; without the check the run started on an unseeded
    energy and nothing raised until the periodic recompute."""
    sim = EngineSession(spec)._ensure_sim()
    clean = np.asarray(sim.context.coords, dtype=np.float32)

    def build(coords: np.ndarray):
        return build_replica_simulation(
            filtered_traj=SimpleNamespace(topology=sim.topology), system=sim.system,
            temperature=0.6, seed=1, replica_idx=0, fixed_residues=[], n_res=0,
            coords_angstroms=coords.T, k_bias=0.0, n_target=0.0, step_size_rad=0.1,
            move_settings=normalize_move_settings(),
        )

    build(clean).step(10)
    with pytest.raises(StericClashError, match="start structure of replica 0"):
        build(_overlap(clean))
