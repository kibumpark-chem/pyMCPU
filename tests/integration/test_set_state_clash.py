"""Coordinates set from outside the moves are checked for hard-core overlaps.

No move can leave a pair more than 0.001 A under its hard-core cutoff, so an
overlap in an accepted state came in with its coordinates: a checkpoint
restore, a replica swap, EngineSession.set_coords. Each of those now checks
the state it sets, the way Simulation.step does: StericClashError by
default, a warning under MCPU_CLASH_FATAL=0. The MPI checks used to ignore
MCPU_CLASH_FATAL, and folding resume, set_coords and replica swaps did not
check at all, so the recompute silently kept the previous energy.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from pymcpu.config import EngineSpec
from pymcpu.sampling import EngineSession, swap_context_coordinates
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
