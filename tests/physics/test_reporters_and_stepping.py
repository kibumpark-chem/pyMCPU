"""Simulation/reporter plumbing: list-sync mechanics and step-offset accounting.

Covers three things about the OpenMM-style ``Simulation`` <-> C++ ``Integrator``
reporter wiring, all fast and actin/chignolin-free (a tiny synthetic
backbone-only system is enough):

1. ``sim.reporters``/``sim.integrator``'s internal reporter list stay in sync
   through add/remove/clear, including the "user mutates ``sim.reporters``
   directly, then ``step()`` resyncs it" OpenMM-compatibility case.
2. ``add_reporter`` rejects unexpected kwargs (fails fast on a typo'd kwarg
   rather than silently ignoring it).
3. ``EnergyReporter`` writes CSV rows at the correct global ``Step`` values,
   including across multiple ``sim.step()`` calls (the REMD case, where the
   global step counter must accumulate rather than reset per call).
4. ``EnergyReporter`` raises instead of silently going quiet on a real write
   failure -- regression test for a production incident (p18.8.3) where a
   transient shared-filesystem write failure silently dropped ~65,000 REMD
   cycles of logged data from three separate multi-day runs, with no
   exception anywhere: the underlying ``std::ofstream`` had no error check
   after ``write()``/``flush()``, so once it failed once it silently no-op'd
   for the rest of the process's life.

This is engine-internal bookkeeping with no legacy MCPU equivalent (the
Python reporter API is new), so it lives in physics/ as an internal
consistency check, not legacy_parity/.
"""

from __future__ import annotations

import resource
import signal
from pathlib import Path

import numpy as np
import pytest

from pymcpu import EnergyReporter, Integrator, Simulation
from tests.physics.helpers.minimal_system_builders import setup_minimal_bb_system

# EnergyReporter's CSV header (src/reporters/EnergyReporter.cpp) -- looked up
# by name below rather than hardcoding column indices, so a future column
# reorder doesn't silently break these assertions.
_MOVE_COUNTER_COLUMNS = (
    "PivotAccepted",
    "PivotAttempted",
    "SidechainAccepted",
    "SidechainAttempted",
    "KicAccepted",
    "KicAttempted",
)


def _tiny_simulation() -> Simulation:
    # Need >= 3 residues for Integrator's pivot-residue distribution.
    n_res, atoms_per = 4, 4
    system, _ctx = setup_minimal_bb_system(n_res, n_res * atoms_per)
    topology = object()  # Dummy: Simulation only stores the reference.
    integrator = Integrator(temperature=1.0, step_size_rad=0.05)
    # This system has no sidechain atoms and no chi angles, so the default mix
    # would spend half its steps on sidechain proposals that return without
    # proposing anything. run() rejects that combination rather than quietly
    # discarding the steps, so say backbone-only explicitly.
    integrator.set_move_weights(0.5, 0.5, 0.0)
    sim = Simulation(topology, system, integrator)  # type: ignore[arg-type]
    n_atoms = system.get_num_atoms()
    coords = np.zeros((3, n_atoms), dtype=np.float32)
    for i in range(n_atoms):
        coords[:, i] = (float(i), 0.0, 0.0)
    sim.context.set_positions(coords)
    return sim


def _read_energy_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Parse an EnergyReporter CSV into (header, [row-dicts]) by column name."""
    lines = path.read_text().strip().splitlines()
    header = lines[0].split(",")
    rows = [dict(zip(header, line.split(","))) for line in lines[1:]]
    return header, rows


def test_add_reporter_rejects_kwargs(tmp_path: Path) -> None:
    sim = _tiny_simulation()
    rep = EnergyReporter(str(tmp_path / "e.csv"), 1)
    with pytest.raises(TypeError):
        sim.add_reporter(rep, interval=1)  # type: ignore[call-arg]


def test_add_remove_clear_reporters(tmp_path: Path) -> None:
    sim = _tiny_simulation()
    assert sim.reporters == []
    assert sim.integrator.num_reporters() == 0

    r1 = sim.add_energy_reporter(str(tmp_path / "a.csv"), 5)
    r2 = sim.add_energy_reporter(str(tmp_path / "b.csv"), 10)
    assert sim.reporters == [r1, r2]
    assert sim.integrator.num_reporters() == 2

    sim.remove_reporter(r1)
    assert sim.reporters == [r2]
    assert sim.integrator.num_reporters() == 1

    sim.clear_reporters()
    assert sim.reporters == []
    assert sim.integrator.num_reporters() == 0


def test_step_resyncs_manual_list_mutation(tmp_path: Path) -> None:
    sim = _tiny_simulation()
    r = EnergyReporter(str(tmp_path / "e.csv"), 2)
    sim.reporters.append(r)  # User mutates the list directly (OpenMM-style).
    assert sim.integrator.num_reporters() == 0  # Not yet synced to the C++ side.
    sim.step(1)
    assert sim.integrator.num_reporters() == 1  # step() must resync before running.


def test_energy_reporter_interval(tmp_path: Path) -> None:
    sim = _tiny_simulation()
    path = tmp_path / "energy.csv"
    sim.add_energy_reporter(str(path), interval=5)
    sim.integrator.set_seed(0)
    sim.step(10)

    header, rows = _read_energy_csv(path)
    assert header[0] == "Step"
    steps = [int(row["Step"]) for row in rows]
    # OpenMM-style: initial frame at 0 (before any moves), then a completed
    # frame every `interval` steps -- for step(10) with interval=5: 0, 5, 10.
    assert steps == [0, 5, 10]

    # Initial (step-0) row must show zero attempts on every move type: no MC
    # move has run yet.
    attempted = [int(rows[0][col]) for col in _MOVE_COUNTER_COLUMNS]
    assert attempted == [0] * len(_MOVE_COUNTER_COLUMNS)
    assert int(rows[0]["WalkerId"]) >= -1  # -1 (unset) is a valid serial-run value.


def test_energy_reporter_step_offset_across_batches(tmp_path: Path) -> None:
    """REMD-style: multiple step() calls must accumulate the global Step column."""
    sim = _tiny_simulation()
    path = tmp_path / "energy.csv"
    sim.add_energy_reporter(str(path), interval=50)
    sim.integrator.set_seed(0)
    for _ in range(3):
        sim.step(50)

    header, rows = _read_energy_csv(path)
    steps = [int(row["Step"]) for row in rows]
    # 3 calls of 50 steps each, interval=50 -> frames at 0, 50, 100, 150;
    # this is the step-offset accumulation contract, not a fresh count per call.
    assert steps == [0, 50, 100, 150]
    assert sim.current_step == 150

    attempted_first = [int(rows[0][col]) for col in _MOVE_COUNTER_COLUMNS]
    assert attempted_first == [0] * len(_MOVE_COUNTER_COLUMNS)  # No moves before step 0.
    # By the second frame some moves have been attempted (exact split across
    # move types is stochastic, so we only check attempts have accumulated).
    assert int(rows[1]["PivotAttempted"]) > 0


def test_energy_reporter_raises_on_write_failure(tmp_path: Path) -> None:
    """A real OS-level write failure must raise, not silently go quiet.

    Forces an actual failed write via RLIMIT_FSIZE (the file is allowed to
    grow past the CSV header but not past the first data row), with SIGXFSZ
    ignored so the process isn't killed outright -- this reproduces exactly
    the "write() fails after some records have already been written" shape
    of the production incident, as opposed to a failure on the very first
    byte.
    """
    path = tmp_path / "energy.csv"
    old_soft, old_hard = resource.getrlimit(resource.RLIMIT_FSIZE)
    old_sigxfsz = signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
    try:
        resource.setrlimit(resource.RLIMIT_FSIZE, (200, old_hard))
        sim = _tiny_simulation()
        sim.add_energy_reporter(str(path), interval=1)
        sim.integrator.set_seed(0)
        with pytest.raises(RuntimeError, match="energy file"):
            sim.step(50)
    finally:
        resource.setrlimit(resource.RLIMIT_FSIZE, (old_soft, old_hard))
        signal.signal(signal.SIGXFSZ, old_sigxfsz)
