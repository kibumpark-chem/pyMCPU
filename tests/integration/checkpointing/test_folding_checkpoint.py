"""Folding-mode checkpoint/resume orchestration tests.

Covers ``FoldingCheckpointState`` save/load round-trips and
``FoldingRunner.run()``'s checkpoint/resume control flow: that the opt-in
early stop (``q_threshold``) saves where it stops and reports the steps it
ran, that a folded start runs every cycle by default, and that convergence
history updates every cycle and survives a resume. Whether a resumed run
writes what an uninterrupted run writes is checked in
``test_resume_matches_uninterrupted.py``.

There is no legacy-MCPU equivalent for any of this -- legacy MCPU has no
checkpointing or resume feature -- so this is pure software-behavior
verification, not a physics question.

The run() tests build a small *real* ``FoldingRunner`` on the chignolin
structure (construction is ~2 s; each test only runs a handful of 2-step
cycles).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

mdtraj = pytest.importorskip(
    "mdtraj", reason="mdtraj not installed -- skipping folding checkpoint tests"
)

from pymcpu.checkpointing import (  # noqa: E402
    CheckpointState,
    FoldingCheckpointState,
    checkpoint_cycle_filename,
    load_checkpoint,
    save_checkpoint,
)
from pymcpu.sampling.folding import FoldingRunner  # noqa: E402


def _build_real_folding_runner(
    pdb_path: str,
    output_dir: Path,
    checkpoint_dir: Path,
    *,
    seed: int = 1,
    resume: bool | str = False,
    checkpoint_interval: int = 1,
    **kwargs,
) -> FoldingRunner:
    """A real (non-mocked) FoldingRunner on the small chignolin structure.

    Construction (~2 s) builds the actual forcefield/system/integrator, and
    ``report_interval``/``steps_per_cycle`` are kept tiny so a handful of
    cycles run in well under a second -- fast enough to exercise run()'s
    real checkpoint/resume control flow end-to-end rather than mocking it.
    """
    return FoldingRunner(
        pdb_path,
        output_dir=str(output_dir),
        checkpoint_dir=str(checkpoint_dir),
        seed=seed,
        report_interval=2,
        steps_per_cycle=2,
        resume=resume,
        checkpoint_interval=checkpoint_interval,
        verbose=False,
        **kwargs,
    )


# ── FoldingCheckpointState construction / round-trip ─────────────


class TestFoldingCheckpointState:
    def test_round_trip_preserves_base_and_folding_fields(self, tmp_path: Path) -> None:
        q_values = np.array([0.45, 0.62, 0.78, 0.91])
        coords = [np.random.rand(10, 3).astype(np.float32) for _ in range(4)]
        state = FoldingCheckpointState(
            cycle=100,
            global_step=100000,
            replica_coords=coords,
            walker_at_state=np.arange(4),
            current_steps=[100000] * 4,
            temperatures=[300.0, 350.0, 400.0, 450.0],
            integrator_rng_states=[""] * 4,
            seed=99,
            native_contacts_fraction=q_values,
            convergence_history=[0.3, 0.4, 0.5, 0.6, 0.7],
        )

        save_checkpoint(state, checkpoint_dir=str(tmp_path), cycle=100)
        loaded = FoldingCheckpointState.from_dict(load_checkpoint(str(tmp_path / "last.chk")))

        assert loaded.cycle == 100
        assert loaded.global_step == 100000
        assert loaded.seed == 99
        for i in range(4):
            np.testing.assert_array_almost_equal(loaded.replica_coords[i], coords[i])
        assert loaded.checkpoint_type == "folding"
        np.testing.assert_array_almost_equal(loaded.native_contacts_fraction, q_values)
        assert loaded.convergence_history == [0.3, 0.4, 0.5, 0.6, 0.7]

    def test_fields_of_removed_skeletons_are_dropped_on_load(self, tmp_path: Path) -> None:
        """Checkpoints from before FoldingBias and BasinTracker were removed
        carry folding_bias_params, basin_assignments and folding_events."""
        state = FoldingCheckpointState(
            cycle=3, global_step=30, replica_coords=[np.zeros((3, 5))], current_steps=[30],
            convergence_history=[0.9, 0.95],
        ).to_dict()
        state.update(
            folding_bias_params={"k": 2.0, "r0": 0.5, "mode": "harmonic"},
            basin_assignments=np.array([0], dtype=np.int32),
            folding_events=[{"cycle": 2, "Q": 0.95}],
        )
        save_checkpoint(state, checkpoint_dir=str(tmp_path), cycle=3)

        loaded = FoldingCheckpointState.from_dict(load_checkpoint(str(tmp_path / "last.chk")))

        assert loaded.cycle == 3
        assert loaded.convergence_history == [0.9, 0.95]
        assert not hasattr(loaded, "folding_bias_params")
        assert not hasattr(loaded, "folding_events")

    def test_base_checkpoint_state_loads_without_folding_fields(self, tmp_path: Path) -> None:
        """A plain (non-folding) CheckpointState must still load cleanly
        through FoldingCheckpointState.from_dict (backward compatibility)."""
        state = CheckpointState(
            cycle=5,
            global_step=5000,
            replica_coords=[np.zeros((5, 3))],
            walker_at_state=np.array([0]),
            current_steps=[5000],
            temperatures=[300.0],
            integrator_rng_states=[""],
            seed=1,
        )
        save_checkpoint(state, checkpoint_dir=str(tmp_path), cycle=5)

        raw = load_checkpoint(str(tmp_path / "last.chk"))
        assert raw["cycle"] == 5
        loaded = FoldingCheckpointState.from_dict(raw)
        assert loaded.cycle == 5
        assert loaded.native_contacts_fraction is None
        assert loaded.rmsd_to_native is None
        assert loaded.convergence_history == []


# ── run() control-flow, exercised end-to-end on a real runner ────


def test_run_updates_convergence_history_after_each_cycle(chignolin_pdb_path: str, tmp_path: Path) -> None:
    """run() must call _update_convergence_history once per cycle, after
    _run_folding_cycle (a convergence check against a history that lags the
    current step would be a one-cycle-stale correctness bug)."""
    ckpt_dir = tmp_path / "ckpt"
    runner = _build_real_folding_runner(chignolin_pdb_path, tmp_path / "out", ckpt_dir, seed=1, checkpoint_interval=1000)

    call_order: list[str] = []
    orig_cycle, orig_update = runner._run_folding_cycle, runner._update_convergence_history
    runner._run_folding_cycle = lambda cycle: (call_order.append("cycle"), orig_cycle(cycle))[1]
    runner._update_convergence_history = lambda: (call_order.append("update"), orig_update())[1]

    runner.run(n_cycles=3)

    assert call_order == ["cycle", "update", "cycle", "update", "cycle", "update"]
    assert len(runner.convergence_history) == 3


# ── The early stop is opt-in ────────────────────────────────────


def test_a_folded_start_runs_every_cycle_by_default(chignolin_pdb_path: str, tmp_path: Path) -> None:
    """Chignolin starts folded (Q = 1 against itself). The run used to stop
    after 10 cycles whatever ``steps`` asked for."""
    runner = _build_real_folding_runner(
        chignolin_pdb_path, tmp_path / "out", tmp_path / "ckpt", checkpoint_interval=1000
    )
    assert runner.q_threshold is None

    runner.run(steps=24)  # 12 cycles of 2 steps

    assert runner.cycle == 12
    assert runner.simulation.current_step == 24
    assert min(runner.convergence_history) >= 0.75  # it would have stopped


def test_the_early_stop_reports_the_steps_it_ran(
    chignolin_pdb_path: str, tmp_path: Path, capsys
) -> None:
    runner = _build_real_folding_runner(
        chignolin_pdb_path, tmp_path / "out", tmp_path / "ckpt",
        checkpoint_interval=1000, q_threshold=0.75, convergence_window=3,
    )
    runner.verbose = True

    runner.run(steps=24)

    assert runner.cycle == 3
    assert runner.simulation.current_step == 6
    out = capsys.readouterr().out
    assert "Stopped early after cycle 2: Q >= 0.75 for 3 cycles in a row" in out
    assert "Ran 6 of 24 MC steps." in out
    # it saves where it stops
    assert (tmp_path / "ckpt" / checkpoint_cycle_filename(2)).is_file()


def test_the_early_stop_is_logged_when_quiet(
    chignolin_pdb_path: str, tmp_path: Path, caplog
) -> None:
    import logging

    runner = _build_real_folding_runner(
        chignolin_pdb_path, tmp_path / "out", tmp_path / "ckpt",
        checkpoint_interval=1000, q_threshold=0.75, convergence_window=2,
    )
    with caplog.at_level(logging.INFO, logger="pymcpu.sampling.folding"):
        runner.run(steps=10)
    assert "Ran 4 of 10 MC steps." in caplog.text


def test_a_resumed_run_restores_the_q_history(chignolin_pdb_path: str, tmp_path: Path) -> None:
    ckpt_dir = tmp_path / "ckpt"
    first = _build_real_folding_runner(chignolin_pdb_path, tmp_path / "out1", ckpt_dir)
    first.run(n_cycles=3)

    second = _build_real_folding_runner(chignolin_pdb_path, tmp_path / "out2", ckpt_dir, resume=True)
    second.run(n_cycles=5)

    assert second.convergence_history[:3] == first.convergence_history
    assert len(second.convergence_history) == 5
