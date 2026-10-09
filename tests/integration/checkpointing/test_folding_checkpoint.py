"""Folding-mode checkpoint/resume orchestration tests.

Covers ``FoldingCheckpointState`` construction and save/load round-trips,
the checkpoint-related parameters on ``FoldingRunner``/``run_folding``, and
``FoldingRunner.run()``'s checkpoint/resume control flow: that a resumed run
loads its checkpoint before reattaching trajectory reporters, that the
opt-in early stop (``q_threshold``) saves where it stops and reports the steps
it ran, that a folded start runs every cycle by default, that a resumed run
continues from the correct cycle, and that convergence history updates every
cycle and survives a resume.

There is no legacy-MCPU equivalent for any of this -- legacy MCPU has no
checkpointing or resume feature -- so this is pure software-behavior
verification, not a physics question.

The four "run-ordering" tests build a small *real* ``FoldingRunner`` on the
chignolin structure (construction is ~2 s; each test only runs a handful of
2-step cycles) and record actual call order/side effects via monkeypatched
spies. The previous version of these tests used ``inspect.getsource(...)``
plus substring position search to check call order without ever executing
the code path -- that passes even if the code is refactored into something
that no longer behaves correctly, and fails on harmless changes like an
added comment or a renamed local variable. Actually running ``run()`` is the
more expensive but the only genuinely behavioral option.
"""

from __future__ import annotations

import inspect
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
    def test_is_subclass_of_checkpoint_state(self) -> None:
        assert issubclass(FoldingCheckpointState, CheckpointState)

    def test_default_checkpoint_type_is_folding(self) -> None:
        state = FoldingCheckpointState(
            cycle=0,
            global_step=0,
            replica_coords=[np.zeros((5, 3))],
            walker_at_state=np.array([0]),
            current_steps=[0],
            temperatures=[300.0],
            integrator_rng_states=[""],
            seed=42,
        )
        assert state.checkpoint_type == "folding"

    def test_optional_fields_default_to_none_or_empty(self) -> None:
        state = FoldingCheckpointState(
            cycle=0,
            global_step=0,
            replica_coords=[np.zeros((5, 3))],
            walker_at_state=np.array([0]),
            current_steps=[0],
            temperatures=[300.0],
            integrator_rng_states=[""],
            seed=42,
        )
        assert state.native_contacts_fraction is None
        assert state.rmsd_to_native is None
        assert state.convergence_history == []

    def test_round_trip_preserves_base_re_fields(self, tmp_path: Path) -> None:
        coords = [np.random.rand(10, 3).astype(np.float32) for _ in range(4)]
        state = FoldingCheckpointState(
            cycle=25,
            global_step=25000,
            replica_coords=coords,
            walker_at_state=np.arange(4),
            current_steps=[25000] * 4,
            temperatures=[300.0, 350.0, 400.0, 450.0],
            integrator_rng_states=[""] * 4,
            seed=42,
            n_targets=4,
            k_bias=1.0,
        )

        save_checkpoint(state, checkpoint_dir=str(tmp_path), cycle=25)
        loaded = FoldingCheckpointState.from_dict(load_checkpoint(str(tmp_path / "last.chk")))

        assert loaded.cycle == 25
        assert loaded.global_step == 25000
        assert loaded.seed == 42
        for i in range(4):
            np.testing.assert_array_almost_equal(loaded.replica_coords[i], coords[i])

    def test_round_trip_preserves_folding_fields(self, tmp_path: Path) -> None:
        q_values = np.array([0.45, 0.62, 0.78, 0.91])
        state = FoldingCheckpointState(
            cycle=100,
            global_step=100000,
            replica_coords=[np.zeros((5, 3))] * 4,
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


# ── Checkpoint parameters on the public API ──────────────────────


def test_folding_runner_init_accepts_checkpoint_config() -> None:
    sig = inspect.signature(FoldingRunner.__init__)
    assert "checkpoint_config" in sig.parameters


def test_folding_runner_run_accepts_checkpoint_kwargs() -> None:
    sig = inspect.signature(FoldingRunner.run)
    required = {
        "checkpoint_dir",
        "checkpoint_interval",
        "keep_last_n",
        "resume",
    }
    missing = required - set(sig.parameters)
    assert not missing, f"FoldingRunner.run() missing checkpoint params: {missing}"


def test_run_folding_module_function_accepts_checkpoint_kwargs() -> None:
    from pymcpu import runners

    sig = inspect.signature(runners.run_folding)
    required = {
        "checkpoint_dir",
        "checkpoint_interval",
        "keep_last_n",
        "resume",
    }
    missing = required - set(sig.parameters)
    assert not missing, f"run_folding missing checkpoint params: {missing}"


# ── run() control-flow, exercised end-to-end on a real runner ────


def test_run_loads_checkpoint_before_attaching_traj_reporters(chignolin_pdb_path: str, tmp_path: Path) -> None:
    """On resume, run() must load the checkpoint before reattaching
    trajectory reporters (reporters need the restored frame counts to know
    whether to append or start fresh)."""
    ckpt_dir = tmp_path / "ckpt"
    seed_runner = _build_real_folding_runner(chignolin_pdb_path, tmp_path / "seed_out", ckpt_dir, seed=1)
    seed_runner.save_checkpoint(cycle=0)  # so the resuming runner has something to load

    runner = _build_real_folding_runner(chignolin_pdb_path, tmp_path / "out", ckpt_dir, seed=2, resume=True)
    call_order: list[str] = []
    orig_load, orig_attach = runner.load_checkpoint, runner._attach_traj_reporters
    runner.load_checkpoint = lambda *a, **k: (call_order.append("load"), orig_load(*a, **k))[1]
    runner._attach_traj_reporters = lambda *a, **k: (call_order.append("attach"), orig_attach(*a, **k))[1]

    runner.run(n_cycles=2)

    assert call_order == ["load", "attach"]


def test_run_saves_checkpoint_on_convergence(chignolin_pdb_path: str, tmp_path: Path) -> None:
    """The early stop must trigger a save distinct from the regular
    checkpoint-interval save (checkpoint_interval is set high enough here
    that only the convergence branch can be responsible for the save)."""
    ckpt_dir = tmp_path / "ckpt"
    seed_runner = _build_real_folding_runner(chignolin_pdb_path, tmp_path / "seed_out", ckpt_dir, seed=1)
    seed_runner.save_checkpoint(cycle=5)  # resume will start at cycle 6

    runner = _build_real_folding_runner(
        chignolin_pdb_path, tmp_path / "out", ckpt_dir, seed=2, resume=True,
        checkpoint_interval=1000, q_threshold=0.75,
    )
    runner._check_folding_convergence = lambda: True
    save_calls: list[int] = []
    orig_save = runner.save_checkpoint
    runner.save_checkpoint = lambda cycle: (save_calls.append(cycle), orig_save(cycle))[1]

    runner.run(n_cycles=20)  # would run cycles 6..19 if not for the forced convergence break

    # cycle=6 is neither a checkpoint_interval multiple nor the final cycle
    # (n_cycles-1=19), so the only thing that can have produced this save
    # entry is the convergence branch.
    assert save_calls == [6]


def test_resume_continues_from_correct_cycle(chignolin_pdb_path: str, tmp_path: Path) -> None:
    """A resumed run must continue at (last checkpointed cycle + 1), not
    restart from 0 or replay the checkpointed cycle again."""
    ckpt_dir = tmp_path / "ckpt"
    first = _build_real_folding_runner(chignolin_pdb_path, tmp_path / "out1", ckpt_dir, seed=1)
    first.run(n_cycles=3)  # cycles 0, 1, 2 -- checkpointed every cycle
    assert first.cycle == 3

    second = _build_real_folding_runner(chignolin_pdb_path, tmp_path / "out2", ckpt_dir, seed=2, resume=True)
    seen_cycles: list[int] = []
    orig_run_cycle = second._run_folding_cycle
    second._run_folding_cycle = lambda cycle: (seen_cycles.append(cycle), orig_run_cycle(cycle))[1]

    second.run(n_cycles=5)

    assert seen_cycles[0] == 3, f"Resume should continue at cycle 3, first ran cycle {seen_cycles[0]}"
    assert second.cycle == 5


def test_convergence_checkpoint_file_exists_on_disk(chignolin_pdb_path: str, tmp_path: Path) -> None:
    """On convergence, a checkpoint_cycle_<N>.chk file for that exact cycle
    must actually exist on disk afterward (not just an in-memory save call).

    Starts from a resumed cycle 6 (rather than cycle 0) and sets
    checkpoint_interval=1000: at cycle 0, ``cycle % interval == 0`` is
    trivially true for *any* interval, which would let the periodic-save
    branch masquerade as the thing that produced the file.
    """
    ckpt_dir = tmp_path / "ckpt"
    seed_runner = _build_real_folding_runner(chignolin_pdb_path, tmp_path / "seed_out", ckpt_dir, seed=1)
    seed_runner.save_checkpoint(cycle=5)  # resume will start at cycle 6

    runner = _build_real_folding_runner(
        chignolin_pdb_path, tmp_path / "out", ckpt_dir, seed=2, resume=True,
        checkpoint_interval=1000, q_threshold=0.75,
    )
    runner._check_folding_convergence = lambda: True

    runner.run(n_cycles=20)

    assert runner.cycle == 7  # converged and broke out after cycle 6
    chk_path = ckpt_dir / checkpoint_cycle_filename(6)
    assert chk_path.is_file(), f"Expected convergence checkpoint at {chk_path}"


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
