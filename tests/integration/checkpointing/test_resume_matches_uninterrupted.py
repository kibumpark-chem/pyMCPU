"""A run stopped and resumed writes what the uninterrupted run writes.

Each test runs one seed straight through, and again in stages: every stage is
a new runner object that resumes from the checkpoint the previous stage left,
the way consecutive jobs do. Every output file (XTC, data CSV, state and
exchange logs, rex_stats.json) must be byte-identical, and so must the final
checkpoint. ``twice`` resumes from a checkpoint that a resumed run wrote.
``crash`` resumes from a checkpoint older than the files, as after a job
killed between checkpoints, so the resume has to cut the files back first.

Both runs use the same checkpoint interval: saving a checkpoint reads the
integrator RNG state, which drops a cached normal draw, so checkpoints at
other cycles would change the trajectory (by design).

The MPI driver has the same tests in ``tests/integration/mpi``.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from pymcpu.checkpointing import checkpoint_cycle_filename, load_checkpoint, save_checkpoint
from tests.helpers.resume_outputs import assert_same_outputs

#: Stages (total cycle counts) per scenario, and the checkpoint cycle the
#: ``crash`` scenario resumes from. Checkpoints every 3 cycles; frames every
#: 2 cycles, so the crash case has a frame and exchange rows to cut.
REX_SCENARIOS = {
    "once": [6, 12],
    "twice": [3, 6, 12],
    "crash": [5, 12],
}
CRASH_CHECKPOINT_CYCLE = 3


def _make_rex(pdb: str, out: Path, ckpt: Path):
    from pymcpu.sampling.replica_exchange import ReplicaExchange

    return ReplicaExchange(
        pdb,
        temperatures=[0.5, 0.6],
        n_targets=[0.0],
        k_bias=0.0,
        log_interval=20,
        output_prefix="rex",
        output_dir=out,
        seed=7,
        checkpoint_dir=ckpt,
        checkpoint_interval=3,
        keep_last_n=10,
        exchange_log="all",
        state_log_interval=1,
    )


def _run_rex(pdb: str, root: Path, stages: list[int], crash: bool = False) -> tuple[Path, Path]:
    out, ckpt = root / "out", root / "ckpt"
    for k, n_cycles in enumerate(stages):
        rex = _make_rex(pdb, out, ckpt)
        resume = None
        if k > 0:
            # A run that ends writes a final checkpoint, so a crash between
            # checkpoints is a resume from the older interval checkpoint.
            name = checkpoint_cycle_filename(CRASH_CHECKPOINT_CYCLE) if crash else "last.chk"
            resume = ckpt / name
        rex.run(n_cycles, 10, verbose=False, write_logs=True, resume=resume)
    return out, ckpt / "last.chk"


@pytest.fixture(scope="module")
def rex_uninterrupted(minimal_pdb_path, tmp_path_factory) -> tuple[Path, Path]:
    return _run_rex(minimal_pdb_path, tmp_path_factory.mktemp("rex_straight"), [12])


@pytest.mark.parametrize("scenario", sorted(REX_SCENARIOS))
def test_replica_exchange_resume_matches_uninterrupted(
    scenario, minimal_pdb_path, rex_uninterrupted, tmp_path
) -> None:
    stages = REX_SCENARIOS[scenario]
    out, last = _run_rex(minimal_pdb_path, tmp_path, stages, crash=scenario == "crash")
    assert_same_outputs(rex_uninterrupted[0], out, rex_uninterrupted[1], last)


def test_replica_exchange_resumes_a_checkpoint_without_exchange_counts(
    minimal_pdb_path, tmp_path
) -> None:
    """A checkpoint written before the exchange counts were saved still
    resumes; rex_stats.json then counts from the resume on, as it used to."""
    out, last = _run_rex(minimal_pdb_path, tmp_path, [6])
    state = load_checkpoint(last)
    assert state.pop("exchange_counts")  # saved by this version...
    save_checkpoint(state, last.parent, filename=last.name)  # ...and now removed

    _make_rex(minimal_pdb_path, out, last.parent).run(9, 10, verbose=False, resume=last)
    stats = json.loads((out / "rex_rex_stats.json").read_text())
    assert stats["temperature"]["attempts"] == 3  # 3 cycles after the resume, 1 pair
    assert stats["cycles_completed"] == 9


def test_replica_exchange_resume_cuts_an_analysis_file_outside_the_output_dir(
    minimal_pdb_path, tmp_path
) -> None:
    """The analysis file can live outside output_dir, where the cut of the
    trajectory files does not reach. A resume from a checkpoint older than
    the file (stopped at cycle 5, saved at 3) now cuts its samples too; they
    used to be kept, and cycles 3 and 4 appeared twice."""
    import numpy as np

    def samples(root: Path, stages: list[int]) -> dict:
        out, ckpt = root / "out", root / "ckpt"
        for k, n_cycles in enumerate(stages):
            resume = ckpt / checkpoint_cycle_filename(CRASH_CHECKPOINT_CYCLE) if k > 0 else None
            _make_rex(pdb=minimal_pdb_path, out=out, ckpt=ckpt).run(
                n_cycles,
                10,
                verbose=False,
                write_logs=False,
                resume=resume,
                analysis_path=root / "analysis" / "an.npz",
            )
        with np.load(root / "analysis" / "an.npz") as data:
            return {key: data[key] for key in data.files}

    want = samples(tmp_path / "straight", [12])
    got = samples(tmp_path / "resumed", [5, 12])
    assert want["cycle"].tolist() == [c for c in range(12) for _ in range(2)]
    for key in want:
        np.testing.assert_array_equal(got[key], want[key], err_msg=key)


#: FoldingRunner cycles are 0-based and it checkpoints when cycle % 3 == 0
#: and at its last cycle, so every stage here ends on a checkpoint cycle the
#: uninterrupted run also saves.
FOLD_SCENARIOS = {
    "once": [7, 13],
    "twice": [4, 7, 13],
    "crash": [6, 13],
}


def _run_folding(pdb: str, root: Path, stages: list[int], crash: bool = False) -> tuple[Path, Path]:
    from pymcpu.sampling.folding import FoldingRunner

    out, ckpt = root / "out", root / "ckpt"
    for k, n_cycles in enumerate(stages):
        if k > 0 and crash:
            # Killed after the cycle-3 checkpoint: last.chk is older than the files.
            shutil.copy(ckpt / checkpoint_cycle_filename(CRASH_CHECKPOINT_CYCLE), ckpt / "last.chk")
        runner = FoldingRunner(
            pdb,
            temperature=0.6,
            report_interval=10,
            steps_per_cycle=5,
            output_dir=out,
            seed=7,
            prefix="fold",
            q_threshold=2.0,  # never converges, so no run stops early
            checkpoint_dir=str(ckpt),
            checkpoint_interval=3,
            keep_last_n=10,
            verbose=False,
        )
        runner.run(n_cycles, resume=True if k > 0 else None)
    return out, ckpt / "last.chk"


@pytest.fixture(scope="module")
def folding_uninterrupted(minimal_pdb_path, tmp_path_factory) -> tuple[Path, Path]:
    return _run_folding(minimal_pdb_path, tmp_path_factory.mktemp("fold_straight"), [13])


@pytest.mark.parametrize("scenario", sorted(FOLD_SCENARIOS))
def test_folding_resume_matches_uninterrupted(
    scenario, minimal_pdb_path, folding_uninterrupted, tmp_path
) -> None:
    stages = FOLD_SCENARIOS[scenario]
    out, last = _run_folding(minimal_pdb_path, tmp_path, stages, crash=scenario == "crash")
    assert_same_outputs(folding_uninterrupted[0], out, folding_uninterrupted[1], last)
