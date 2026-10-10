"""MPI replica-exchange checkpointing: fast unit tests that don't need a
working ``mpirun``.

Every test here drives ``MPIReplicaExchange`` through a mock single-rank
communicator (``mock_comm_rank0``, from ``tests/integration/conftest.py``)
and checks what actually happened: files written, collective calls made,
and outputs identical to an uninterrupted run. Real ``mpirun -n 2`` tests
live in ``test_mpi_checkpointing_multirank.py``.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest

from tests.helpers.resume_outputs import assert_same_outputs


def _ensure_mpi_module_for_construction(monkeypatch: pytest.MonkeyPatch):
    """``MPIReplicaExchange.__init__`` needs ``MPI.TAG_UB`` even with a mock
    comm. If libmpi failed to load (module ``MPI`` is None), install a
    minimal stand-in -- 32767 is the MPI standard's guaranteed *minimum*
    TAG_UB, so any real implementation satisfies it too."""
    import pymcpu.sampling.mpi_replica_exchange as mmod

    if mmod.MPI is not None:
        return mmod.MPI

    fake_mpi = MagicMock()
    fake_mpi.TAG_UB = 32767
    fake_mpi.LOR = object()
    monkeypatch.setattr(mmod, "MPI", fake_mpi)
    monkeypatch.setattr(mmod, "_require_mpi", lambda: fake_mpi)
    return fake_mpi


def _make_two_temp_mpi_re(comm, pdb_path, checkpoint_config, output_dir, **kwargs):
    """Minimal 2-temperature / 1-window MPI RE (2 replicas, both local to a
    single mock rank)."""
    from pymcpu.sampling.mpi_replica_exchange import MPIReplicaExchange

    return MPIReplicaExchange(
        comm,
        str(pdb_path),
        temperatures=[0.5, 0.6],
        n_targets=[0.0],
        k_bias=0.0,
        log_interval=1,
        output_dir=str(output_dir),
        output_prefix="rex",
        seed=42,
        checkpoint_config=checkpoint_config,
        **kwargs,
    )


def test_mpi_checkpoint_save_load_round_trip_with_mock_comm(
    mock_comm_rank0, tmp_path, minimal_pdb_path, monkeypatch
):
    """
    Round-trip ``_mpi_save_checkpoint`` / ``_mpi_load_checkpoint`` with a
    mock single-rank comm (no mpirun required). Also documents the save
    protocol behaviorally: >=2 Barrier() calls (one before write, one
    after) and that gather/bcast are actually invoked, replacing what used
    to be three separate source-grep tests.
    """
    from pymcpu.checkpointing import CheckpointConfig

    _ensure_mpi_module_for_construction(monkeypatch)

    chk_dir = str(tmp_path / "mock_chk")
    os.makedirs(chk_dir, exist_ok=True)
    out_dir = tmp_path / "mock_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = CheckpointConfig(
        checkpoint_dir=chk_dir,
        checkpoint_interval=1,
        keep_last_n=3,
        resume=False,
    )

    re = _make_two_temp_mpi_re(mock_comm_rank0, minimal_pdb_path, cfg, out_dir)

    # Attach reporters so traj frame bookkeeping does not explode on save.
    re._attach_traj_reporters(resume=False)
    re._mc_replica_steps = 1

    mock_comm_rank0.reset_mock()
    mock_comm_rank0.Get_rank.return_value = 0
    mock_comm_rank0.Get_size.return_value = 1
    mock_comm_rank0.gather.side_effect = lambda data, root=0: [data]
    mock_comm_rank0.bcast.side_effect = lambda data, root=0: data
    mock_comm_rank0.Barrier.return_value = None

    re._mpi_save_checkpoint(cycle=7, comm=mock_comm_rank0)

    assert os.path.exists(os.path.join(chk_dir, "last.chk")), (
        "last.chk must be written by rank 0 (mock)"
    )
    # One Barrier before the write, one after -- see _mpi_save_checkpoint's
    # own docstring for the protocol this pins.
    assert mock_comm_rank0.Barrier.call_count >= 2, (
        f"Expected >= 2 Barrier() calls in _mpi_save_checkpoint, "
        f"got {mock_comm_rank0.Barrier.call_count}"
    )
    assert mock_comm_rank0.gather.called, "gather must collect local state"

    mock_comm_rank0.reset_mock()
    mock_comm_rank0.Get_rank.return_value = 0
    mock_comm_rank0.Get_size.return_value = 1
    mock_comm_rank0.bcast.side_effect = lambda data, root=0: data
    mock_comm_rank0.Barrier.return_value = None

    cfg.resume = True
    state = re._mpi_load_checkpoint(comm=mock_comm_rank0)

    assert state is not None, "load_checkpoint returned None unexpectedly"
    # load returns a dict (pickle payload), not CheckpointState
    assert int(state["cycle"]) == 7, f"Expected cycle=7, got {state.get('cycle')}"
    assert mock_comm_rank0.bcast.called, "bcast must distribute state to all ranks"


#: Stages (total cycle counts) per scenario; checkpoints every 3 cycles. The
#: MPI driver saves only on that interval, so the ``crash`` run stops at
#: cycle 5 with last.chk at cycle 3 and files that hold 5 cycles, as after a
#: job killed between checkpoints. ``twice`` resumes from a checkpoint that a
#: resumed run wrote.
RESUME_SCENARIOS = {"once": [6, 12], "twice": [3, 6, 12], "crash": [5, 12]}


def _run_stages(comm, pdb_path, root, stages):
    """Run each stage as a new driver; every stage after the first resumes."""
    from pymcpu.checkpointing import CheckpointConfig

    out, ckpt = root / "out", root / "ckpt"
    for k, n_cycles in enumerate(stages):
        cfg = CheckpointConfig(
            checkpoint_dir=str(ckpt), checkpoint_interval=3, keep_last_n=10, resume=k > 0
        )
        re = _make_two_temp_mpi_re(
            comm, pdb_path, cfg, out, exchange_log="all", state_log_interval=1
        )
        re.run(n_cycles, 10, verbose=False, write_logs=True)
    return out, ckpt / "last.chk"


@pytest.mark.parametrize("scenario", sorted(RESUME_SCENARIOS))
def test_resumed_run_matches_uninterrupted_with_mock_comm(
    scenario, mock_comm_rank0, tmp_path, minimal_pdb_path, monkeypatch
):
    """
    A run resumed once, twice, or after a crash between checkpoints leaves
    the files and final checkpoint of the uninterrupted run. Before this was
    fixed, a second resume cut each XTC and data CSV back to the frames
    written since the first resume, rex_stats.json counted only the last
    job's exchanges, and a crash left duplicate exchange and state rows.
    """
    _ensure_mpi_module_for_construction(monkeypatch)
    straight = _run_stages(mock_comm_rank0, minimal_pdb_path, tmp_path / "straight", [12])
    resumed = _run_stages(
        mock_comm_rank0, minimal_pdb_path, tmp_path / "resumed", RESUME_SCENARIOS[scenario]
    )
    assert_same_outputs(straight[0], resumed[0], straight[1], resumed[1])


@pytest.mark.parametrize("resume_from", ["flag", "directory"])
def test_resume_from_a_checkpoint_without_exchange_counts_with_mock_comm(
    resume_from, mock_comm_rank0, tmp_path, minimal_pdb_path, monkeypatch
):
    """A checkpoint written before the exchange counts were saved still
    resumes, with ``resume=True`` or with ``resume`` naming the checkpoint
    directory; rex_stats.json then counts from the resume on, as it used to.
    A run that ignored the resume would count all 9 cycles. The
    byte-identical resume tests above cannot see that: a fresh run with the
    same seed writes the same files."""
    import json

    from pymcpu.checkpointing import CheckpointConfig, load_checkpoint, save_checkpoint

    _ensure_mpi_module_for_construction(monkeypatch)
    out, last = _run_stages(mock_comm_rank0, minimal_pdb_path, tmp_path, [6])
    state = load_checkpoint(last)
    assert state.pop("exchange_counts")  # saved by this version...
    save_checkpoint(state, last.parent, filename=last.name)  # ...and now removed

    resume = True if resume_from == "flag" else str(last.parent)
    cfg = CheckpointConfig(checkpoint_dir=str(last.parent), checkpoint_interval=3, resume=resume)
    re = _make_two_temp_mpi_re(
        mock_comm_rank0, minimal_pdb_path, cfg, out, exchange_log="all", state_log_interval=1
    )
    re.run(9, 10, verbose=False, write_logs=True)
    stats = json.loads((out / "rex_rex_stats.json").read_text())
    assert stats["temperature"]["attempts"] == 3  # 3 cycles after the resume, 1 pair
    assert stats["cycles_completed"] == 9


def test_resume_cuts_analysis_samples_past_the_checkpoint_with_mock_comm(
    mock_comm_rank0, tmp_path, minimal_pdb_path, monkeypatch
):
    """A run that went on past its last checkpoint (stopped at cycle 5, saved
    at 3) wrote analysis samples the resumed run writes again. Rank 0 now
    cuts them on resume, so the samples match the uninterrupted run's; they
    used to be kept, and cycles 3 and 4 appeared twice."""
    import numpy as np

    from pymcpu.checkpointing import CheckpointConfig

    _ensure_mpi_module_for_construction(monkeypatch)

    def samples(root, stages):
        for k, n_cycles in enumerate(stages):
            cfg = CheckpointConfig(
                checkpoint_dir=str(root / "ckpt"), checkpoint_interval=3, resume=k > 0
            )
            re = _make_two_temp_mpi_re(mock_comm_rank0, minimal_pdb_path, cfg, root / "out")
            re.run(n_cycles, 10, verbose=False, write_logs=False, analysis_path=root / "an.npz")
        with np.load(root / "an.npz") as data:
            return {key: data[key] for key in data.files}

    want = samples(tmp_path / "straight", [12])
    got = samples(tmp_path / "resumed", [5, 12])
    assert want["cycle"].tolist() == [c for c in range(12) for _ in range(2)]
    for key in want:
        np.testing.assert_array_equal(got[key], want[key], err_msg=key)
