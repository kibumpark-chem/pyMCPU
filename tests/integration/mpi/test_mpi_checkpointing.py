"""MPI replica-exchange checkpointing: fast unit tests that don't need a
working ``mpirun``.

Every behavioral assertion here drives ``MPIReplicaExchange`` through a mock
single-rank communicator (``mock_comm_rank0``, from
``tests/integration/conftest.py``) and checks *what actually happened* --
files written, collective calls made, call order, row counts -- rather than
grepping the implementation's source text for call names. The previous
version of this file used ``inspect.getsource(...)`` plus substring/position
search to assert code structure; that style is brittle (breaks on harmless
refactors like renaming a call site or wrapping it in a helper) and can pass
even when the asserted call never actually executes. Real ``mpirun -n 2``
tests live in ``test_mpi_checkpointing_multirank.py``.
"""

from __future__ import annotations

import csv
import glob
import inspect
import os
from unittest.mock import MagicMock, Mock

import pytest


def _import_mpi_re():
    """Import MPIReplicaExchange even when libmpi is missing at runtime."""
    from pymcpu.sampling.mpi_replica_exchange import MPIReplicaExchange

    return MPIReplicaExchange


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


def _csv_row_count(path: str) -> int:
    with open(path, newline="") as fh:
        return sum(1 for _ in csv.reader(fh))


def test_mpi_re_accepts_checkpoint_config() -> None:
    """MPIReplicaExchange.__init__ must accept a CheckpointConfig."""
    MPIReplicaExchange = _import_mpi_re()
    params = set(inspect.signature(MPIReplicaExchange.__init__).parameters)
    assert "checkpoint_config" in params or "checkpoint_dir" in params, (
        "MPIReplicaExchange.__init__ must accept checkpoint_config "
        "or individual checkpoint kwargs"
    )


def test_mpi_re_run_accepts_checkpoint_kwargs() -> None:
    """MPIReplicaExchange.run() must accept all 7 checkpoint kwargs."""
    MPIReplicaExchange = _import_mpi_re()
    params = set(inspect.signature(MPIReplicaExchange.run).parameters)
    required = {
        "checkpoint_dir",
        "checkpoint_interval",
        "keep_last_n",
        "resume",
        "cloud_sync",
        "cloud_bucket",
        "cloud_sync_cmd",
    }
    missing = required - params
    assert not missing, f"MPIReplicaExchange.run() missing checkpoint params: {missing}"


def test_runner_mpi_accepts_checkpoint_params() -> None:
    """run_mpi_replica_exchange_2d must accept the 7 checkpoint parameters."""
    from pymcpu import runners

    params = set(inspect.signature(runners.run_mpi_replica_exchange_2d).parameters)
    required = {
        "checkpoint_dir",
        "checkpoint_interval",
        "keep_last_n",
        "resume",
        "cloud_sync",
        "cloud_bucket",
        "cloud_sync_cmd",
    }
    missing = required - params
    assert not missing, f"runner missing checkpoint params: {missing}"


def test_mpi4py_optional_for_unit_suite() -> None:
    """Unit suite must not hard-require a working libmpi: importing the
    module for construction/inspection must succeed even if mpi4py cannot
    dlopen libmpi (MPI remains None in that case)."""
    mod = __import__("pymcpu.sampling.mpi_replica_exchange", fromlist=["MPI"])
    assert hasattr(mod, "MPIReplicaExchange")


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

    try:
        re = _make_two_temp_mpi_re(mock_comm_rank0, minimal_pdb_path, cfg, out_dir)
    except Exception as exc:
        pytest.skip(f"Could not construct MPIReplicaExchange with mock comm: {exc}")

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


def test_run_resume_loads_truncates_attaches_in_order_and_appends(
    mock_comm_rank0, tmp_path, minimal_pdb_path, monkeypatch
):
    """
    Behavioral replacement for the old source-grep order tests
    (``_mpi_load_checkpoint`` before ``_attach_traj_reporters``, truncation
    between the two) plus the "append passes through on resume" check.

    Rather than searching ``run()``'s source text for call names in order,
    wrap the real bound methods with ``Mock(wraps=...)`` and read back the
    actual call order from a shared manager -- this only passes if the
    calls actually happen, in the actual order, at runtime.
    """
    from pymcpu.checkpointing import CheckpointConfig

    _ensure_mpi_module_for_construction(monkeypatch)

    chk_dir = str(tmp_path / "chk")
    out_dir = tmp_path / "out"
    cfg = CheckpointConfig(checkpoint_dir=chk_dir, checkpoint_interval=1, resume=False)

    try:
        re1 = _make_two_temp_mpi_re(mock_comm_rank0, minimal_pdb_path, cfg, out_dir)
    except Exception as exc:
        pytest.skip(f"Could not construct MPIReplicaExchange with mock comm: {exc}")

    # First run: 1 real cycle, writing some rows to each replica's data CSV.
    re1.run(1, 2, verbose=False, write_logs=False, checkpoint_interval=1)
    assert re1.cycle == 1
    data_csvs = sorted(glob.glob(str(out_dir / "*_data.csv")))
    assert data_csvs, "expected per-replica data CSVs after the first run"
    rows_after_first_run = {p: _csv_row_count(p) for p in data_csvs}
    assert all(n > 0 for n in rows_after_first_run.values())

    # Second instance, resuming from the checkpoint the first run wrote.
    cfg2 = CheckpointConfig(checkpoint_dir=chk_dir, checkpoint_interval=1, resume=True)
    re2 = _make_two_temp_mpi_re(mock_comm_rank0, minimal_pdb_path, cfg2, out_dir)

    manager = Mock()
    for name in ("_mpi_load_checkpoint", "_truncate_local_trajectories", "_attach_traj_reporters"):
        wrapped = Mock(wraps=getattr(re2, name))
        monkeypatch.setattr(re2, name, wrapped)
        manager.attach_mock(wrapped, name)

    re2.run(2, 2, verbose=False, write_logs=False, checkpoint_interval=1, resume=chk_dir)
    assert re2.cycle == 2

    call_order = [c[0] for c in manager.mock_calls if c[0] in {
        "_mpi_load_checkpoint", "_truncate_local_trajectories", "_attach_traj_reporters"
    }]
    assert call_order.index("_mpi_load_checkpoint") < call_order.index("_truncate_local_trajectories")
    assert call_order.index("_truncate_local_trajectories") < call_order.index("_attach_traj_reporters")

    # The resume flag actually reaching _attach_traj_reporters is what makes
    # mcpu_core open the CSV/XTC reporters in append mode.
    attach_call = manager._attach_traj_reporters.call_args
    resume_arg = attach_call.kwargs.get("resume", attach_call.args[0] if attach_call.args else None)
    assert resume_arg is True

    # And the actual behavioral consequence: row counts grew, they weren't
    # reset to a fresh header-only file.
    for path in data_csvs:
        assert _csv_row_count(path) > rows_after_first_run[path], (
            f"{path} should have grown after resumed run, "
            "not been truncated/overwritten"
        )
