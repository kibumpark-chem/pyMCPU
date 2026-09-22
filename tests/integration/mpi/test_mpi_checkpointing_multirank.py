"""Real multi-rank MPI checkpointing tests: require ``mpirun -n 2``.

Split out of ``test_mpi_checkpointing.py`` so the fast mock-comm unit tests
can run under plain ``pytest`` while these -- the only tests in the suite
that actually need a working ``libmpi`` and 2 live ranks -- stay clearly
marked and easy to run in isolation::

    mpirun -n 2 pytest tests/integration/mpi/test_mpi_checkpointing_multirank.py -v

Under plain single-process pytest (no mpirun) every test here skips via the
``comm.Get_size() < 2`` guard.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import pytest


def _make_two_temp_mpi_re(comm, pdb_path, checkpoint_config, output_dir):
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
    )


@pytest.mark.mpi_integration
def test_only_rank0_creates_checkpoint_file(broadcast_tmp_path, minimal_pdb_path):
    """
    Run 1 MPI RE cycle with 2 ranks. Assert 1-2 .chk files exist
    (last.chk + versioned cycle file), and that no rank other than 0
    wrote one.

    Run with: mpirun -n 2 pytest tests/integration/mpi/test_mpi_checkpointing_multirank.py
                      -v -k test_only_rank0_creates_checkpoint_file
    """
    try:
        from mpi4py import MPI
    except Exception:
        pytest.skip("mpi4py/libmpi not available")

    from pymcpu.checkpointing import CheckpointConfig

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if comm.Get_size() < 2:
        pytest.skip("Need mpirun -n 2 for this test")

    chk_dir = str(broadcast_tmp_path / "checkpoints")
    out_dir = broadcast_tmp_path / "out_rank0_chk"
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        os.makedirs(chk_dir, exist_ok=True)
    comm.Barrier()

    cfg = CheckpointConfig(
        checkpoint_dir=chk_dir,
        checkpoint_interval=1,
        keep_last_n=5,
        resume=False,
    )

    re = _make_two_temp_mpi_re(comm, minimal_pdb_path, cfg, out_dir)
    # 1 MC step, 1 cycle -> checkpoint at cycle=1 after exchange_all
    re.run(1, 1, verbose=(rank == 0), write_logs=False)

    comm.Barrier()

    if rank == 0:
        chk_files = glob.glob(chk_dir + "/*.chk")
        assert len(chk_files) >= 1, f"Expected at least 1 .chk file, found {len(chk_files)}"
        assert len(chk_files) <= 2, (
            f"Expected at most 2 .chk files (last.chk + versioned), "
            f"found {len(chk_files)}: {chk_files}. "
            "Multiple ranks may have written independently."
        )
        assert any(Path(f).name == "last.chk" for f in chk_files), (
            "last.chk must always be written"
        )

    comm.Barrier()


@pytest.mark.mpi_integration
def test_all_ranks_start_from_same_cycle_after_resume(broadcast_tmp_path, minimal_pdb_path):
    """
    Run MPI RE for 5 cycles, then resume. All ranks must agree
    start_cycle == 6 (last saved cycle is 5; resume starts at cycle+1).

    Run with: mpirun -n 2 pytest tests/integration/mpi/test_mpi_checkpointing_multirank.py
                      -v -k test_all_ranks_start_from_same_cycle
    """
    try:
        from mpi4py import MPI
    except Exception:
        pytest.skip("mpi4py/libmpi not available")

    from pymcpu.checkpointing import CheckpointConfig

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if comm.Get_size() < 2:
        pytest.skip("Need mpirun -n 2 for this test")

    chk_dir = str(broadcast_tmp_path / "checkpoints_resume")
    out_dir = broadcast_tmp_path / "out_resume"
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        os.makedirs(chk_dir, exist_ok=True)
    comm.Barrier()

    cfg_run = CheckpointConfig(
        checkpoint_dir=chk_dir,
        checkpoint_interval=1,
        keep_last_n=10,
        resume=False,
    )

    # Phase 1: run 5 cycles. After each cycle exchange_all increments
    # _cycle (1..5); saves use cycle=_cycle, so last.chk has cycle=5.
    re_initial = _make_two_temp_mpi_re(comm, minimal_pdb_path, cfg_run, out_dir)
    re_initial.run(5, 1, verbose=(rank == 0), write_logs=False)
    comm.Barrier()

    # Phase 2: new instance, load only (no run()).
    cfg_resume = CheckpointConfig(
        checkpoint_dir=chk_dir,
        checkpoint_interval=1,
        keep_last_n=10,
        resume=True,
    )
    re_resumed = _make_two_temp_mpi_re(comm, minimal_pdb_path, cfg_resume, out_dir)
    state = re_resumed._mpi_load_checkpoint(comm=comm)
    # Matches run()'s own convention: start_cycle = checkpoint_state.cycle + 1
    this_rank_start_cycle = (int(state["cycle"]) + 1) if state is not None else 0

    all_start_cycles = comm.gather(this_rank_start_cycle, root=0)

    if rank == 0:
        assert all_start_cycles is not None
        assert len(all_start_cycles) == comm.Get_size()
        for r_idx, sc in enumerate(all_start_cycles):
            assert sc == 6, (
                f"Rank {r_idx} has start_cycle={sc}, expected 6. "
                "After 5 completed cycles, last.chk stores cycle=5; "
                "run() resumes at cycle+1 -> 6."
            )
        assert os.path.exists(os.path.join(chk_dir, "last.chk")), (
            "last.chk must exist after 5 cycles"
        )

    comm.Barrier()
