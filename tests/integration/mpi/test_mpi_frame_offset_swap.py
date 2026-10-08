"""A cross-rank replica swap carries each walker's engine frame: requires
``mpirun -n 2``::

    mpirun -n 2 pytest tests/integration/mpi/test_mpi_frame_offset_swap.py -v

The frame offset moves with the chain (Context.recenter), so two walkers can
sit in different frames; the swap sends the offset with the coordinates, and
each walker re-enters on the other rank bit for bit. A checkpoint save
recentres each walker before it stores the walker's offset. Under plain
single-process pytest it skips.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.mpi_integration
def test_a_cross_rank_swap_keeps_each_walker_frame(
    broadcast_tmp_path, minimal_pdb_path, monkeypatch
) -> None:
    try:
        from mpi4py import MPI
    except Exception:
        pytest.skip("mpi4py/libmpi not available")

    from pymcpu.checkpointing import CheckpointConfig
    from pymcpu.sampling import mpi_replica_exchange
    from pymcpu.sampling.replica_exchange_core import get_coords

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if comm.Get_size() != 2:
        pytest.skip("Need mpirun -n 2 for this test")

    out_dir = broadcast_tmp_path / "out_frame_swap"
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
    comm.Barrier()
    rex = mpi_replica_exchange.MPIReplicaExchange(
        comm,
        str(minimal_pdb_path),
        temperatures=[0.5, 0.6],
        n_targets=[0.0],
        k_bias=0.0,
        log_interval=1,
        output_dir=str(out_dir),
        output_prefix="rex",
        seed=42,
        checkpoint_config=CheckpointConfig(
            checkpoint_dir=str(broadcast_tmp_path / "ck_frame_swap"),
            checkpoint_interval=100,
            resume=False,
        ),
    )
    assert [int(r) for r in rex.owner_rank] == [0, 1]
    (rid,) = list(rex.local_replica_indices)
    ctx = rex.replicas[rid].simulation.context

    # Each rank moves its walker to its own far place and recentres it there.
    ctx.set_positions(get_coords(ctx) + 100.0 * (rank + 1), frame_offset=(0.0, 0.0, 0.0))
    ctx.calculate_total_energy(-1)
    assert np.all(ctx.recenter() > 90.0)
    mine = (np.asarray(ctx.get_state().coords).copy(), ctx.frame_offset.copy())
    theirs = comm.allgather(mine)[1 - rank]
    assert not np.array_equal(mine[1], theirs[1])

    monkeypatch.setattr(
        mpi_replica_exchange, "evaluate_exchange_acceptance", lambda *a, **k: True
    )
    rex._attempt_exchange(0, 1, dim="T")
    assert np.array_equal(np.asarray(ctx.get_state().coords), theirs[0])
    assert np.array_equal(ctx.frame_offset, theirs[1])
    assert not ctx.has_steric_clash()

    rex._attempt_exchange(0, 1, dim="T")
    assert np.array_equal(np.asarray(ctx.get_state().coords), mine[0])
    assert np.array_equal(ctx.frame_offset, mine[1])
    comm.Barrier()


@pytest.mark.mpi_integration
def test_a_checkpoint_save_recentres_each_walker(broadcast_tmp_path, minimal_pdb_path) -> None:
    try:
        from mpi4py import MPI
    except Exception:
        pytest.skip("mpi4py/libmpi not available")

    from pymcpu.checkpointing import CheckpointConfig, load_checkpoint, saved_frame_offset
    from pymcpu.sampling import mpi_replica_exchange
    from pymcpu.sampling.replica_exchange_core import get_coords

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if comm.Get_size() != 2:
        pytest.skip("Need mpirun -n 2 for this test")

    out_dir = broadcast_tmp_path / "out_frame_save"
    ck_dir = broadcast_tmp_path / "ck_frame_save"
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
    comm.Barrier()
    rex = mpi_replica_exchange.MPIReplicaExchange(
        comm,
        str(minimal_pdb_path),
        temperatures=[0.5, 0.6],
        n_targets=[0.0],
        k_bias=0.0,
        log_interval=1,
        output_dir=str(out_dir),
        output_prefix="rex",
        seed=42,
        checkpoint_config=CheckpointConfig(
            checkpoint_dir=str(ck_dir), checkpoint_interval=1, resume=False
        ),
    )
    (rid,) = list(rex.local_replica_indices)
    ctx = rex.replicas[rid].simulation.context

    # Each walker drifted far out in its unshifted frame; the save moves it.
    ctx.set_positions(get_coords(ctx) + 100.0 * (rank + 1), frame_offset=(0.0, 0.0, 0.0))
    ctx.calculate_total_energy(-1)
    rex._mpi_save_checkpoint(1, comm)
    assert np.all(ctx.frame_offset > 90.0)
    assert np.abs(np.asarray(ctx.get_state().coords)).max() < 64.0
    offsets = comm.allgather((rid, ctx.frame_offset.copy()))
    if rank == 0:
        state = load_checkpoint(ck_dir / "last.chk")
        for r, offset in offsets:
            assert np.array_equal(saved_frame_offset(state, r), offset)
    comm.Barrier()
