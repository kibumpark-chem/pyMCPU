"""MPI replica exchange with more slots than ranks: requires ``mpirun -n 2`` or more.

A rank that owns two neighbouring slots exchanges them locally, and every
rank must still meet at the same collectives, or the next cross-rank exchange
deadlocks (3 or 4 temperatures on 2 ranks used to hang in the first cycle)::

    mpirun -n 2 pytest tests/integration/mpi/test_mpi_uneven_ranks.py -v
    mpirun -n 3 pytest tests/integration/mpi/test_mpi_uneven_ranks.py -v

Under plain single-process pytest the test skips. After the run, rank 0
repeats it alone, owning every slot, and the exchange records must be
identical: who owns a slot does not change the run.
"""

from __future__ import annotations

import csv
import faulthandler

import pytest


@pytest.mark.mpi_integration
@pytest.mark.parametrize("layout", ["one_extra_slot", "two_slots_each"])
def test_ranks_owning_neighbouring_slots(broadcast_tmp_path, minimal_pdb_path, layout):
    try:
        from mpi4py import MPI
    except Exception:
        pytest.skip("mpi4py/libmpi not available")

    from pymcpu.sampling.mpi_replica_exchange import MPIReplicaExchange

    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    if size < 2:
        pytest.skip("Need mpirun -n 2 (or more) for this test")

    # Rank 0 owns slots 0 and 1 in both layouts, so pair (0, 1) is local to it
    # and pair (1, 2) crosses ranks. With one extra slot every other rank owns
    # one slot (3 temperatures on 2 ranks, 4 on 3); with two slots each, every
    # rank exchanges a pair locally (4 temperatures on 2 ranks).
    if layout == "one_extra_slot":
        n_temps = size + 1
        expected = [0, 1] if rank == 0 else [rank + 1]
    else:
        n_temps = 2 * size
        expected = [2 * rank, 2 * rank + 1]

    out = broadcast_tmp_path / "out_uneven"
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
    comm.Barrier()

    def run(run_comm, run_out):
        rex = MPIReplicaExchange(
            run_comm,
            str(minimal_pdb_path),
            temperatures=[0.5 + 0.05 * k for k in range(n_temps)],
            n_targets=[0.0],
            k_bias=0.0,
            log_interval=1_000_000,
            output_dir=str(run_out),
            output_prefix="rex",
            seed=3,
            exchange_log="all",
            checkpoint_dir=str(run_out / "checkpoints"),
        )
        # A deadlock would otherwise hang until the job's time limit.
        faulthandler.dump_traceback_later(300, exit=True)
        try:
            rex.run(3, 20, verbose=False, write_logs=True)
        finally:
            faulthandler.cancel_dump_traceback_later()
        return rex

    def exchange_rows(run_out):
        with (run_out / "rex_exchange.csv").open() as handle:
            return list(csv.DictReader(handle))

    rex = run(comm, out)
    assert sorted(rex.replicas) == expected
    comm.Barrier()
    if rank != 0:
        return
    rows = exchange_rows(out)
    pairs = sorted({(int(r["replica_i"]), int(r["replica_j"])) for r in rows})
    assert pairs == [(k, k + 1) for k in range(n_temps - 1)]
    assert len(rows) == 3 * (n_temps - 1)

    class OneRank:
        """MPI.COMM_SELF, with COMM_WORLD's TAG_UB (MPI defines it on COMM_WORLD only)."""

        def Get_attr(self, key):
            return MPI.COMM_WORLD.Get_attr(key) if key == MPI.TAG_UB else MPI.COMM_SELF.Get_attr(key)

        def __getattr__(self, name):
            return getattr(MPI.COMM_SELF, name)

    alone = broadcast_tmp_path / "out_one_rank"
    alone.mkdir(parents=True, exist_ok=True)
    run(OneRank(), alone)
    alone_rows = exchange_rows(alone)

    def key(row):
        return int(row["cycle"]), int(row["replica_i"]), int(row["replica_j"])

    assert sorted(rows, key=key) == sorted(alone_rows, key=key)
