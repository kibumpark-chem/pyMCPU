"""MPI replica exchange at one temperature with several umbrella windows.

A one-temperature YAML config with targets runs this layout, so it must work
under ``mpirun`` too: window swaps only, some of them between two slots of
one rank and some across ranks::

    mpirun -n 2 pytest tests/integration/mpi/test_mpi_one_temperature_windows.py -v
    mpirun -n 3 pytest tests/integration/mpi/test_mpi_one_temperature_windows.py -v

Under plain single-process pytest the test skips.
"""

from __future__ import annotations

import csv
import faulthandler
import json

import pytest


@pytest.mark.mpi_integration
def test_one_temperature_windows(broadcast_tmp_path, minimal_pdb_path):
    try:
        from mpi4py import MPI
    except Exception:
        pytest.skip("mpi4py/libmpi not available")

    from pymcpu.runners import run_from_config
    from pymcpu.config import yaml_dict_to_config

    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    if size < 2:
        pytest.skip("Need mpirun -n 2 (or more) for this test")

    n_windows = size + 1  # rank 0 owns windows 0 and 1, the others one each
    out = broadcast_tmp_path / "out_windows"
    cfg = yaml_dict_to_config(
        {
            "pdb": str(minimal_pdb_path),
            "temperatures": [0.5],
            "n_targets": [2.0 * k for k in range(n_windows)],
            "k_bias": 0.5,
            "num_cycles": 3,
            "mc_replica_steps": 20,
            "log_interval": 1_000_000,
            "seed": 5,
            "exchange_log": "all",
            "output_prefix": str(out / "umb"),
            "checkpointing": {"enabled": False},
        }
    )
    assert cfg.mode == "replica_exchange_2d"

    faulthandler.dump_traceback_later(300, exit=True)
    try:
        run_from_config(cfg, comm=comm, verbose=False)
    finally:
        faulthandler.cancel_dump_traceback_later()
    comm.Barrier()
    if rank != 0:
        return
    with (out / "umb_exchange.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert {row["dim"] for row in rows} == {"Q"}
    pairs = sorted({(int(r["replica_i"]), int(r["replica_j"])) for r in rows})
    assert pairs == [(k, k + 1) for k in range(n_windows - 1)]
    assert len(rows) == 3 * (n_windows - 1)
    stats = json.loads((out / "umb_rex_stats.json").read_text())
    assert stats["temperature"]["attempts"] == 0
    assert stats["Q"]["attempts"] == 3 * (n_windows - 1)
    assert not list(broadcast_tmp_path.rglob("*.chk"))
