"""Fixtures scoped to the integration subtree that need MPI or a real
checkpoint round-trip -- kept out of the root ``tests/conftest.py`` since
nothing outside ``tests/integration/`` needs them.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def broadcast_tmp_path(tmp_path):
    """Under mpirun, pytest's ``tmp_path`` is only reliable on rank 0.
    Broadcast that path so all ranks share one directory. Under plain
    pytest (no MPI), returns ``tmp_path`` unchanged."""
    try:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        path_str = str(tmp_path) if comm.Get_rank() == 0 else None
        path_str = comm.bcast(path_str, root=0)
        return Path(path_str)
    except Exception:
        return tmp_path


@pytest.fixture
def mock_comm_rank0():
    """Mock MPI communicator simulating a single-rank (rank 0) job. Safe
    under plain pytest -- no mpirun / libmpi required."""
    comm = MagicMock()
    comm.Get_rank.return_value = 0
    comm.Get_size.return_value = 1
    # TAG_UB lookup used in MPIReplicaExchange.__init__.
    comm.Get_attr.return_value = 32767
    comm.gather.side_effect = lambda data, root=0: [data]
    comm.bcast.side_effect = lambda data, root=0: data
    comm.Barrier.return_value = None
    comm.scatter.side_effect = lambda data, root=0: data[0]
    comm.allreduce.side_effect = lambda val, op=None: val
    return comm
