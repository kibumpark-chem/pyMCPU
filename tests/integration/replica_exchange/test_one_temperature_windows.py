"""A YAML config with one temperature and targets runs umbrella sampling.

Such a config used to run plain folding and drop its targets without a
message. It now runs replica exchange over the windows at that temperature,
which both drivers support: the windows swap with their neighbours and there
are no temperature swaps. Run here in process and through the MPI driver
with a single-rank mock communicator; ``tests/integration/mpi`` runs it under
``mpirun``. The umbrella defaults are checked too: no targets means no
umbrella, unless ``k_bias`` is given.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

pytest.importorskip("mdtraj")

from pymcpu import runners  # noqa: E402
from pymcpu.config import yaml_dict_to_config  # noqa: E402


def _config(tmp_path: Path, **keys):
    return yaml_dict_to_config(
        {
            "pdb": str(runners.default_example_pdb()),
            "temperatures": [0.5],
            "n_targets": [0, 3, 6],
            "k_bias": 0.5,
            "num_cycles": 2,
            "mc_replica_steps": 5,
            "exchange_log": "all",
            "output_prefix": str(tmp_path / "out" / "umb"),
            "checkpointing": {"enabled": False},
            **keys,
        }
    )


@pytest.mark.parametrize("mpi", [False, True], ids=["serial", "mpi"])
def test_one_temperature_umbrella_run(tmp_path: Path, monkeypatch, mpi: bool, request) -> None:
    monkeypatch.chdir(tmp_path)
    comm = None
    if mpi:
        from tests.integration.mpi.test_mpi_checkpointing import (
            _ensure_mpi_module_for_construction,
        )

        _ensure_mpi_module_for_construction(monkeypatch)
        comm = request.getfixturevalue("mock_comm_rank0")
    cfg = _config(tmp_path)
    assert cfg.mode == "replica_exchange_2d"

    runners.run_from_config(cfg, comm=comm, verbose=False)

    out = tmp_path / "out"
    assert len(sorted(out.glob("umb_*_data.csv"))) == 3  # one per window
    with (out / "umb_exchange.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert {row["dim"] for row in rows} == {"Q"}
    assert sorted({(int(r["replica_i"]), int(r["replica_j"])) for r in rows}) == [(0, 1), (1, 2)]
    assert len(rows) == 2 * 2  # two neighbour pairs, two cycles
    stats = json.loads((out / "umb_rex_stats.json").read_text())
    assert stats["temperature"]["attempts"] == 0
    assert stats["Q"]["attempts"] == 4


class _Stop(Exception):
    pass


@pytest.mark.parametrize(
    "keys, k_bias",
    [
        ({"temperatures": [0.5, 0.6]}, 0.0),
        ({"temperatures": [0.5], "n_targets": [0, 3, 6]}, 1.0),
        ({"temperatures": [0.5], "n_targets": [0, 3, 6], "k_bias": 0.5}, 0.5),
        ({"temperatures": [0.5, 0.6], "k_bias": 0.25}, 0.25),
    ],
    ids=["no-targets", "targets", "targets-and-k", "k-without-targets"],
)
@pytest.mark.parametrize("mpi", [False, True], ids=["serial", "mpi"])
def test_the_sampler_gets_the_umbrella_default(
    tmp_path: Path, monkeypatch, keys: dict, k_bias: float, mpi: bool
) -> None:
    from unittest.mock import MagicMock

    seen: dict = {}

    class Recorder:
        def __init__(self, *args, **kwargs) -> None:
            seen.update(kwargs)
            raise _Stop

    comm = None
    if mpi:
        import pymcpu.sampling.mpi_replica_exchange as mpi_rex

        monkeypatch.setattr(mpi_rex, "MPIReplicaExchange", Recorder)
        comm = MagicMock()
        comm.Get_rank.return_value = 0
    else:
        monkeypatch.setattr(runners, "ReplicaExchange", Recorder)
    monkeypatch.chdir(tmp_path)
    cfg = yaml_dict_to_config(
        {"pdb": str(runners.default_example_pdb()), "output_prefix": str(tmp_path / "o" / "r"),
         **keys}
    )

    with pytest.raises(_Stop):
        runners.run_from_config(cfg, comm=comm, verbose=False)

    assert seen["k_bias"] == k_bias


def test_the_runner_function_defaults_to_no_umbrella_without_targets(
    tmp_path: Path, monkeypatch
) -> None:
    """``run_replica_exchange_2d`` used to default to k_bias 1.0 whatever the
    targets, so a direct call without targets pulled toward N = 0."""
    seen: dict = {}

    class Recorder:
        def __init__(self, *args, **kwargs) -> None:
            seen.update(kwargs)
            raise _Stop

    monkeypatch.setattr(runners, "ReplicaExchange", Recorder)
    pdb = runners.default_example_pdb()
    for targets, expected in (({}, 0.0), ({"n_targets": [0.0, 4.0]}, 1.0)):
        with pytest.raises(_Stop):
            runners.run_replica_exchange_2d(
                pdb=pdb, temperatures=[0.5, 0.6], output_dir=tmp_path, verbose=False, **targets
            )
        assert seen["k_bias"] == expected
