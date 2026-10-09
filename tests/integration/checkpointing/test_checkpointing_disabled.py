"""``enabled: false`` in a config's checkpointing block writes no checkpoint.

``run_from_config`` never passed ``enabled`` on, and only ``FoldingRunner``
read it, so ``mcpu run`` wrote checkpoints anyway. Each test runs a short
simulation from a config with checkpointing off and looks for checkpoint
files: folding, serial replica exchange, and MPI replica exchange through a
single-rank mock communicator. The last test checks that the same setup with
checkpointing on does write them, so an empty directory means something.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("mdtraj")

from pymcpu import runners  # noqa: E402
from pymcpu.checkpointing import CheckpointConfig  # noqa: E402
from pymcpu.config import (  # noqa: E402
    IntegratorConfig,
    OutputsConfig,
    ReplicaExchangeConfig,
    SimulationConfig,
    yaml_dict_to_config,
)


def _checkpoints(tmp_path: Path) -> list[Path]:
    return sorted(tmp_path.rglob("*.chk"))


def _config(tmp_path: Path, mode: str, *, enabled: bool) -> SimulationConfig:
    rex = None
    if mode == "replica_exchange_2d":
        rex = ReplicaExchangeConfig(
            temperatures=[0.5, 0.6], cycles=2, steps_per_cycle=5, log_interval=5
        )
    return SimulationConfig(
        mode=mode,
        pdb=str(runners.default_example_pdb()),
        integrator=IntegratorConfig(steps=10, report_interval=5),
        outputs=OutputsConfig(output_dir=str(tmp_path / "out")),
        replica_exchange=rex,
        checkpoint=CheckpointConfig(
            checkpoint_dir=str(tmp_path / "ck"), checkpoint_interval=1, enabled=enabled
        ),
    )


@pytest.mark.parametrize("mode", ["folding", "replica_exchange_2d"])
def test_no_checkpoint_when_disabled(tmp_path: Path, monkeypatch, mode: str) -> None:
    monkeypatch.chdir(tmp_path)

    runners.run_from_config(_config(tmp_path, mode, enabled=False), verbose=False)

    assert _checkpoints(tmp_path) == []
    assert not (tmp_path / "ck").exists()


def test_no_checkpoint_when_disabled_under_mpi(
    tmp_path: Path, monkeypatch, mock_comm_rank0
) -> None:
    from tests.integration.mpi.test_mpi_checkpointing import (
        _ensure_mpi_module_for_construction,
    )

    _ensure_mpi_module_for_construction(monkeypatch)
    monkeypatch.chdir(tmp_path)

    runners.run_from_config(
        _config(tmp_path, "replica_exchange_2d", enabled=False),
        comm=mock_comm_rank0,
        verbose=False,
    )

    assert _checkpoints(tmp_path) == []
    assert not (tmp_path / "ck").exists()


def test_a_yaml_checkpointing_block_turns_it_off(tmp_path: Path) -> None:
    cfg = yaml_dict_to_config(
        {"pdb": "x.pdb", "temperatures": [0.5, 0.6], "checkpointing": {"enabled": False}}
    )
    assert cfg.checkpoint.enabled is False


@pytest.mark.parametrize("mode", ["folding", "replica_exchange_2d"])
def test_checkpoints_are_written_when_enabled(tmp_path: Path, monkeypatch, mode: str) -> None:
    monkeypatch.chdir(tmp_path)

    runners.run_from_config(_config(tmp_path, mode, enabled=True), verbose=False)

    assert (tmp_path / "ck" / "last.chk").is_file()
