"""Config-driven replica exchange applies the configured move settings.

``run_from_config`` -> ``run_replica_exchange_2d`` / ``run_mpi_replica_exchange_2d``
-> ``ReplicaExchange`` / ``MPIReplicaExchange`` used to drop every move
setting, so REMD ran with the defaults whatever the config said. One test
runs a real (tiny) serial REMD from a config and reads its CSVs; the other
checks that each of the five settings is forwarded, on the serial and the
MPI path.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

pytest.importorskip("mdtraj")

from pymcpu import runners  # noqa: E402
from pymcpu.config import (  # noqa: E402
    CheckpointConfig,
    IntegratorConfig,
    OutputsConfig,
    ReplicaExchangeConfig,
    SimulationConfig,
)


def _remd_config(tmp_path: Path, integrator: IntegratorConfig) -> SimulationConfig:
    return SimulationConfig(
        mode="replica_exchange_2d",
        pdb=str(runners.default_example_pdb()),
        integrator=integrator,
        outputs=OutputsConfig(output_dir=str(tmp_path / "out")),
        replica_exchange=ReplicaExchangeConfig(
            temperatures=[0.5, 0.6],
            native_contact_targets=[0.0],
            cycles=2,
            steps_per_cycle=10,
            log_interval=10,
        ),
        checkpoint=CheckpointConfig(checkpoint_dir=str(tmp_path / "checkpoints")),
    )


def test_remd_config_runs_and_applies_move_settings(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    cfg = _remd_config(
        tmp_path, IntegratorConfig(move_weights=(1.0, 0.0, 0.0), pivot_rama_probability=1.0)
    )

    runners.run_from_config(cfg, verbose=False)

    files = sorted((tmp_path / "out").glob("rex_*_data.csv"))
    assert len(files) == 2
    for path in files:
        with path.open() as fh:
            last = list(csv.DictReader(fh))[-1]
        # 2 cycles x 10 steps, all of them rama pivots; only pivot-slot columns.
        assert int(last["rama_pivot_attempted"]) == 20
        assert int(last["pivot_attempted"]) == 0
        assert "kic_attempted" not in last
        assert "rotamer_attempted" not in last


class _Stop(Exception):
    """Raised by the stand-in sampler once it has recorded its arguments."""


def _recorder(seen: dict):
    class Recorder:
        def __init__(self, *args, **kwargs) -> None:
            seen.update(kwargs)
            raise _Stop

    return Recorder


class _SingleRankComm:
    def Get_rank(self) -> int:  # noqa: N802 (mpi4py spelling)
        return 0

    def Get_size(self) -> int:  # noqa: N802
        return 1

    def Barrier(self) -> None:  # noqa: N802
        pass


_SETTINGS = dict(
    step_size_rad=0.07,
    move_weights=(0.5, 0.3, 0.2),
    sidechain_move_mode="continuous",
    pivot_rama_probability=0.4,
    pivot_rama_schedule={"t_low": 0.5, "t_high": 0.6, "p_min": 0.1, "p_max": 0.9},
)


@pytest.mark.parametrize("mpi", [False, True], ids=["serial", "mpi"])
def test_every_move_setting_is_forwarded(tmp_path: Path, monkeypatch, mpi: bool) -> None:
    monkeypatch.chdir(tmp_path)
    seen: dict = {}
    if mpi:
        import pymcpu.sampling.mpi_replica_exchange as mpi_rex

        monkeypatch.setattr(mpi_rex, "MPIReplicaExchange", _recorder(seen))
        comm = _SingleRankComm()
    else:
        monkeypatch.setattr(runners, "ReplicaExchange", _recorder(seen))
        comm = None
    integrator = IntegratorConfig(**_SETTINGS)

    with pytest.raises(_Stop):
        runners.run_from_config(_remd_config(tmp_path, integrator), comm=comm, verbose=False)

    assert seen["step_size_rad"] == pytest.approx(0.07)
    assert seen["move_weights"] == pytest.approx(integrator.move_weights)
    assert seen["sidechain_move_mode"] == "continuous"
    assert seen["pivot_rama_probability"] == pytest.approx(0.4)
    assert seen["pivot_rama_schedule"] == _SETTINGS["pivot_rama_schedule"]
