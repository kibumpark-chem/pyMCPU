"""``run_from_config`` MPI dispatch: it must be additive-only.

Every existing caller (``mcpu run`` and every test that predates this file)
calls ``run_from_config(cfg, verbose=...)`` with no ``comm`` -- these tests pin
that the no-``comm`` path is unchanged, and that passing an explicit ``comm``
(never reading ``cfg.mpi``, which is intentionally-ignored per
docs/running_remd.md) is what selects the MPI replica-exchange runner.
Monkeypatches the ``run_*`` functions directly rather than exercising a full
simulation, since this file is about dispatch wiring, not replica-exchange
physics (that's covered elsewhere).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from pymcpu import runners
from pymcpu.config import (
    CheckpointConfig,
    ConstraintsConfig,
    IntegratorConfig,
    OutputsConfig,
    ReplicaExchangeConfig,
    SimulationConfig,
)


def _rex_config(**overrides) -> SimulationConfig:
    return SimulationConfig(
        mode="replica_exchange_2d",
        pdb="/nonexistent/dummy.pdb",
        integrator=IntegratorConfig(),
        outputs=OutputsConfig(),
        replica_exchange=ReplicaExchangeConfig(),
        constraints=ConstraintsConfig(),
        checkpoint=CheckpointConfig(),
        **overrides,
    )


def _folding_config(**overrides) -> SimulationConfig:
    return SimulationConfig(
        mode="folding",
        pdb="/nonexistent/dummy.pdb",
        integrator=IntegratorConfig(),
        outputs=OutputsConfig(),
        checkpoint=CheckpointConfig(),
        **overrides,
    )


def test_replica_exchange_without_comm_calls_serial_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    serial = MagicMock(return_value="serial-result")
    mpi = MagicMock(return_value="mpi-result")
    monkeypatch.setattr(runners, "run_replica_exchange_2d", serial)
    monkeypatch.setattr(runners, "run_mpi_replica_exchange_2d", mpi)

    result = runners.run_from_config(_rex_config(), verbose=False)

    assert result == "serial-result"
    serial.assert_called_once()
    mpi.assert_not_called()
    assert "comm" not in serial.call_args.kwargs
    assert serial.call_args.kwargs["backend"] == "serial"


def test_replica_exchange_with_comm_calls_mpi_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    serial = MagicMock(return_value="serial-result")
    mpi = MagicMock(return_value="mpi-result")
    monkeypatch.setattr(runners, "run_replica_exchange_2d", serial)
    monkeypatch.setattr(runners, "run_mpi_replica_exchange_2d", mpi)
    comm = MagicMock()

    result = runners.run_from_config(_rex_config(), comm=comm, verbose=False)

    assert result == "mpi-result"
    mpi.assert_called_once()
    serial.assert_not_called()
    assert mpi.call_args.kwargs["comm"] is comm
    # backend is serial-only; the MPI runner doesn't accept it.
    assert "backend" not in mpi.call_args.kwargs


def test_replica_exchange_mpi_dispatch_ignores_cfg_mpi_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``cfg.mpi`` must stay a launch-mechanics no-op (docs/running_remd.md):
    MPI-ness comes only from passing ``comm``, never from the config file."""
    serial = MagicMock(return_value="serial-result")
    mpi = MagicMock(return_value="mpi-result")
    monkeypatch.setattr(runners, "run_replica_exchange_2d", serial)
    monkeypatch.setattr(runners, "run_mpi_replica_exchange_2d", mpi)

    result = runners.run_from_config(_rex_config(mpi=True), verbose=False)

    assert result == "serial-result"
    serial.assert_called_once()
    mpi.assert_not_called()


def test_folding_without_comm_is_unaffected(monkeypatch: pytest.MonkeyPatch) -> None:
    folding = MagicMock(return_value="folding-result")
    monkeypatch.setattr(runners, "run_folding", folding)

    result = runners.run_from_config(_folding_config(), verbose=False)

    assert result == "folding-result"
    folding.assert_called_once()


def test_folding_with_comm_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    folding = MagicMock(return_value="folding-result")
    monkeypatch.setattr(runners, "run_folding", folding)

    with pytest.raises(ValueError, match="MPI is not supported"):
        runners.run_from_config(_folding_config(), comm=MagicMock(), verbose=False)

    folding.assert_not_called()
