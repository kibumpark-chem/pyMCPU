"""Replica exchange applies the configured move settings to every replica.

``build_replica_simulation`` used to build each replica's integrator from its
temperature alone, so move weights, the sidechain mode, the rama-pivot
settings and the step size were silently ignored under REMD, while the same
settings worked for a single-trajectory folding run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("mdtraj")

from pymcpu import mcpu_core  # noqa: E402
from pymcpu.sampling.replica_exchange import ReplicaExchange  # noqa: E402

_TEMPERATURES = [0.5, 0.6]


def _rex(tmp_path: Path, pdb: str, monkeypatch, **kwargs) -> ReplicaExchange:
    # The default CheckpointConfig points at ./checkpoints; keep it out of the repo.
    monkeypatch.chdir(tmp_path)
    return ReplicaExchange(
        str(pdb),
        temperatures=_TEMPERATURES,
        n_targets=[0.0],
        k_bias=0.0,
        log_interval=5,
        output_prefix="rex",
        output_dir=tmp_path / "out",
        seed=0,
        **kwargs,
    )


def test_move_weights_reach_every_replica(tmp_path, chignolin_pdb_path, monkeypatch) -> None:
    rex = _rex(tmp_path, chignolin_pdb_path, monkeypatch, move_weights=(1.0, 0.0, 0.0))
    rex.run(2, 10, verbose=False, checkpoint_dir=None)

    assert len(rex.replicas) == len(_TEMPERATURES)
    for replica in rex.replicas:
        integrator = replica.simulation.integrator
        assert integrator.move_weights() == pytest.approx((1.0, 0.0, 0.0))
        assert integrator.get_bb_attempted() == 20
        assert integrator.get_kic_attempted() == 0
        assert integrator.get_sc_attempted() == 0


def test_mode_step_size_and_rama_schedule_reach_every_replica(
    tmp_path, chignolin_pdb_path, monkeypatch
) -> None:
    rex = _rex(
        tmp_path,
        chignolin_pdb_path,
        monkeypatch,
        step_size_rad=0.05,
        kic_step_size_rad=0.25,
        sidechain_move_mode="continuous",
        pivot_rama_schedule={"t_low": 0.5, "t_high": 0.6, "p_min": 0.2, "p_max": 0.8},
    )
    for replica in rex.replicas:
        integrator = replica.simulation.integrator
        assert integrator.backbone_step_size_rad() == pytest.approx(0.05)
        assert integrator.kic_step_size_rad() == pytest.approx(0.25)
        # The schedule is evaluated against each replica's own temperature.
        expected_p = 0.2 if replica.temperature <= 0.5 else 0.8
        assert integrator.pivot_rama_probability() == pytest.approx(expected_p)

    rex.run(2, 20, verbose=False, checkpoint_dir=None)
    for replica in rex.replicas:
        integrator = replica.simulation.integrator
        assert integrator.get_sc_attempted() > 0
        assert integrator.get_rotamer_attempted() == 0


def test_defaults_match_a_bare_integrator(tmp_path, chignolin_pdb_path, monkeypatch) -> None:
    """Default REMD settings are exactly what an unconfigured Integrator gets,
    so runs that never set them are unchanged by the fix."""
    rex = _rex(tmp_path, chignolin_pdb_path, monkeypatch)
    for replica in rex.replicas:
        integrator = replica.simulation.integrator
        bare = mcpu_core.Integrator(replica.temperature)
        assert integrator.move_weights() == bare.move_weights()
        assert integrator.backbone_step_size_rad() == bare.backbone_step_size_rad()
        assert integrator.sidechain_step_size_rad() == bare.sidechain_step_size_rad()
        assert integrator.kic_step_size_rad() == bare.kic_step_size_rad()
        assert integrator.pivot_rama_probability() == bare.pivot_rama_probability()


def test_the_mpi_driver_builds_its_replicas_with_the_widths(
    tmp_path, chignolin_pdb_path, monkeypatch, mock_comm_rank0
) -> None:
    """MPIReplicaExchange hands both widths to every replica it builds (one
    mock rank owns both slots)."""
    from pymcpu.sampling.mpi_replica_exchange import MPIReplicaExchange
    from tests.integration.mpi.test_mpi_checkpointing import (
        _ensure_mpi_module_for_construction,
    )

    _ensure_mpi_module_for_construction(monkeypatch)
    monkeypatch.chdir(tmp_path)
    rex = MPIReplicaExchange(
        mock_comm_rank0,
        str(chignolin_pdb_path),
        temperatures=_TEMPERATURES,
        n_targets=[0.0],
        k_bias=0.0,
        log_interval=5,
        output_prefix="rex",
        output_dir=str(tmp_path / "out"),
        seed=0,
        step_size_rad=0.05,
        kic_step_size_rad=0.25,
    )
    assert sorted(rex.replicas) == [0, 1]
    for replica in rex.replicas.values():
        integrator = replica.simulation.integrator
        assert integrator.backbone_step_size_rad() == pytest.approx(0.05)
        assert integrator.kic_step_size_rad() == pytest.approx(0.25)
