"""The runners build the force field a config names.

The folding runner and both replica-exchange engines used to build
MCPUForceField whatever the config said, so ``forcefield: korp`` silently ran
mcpu08. They now build through ``pymcpu.forcefields.load_forcefield``, the
same path as EngineSession. These tests run KORP through each runner, check
the guards that come with a backbone-only force field (no sidechain moves, no
CB contacts, its own trajectory topology), and check that a checkpoint
refuses to resume under a different force field.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

md = pytest.importorskip("mdtraj")

from pymcpu import runners  # noqa: E402
from pymcpu.checkpointing import (  # noqa: E402
    CheckpointConfig,
    checkpoint_forcefield_error,
    load_checkpoint,
)
from pymcpu.config import (  # noqa: E402
    ConstraintsConfig,
    EngineSpec,
    IntegratorConfig,
    OutputsConfig,
    ReplicaExchangeConfig,
    SimulationConfig,
)
from pymcpu.forcefields import load_forcefield  # noqa: E402
from pymcpu.forcefields.korp import KORPForceField  # noqa: E402

BACKBONE_MOVES = [0.5, 0.5, 0.0]


def _map_path() -> Path:
    path = os.environ.get("KORP_MAP_PATH")
    if not path or not Path(path).is_file():
        pytest.skip("set KORP_MAP_PATH to the korp6Dv1.bin energy map")
    return Path(path)


@pytest.fixture(scope="module")
def korp_pdb() -> str:
    pdb = _map_path().parent / "CASP12DCsel20" / "T0860D1.pdb"
    if not pdb.is_file():
        pytest.skip(f"{pdb} not found next to the KORP map")
    return str(pdb)


def _korp_options() -> dict:
    return {"map_path": str(_map_path()), "map_mmap": True}


def _remd_config(
    pdb: str,
    out: Path,
    *,
    forcefield: str = "korp",
    move_weights=BACKBONE_MOVES,
    resume: bool = False,
    linker_residues: list[int] | None = None,
    contact_atom_mode: str = "ca",
) -> SimulationConfig:
    return SimulationConfig(
        mode="replica_exchange_2d",
        pdb=pdb,
        forcefield=forcefield,
        forcefield_options=_korp_options() if forcefield == "korp" else {},
        integrator=IntegratorConfig(seed=5, move_weights=move_weights),
        outputs=OutputsConfig(output_dir=str(out)),
        replica_exchange=ReplicaExchangeConfig(
            temperatures=[0.5, 0.6],
            native_contact_targets=[0.0],
            k_bias=0.0,
            cycles=2,
            steps_per_cycle=20,
            log_interval=10,
            contact_atom_mode=contact_atom_mode,
        ),
        constraints=ConstraintsConfig(linker_residues=linker_residues or []),
        checkpoint=CheckpointConfig(
            checkpoint_dir=str(out / "chk"), checkpoint_interval=1, resume=resume
        ),
    )


def _folding_config(pdb: str, out: Path, **constraints) -> SimulationConfig:
    return SimulationConfig(
        mode="folding",
        pdb=pdb,
        forcefield="korp",
        forcefield_options=_korp_options(),
        integrator=IntegratorConfig(
            seed=3, steps=40, report_interval=10, move_weights=BACKBONE_MOVES
        ),
        outputs=OutputsConfig(output_dir=str(out)),
        constraints=ConstraintsConfig(**constraints),
    )


def _korp_atoms(pdb: str) -> int:
    return KORPForceField(md.load(pdb), **_korp_options()).n_atoms


# ── config validation ───────────────────────────────────────────────


def test_unknown_forcefield_is_rejected_when_the_config_is_built(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown force field 'korpp'"):
        SimulationConfig(mode="folding", pdb=str(tmp_path / "x.pdb"), forcefield="korpp")


def test_unknown_forcefield_is_rejected_from_yaml(tmp_path: Path) -> None:
    from pymcpu.config import load_yaml_config

    path = tmp_path / "c.yaml"
    path.write_text("pdb: x.pdb\ntemperatures: [0.5]\nforcefield: amber\n")
    with pytest.raises(ValueError, match="unknown force field 'amber'"):
        load_yaml_config(path)


def test_mcpu_options_are_refused_for_korp(korp_pdb: str, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="param_set/param_dir"):
        load_forcefield(korp_pdb, "korp", _korp_options(), param_dir=str(tmp_path))
    with pytest.raises(ValueError, match="param_set/param_dir"):
        load_forcefield(korp_pdb, "korp", _korp_options(), param_set="mcpu_other")


def test_korp_with_sidechain_moves_fails_before_any_output(
    korp_pdb: str, tmp_path: Path
) -> None:
    cfg = _remd_config(korp_pdb, tmp_path / "out", move_weights=None)
    with pytest.raises(ValueError, match=r"no sidechains.*\[pivot, kic, 0\.0\]"):
        runners.run_from_config(cfg, verbose=False)
    assert not list((tmp_path / "out").glob("*.xtc"))


def test_cb_contacts_are_refused_for_korp(korp_pdb: str, tmp_path: Path) -> None:
    cfg = _remd_config(korp_pdb, tmp_path / "out", contact_atom_mode="cb")
    with pytest.raises(ValueError, match="no sidechain atoms"):
        runners.run_from_config(cfg, verbose=False)


# ── each runner builds KORP ──────────────────────────────────────────


def test_serial_remd_runs_korp_and_writes_a_matching_topology(
    korp_pdb: str, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    runners.run_from_config(_remd_config(korp_pdb, out), verbose=False)

    n_atoms = _korp_atoms(korp_pdb)
    top = out / "rex_topology.pdb"
    assert top.is_file(), "a backbone-only run must write the topology its XTC needs"
    xtc = sorted(out.rglob("*.xtc"))
    assert xtc
    traj = md.load(str(xtc[0]), top=str(top))
    assert traj.n_atoms == n_atoms
    assert {a.name for a in traj.topology.atoms} <= {"N", "CA", "C", "O", "OXT"}

    state = load_checkpoint(out / "chk" / "last.chk")
    assert state["forcefield"] == "korp"
    assert np.asarray(state["replica_coords"][0]).shape[1] == n_atoms


def test_folding_runs_korp_with_the_same_start_energy_as_engine_session(
    korp_pdb: str, tmp_path: Path
) -> None:
    from pymcpu.sampling import EngineSession
    from pymcpu.sampling.folding import FoldingRunner

    runner = FoldingRunner(
        korp_pdb, forcefield="korp", forcefield_options=_korp_options(),
        move_weights=tuple(BACKBONE_MOVES), output_dir=tmp_path / "f", verbose=False,
    )
    assert isinstance(runner.forcefield, KORPForceField)
    session = EngineSession(EngineSpec(
        pdb=korp_pdb, forcefield="korp", forcefield_options=_korp_options(),
        move_weights=tuple(BACKBONE_MOVES),
    ))
    e_session = session._ensure_sim().context.calculate_total_energy(-1)
    assert runner.simulation.context.calculate_total_energy(-1) == e_session

    runners.run_from_config(_folding_config(korp_pdb, tmp_path / "cfg"), verbose=False)
    top = tmp_path / "cfg" / "folding_topology.pdb"
    assert top.is_file()
    xtc = sorted((tmp_path / "cfg").rglob("*.xtc"))
    assert xtc and md.load(str(xtc[0]), top=str(top)).n_atoms == runner.forcefield.n_atoms


def test_run_from_config_hands_the_forcefield_to_the_mpi_runner(
    korp_pdb: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = {}
    monkeypatch.setattr(
        runners, "run_mpi_replica_exchange_2d", lambda **kw: seen.update(kw))
    runners.run_from_config(_remd_config(korp_pdb, tmp_path), comm=object(), verbose=False)
    assert seen["forcefield"] == "korp"
    assert seen["forcefield_options"] == _korp_options()


def _mpi_korp_engine(korp_pdb: str, tmp_path: Path, comm, monkeypatch):
    import pymcpu.sampling.mpi_replica_exchange as mmod
    from unittest.mock import MagicMock

    if mmod.MPI is None:  # libmpi missing: same stand-in as the MPI unit tests
        fake = MagicMock(TAG_UB=32767, LOR=object())
        monkeypatch.setattr(mmod, "MPI", fake)
        monkeypatch.setattr(mmod, "_require_mpi", lambda: fake)
    return mmod.MPIReplicaExchange(
        comm, korp_pdb, temperatures=[0.5, 0.6], n_targets=[0.0],
        k_bias=0.0, log_interval=1, output_dir=str(tmp_path), output_prefix="rex",
        seed=1, move_weights=tuple(BACKBONE_MOVES),
        forcefield="korp", forcefield_options=_korp_options(),
    )


def test_mpi_engine_builds_korp(
    korp_pdb: str, tmp_path: Path, mock_comm_rank0, monkeypatch: pytest.MonkeyPatch
) -> None:
    rex = _mpi_korp_engine(korp_pdb, tmp_path, mock_comm_rank0, monkeypatch)
    assert isinstance(rex.forcefield, KORPForceField)
    assert rex.system.get_num_atoms() == rex.forcefield.n_atoms
    assert rex.top_path == str(tmp_path / "rex_topology.pdb")
    assert md.load(rex.top_path).n_atoms == rex.forcefield.n_atoms


def test_mpi_topology_write_error_reaches_every_rank(
    korp_pdb: str, tmp_path: Path, mock_comm_rank0, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rank 0's write error is broadcast, so no rank waits in a barrier."""
    import pymcpu.sampling.mpi_replica_exchange as mmod

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(mmod, "trajectory_topology_path", fail)
    with pytest.raises(RuntimeError, match="disk full"):
        _mpi_korp_engine(korp_pdb, tmp_path, mock_comm_rank0, monkeypatch)
    sent = [c.args[0] for c in mock_comm_rank0.bcast.call_args_list]
    assert any(isinstance(x, str) and "disk full" in x for x in sent)


# ── resume ───────────────────────────────────────────────────────────


def test_resume_under_a_different_forcefield_is_refused(
    korp_pdb: str, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    runners.run_from_config(_remd_config(korp_pdb, out), verbose=False)
    cfg = _remd_config(korp_pdb, out, forcefield="mcpu08", resume=True)
    with pytest.raises(ValueError, match="written by a 'korp' run.*'mcpu08'"):
        runners.run_from_config(cfg, verbose=False)


def test_resume_under_the_same_forcefield_continues(korp_pdb: str, tmp_path: Path) -> None:
    out = tmp_path / "out"
    runners.run_from_config(_remd_config(korp_pdb, out), verbose=False)
    runners.run_from_config(_remd_config(korp_pdb, out, resume=True), verbose=False)


def test_forcefield_check_treats_old_checkpoints_as_mcpu08() -> None:
    assert checkpoint_forcefield_error("", "mcpu08") is None
    assert checkpoint_forcefield_error(None, "mcpu") is None  # alias
    assert checkpoint_forcefield_error("mcpu", "mcpu08") is None
    assert "written by a 'mcpu08' run" in checkpoint_forcefield_error("", "korp")
    assert checkpoint_forcefield_error("retired", "korp") is not None


# ── linker masks reach KORP ──────────────────────────────────────────


def test_linker_mask_reaches_korp_in_every_runner(korp_pdb: str, tmp_path: Path) -> None:
    """Needs the KORP potentials to honour energy masks (korp-masks)."""
    from pymcpu.sampling.folding import FoldingRunner
    from pymcpu.sampling.replica_exchange import ReplicaExchange

    linker = list(range(0, 12))
    common = dict(forcefield="korp", forcefield_options=_korp_options(),
                  move_weights=tuple(BACKBONE_MOVES))
    plain = FoldingRunner(korp_pdb, output_dir=tmp_path / "a", verbose=False, **common)
    masked = FoldingRunner(korp_pdb, output_dir=tmp_path / "b", verbose=False,
                           linker_residues=linker, **common)
    e_plain = plain.simulation.context.calculate_total_energy(-1)
    e_masked = masked.simulation.context.calculate_total_energy(-1)
    assert e_masked != e_plain

    rex = ReplicaExchange(
        korp_pdb, temperatures=[0.5], n_targets=[0.0], k_bias=0.0,
        output_dir=str(tmp_path / "r"), linker_residues=linker, **common)
    e_rex = rex.replicas[0].simulation.context.calculate_total_energy(-1)
    assert e_rex == e_masked
