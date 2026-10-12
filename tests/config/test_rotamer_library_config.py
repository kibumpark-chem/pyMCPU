"""``rotamer_library``: defaults, config round trips, where it is applied, and
checkpoints.

Every entry point (YAML, JSON, IntegratorConfig, EngineSpec, the runner
keywords, the sampler classes) must carry the setting to the System the run
builds, every default must be the same library (also the one a System built
straight from the force field holds), and a checkpoint must refuse to resume
under another library. The MPI driver's resume check is in
``tests/integration/mpi/test_mpi_checkpointing.py``.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from pymcpu.checkpointing import (
    checkpoint_rotamer_library_error,
    load_checkpoint,
    save_checkpoint,
)
from pymcpu.config import (
    DEFAULT_ROTAMER_LIBRARY,
    EngineSpec,
    IntegratorConfig,
    config_from_dict,
    normalize_rotamer_library,
    yaml_dict_to_config,
)
from pymcpu.runners import default_example_pdb

md = pytest.importorskip("mdtraj")

_BASE_YAML = {"pdb": "examples/data/1uao.pdb", "temperatures": [0.6], "num_cycles": 1,
              "mc_replica_steps": 10}
_BASE_JSON = {"mode": "folding", "pdb": "examples/data/1uao.pdb"}
_PDB = str(default_example_pdb())


def _fingerprint(system) -> str:
    return system.get_rotamer_library().fingerprint


def test_every_default_is_dunbrack2002_sym() -> None:
    from pymcpu import runners
    from pymcpu.forcefields import load_forcefield
    from pymcpu.forcefields.builders.rotamer_builder import load_rotamer_library
    from pymcpu.sampling.folding import FoldingRunner
    from pymcpu.sampling.mpi_replica_exchange import MPIReplicaExchange
    from pymcpu.sampling.replica_exchange import ReplicaExchange
    from pymcpu.sampling.replica_exchange_core import build_system_and_cv

    assert DEFAULT_ROTAMER_LIBRARY == "dunbrack2002_sym"
    assert IntegratorConfig().rotamer_library == DEFAULT_ROTAMER_LIBRARY
    assert yaml_dict_to_config(dict(_BASE_YAML)).integrator.rotamer_library == DEFAULT_ROTAMER_LIBRARY
    assert config_from_dict(dict(_BASE_JSON)).integrator.rotamer_library == DEFAULT_ROTAMER_LIBRARY
    assert EngineSpec(pdb=_PDB).rotamer_library == DEFAULT_ROTAMER_LIBRARY
    assert normalize_rotamer_library(None) == normalize_rotamer_library("") == DEFAULT_ROTAMER_LIBRARY
    for entry in (runners.run_folding, runners.run_replica_exchange_2d,
                  runners.run_mpi_replica_exchange_2d, FoldingRunner, ReplicaExchange,
                  MPIReplicaExchange, build_system_and_cv):
        assert inspect.signature(entry).parameters["rotamer_library"].default == DEFAULT_ROTAMER_LIBRARY
    assert inspect.signature(load_rotamer_library).parameters["name"].default == DEFAULT_ROTAMER_LIBRARY
    # What the engine gets from a System built straight from the force field
    # (Simulation, notebooks, scripts): the same library.
    ff = load_forcefield(_PDB)
    system = ff.create_system(ff.output_topology)
    assert _fingerprint(system) == load_rotamer_library().fingerprint
    assert _fingerprint(system).startswith("dunbrack2002_sym md5=")
    assert system.get_rotamer_library().has_truncated_tables()


@pytest.mark.parametrize("name", ["dunbrack2002", "dunbrack2002_sym"])
def test_every_config_entry_point_round_trips(name: str) -> None:
    yaml_cfg = yaml_dict_to_config({**_BASE_YAML, "rotamer_library": name})
    json_cfg = config_from_dict({**_BASE_JSON, "integrator": {"rotamer_library": name}})
    assert yaml_cfg.integrator.rotamer_library == json_cfg.integrator.rotamer_library == name
    assert IntegratorConfig(rotamer_library=name).rotamer_library == name
    assert EngineSpec(pdb=_PDB, rotamer_library=f" {name.upper()} ").rotamer_library == name
    yaml_cfg.pdb = _PDB
    assert EngineSpec.from_simulation_config(yaml_cfg).rotamer_library == name


def test_an_unknown_library_is_refused_everywhere() -> None:
    for build in (
        lambda: yaml_dict_to_config({**_BASE_YAML, "rotamer_library": "dunbrack2010"}),
        lambda: config_from_dict({**_BASE_JSON, "integrator": {"rotamer_library": "bbind"}}),
        lambda: IntegratorConfig(rotamer_library="bbind"),
        lambda: EngineSpec(pdb=_PDB, rotamer_library="bbind"),
    ):
        with pytest.raises(ValueError, match="rotamer_library must be one of"):
            build()


@pytest.fixture
def stop_at_construction(monkeypatch):
    """Replaces the sampler classes with stubs that record their keyword
    arguments and stop the run there."""
    seen = {}

    class Stop(Exception):
        pass

    def stub(name):
        def build(*args, **kwargs):
            seen[name] = kwargs
            raise Stop
        return build

    import pymcpu.sampling.folding as folding

    monkeypatch.setattr(folding, "FoldingRunner", stub("folding"))
    monkeypatch.setattr("pymcpu.runners.ReplicaExchange", stub("rex"))
    return seen, Stop


def test_runner_keywords_reach_the_samplers(stop_at_construction, tmp_path, monkeypatch) -> None:
    from pymcpu.runners import run_folding, run_replica_exchange_2d

    monkeypatch.chdir(tmp_path)  # default checkpoint directories
    seen, stop = stop_at_construction
    with pytest.raises(stop):
        run_folding(pdb=_PDB, output_dir=tmp_path, rotamer_library="dunbrack2002")
    assert seen["folding"]["rotamer_library"] == "dunbrack2002"
    with pytest.raises(stop):
        run_replica_exchange_2d(pdb=_PDB, temperatures=[0.5, 0.6], output_dir=tmp_path,
                                rotamer_library="dunbrack2002")
    assert seen["rex"]["rotamer_library"] == "dunbrack2002"


def test_the_system_a_run_builds_holds_the_selected_library(tmp_path, monkeypatch) -> None:
    from pymcpu.sampling.engine_session import EngineSession
    from pymcpu.sampling.folding import FoldingRunner
    from pymcpu.sampling.replica_exchange import ReplicaExchange

    monkeypatch.chdir(tmp_path)
    default = FoldingRunner(_PDB, output_dir=str(tmp_path / "d"), verbose=False)
    assert _fingerprint(default.system).startswith("dunbrack2002_sym md5=")
    runner = FoldingRunner(_PDB, output_dir=str(tmp_path / "f"), rotamer_library="dunbrack2002",
                           verbose=False)
    assert _fingerprint(runner.system).startswith("dunbrack2002 md5=")
    assert not runner.system.get_rotamer_library().has_truncated_tables()
    rex = ReplicaExchange(_PDB, temperatures=[0.5, 0.6], n_targets=[0.0], k_bias=0.0,
                          output_prefix="rex", output_dir=tmp_path / "r", seed=0,
                          rotamer_library="dunbrack2002")
    assert _fingerprint(rex.system).startswith("dunbrack2002 md5=")
    for replica in rex.replicas:  # each replica's engine runs that System
        assert _fingerprint(replica.simulation.context.get_system()).startswith("dunbrack2002 md5=")
    session = EngineSession(EngineSpec(pdb=_PDB, rotamer_library="dunbrack2002"))
    assert _fingerprint(session._ensure_sim().system).startswith("dunbrack2002 md5=")
    session = EngineSession(EngineSpec(pdb=_PDB))
    assert _fingerprint(session._ensure_sim().system).startswith("dunbrack2002_sym md5=")


def test_checkpoint_library_check() -> None:
    sym = "dunbrack2002_sym md5=0123 builder=1"
    legacy = "dunbrack2002 md5=0123"
    assert checkpoint_rotamer_library_error(sym, sym, "rotamer_library") is None
    assert checkpoint_rotamer_library_error(legacy, legacy, "rotamer_library") is None
    # Checkpoints from before the field: written by a dunbrack2002 run.
    assert checkpoint_rotamer_library_error("", "dunbrack2002 md5=4567", "rotamer_library") is None
    assert checkpoint_rotamer_library_error(None, legacy, "rotamer_library") is None
    err = checkpoint_rotamer_library_error("", sym, "rotamer_library", "x.chk")
    assert err.startswith("checkpoint x.chk was written with rotamer library 'dunbrack2002', but "
                          f"this run uses {sym!r}; resume it with rotamer_library: dunbrack2002")
    err = checkpoint_rotamer_library_error(sym, legacy, "rotamer_library")
    assert "the checkpoint was written with rotamer library" in err
    assert "resume it with rotamer_library: dunbrack2002_sym" in err
    # Same name, other data or builder: still refused.
    assert checkpoint_rotamer_library_error(legacy, "dunbrack2002 md5=4567", "rotamer_library")
    assert checkpoint_rotamer_library_error(sym, "dunbrack2002_sym md5=0123 builder=2",
                                            "rotamer_library")
    # The continuous sidechain move uses no library.
    assert checkpoint_rotamer_library_error(legacy, sym, "continuous") is None


def _folding(root: Path, library: str | None = None, *, resume: bool = False):
    from pymcpu.sampling.folding import FoldingRunner

    return FoldingRunner(
        _PDB, output_dir=str(root / "out"), checkpoint_dir=str(root / "ck"), seed=1,
        report_interval=5, steps_per_cycle=5, checkpoint_interval=1, resume=resume,
        verbose=False, **({"rotamer_library": library} if library else {}),
    )


def _attempted(runner) -> int:
    integ = runner.simulation.integrator
    return integ.get_bb_attempted() + integ.get_kic_attempted() + integ.get_sc_attempted()


def test_folding_records_the_library_and_refuses_another(tmp_path: Path) -> None:
    _folding(tmp_path).run(n_cycles=2)
    saved = load_checkpoint(tmp_path / "ck" / "last.chk")["rotamer_library"]
    assert saved.startswith("dunbrack2002_sym md5=")
    other = _folding(tmp_path, "dunbrack2002", resume=True)
    with pytest.raises(ValueError, match="resume it with rotamer_library: dunbrack2002_sym"):
        other.run(n_cycles=4)
    assert _attempted(other) == 0
    same = _folding(tmp_path, resume=True)
    same.run(n_cycles=4)
    assert _attempted(same) == 4 * 5  # the counters continue from the checkpoint


def test_an_old_folding_checkpoint_resumes_only_with_dunbrack2002(tmp_path: Path) -> None:
    _folding(tmp_path, "dunbrack2002").run(n_cycles=2)
    for chk in (tmp_path / "ck").glob("*.chk"):
        state = load_checkpoint(chk)
        assert state.pop("rotamer_library").startswith("dunbrack2002 md5=")  # saved...
        save_checkpoint(state, chk.parent, filename=chk.name)  # ...and now removed
    default = _folding(tmp_path, resume=True)
    with pytest.raises(ValueError, match="written with rotamer library 'dunbrack2002'"):
        default.run(n_cycles=4)
    legacy = _folding(tmp_path, "dunbrack2002", resume=True)
    legacy.run(n_cycles=4)
    assert _attempted(legacy) == 4 * 5


def test_serial_replica_exchange_refuses_another_library(tmp_path: Path, monkeypatch) -> None:
    from pymcpu.sampling.replica_exchange import ReplicaExchange

    monkeypatch.chdir(tmp_path)

    def rex(**kwargs):
        return ReplicaExchange(
            _PDB, temperatures=[0.5, 0.6], n_targets=[0.0], k_bias=0.0, log_interval=5,
            output_prefix="rex", output_dir=tmp_path / "out", seed=3,
            checkpoint_dir=tmp_path / "ck", checkpoint_interval=1, **kwargs,
        )

    rex().run(1, 5, verbose=False, checkpoint_dir=tmp_path / "ck", checkpoint_interval=1)
    saved = load_checkpoint(tmp_path / "ck" / "last.chk")["rotamer_library"]
    assert saved.startswith("dunbrack2002_sym md5=")
    with pytest.raises(ValueError, match="resume it with rotamer_library: dunbrack2002_sym"):
        rex(rotamer_library="dunbrack2002").run(
            2, 5, verbose=False, checkpoint_dir=tmp_path / "ck", checkpoint_interval=1,
            resume=tmp_path / "ck" / "last.chk")
