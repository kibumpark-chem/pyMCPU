"""Coordinates for the wrong number of atoms are rejected, not silently resized.

Loading coordinates used to resize the engine's state to whatever it was
given. A checkpoint or restart file written with a different atom layout
then loaded with every atom after the first difference shifted, and only
replica exchange noticed, by accident, through a steric clash. Every coordinate setter now
checks the count, and each restore path names the checkpoint in its error.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

md = pytest.importorskip("mdtraj")

import pymcpu as mc  # noqa: E402
from pymcpu.checkpointing import (  # noqa: E402
    CheckpointConfig,
    checkpoint_layout_error,
    load_checkpoint,
    save_checkpoint,
)
from pymcpu.runners import default_example_pdb  # noqa: E402

LAYOUT_ERROR = "different atom layout"


@pytest.fixture(scope="module")
def heavy():
    traj = md.load(str(default_example_pdb()))
    return traj.atom_slice(traj.topology.select("not element H"))


@pytest.fixture()
def sim(heavy):
    ff = mc.MCPUForceField(heavy)
    system = ff.create_system(heavy.topology)
    simulation = mc.Simulation(heavy.topology, system, mc.Integrator(0.6))
    coords = (ff.coords[0] * 10.0).T.astype(np.float32)
    simulation.context.set_positions(coords)
    return simulation, coords


def _with_extra_column(coords):
    arr = np.asarray(coords)
    return np.hstack([arr, arr[:, :1]])


# --------------------------------------------------------------------- C++


@pytest.mark.parametrize("setter", ["set_positions", "context.coords", "state.coords"])
@pytest.mark.parametrize("change", [-1, +1])
def test_every_setter_rejects_a_wrong_atom_count(sim, setter: str, change: int) -> None:
    simulation, coords = sim
    wrong = coords[:, :change] if change < 0 else _with_extra_column(coords).astype(np.float32)
    with pytest.raises(ValueError, match="atoms, but th"):
        if setter == "set_positions":
            simulation.context.set_positions(wrong)
        elif setter == "context.coords":
            simulation.context.coords = wrong
        else:
            simulation.context.get_state().coords = wrong
    assert np.array_equal(simulation.context.coords, coords)  # nothing was loaded


def test_the_right_atom_count_still_loads(sim) -> None:
    simulation, coords = sim
    moved = coords + np.float32(0.5)
    simulation.context.set_positions(moved)
    assert np.allclose(simulation.context.coords, moved)


def test_debug_force_pivot_reports_the_moved_atoms(sim) -> None:
    simulation, _ = sim
    integrator = simulation.integrator
    assert integrator.debug_force_pivot(simulation.context, 5, True)
    moved = list(integrator.last_moved_indices())
    assert moved and len(set(moved)) == len(moved)
    assert np.isfinite(integrator.last_delta_energy())


# ----------------------------------------------------------- restore paths


def test_checkpoint_layout_error_names_the_counts() -> None:
    assert checkpoint_layout_error([np.zeros((3, 80))], 80) is None
    assert checkpoint_layout_error([np.zeros((80, 3))], 80) is None  # either orientation
    message = checkpoint_layout_error([np.zeros((3, 80))], 77, "run/last.chk")
    assert "run/last.chk" in message and "80 atoms" in message and "has 77" in message


def _grow_checkpoints(directory: Path) -> None:
    """Rewrite every checkpoint as if it came from a layout with one more atom."""
    for chk in directory.glob("*.chk"):
        state = load_checkpoint(chk)
        state["replica_coords"] = [_with_extra_column(c) for c in state["replica_coords"]]
        save_checkpoint(state, chk.parent, filename=chk.name)


def test_folding_resume_rejects_an_old_layout(chignolin_pdb_path: str, tmp_path: Path) -> None:
    from pymcpu.sampling.folding import FoldingRunner

    def runner(resume: bool) -> FoldingRunner:
        return FoldingRunner(
            chignolin_pdb_path, output_dir=str(tmp_path / "out"),
            checkpoint_dir=str(tmp_path / "ck"), seed=1, report_interval=5,
            steps_per_cycle=5, checkpoint_interval=1, resume=resume, verbose=False,
        )

    runner(resume=False).run(n_cycles=2)
    _grow_checkpoints(tmp_path / "ck")
    resumed = runner(resume=True)
    with pytest.raises(ValueError, match=LAYOUT_ERROR):
        resumed.run(n_cycles=4)
    integ = resumed.simulation.integrator
    assert integ.get_bb_attempted() + integ.get_kic_attempted() + integ.get_sc_attempted() == 0


def test_serial_remd_resume_rejects_an_old_layout(
    chignolin_pdb_path: str, tmp_path: Path, monkeypatch
) -> None:
    from pymcpu.sampling.replica_exchange import ReplicaExchange

    monkeypatch.chdir(tmp_path)

    def rex() -> ReplicaExchange:
        return ReplicaExchange(
            chignolin_pdb_path, temperatures=[0.5, 0.6], n_targets=[0.0], k_bias=0.0,
            log_interval=5, output_prefix="rex", output_dir=tmp_path / "out", seed=3,
            checkpoint_dir=tmp_path / "ck", checkpoint_interval=1,
        )

    rex().run(1, 5, verbose=False, checkpoint_dir=tmp_path / "ck", checkpoint_interval=1)
    _grow_checkpoints(tmp_path / "ck")
    with pytest.raises(ValueError, match=LAYOUT_ERROR):
        rex().run(2, 5, verbose=False, checkpoint_dir=tmp_path / "ck",
                  checkpoint_interval=1, resume=tmp_path / "ck" / "last.chk")


class _OneRankComm:
    def __init__(self, rank: int, broadcast_value=None) -> None:
        self.rank = rank
        self.broadcast_value = broadcast_value

    def Get_rank(self) -> int:  # noqa: N802 (mpi4py spelling)
        return self.rank

    def bcast(self, obj, root: int = 0):
        return obj if self.rank == root else self.broadcast_value


def _bare_mpi_rex(system, checkpoint_dir: Path):
    from pymcpu.sampling.mpi_replica_exchange import MPIReplicaExchange

    rex = object.__new__(MPIReplicaExchange)  # only what _mpi_load_checkpoint reads
    rex.checkpoint_config = CheckpointConfig(checkpoint_dir=str(checkpoint_dir))
    rex.system = system
    rex.forcefield_name = "mcpu08"
    rex.local_replica_indices = []
    rex.replicas = {}
    return rex


def test_mpi_resume_decides_on_rank_0_and_every_rank_raises(sim, tmp_path: Path) -> None:
    simulation, coords = sim
    system = simulation.system
    old = {"replica_coords": [_with_extra_column(coords)], "walker_at_state": [0], "cycle": 1}
    save_checkpoint(old, tmp_path, filename="last.chk")

    rex = _bare_mpi_rex(system, tmp_path)
    with pytest.raises(ValueError, match=LAYOUT_ERROR):
        rex._mpi_load_checkpoint(_OneRankComm(rank=0))

    # A non-root rank gets rank 0's verdict from the broadcast and raises too,
    # instead of carrying on into the next collective.
    other = _OneRankComm(rank=1, broadcast_value=(None, "checkpoint x: different atom layout"))
    with pytest.raises(ValueError, match=LAYOUT_ERROR):
        _bare_mpi_rex(system, tmp_path)._mpi_load_checkpoint(other)


def test_mpi_resume_of_a_matching_checkpoint_gets_past_the_check(sim, tmp_path: Path) -> None:
    simulation, coords = sim
    save_checkpoint(
        {"replica_coords": [coords], "walker_at_state": [0], "cycle": 3}, tmp_path, filename="last.chk"
    )
    rex = _bare_mpi_rex(simulation.system, tmp_path)
    state = rex._mpi_load_checkpoint(_OneRankComm(rank=0))
    assert state["cycle"] == 3 and rex.cycle == 3


def test_engine_session_rejects_a_restart_state_of_the_wrong_size(
    engine_spec_factory, tmp_path: Path
) -> None:
    from pymcpu.sampling import EngineSession

    session = EngineSession(engine_spec_factory())
    coords = session.coords_from_auxref(session.spec.pdb)
    npz = tmp_path / "state.npz"
    np.savez(npz, coords=_with_extra_column(coords))
    with pytest.raises(ValueError, match="atoms, but the system has"):
        session.set_coords(session.coords_from_auxref(str(npz)))


def test_resume_paths_reject_before_touching_the_system(sim) -> None:
    """The shared check runs on the stored coordinates alone."""
    simulation, coords = sim
    n = simulation.system.get_num_atoms()
    assert checkpoint_layout_error([coords], n) is None
    assert LAYOUT_ERROR in checkpoint_layout_error([_with_extra_column(coords)], n)
