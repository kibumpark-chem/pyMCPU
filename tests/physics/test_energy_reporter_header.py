"""The energy CSV header comes from the registered terms and the moves in use.

It is written when the first run() starts. In append mode an existing header
must match, and any mismatch raises before a single move is made -- the
failure the old fixed header allowed was columns silently shifting under a
resumed run.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

md = pytest.importorskip("mdtraj")

import pymcpu as mc  # noqa: E402
from pymcpu import mcpu_core  # noqa: E402
from pymcpu.runners import default_example_pdb  # noqa: E402

MCPU_HEADER = (
    "step,total,mu,backbone_torsion,sidechain_torsion,hydrogen_bond,aromatic,"
    "pivot_accepted,pivot_attempted,rama_pivot_accepted,rama_pivot_attempted,"
    "kic_accepted,kic_attempted,rotamer_accepted,rotamer_attempted,walker_id"
)


@pytest.fixture(scope="module")
def heavy():
    traj = md.load(str(default_example_pdb()))
    return traj.atom_slice(traj.topology.select("not element H"))


def _sim(heavy):
    ff = mc.MCPUForceField(heavy)
    system = ff.create_system(heavy.topology)
    integrator = mc.Integrator(temperature=0.6, step_size_rad=0.1)
    integrator.set_seed(5)
    sim = mc.Simulation(heavy.topology, system, integrator)
    sim.context.set_positions((ff.coords[0] * 10.0).T.astype(np.float32))
    return sim, system, integrator


def _lines(path: Path) -> list[str]:
    return path.read_text().splitlines()


def test_mcpu_header(tmp_path: Path, heavy) -> None:
    sim, _, _ = _sim(heavy)
    path = tmp_path / "e.csv"
    sim.add_energy_reporter(str(path), interval=10)
    assert path.read_text() == ""  # nothing is known about the columns yet
    sim.step(20)
    lines = _lines(path)
    assert lines[0] == MCPU_HEADER
    assert len(lines) == 1 + 3  # steps 0, 10, 20
    assert all(len(row.split(",")) == len(MCPU_HEADER.split(",")) for row in lines[1:])


def test_append_to_matching_file_continues_without_a_second_header(tmp_path: Path, heavy) -> None:
    path = tmp_path / "e.csv"
    sim, _, _ = _sim(heavy)
    sim.add_energy_reporter(str(path), interval=10)
    sim.step(10)

    sim2, _, _ = _sim(heavy)
    sim2.add_reporter(mc.EnergyReporter(str(path), 10, True))
    sim2.step(10)
    lines = _lines(path)
    assert lines.count(MCPU_HEADER) == 1
    assert len(lines) == 1 + 2 + 2


def test_append_to_missing_or_empty_file_writes_the_header(tmp_path: Path, heavy) -> None:
    for path in [tmp_path / "missing.csv", tmp_path / "empty.csv"]:
        if path.name == "empty.csv":
            path.write_text("")
        sim, _, _ = _sim(heavy)
        sim.add_reporter(mc.EnergyReporter(str(path), 10, True))
        sim.step(10)
        assert _lines(path)[0] == MCPU_HEADER


@pytest.mark.parametrize(
    "existing",
    [
        "step,total,mu,walker_id\n0,1.0,1.0,-1\n",  # different columns
        MCPU_HEADER,  # right text but no line ending: an incomplete write
    ],
)
def test_mismatched_append_raises_before_any_move(tmp_path: Path, heavy, existing: str) -> None:
    path = tmp_path / "e.csv"
    path.write_text(existing)
    sim, _, integrator = _sim(heavy)
    sim.add_reporter(mc.EnergyReporter(str(path), 10, True))
    with pytest.raises(RuntimeError, match="already has a different header"):
        sim.step(10)
    assert integrator.get_bb_attempted() + integrator.get_kic_attempted() + integrator.get_sc_attempted() == 0
    assert path.read_text() == existing


# The fixed header every energy file had before per-term columns.
LEGACY_HEADER = (
    "Step,Total,Mu,BackboneTorsion,SidechainTorsion,HydrogenBond,Aromatic,"
    "NativeContactsBias,PivotAccepted,PivotAttempted,SidechainAccepted,"
    "SidechainAttempted,KicAccepted,KicAttempted,WalkerId"
)


def test_a_legacy_header_gets_its_own_error(tmp_path: Path, heavy) -> None:
    """No setting can make a run match the old fixed columns, so the error says
    the file is from an earlier version instead of blaming the run's settings."""
    path = tmp_path / "e.csv"
    existing = LEGACY_HEADER + "\n0,-14.7,-3.4,-3.2,-7.8,-0.1,0.0,0.0,0,0,0,0,0,0,-1\n"
    path.write_text(existing)
    sim, _, integrator = _sim(heavy)
    sim.add_reporter(mc.EnergyReporter(str(path), 10, True))
    with pytest.raises(RuntimeError, match="written by an earlier pyMCPU"):
        sim.step(10)
    assert integrator.get_bb_attempted() + integrator.get_kic_attempted() + integrator.get_sc_attempted() == 0
    assert path.read_text() == existing


def test_resuming_a_run_with_a_legacy_csv_raises_before_any_move(
    tmp_path: Path, chignolin_pdb_path: str
) -> None:
    """End to end: a folding run checkpoints, its CSV is swapped for one in the
    old format (as a run started before this change would have), and the
    resume stops with the specific error."""
    from pymcpu.checkpointing import load_checkpoint
    from pymcpu.sampling.folding import FoldingRunner

    def runner(resume: bool) -> FoldingRunner:
        return FoldingRunner(
            chignolin_pdb_path, output_dir=str(tmp_path / "out"),
            checkpoint_dir=str(tmp_path / "ck"), seed=3, report_interval=5,
            steps_per_cycle=5, checkpoint_interval=1, resume=resume, verbose=False,
        )

    runner(resume=False).run(n_cycles=2)
    csv_path = tmp_path / "out" / "folding_data.csv"
    rows = _lines(csv_path)[1:]
    csv_path.write_text("\n".join([LEGACY_HEADER, *rows]) + "\n")

    resumed = runner(resume=True)
    with pytest.raises(RuntimeError, match="written by an earlier pyMCPU"):
        resumed.run(n_cycles=4)
    # The resume restored the checkpoint's counters; no move was made after that.
    saved = load_checkpoint(tmp_path / "ck" / "last.chk")["integrator_move_counters"][0]
    assert resumed.simulation.integrator.get_move_counters() == saved


def test_a_term_added_after_the_header_raises(tmp_path: Path, heavy) -> None:
    sim, system, _ = _sim(heavy)
    sim.add_energy_reporter(str(tmp_path / "e.csv"), interval=10)
    sim.step(10)
    extra = mcpu_core.NativeContactsBiasPotential([0], [5], 5.0)
    extra.set_energy_group(6)
    extra.set_name("native_contacts_bias")
    system.add_potential(extra)
    with pytest.raises(RuntimeError, match="energy terms changed"):
        sim.step(10)


def test_a_move_kind_enabled_after_the_header_raises(tmp_path: Path, heavy) -> None:
    sim, _, integrator = _sim(heavy)
    sim.add_energy_reporter(str(tmp_path / "e.csv"), interval=10)
    sim.step(10)
    integrator.set_sidechain_move_mode("continuous")  # 'sidechain' has no column
    with pytest.raises(RuntimeError, match="no columns for the 'sidechain' move"):
        sim.step(10)


def test_remd_replicas_share_one_header_under_a_rama_schedule(
    tmp_path: Path, chignolin_pdb_path: str, monkeypatch
) -> None:
    """A schedule gives each replica its own rama probability (0 and 1 here);
    the columns depend on slot weights only, so every file matches."""
    from pymcpu.sampling.replica_exchange import ReplicaExchange

    monkeypatch.chdir(tmp_path)
    out = tmp_path / "out"
    rex = ReplicaExchange(
        str(chignolin_pdb_path),
        temperatures=[0.4, 0.7],
        n_targets=[0.0],
        k_bias=0.0,
        log_interval=5,
        output_prefix="rex",
        output_dir=out,
        seed=0,
        pivot_rama_schedule={"t_low": 0.4, "t_high": 0.7, "p_min": 0.0, "p_max": 1.0},
    )
    rex.run(1, 10, verbose=False, checkpoint_dir=None)
    headers = {_lines(f)[0] for f in out.glob("rex_*_data.csv")}
    # REMD attaches the umbrella bias, so its term gets a column too.
    assert headers == {MCPU_HEADER.replace(",aromatic,", ",aromatic,native_contacts_bias,")}
