"""Start overlaps: exempt for the whole run, or penalised (``clash_start_exemption``).

A pair that can clash and is already inside its hard core in the structure the
force field is built from (a start overlap) is exempt from the clash check for
the whole run by default, as every move that re-decided it would be rejected.
With ``clash_start_exemption=False`` it is penalised instead: it scores
``MuPotential.start_overlap_penalty`` (1000, raw Mu units) on top of its
contact energy while it stays inside, so a move that takes it out lowers the
energy by that much, one that puts it back raises it, and a cold run removes
it if a local move can reach it. Every other pair keeps the hard core.

The structures are chignolin with TYR1's side chain turned about CA-CB. At a
hard core of 0.85 radius sums, with pairs up to three bonds apart left out,
the deposited structure has no start overlap; turned by 200 degrees it has one
(TYR1 CD2 with TRP8 CZ3, at 0.78 radius sums), and by 20 degrees three (TYR1
OH with GLY6 O, and TYR1 CD1 with TRP8 CD2 and CE3), none of them inside 0.75.
"""

from __future__ import annotations

import logging
from pathlib import Path

import mdtraj as md
import numpy as np
import pytest

from pymcpu import mcpu_core
from pymcpu.checkpointing import CheckpointConfig
from pymcpu.forcefields.mcpu import MCPUForceField
from pymcpu.runners import default_example_pdb
from pymcpu.sampling.folding import FoldingRunner
from pymcpu.simulation import check_state_clash, start_overlaps
from tests.fixtures.context_builders import ATOL

HARD = {"clash_alpha": 0.85, "clash_bond_exclusion": 3}
PENALISED = {**HARD, "clash_start_exemption": False}
PENALTY = 1000.0  # raw Mu units
MU_WEIGHT = 0.4


@pytest.fixture(scope="module")
def heavy() -> md.Trajectory:
    traj = md.load(str(default_example_pdb()))
    return traj.atom_slice(traj.topology.select("not element H"))


def _turned(heavy: md.Trajectory, residue: int, degrees: float) -> md.Trajectory:
    """``heavy`` with ``residue``'s side chain beyond CB turned about CA-CB:
    a change of chi1 alone, so every bond and angle stays as it was."""
    out = heavy[0]
    xyz = out.xyz[0].astype(np.float64)
    atoms = {a.name: a.index for a in out.topology.residue(residue).atoms}
    ca, cb = xyz[atoms["CA"]], xyz[atoms["CB"]]
    axis = (cb - ca) / np.linalg.norm(cb - ca)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    t = np.deg2rad(degrees)
    rot = np.eye(3) + np.sin(t) * k + (1 - np.cos(t)) * k @ k
    moved = [i for name, i in atoms.items() if name not in ("N", "CA", "C", "O", "OXT", "CB")]
    xyz[moved] = (rot @ (xyz[moved] - cb).T).T + cb
    out.xyz = xyz[None].astype(np.float32)
    return out


@pytest.fixture(scope="module")
def one_overlap(heavy: md.Trajectory) -> md.Trajectory:
    return _turned(heavy, 1, 200.0)


@pytest.fixture(scope="module")
def three_overlaps(heavy: md.Trajectory) -> md.Trajectory:
    return _turned(heavy, 1, 20.0)


def _context(structure: md.Trajectory, **options) -> mcpu_core.Context:
    ff = MCPUForceField(structure, **options)
    context = mcpu_core.Context(ff.create_system(structure.topology))
    context.set_positions((ff.coords[0] * 10.0).T.astype(np.float32))
    context.calculate_total_energy(-1)
    return context


def _start_coords(structure: md.Trajectory) -> np.ndarray:
    return (MCPUForceField(structure).coords[0] * 10.0).T.astype(np.float32)


def _mu(context: mcpu_core.Context, weighted: bool = True) -> float:
    return context.energy_breakdown(weighted=weighted)["by_name"]["mu"]


def test_the_default_exempts_them(heavy, one_overlap) -> None:
    assert MCPUForceField(heavy).clash_start_exemption is True
    context = _context(one_overlap, **HARD)
    mu = context.mu_potential
    assert mu.start_overlap_pairs == 1
    assert mu.start_overlaps_penalised is False
    assert start_overlaps(context) == (1, 1)  # still inside, and exempt
    assert not context.has_steric_clash()
    # The deposited structure has none under these options.
    assert start_overlaps(_context(heavy, **HARD)) == (0, 0)


def test_a_penalised_overlap_adds_the_penalty(one_overlap) -> None:
    exempt = _context(one_overlap, **HARD)
    penalised = _context(one_overlap, **PENALISED)
    mu = penalised.mu_potential
    assert mu.start_overlaps_penalised is True
    assert mu.start_overlap_penalty == PENALTY
    assert start_overlaps(penalised) == (1, 1)
    assert not penalised.has_steric_clash()
    assert check_state_clash(penalised, "start structure") is False
    assert _mu(penalised, weighted=False) - _mu(exempt, weighted=False) == pytest.approx(
        PENALTY, abs=1e-3)
    assert penalised.calculate_total_energy(-1) - exempt.calculate_total_energy(-1) == (
        pytest.approx(MU_WEIGHT * PENALTY, abs=1e-3))


def test_the_penalty_follows_the_pair(heavy, one_overlap) -> None:
    """Out of the core the pair scores as under the exemption; back in, it
    pays the penalty again, and is no clash."""
    penalised = _context(one_overlap, **PENALISED)
    exempt = _context(one_overlap, **HARD)
    turned_energy = penalised.calculate_total_energy(-1)
    deposited = _start_coords(heavy)
    for context in (penalised, exempt):
        context.set_positions(deposited)
        context.calculate_total_energy(-1)
        assert not context.has_steric_clash()
    assert start_overlaps(penalised) == (0, 1)
    assert penalised.calculate_total_energy(-1) == pytest.approx(
        exempt.calculate_total_energy(-1), abs=1e-9)
    penalised.set_positions(_start_coords(one_overlap))
    assert penalised.calculate_total_energy(-1) == pytest.approx(turned_energy, abs=1e-9)
    assert not penalised.has_steric_clash()
    assert start_overlaps(penalised) == (1, 1)


def test_moves_agree_with_full_recomputes_while_penalised(three_overlaps) -> None:
    """Every term's delta matches a full recompute on proposals of the
    default move mix (pivots included) from a state that holds three
    penalised overlaps."""
    context = _context(three_overlaps, **PENALISED)
    assert start_overlaps(context) == (3, 3)
    integrator = mcpu_core.Integrator(temperature=300.0, step_size_rad=0.1)
    integrator.set_seed(2026)
    integrator.verify_physics_consistency(context, num_steps=500, atol=ATOL)


def test_a_cold_run_takes_them_out(three_overlaps, tmp_path: Path, caplog) -> None:
    """The relaxation: a cold run of local moves (KIC and side chains, no
    pivots) from a start with three penalised overlaps ends with none, its
    running energy equal to a full recompute, and the same energy as the
    exempt force field gives the final coordinates."""
    pdb = tmp_path / "turned.pdb"
    three_overlaps.save_pdb(str(pdb))
    runner = FoldingRunner(
        str(pdb), temperature=0.1, report_interval=10_000, steps_per_cycle=10_000,
        output_dir=str(tmp_path / "out"), seed=11, move_weights=(0.0, 0.5, 0.5),
        forcefield_options=PENALISED, verbose=False,
        checkpoint_config=CheckpointConfig(checkpoint_dir=None, enabled=False),
    )
    context = runner.simulation.context
    assert start_overlaps(context) == (3, 3)
    with caplog.at_level(logging.INFO, logger="pymcpu.sampling.folding"):
        runner.run(steps=50_000)
    assert start_overlaps(context) == (0, 3)
    assert any("0 of 3 still inside the hard core" in r.getMessage() for r in caplog.records)
    incremental = float(context.get_state().current_energy)
    assert runner.simulation.recompute_energy() == pytest.approx(incremental, abs=1e-6)
    final = np.asarray(context.coords, dtype=np.float64)
    exempt = _context(three_overlaps, **HARD)
    exempt.set_positions(final)
    assert exempt.calculate_total_energy(-1) == pytest.approx(
        float(context.get_state().current_energy), abs=1e-6)
    assert not exempt.has_steric_clash()


def test_more_penalty_than_the_bare_clash_threshold(heavy) -> None:
    """At a hard core of 1.2 radius sums the deposited structure has over a
    hundred start overlaps: their penalties pass half the clash sentinel
    (49999.5), which alone would read as a clash. The threshold rises with
    them, and a real overlap is still caught."""
    context = _context(heavy, clash_alpha=1.2, clash_bond_exclusion=3, clash_start_exemption=False)
    mu = context.mu_potential
    inside, found = start_overlaps(context)
    assert inside == found >= 50
    assert _mu(context, weighted=False) > 0.5 * 99999.0
    assert not context.has_steric_clash()
    integrator = mcpu_core.Integrator(temperature=300.0, step_size_rad=0.1)
    integrator.set_seed(7)
    integrator.verify_physics_consistency(context, num_steps=100, atol=ATOL)
    # TRP8 CD2 put 0.5 A from PRO3 CA (atoms 70 and 10 in engine order).
    coords = np.array(context.coords, dtype=np.float32)
    coords[:, 70] = coords[:, 10] + np.array([0.5, 0.0, 0.0], dtype=np.float32)
    context.set_positions(coords)
    context.calculate_total_energy(-1)
    assert context.has_steric_clash()
    assert mu.start_overlap_pairs == found


def test_the_count_survives_an_atom_reorder(three_overlaps) -> None:
    ff = MCPUForceField(three_overlaps, **PENALISED)
    context = mcpu_core.Context(ff.create_system(three_overlaps.topology))
    context.set_atom_reorder_mode("init_only")
    context.set_positions((ff.coords[0] * 10.0).T.astype(np.float32))
    energy = context.calculate_total_energy(-1)
    assert context.atom_permutation_info()["enabled"]
    assert start_overlaps(context) == (3, 3)
    assert energy == pytest.approx(
        _context(three_overlaps, **PENALISED).calculate_total_energy(-1), abs=1e-6)


@pytest.mark.parametrize("value", ["false", 0, 1, None], ids=repr)
def test_clash_start_exemption_is_checked(heavy: md.Trajectory, value: object) -> None:
    with pytest.raises(ValueError, match="clash_start_exemption must be True or False"):
        MCPUForceField(heavy, clash_start_exemption=value)
