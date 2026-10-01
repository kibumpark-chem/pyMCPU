"""MCPUForceField's engine layout: one slot per atom, glycine included.

The engine stores atoms as ``[N,CA,C]`` per residue, then every O, then every
sidechain, then any explicit amide H. Glycine has no sidechain atoms, so its
block has ``sc_start == -1`` and ``sc_count == 0``; its CA lives only in the
backbone segment.

The pivot checks derive the atoms a pivot must move from the topology's bond
graph -- cut the rotated bond and take the moving side -- rather than from the
engine's own index ranges, so a range that picks up or drops an atom is
caught. Internal consistency only: no legacy MCPU output is read.
"""

from __future__ import annotations

from collections import defaultdict

import mdtraj as md
import numpy as np
import pytest

import pymcpu as mc
from pymcpu.forcefields.mcpu import MCPUForceField
from pymcpu.runners import default_example_pdb
from tests.fixtures.context_builders import resolve_test_pdb

BACKBONE = ("N", "CA", "C", "O", "OXT", "OCT")


def _heavy(path) -> md.Trajectory:
    traj = md.load(str(path))
    return traj.atom_slice(traj.topology.select("not element H"))


@pytest.fixture(scope="module")
def chignolin() -> md.Trajectory:
    return _heavy(default_example_pdb())


@pytest.fixture(scope="module")
def actin() -> md.Trajectory:
    return _heavy(resolve_test_pdb())


def _simulation(ff: MCPUForceField, traj: md.Trajectory) -> mc.Simulation:
    sim = mc.Simulation(traj.topology, ff.create_system(traj.topology), mc.Integrator(0.6))
    sim.context.set_positions((ff.coords[0] * 10.0).T.astype(np.float32))
    sim.context.calculate_total_energy(-1)
    return sim


def _glycines(ff: MCPUForceField) -> list[int]:
    names = {a.residue_index: a.residue_name for a in ff.ordered_atom_list}
    return [r for r in range(ff.n_res) if names[r] == "GLY"]


# ------------------------------------------------------------------- layout


@pytest.mark.parametrize("which", ["chignolin", "actin"])
def test_one_slot_per_heavy_atom(which: str, request) -> None:
    traj = request.getfixturevalue(which)
    ff = MCPUForceField(traj)
    assert ff.n_atoms == traj.n_atoms
    assert sorted(ff.inverse_mapping) == list(range(traj.n_atoms))
    assert len(_glycines(ff)) > 0


def test_explicit_amide_h_adds_only_the_h_slots(chignolin: md.Trajectory) -> None:
    ff = MCPUForceField(chignolin, virtual_amide_h=False)
    assert ff.total_h_atoms > 0
    assert ff.n_atoms == chignolin.n_atoms + ff.total_h_atoms
    mapping = ff.inverse_mapping
    assert mapping.count(-1) == ff.total_h_atoms
    assert sorted(i for i in mapping if i >= 0) == list(range(chignolin.n_atoms))


@pytest.mark.parametrize("virtual_amide_h", [True, False])
def test_glycine_has_no_sidechain_block(actin: md.Trajectory, virtual_amide_h: bool) -> None:
    ff = MCPUForceField(actin, virtual_amide_h=virtual_amide_h)
    sc_end = ff.total_bb_atoms + ff.total_o_atoms + ff.total_sc_atoms
    first_sc = list(ff.downstream.first_sc_of_residue)
    atoms = ff.ordered_atom_list
    for r in _glycines(ff):
        block = ff.blocks[r]
        assert block.sc_start == -1 and block.sc_count == 0
        cas = [i for i, a in enumerate(atoms) if a.residue_index == r and a.name == "CA"]
        assert cas == [block.bb_start + 1]
        # Where the sidechains after this residue begin, which is where a
        # pivot here starts rotating sidechains.
        later = [ff.blocks[s].sc_start for s in range(r + 1, ff.n_res) if ff.blocks[s].sc_start >= 0]
        assert first_sc[r] == (later[0] if later else sc_end)


def test_every_other_residue_counts_its_own_sidechain(actin: md.Trajectory) -> None:
    ff = MCPUForceField(actin)
    n_sc = defaultdict(int)
    for atom in actin.topology.atoms:
        if atom.name not in BACKBONE:
            n_sc[atom.residue.index] += 1
    for r in range(ff.n_res):
        if r in _glycines(ff):
            continue
        assert ff.blocks[r].sc_count == n_sc[r], r


def test_sidechain_moves_at_a_glycine_are_still_rejected(chignolin: md.Trajectory) -> None:
    ff = MCPUForceField(chignolin)
    sim = _simulation(ff, chignolin)
    for r in _glycines(ff):
        assert not sim.integrator.debug_force_sc(sim.context, r)
        assert not sim.integrator.debug_force_rotamer(sim.context, r)


# -------------------------------------------------------------------- pivot


def _bond_graph(ff: MCPUForceField, traj: md.Trajectory) -> dict[int, set[int]]:
    """Bonds between engine slots, from the topology plus each explicit H-N."""
    engine_of = {a.original_index: i for i, a in enumerate(ff.ordered_atom_list) if a.original_index >= 0}
    graph: dict[int, set[int]] = defaultdict(set)
    for a, b in traj.topology.bonds:
        i, j = engine_of[a.index], engine_of[b.index]
        graph[i].add(j)
        graph[j].add(i)
    for i, atom in enumerate(ff.ordered_atom_list):
        if atom.name == "H":
            n = ff.blocks[atom.residue_index].bb_start
            graph[i].add(n)
            graph[n].add(i)
    assert len(graph) == ff.n_atoms
    return graph


def _side(graph: dict[int, set[int]], start: int, cut: tuple[int, int]) -> set[int]:
    """Atoms reachable from ``start`` without crossing the ``cut`` bond."""
    seen, stack = {start}, [start]
    while stack:
        i = stack.pop()
        for j in graph[i]:
            if {i, j} == set(cut) or j in seen:
                continue
            seen.add(j)
            stack.append(j)
    return seen


def _pivot_cases(n_res: int, residues):
    """(residue, is_phi, fixed, rotates_n_term) for both directions where the
    engine allows them. The direction follows the engine's rule: the shorter
    side, unless a fixed residue rules one side out."""
    for r in residues:
        if not 1 <= r <= n_res - 2:
            continue
        for fixed in ([], [1], [n_res - 1]):
            c_ok = not any(r + 1 <= f < n_res for f in fixed)
            n_ok = not any(1 <= f < r for f in fixed)
            if not (c_ok or n_ok):
                continue
            n_term = (r < n_res // 2) if (c_ok and n_ok) else not c_ok
            for is_phi in (True, False):
                yield r, is_phi, fixed, n_term


def _check_pivots(ff: MCPUForceField, traj: md.Trajectory, residues) -> set[tuple[int, bool, bool]]:
    sim = _simulation(ff, traj)
    integ = sim.integrator
    graph = _bond_graph(ff, traj)
    names = {a.residue_index: a.residue_name for a in ff.ordered_atom_list}
    covered = set()
    for r, is_phi, fixed, n_term in _pivot_cases(ff.n_res, residues):
        if is_phi and names[r] == "PRO":
            continue
        block = ff.blocks[r]
        n, ca, c = block.bb_start, block.bb_start + 1, block.c_start
        axis = (n, ca) if is_phi else (ca, c)
        moving_end = axis[0] if n_term else axis[1]
        expected = _side(graph, moving_end, axis)
        assert axis[1 - axis.index(moving_end)] not in expected, "the cut must split the chain"

        integ.clear_fixed_residues()
        if fixed:
            integ.set_fixed_residues(fixed, ff.n_res)
        assert integ.debug_force_pivot(sim.context, r, is_phi)
        moved = set(integ.last_moved_indices())
        # Atoms on the rotation axis do not move whether or not the engine
        # lists them, so compare everything else.
        assert moved - set(axis) == expected - set(axis), (
            f"{names[r]}{r} {'phi' if is_phi else 'psi'} "
            f"{'N-term' if n_term else 'C-term'}: extra {sorted(moved - expected - set(axis))}, "
            f"missing {sorted(expected - moved - set(axis))}"
        )
        assert np.isfinite(integ.last_delta_energy())
        covered.add((r, is_phi, n_term))
    integ.clear_fixed_residues()
    return covered


@pytest.mark.parametrize("virtual_amide_h", [True, False])
def test_pivots_move_exactly_one_side_of_the_bond(chignolin: md.Trajectory, virtual_amide_h: bool) -> None:
    ff = MCPUForceField(chignolin, virtual_amide_h=virtual_amide_h)
    covered = _check_pivots(ff, chignolin, range(ff.n_res))
    for r in _glycines(ff):
        if 1 <= r <= ff.n_res - 2:
            assert {(r, p, d) for p in (True, False) for d in (True, False)} <= covered


def test_glycine_pivots_in_actin(actin: md.Trajectory) -> None:
    """Every glycine context in a large protein, including glycine next to
    proline and to another glycine."""
    ff = MCPUForceField(actin)
    glycines = _glycines(ff)
    covered = _check_pivots(ff, actin, glycines)
    assert {r for r, _, _ in covered} == {r for r in glycines if 1 <= r <= ff.n_res - 2}
