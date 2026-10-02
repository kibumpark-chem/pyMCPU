"""KORPForceField keeps chain IDs and residue numbers through its backbone slice.

KORP keys sequence separation on (chain, residue number): pairs on different
chains are non-bonded, and the CA-CA steric guard exempts only near
neighbours on the same chain. The force field slices the input down to its
backbone with ``atom_slice``, and mdtraj's ``Topology.subset`` drops every
chain ID and replaces a residue number of 0 with the residue's index. So
every chain read as ``' '``: two chains were numbering-checked and scored as
one, and the guard excused cross-chain contacts as bonded neighbours.

``KORPForceField.__init__`` loads the 316 MiB map first, so these tests run
the construction steps that come after it directly; they need no map.
"""

from __future__ import annotations

from pathlib import Path

import mdtraj as md
import numpy as np
import pytest

from pymcpu.forcefields.korp import KORPForceField
from pymcpu.runners import default_example_pdb


def _two_chains(tmp_path: Path, b_first: int, shift_z: float = 12.0) -> md.Trajectory:
    """1UAO as chain A, and as chain B a copy turned 1 rad about z and moved
    along z, numbered from ``b_first``. The turn keeps every inter-chain pair
    off KORP's bin edges, where a pure translation would put them."""
    atoms = [ln for ln in default_example_pdb().read_text().splitlines() if ln.startswith("ATOM")]
    xyz = np.array([[float(ln[30:38]), float(ln[38:46]), float(ln[46:54])] for ln in atoms])
    centre = xyz.mean(axis=0)
    c, s = np.cos(1.0), np.sin(1.0)
    turn = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    lines = atoms + ["TER"]
    for ln, pos in zip(atoms, xyz):
        x, y, z = turn @ (pos - centre) + centre + (0.0, 0.0, shift_z)
        seq = int(ln[22:26]) - 1 + b_first
        lines.append(ln[:21] + "B" + f"{seq:4d}" + ln[26:30] + f"{x:8.3f}{y:8.3f}{z:8.3f}" + ln[54:])
    path = tmp_path / "two_chains.pdb"
    path.write_text("\n".join(lines + ["TER", "END"]) + "\n")
    return md.load(str(path))


def _layout(traj: md.Trajectory) -> KORPForceField:
    """KORPForceField's construction, minus loading the map."""
    ff = KORPForceField.__new__(KORPForceField)
    ff.min_separation, ff.min_distance = 3, 3.2
    sliced = KORPForceField._slice_backbone(traj)
    ff._collect_residues(sliced)
    ff._validate_residue_numbering()
    ff._build_layout(sliced)
    ff._build_output_topology(sliced)
    ff._check_initial_sterics()
    return ff


def test_each_chain_keeps_its_id(tmp_path) -> None:
    ff = _layout(_two_chains(tmp_path, b_first=11))
    assert ff.chain_ids == ["A"] * 10 + ["B"] * 10
    assert [chain.chain_id for chain in ff.output_topology.chains] == ["A", "B"]


def test_two_chains_may_share_residue_numbers(tmp_path) -> None:
    """Homo-oligomers number each chain from 1; that used to read as one
    chain going backwards and was refused."""
    ff = _layout(_two_chains(tmp_path, b_first=1))
    assert ff.res_seq == list(range(1, 11)) * 2


def test_a_residue_numbered_zero_keeps_its_number(tmp_path) -> None:
    ff = _layout(_two_chains(tmp_path, b_first=0))
    assert ff.res_seq[10:] == list(range(10))


def test_a_cross_chain_overlap_is_not_excused(tmp_path) -> None:
    """CA(B11) 2 A from CA(A10): adjacent numbers, but different chains.
    Placed on the far side of A10 from the rest of chain A, so A10 is the
    only CA it comes near."""
    traj = _two_chains(tmp_path, b_first=11)
    top = traj.topology
    ca_a = top.select("chainid 0 and name CA")
    ca_a10 = top.select("chainid 0 and resSeq 10 and name CA")[0]
    b11 = top.select("chainid 1 and resSeq 11")
    ca_b11 = top.select("chainid 1 and resSeq 11 and name CA")[0]
    outward = traj.xyz[0, ca_a10] - traj.xyz[0, ca_a].mean(axis=0)
    target = traj.xyz[0, ca_a10] + 0.2 * outward / np.linalg.norm(outward)  # nm
    traj.xyz[0, b11] += target - traj.xyz[0, ca_b11]
    with pytest.raises(ValueError, match="already violates"):
        _layout(traj)


def test_a_multi_chain_input_warns_that_the_moves_join_the_chains(tmp_path, caplog) -> None:
    with caplog.at_level("WARNING", logger="pymcpu.forcefields.korp"):
        _layout(_two_chains(tmp_path, b_first=11))
    assert any("one bonded backbone" in record.getMessage() for record in caplog.records)
