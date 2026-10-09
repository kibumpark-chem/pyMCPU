"""``MCPUForceField`` checks its input structure.

Several chains, or a chain with missing residues, used to build and run
without a message, joined as if bonded; a missing side-chain atom ended in a
bare ``KeyError`` that named only the atom. Each case is built from the
bundled chignolin (one chain, residues 1-10).
"""

from __future__ import annotations

from pathlib import Path

import mdtraj as md
import numpy as np
import pytest

from pymcpu.forcefields.mcpu import MCPUForceField
from pymcpu.runners import default_example_pdb


@pytest.fixture(scope="module")
def chignolin() -> md.Trajectory:
    traj = md.load(str(default_example_pdb()))
    return traj.atom_slice(traj.topology.select("not element H"))


def _two_chains(path: Path) -> md.Trajectory:
    """Chignolin twice, as chains A and B, 30 Å apart."""
    lines = [line for line in default_example_pdb().read_text().splitlines()
             if line.startswith("ATOM") and line[76:78].strip() != "H"]
    out = []
    for chain, shift in (("A", 0.0), ("B", 30.0)):
        for line in lines:
            x = float(line[30:38]) + shift
            out.append(f"{line[:21]}{chain}{line[22:30]}{x:8.3f}{line[38:]}")
        out.append("TER")
    path.write_text("\n".join(out) + "\nEND\n")
    return md.load(str(path))


def test_one_chain_builds(chignolin: md.Trajectory) -> None:
    MCPUForceField(chignolin)


def test_several_chains_raise(chignolin: md.Trajectory, tmp_path: Path) -> None:
    with pytest.raises(
        ValueError,
        match="one continuous chain.*it has 2 chains.*allow_chain_breaks=True.*"
        "forcefield_options: {allow_chain_breaks: true}",
    ):
        MCPUForceField(_two_chains(tmp_path / "two.pdb"))


def test_a_missing_residue_raises(chignolin: md.Trajectory) -> None:
    gapped = chignolin.atom_slice(chignolin.topology.select("not resSeq 5"))
    with pytest.raises(ValueError, match="numbering jumps from PRO 4 to THR 6"):
        MCPUForceField(gapped)


def test_a_long_peptide_bond_raises(chignolin: md.Trajectory) -> None:
    """Numbering intact, but residues 6-10 moved 5 Å away from 1-5."""
    moved = chignolin[:]
    later = chignolin.topology.select("resSeq 6 to 10")
    moved.xyz = moved.xyz.copy()
    moved.xyz[0, later, 0] += 0.5
    with pytest.raises(ValueError, match=r"C of GLU 5 is \d+\.\d Å from N of THR 6"):
        MCPUForceField(moved)


def test_the_opt_in_joins_the_pieces(chignolin: md.Trajectory, tmp_path: Path) -> None:
    gapped = chignolin.atom_slice(chignolin.topology.select("not resSeq 5"))
    assert MCPUForceField(gapped, allow_chain_breaks=True).n_res == 9
    assert MCPUForceField(_two_chains(tmp_path / "two.pdb"), allow_chain_breaks=True).n_res == 20


@pytest.mark.parametrize("atom", ["OE2", "CB", "O"])
def test_a_missing_atom_names_its_residue(chignolin: md.Trajectory, atom: str) -> None:
    pruned = chignolin.atom_slice(
        chignolin.topology.select(f"not (resSeq 5 and name {atom})")
    )
    with pytest.raises(ValueError, match=f"GLU 5 lacks {atom}\\b.*PDBFixer"):
        MCPUForceField(pruned)


def test_every_residue_with_missing_atoms_is_listed(chignolin: md.Trajectory) -> None:
    drop = {(2, "OH"), (9, "CZ2"), (9, "CH2")}
    pruned = chignolin.atom_slice(
        [a.index for a in chignolin.topology.atoms if (a.residue.resSeq, a.name) not in drop]
    )
    with pytest.raises(ValueError, match="from 2 residue") as error:
        MCPUForceField(pruned)
    assert "TYR 2 lacks OH" in str(error.value)
    assert "TRP 9 lacks CZ2, CH2" in str(error.value)


def test_insertion_style_repeated_numbers_are_not_a_break(chignolin: md.Trajectory) -> None:
    """Numbering that repeats (insertion codes) is not a jump; the C-N
    distance decides."""
    copy = chignolin[:]
    copy.topology = chignolin.topology.copy()
    for residue in copy.topology.residues:
        if residue.resSeq >= 6:
            residue.resSeq -= 1  # 1, 2, 3, 4, 5, 5, 6, 7, 8, 9
    assert np.isfinite(MCPUForceField(copy).coords).all()
