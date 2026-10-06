"""What the Mu potential gets for glycine's CA.

Glycine's CA has one engine slot, in the backbone segment. For Mu it must be
ordinary backbone for clash/contact eligibility -- legacy MCPU's
``IsSidechainAtom("CA")`` is false for every residue -- with the
glycine-specific Mu *type*.

Earlier versions stored the CA twice and muted the backbone copy, so only the
sidechain-segment copy was scored. These tests pin the two inputs Mu reads
for the single slot (eligibility masks, atom type), so that a
leftover mute, which would silently drop glycine's Mu energy, cannot pass.

Internal consistency only: every assertion checks pyMCPU's own builder
output. No legacy MCPU binary, log or number is read.
"""

from __future__ import annotations

import mdtraj as md
import pytest

from pymcpu.forcefields.builders.mu_builder import MuPotentialBuilder
from pymcpu.forcefields.mcpu import MCPUForceField
from pymcpu.runners import default_example_pdb
from tests.fixtures.context_builders import resolve_test_pdb

LONG_RANGE = 4  # build_topology_masks' default skip_local_contact_range


def _forcefield(path) -> MCPUForceField:
    traj = md.load(str(path))
    heavy = traj.atom_slice(traj.topology.select("not element H"))
    return MCPUForceField(heavy, param_set="mcpu08")


@pytest.fixture(scope="module")
def acta_forcefield() -> MCPUForceField:
    return _forcefield(resolve_test_pdb())  # actin, the suite default


@pytest.fixture(scope="module")
def acta_masks(acta_forcefield: MCPUForceField):
    return MuPotentialBuilder.build_topology_masks(
        acta_forcefield.ordered_atom_list, acta_forcefield.n_atoms
    )


def _gly_cas(ordered_atom_list) -> list[int]:
    return [
        i for i, a in enumerate(ordered_atom_list) if a.residue_name == "GLY" and a.name == "CA"
    ]


def test_each_glycine_ca_has_one_backbone_slot(acta_forcefield: MCPUForceField) -> None:
    ff = acta_forcefield
    gly_cas = _gly_cas(ff.ordered_atom_list)
    gly_residues = {ff.ordered_atom_list[i].residue_index for i in gly_cas}
    assert gly_cas, "test PDB must contain GLY residues for this test to be meaningful"
    assert len(gly_cas) == len(gly_residues)
    assert all(i < ff.total_bb_atoms for i in gly_cas)


def test_glycine_ca_is_backbone_for_eligibility(acta_forcefield: MCPUForceField) -> None:
    ordered = acta_forcefield.ordered_atom_list
    for i in _gly_cas(ordered):
        assert MuPotentialBuilder._is_sidechain_for_eligibility(ordered[i]) is False


class _RecordingLookup(dict):
    def __init__(self, base) -> None:
        super().__init__(base)
        self.keys_read: list[tuple[str, str]] = []

    def __getitem__(self, key):
        self.keys_read.append(key)
        return super().__getitem__(key)


def test_glycine_ca_gets_the_glycine_mu_type() -> None:
    ff = _forcefield(default_example_pdb())  # small: build() fills an N x N matrix
    lookup = _RecordingLookup(ff.atom_type_lookup)
    MuPotentialBuilder.build(
        atom_list=ff.ordered_atom_list,
        atom_to_residue=ff.atom_to_res,
        mu_potential_matrix=ff.mu_energies,
        atom_type_lookup=lookup,
    )
    assert len(lookup.keys_read) == ff.n_atoms  # one read per atom, no H here
    gly_cas = _gly_cas(ff.ordered_atom_list)
    assert gly_cas
    assert {lookup.keys_read[i] for i in gly_cas} == {("GLY", "CA")}
    assert ff.atom_type_lookup[("GLY", "CA")][0] != ff.atom_type_lookup[("XXX", "CA")][0]


def test_glycine_ca_is_scored_against_distant_atoms(acta_forcefield: MCPUForceField, acta_masks) -> None:
    """A muted atom has every mask entry zero. Glycine's CA must be able to
    clash with distant atoms and make Mu contacts with distant sidechains."""
    ordered = acta_forcefield.ordered_atom_list
    n = acta_forcefield.n_atoms
    contact, clash = acta_masks
    for i in _gly_cas(ordered):
        res = ordered[i].residue_index
        far = [j for j, a in enumerate(ordered) if abs(a.residue_index - res) > LONG_RANGE]
        far_sc = [j for j in far if MuPotentialBuilder._is_sidechain_for_eligibility(ordered[j])]
        assert all(clash[i * n + j] == 1 for j in far)
        assert far_sc and all(contact[i * n + j] == 1 for j in far_sc)


def test_glycine_ca_long_range_backbone_pair_excludes_contact(
    acta_forcefield: MCPUForceField, acta_masks
) -> None:
    """Long-range backbone-backbone pairs are excluded from Mu contacts, and
    glycine's CA is no exception."""
    ordered = acta_forcefield.ordered_atom_list
    n = acta_forcefield.n_atoms
    contact, _ = acta_masks
    backbone_names = MuPotentialBuilder._ELIGIBILITY_BACKBONE_NAMES
    gly_idx = _gly_cas(ordered)[0]
    gly_atom = ordered[gly_idx]

    found_long_range_backbone_pair = False
    for j, other in enumerate(ordered):
        if other.name not in backbone_names:
            continue
        if abs(other.residue_index - gly_atom.residue_index) < LONG_RANGE:
            continue
        found_long_range_backbone_pair = True
        assert contact[gly_idx * n + j] == 0, (
            f"{gly_atom.residue_name}{gly_atom.residue_index}.CA vs "
            f"{other.residue_name}{other.residue_index}.{other.name} "
            "should be contact-excluded (BB-BB long range)"
        )
    assert found_long_range_backbone_pair
