"""GLY-CA clash/contact ELIGIBILITY regression test.

``MCPUForceField._order_atoms`` duplicates GLY's CA into the sidechain
segment purely so the sidechain block-rotation move has a valid index range
for every residue (an array-layout fact -- see
``MCPUForceField._initialize_attributes``'s comment on GLY handling). Legacy
MCPU keeps that separate from clash/contact ELIGIBILITY: ``IsSidechainAtom
("CA")`` is always false there, regardless of residue -- GLY's CA only gets a
distinct Mu-potential *type* value, never a different eligibility
classification.

``MuPotentialBuilder`` used to read the array-position ``is_sidechain`` flag
directly for eligibility too, which meant the GLY-CA sidechain-segment
duplicate was treated as a real sidechain atom for clash/contact rules --
notably escaping the "both atoms backbone" exclusion from the long-range mu
contact term that every other residue's CA correctly gets. This test locks in
the fix (``MuPotentialBuilder._is_sidechain_for_eligibility``, name-based,
matching legacy) so it can't silently regress.

This is an internal-consistency regression test: every assertion compares
pyMCPU's own ``MuPotentialBuilder``/``layer1_atom_meta``/
``build_topology_masks`` against its own array-position flags on a real actin
PDB -- no legacy MCPU binary, log, or hardcoded legacy numeric output is read
anywhere, even though the *rationale* is legacy-motivated design intent.
"""

from __future__ import annotations

import mdtraj as md
import pytest

from pymcpu.forcefields.builders.mu_builder import MuPotentialBuilder
from pymcpu.forcefields.mcpu import MCPUForceField
from tests.fixtures.context_builders import resolve_test_pdb


@pytest.fixture(scope="module")
def acta_forcefield() -> MCPUForceField:
    pdb = resolve_test_pdb()  # actin (acta.pdb) is the suite default
    traj = md.load(str(pdb))
    heavy = traj.atom_slice(traj.topology.select("not element H"))
    return MCPUForceField(heavy, param_set="mcpu08")


def _gly_ca_atoms(ordered_atom_list) -> list[tuple[int, object]]:
    return [
        (idx, a)
        for idx, a in enumerate(ordered_atom_list)
        if a.residue_name == "GLY" and a.name == "CA"
    ]


def test_gly_ca_is_never_eligibility_sidechain(acta_forcefield: MCPUForceField) -> None:
    """Both the backbone-segment and sidechain-segment GLY-CA copies must be
    backbone for eligibility purposes, regardless of which array segment
    (``atom.is_sidechain``) they live in."""
    gly_cas = _gly_ca_atoms(acta_forcefield.ordered_atom_list)
    assert len(gly_cas) > 0, "test PDB must contain GLY residues for this test to be meaningful"
    for _, atom in gly_cas:
        assert MuPotentialBuilder._is_sidechain_for_eligibility(atom) is False


def test_gly_ca_array_position_flag_unaffected(acta_forcefield: MCPUForceField) -> None:
    """The array-position ``is_sidechain`` flag (used for sc_start/sc_count
    block-rotation bookkeeping, NOT eligibility) must still distinguish the
    two duplicate copies -- the fix must not touch this."""
    gly_cas = _gly_ca_atoms(acta_forcefield.ordered_atom_list)
    backbone_copies = [a for _, a in gly_cas if not a.is_sidechain]
    sidechain_copies = [a for _, a in gly_cas if a.is_sidechain]
    assert len(backbone_copies) == len(sidechain_copies) > 0


def test_layer1_atom_meta_is_sidechain_matches_eligibility(
    acta_forcefield: MCPUForceField,
) -> None:
    """layer1_atom_meta's returned is_sidechain array (fed to the C++ Layer-1
    eligibility machinery) must read 0 for GLY's CA, in both segments."""
    ordered = acta_forcefield.ordered_atom_list
    _, is_sidechain, _, _ = MuPotentialBuilder.layer1_atom_meta(ordered)
    for idx, _ in _gly_ca_atoms(ordered):
        assert is_sidechain[idx] == 0


def test_gly_ca_long_range_backbone_pair_excludes_contact(
    acta_forcefield: MCPUForceField,
) -> None:
    """The long-range (res_diff >= skip_local_contact_range) mu-contact term
    excludes BB-BB pairs. A GLY-CA (sidechain-segment duplicate) paired with
    another residue's backbone atom, far away in sequence, must be excluded
    from contact eligibility just like any other backbone-backbone pair --
    this is exactly the case the conflation bug broke."""
    ordered = acta_forcefield.ordered_atom_list
    n = acta_forcefield.n_atoms
    contact_flat, _clash_flat = MuPotentialBuilder.build_topology_masks(ordered, n)

    # Canonical eligibility-backbone name set, imported rather than
    # re-hardcoded, so this test can't silently drift from the source it's
    # regression-testing.
    backbone_names = MuPotentialBuilder._ELIGIBILITY_BACKBONE_NAMES
    gly_ca_sc = [
        (idx, a)
        for idx, a in enumerate(ordered)
        if a.residue_name == "GLY" and a.name == "CA" and a.is_sidechain
    ]
    assert gly_ca_sc, "expected at least one sidechain-segment GLY-CA duplicate"
    gly_idx, gly_atom = gly_ca_sc[0]

    found_long_range_backbone_pair = False
    for j, other in enumerate(ordered):
        if other.name not in backbone_names:
            continue
        if abs(other.residue_index - gly_atom.residue_index) < 4:
            continue
        found_long_range_backbone_pair = True
        assert contact_flat[gly_idx * n + j] == 0, (
            f"{gly_atom.residue_name}{gly_atom.residue_index}.CA vs "
            f"{other.residue_name}{other.residue_index}.{other.name} "
            "should be contact-excluded (BB-BB long range)"
        )
    assert found_long_range_backbone_pair
