"""``clash_bond_exclusion``: pairs a few covalent bonds apart cannot clash.

By default the clash rules are mcpu08's (``build_topology_masks``), under which
many pairs two or three bonds apart can clash: CA-CG, whose rule tests for a
side-chain name starting with 'G' that no atom has, N(i)-N(i+1), set by psi,
and so on. They sit at distances their bonds and one torsion fix, so a harder
core forbids some torsions outright or, for the pairs already inside it at
the start, exempts them for the whole run. With ``clash_bond_exclusion=n``
every pair at most n bonds apart is left out of the clash check. The bonds
come from residue templates by atom name, never from coordinates, plus the
structure's disulfide bridges (two CYS SG atoms under 2.5 A apart); the
contact mask does not change, and a pair bonded through the templates that
could make a contact is refused.
"""

from __future__ import annotations

import logging
from pathlib import Path

import mdtraj as md
import numpy as np
import pytest

from pymcpu import mcpu_core
from pymcpu.forcefields.builders.mu_builder import MuPotentialBuilder
from pymcpu.forcefields.mcpu import MCPUAtom, MCPUForceField
from pymcpu.runners import default_example_pdb
from tests.fixtures.context_builders import TEST_PDB

FAR = 4  # bond_distances(..., 3) gives 3 + 1 for pairs farther apart

#: Glycine, proline and a C-terminal serine with its OXT.
CHAIN = {
    0: ("GLY", ["N", "CA", "C", "O"]),
    1: ("PRO", ["N", "CA", "C", "CB", "CG", "CD", "O"]),
    2: ("SER", ["N", "CA", "C", "CB", "OG", "O", "OXT"]),
}

#: (residue, atom), (residue, atom), bonds apart (FAR: more than three).
EXPECTED = [
    ((0, "N"), (0, "CA"), 1), ((0, "N"), (0, "C"), 2), ((0, "N"), (0, "O"), 3),
    ((0, "CA"), (0, "O"), 2),
    # the peptide bond, and proline's N bonded to both CA and CD
    ((0, "C"), (1, "N"), 1), ((0, "CA"), (1, "N"), 2), ((0, "O"), (1, "N"), 2),
    ((0, "N"), (1, "N"), 3), ((0, "C"), (1, "CD"), 2), ((0, "C"), (1, "CA"), 2),
    ((0, "CA"), (1, "CD"), 3), ((0, "O"), (1, "CD"), 3), ((0, "O"), (1, "CA"), 3),
    ((0, "C"), (1, "CG"), 3), ((0, "C"), (1, "CB"), 3), ((0, "C"), (1, "C"), 3),
    ((0, "N"), (1, "CD"), FAR), ((0, "N"), (1, "CA"), FAR),
    # proline's ring
    ((1, "N"), (1, "CD"), 1), ((1, "CA"), (1, "CD"), 2), ((1, "CB"), (1, "CD"), 2),
    ((1, "N"), (1, "CG"), 2), ((1, "CA"), (1, "CG"), 2), ((1, "N"), (1, "CB"), 2),
    ((1, "C"), (1, "CD"), 3), ((1, "O"), (1, "N"), 3),
    # the C-terminus
    ((2, "C"), (2, "OXT"), 1), ((2, "O"), (2, "OXT"), 2), ((2, "CA"), (2, "OXT"), 2),
    ((2, "N"), (2, "OXT"), 3), ((2, "CB"), (2, "OXT"), 3), ((2, "OG"), (2, "OXT"), FAR),
    ((2, "N"), (2, "OG"), 3), ((2, "C"), (2, "OG"), 3), ((2, "O"), (2, "CB"), 3),
    ((2, "O"), (2, "OG"), FAR), ((1, "O"), (2, "CA"), 3), ((1, "CA"), (2, "N"), 2),
    ((1, "C"), (2, "OXT"), FAR), ((0, "C"), (2, "N"), FAR),
]


def _chain(extra: list[MCPUAtom] | None = None) -> list[MCPUAtom]:
    atoms = [
        MCPUAtom(original_index=-1, name=name, residue_name=residue, residue_index=r)
        for r, (residue, names) in CHAIN.items()
        for name in names
    ]
    return atoms + list(extra or [])


def _index(atoms: list[MCPUAtom], residue: int, name: str) -> int:
    return next(k for k, a in enumerate(atoms) if a.residue_index == residue and a.name == name)


@pytest.mark.parametrize(("a", "b", "bonds"), EXPECTED, ids=[f"{a}-{b}" for a, b, _ in EXPECTED])
def test_bonds_apart_on_a_glycine_proline_serine_chain(a, b, bonds) -> None:
    atoms = _chain()
    distances = MuPotentialBuilder.bond_distances(atoms, 3)
    i, j = _index(atoms, *a), _index(atoms, *b)
    assert distances[i, j] == distances[j, i] == bonds


def test_every_atom_is_zero_bonds_from_itself() -> None:
    distances = MuPotentialBuilder.bond_distances(_chain(), 3)
    assert distances.dtype == np.int16
    assert np.all(np.diag(distances) == 0)
    assert np.array_equal(distances, distances.T)


def test_a_glycine_ca_listed_twice_is_one_atom() -> None:
    """Layouts that stored glycine's CA in two slots: both slots are the
    backbone CA's node."""
    copy = MCPUAtom(original_index=-1, name="CA", residue_name="GLY", residue_index=0)
    atoms = _chain([copy])
    distances = MuPotentialBuilder.bond_distances(atoms, 3)
    ca, dup = _index(atoms, 0, "CA"), len(atoms) - 1
    assert distances[ca, dup] == 0
    assert np.array_equal(distances[dup], distances[ca])


def test_an_explicit_amide_hydrogen_is_bonded_to_nothing() -> None:
    hydrogen = MCPUAtom(original_index=-1, name="H", residue_name="SER", residue_index=2)
    atoms = _chain([hydrogen])
    distances = MuPotentialBuilder.bond_distances(atoms, 3)
    assert np.all(distances[-1] == FAR) and np.all(distances[:, -1] == FAR)


def test_a_residue_without_a_bond_template_is_refused() -> None:
    odd = [MCPUAtom(original_index=-1, name=name, residue_name="XYZ", residue_index=3)
           for name in ("N", "CA", "C", "O")]
    with pytest.raises(ValueError, match=r"residue 3 \(XYZ\) has no bond template"):
        MuPotentialBuilder.bond_distances(_chain(odd), 3)


def test_an_atom_no_bond_reaches_is_refused() -> None:
    odd = [MCPUAtom(original_index=-1, name="SE", residue_name="SER", residue_index=2)]
    with pytest.raises(ValueError, match="no template bond reaches SE of SER 2"):
        MuPotentialBuilder.bond_distances(_chain(odd), 3)


def test_a_contact_eligible_bonded_pair_is_refused() -> None:
    atoms = _chain()
    n = len(atoms)
    contact = np.zeros((n, n), dtype=np.int8)
    clash = np.ones((n, n), dtype=np.int8)
    i, j = _index(atoms, 0, "C"), _index(atoms, 1, "CG")  # three bonds apart
    contact[i, j] = contact[j, i] = 1
    with pytest.raises(ValueError, match="are 3 bonds apart and contact-eligible"):
        MuPotentialBuilder._exclude_bonded_pairs(atoms, contact, clash, 3)
    assert np.all(clash == 1)  # nothing changed


# --- the masks of a real structure -------------------------------------------

@pytest.fixture(scope="module")
def heavy() -> md.Trajectory:
    traj = md.load(str(default_example_pdb()))  # chignolin: GLY, PRO, an OXT
    return traj.atom_slice(traj.topology.select("not element H"))


@pytest.fixture(scope="module")
def chignolin(heavy: md.Trajectory) -> MCPUForceField:
    return MCPUForceField(heavy)


def _masks(ff: MCPUForceField, exclusion: int) -> tuple[np.ndarray, np.ndarray]:
    contact, clash = MuPotentialBuilder.build_topology_masks(
        ff.ordered_atom_list, ff.n_atoms, clash_bond_exclusion=exclusion)
    shape = (ff.n_atoms, ff.n_atoms)
    return np.reshape(contact, shape), np.reshape(clash, shape)


def test_the_default_is_mcpu08s_rules(chignolin: MCPUForceField, heavy: md.Trajectory) -> None:
    assert MCPUForceField(heavy).clash_bond_exclusion == 0
    default = MuPotentialBuilder.build_topology_masks(chignolin.ordered_atom_list, chignolin.n_atoms)
    explicit = MuPotentialBuilder.build_topology_masks(
        chignolin.ordered_atom_list, chignolin.n_atoms, clash_bond_exclusion=0)
    assert default == explicit


@pytest.mark.parametrize("exclusion", [1, 2, 3])
def test_exactly_the_bonded_pairs_leave_the_clash_check(
    chignolin: MCPUForceField, exclusion: int
) -> None:
    contact0, clash0 = _masks(chignolin, 0)
    contact, clash = _masks(chignolin, exclusion)
    near = MuPotentialBuilder.bond_distances(chignolin.ordered_atom_list, exclusion) <= exclusion
    np.fill_diagonal(near, False)
    assert np.array_equal(contact, contact0)
    assert np.all(clash[near] == 0)
    assert np.array_equal(clash[~near], clash0[~near])
    # mcpu08's rules already exempt every bonded pair, and let some pairs two
    # or three bonds apart clash.
    assert bool(np.count_nonzero(clash0[near])) == (exclusion > 1)


def test_the_pairs_mcpu08s_rules_let_clash(chignolin: MCPUForceField) -> None:
    """Pairs two and three bonds apart that mcpu08's rules let clash, and one
    they already exempt."""
    atoms = chignolin.ordered_atom_list

    def pair(a, b):
        return _index(atoms, *a), _index(atoms, *b)

    _, clash0 = _masks(chignolin, 0)
    _, clash3 = _masks(chignolin, 3)
    for a, b in [((1, "CA"), (1, "CG")),   # TYR1, two bonds: 'CG' does not start with 'G'
                 ((1, "N"), (1, "CG")),    # three bonds, set by chi1
                 ((0, "N"), (1, "N")),     # three bonds, set by psi of GLY0
                 ((2, "CB"), (3, "N"))]:   # ASP2 CB and PRO3 N, three bonds
        i, j = pair(a, b)
        assert clash0[i, j] == 1 and clash3[i, j] == 0, (a, b)
    i, j = pair((2, "C"), (3, "CD"))  # PRO3's CD, two bonds from the previous C
    assert clash0[i, j] == 0 and clash3[i, j] == 0


# None is not here: like hbond_beta_favor, it asks for the parameter set's own
# (test_paramset_manifest.py), 0 for mcpu08.
@pytest.mark.parametrize("value", [-1, 4, 3.0, True, "3"], ids=repr)
def test_clash_bond_exclusion_is_checked(heavy: md.Trajectory, value: object) -> None:
    with pytest.raises(ValueError, match="clash_bond_exclusion must be 0"):
        MCPUForceField(heavy, clash_bond_exclusion=value)


def test_a_bonded_pair_pulled_inside_the_core_clashes_only_by_default(heavy: md.Trajectory) -> None:
    """GLY0 N and TYR1 N, three bonds apart, are outside the core in the
    deposited structure, so mcpu08's rules let them clash. Pulled to 0.6
    radius sums apart, the state clashes under those rules and not with
    ``clash_bond_exclusion=3``."""
    verdicts = {}
    for exclusion in (0, 3):
        ff = MCPUForceField(heavy, clash_bond_exclusion=exclusion)
        n0, n1 = ff.blocks[0].bb_start, ff.blocks[1].bb_start
        start = np.ascontiguousarray((ff.coords[0] * 10.0).T.astype(np.float32))
        radius = ff.atom_type_lookup[("XXX", "N")][1]
        xyz = start.astype(np.float64)
        u = xyz[:, n1] - xyz[:, n0]
        coords = start.copy()
        coords[:, n1] = (xyz[:, n0] + 0.6 * 2 * radius * u / np.linalg.norm(u)).astype(np.float32)
        ctx = mcpu_core.Context(ff.create_system(heavy.topology))
        ctx.set_positions(start)
        ctx.calculate_total_energy(-1)
        assert not ctx.has_steric_clash()
        ctx.set_positions(coords)
        ctx.calculate_total_energy(-1)
        verdicts[exclusion] = bool(ctx.has_steric_clash())
    assert verdicts == {0: True, 3: False}


# --- disulfide bridges -------------------------------------------------------

#: Crambin (PDB 3NIR, X-ray, 0.48 A), protein heavy atoms: three bridges.
CRAMBIN_PDB = Path(__file__).parent / "data" / "3nir_heavy.pdb"
CRAMBIN_BRIDGES = [(2, 39), (3, 31), (15, 25)]

#: Two cysteines far apart in sequence.
CYSTEINES = {r: ("CYS", ["N", "CA", "C", "CB", "SG", "O"]) for r in (10, 20)}

#: (atom of CYS 10, atom of CYS 20, bonds apart through a bridge between them)
ACROSS = [
    ("SG", "SG", 1), ("CB", "SG", 2), ("SG", "CB", 2), ("CA", "SG", 3), ("SG", "CA", 3),
    ("CB", "CB", 3), ("N", "SG", FAR), ("C", "SG", FAR), ("CA", "CB", FAR),
]


def _cysteines() -> list[MCPUAtom]:
    return [
        MCPUAtom(original_index=-1, name=name, residue_name=residue, residue_index=r)
        for r, (residue, names) in CYSTEINES.items()
        for name in names
    ]


@pytest.mark.parametrize(("a", "b", "bonds"), ACROSS, ids=[f"{a}-{b}'" for a, b, _ in ACROSS])
def test_a_disulfide_is_a_bond(a, b, bonds) -> None:
    atoms = _cysteines()
    i, j = _index(atoms, 10, a), _index(atoms, 20, b)
    bridged = MuPotentialBuilder.bond_distances(atoms, 3, [(10, 20)])
    assert bridged[i, j] == bridged[j, i] == bonds
    assert MuPotentialBuilder.bond_distances(atoms, 3)[i, j] == FAR


@pytest.mark.parametrize("pair", [(2, 10), (10, 10)], ids=["a-serine", "one-residue"])
def test_a_disulfide_joins_two_cysteines(pair) -> None:
    with pytest.raises(ValueError, match="joins the SG atoms of two CYS residues"):
        MuPotentialBuilder.bond_distances(_chain(_cysteines()), 3, [pair])


def test_a_pair_across_a_bridge_keeps_its_contact() -> None:
    """A bridge can join residues far apart in sequence, so a pair across it
    may be contact-eligible. It leaves the clash check and keeps its contact,
    where a contact-eligible pair bonded through the templates is refused."""
    atoms = _cysteines()
    n = len(atoms)
    contact = np.zeros((n, n), dtype=np.int8)
    i, j = _index(atoms, 10, "CB"), _index(atoms, 20, "SG")
    contact[i, j] = contact[j, i] = 1
    plain, bridged = np.ones((n, n), dtype=np.int8), np.ones((n, n), dtype=np.int8)
    MuPotentialBuilder._exclude_bonded_pairs(atoms, contact, plain, 3)
    MuPotentialBuilder._exclude_bonded_pairs(atoms, contact, bridged, 3, [(10, 20)])
    assert plain[i, j] == plain[j, i] == 1
    assert bridged[i, j] == bridged[j, i] == 0
    assert contact[i, j] == contact[j, i] == 1


@pytest.mark.parametrize(("sg_sg", "bridged"), [(2.04, True), (2.45, True), (2.55, False), (3.75, False)])
def test_a_bridge_is_two_sg_atoms_under_2_5_angstrom(sg_sg: float, bridged: bool) -> None:
    atoms = _cysteines()
    coords = np.zeros((len(atoms), 3))
    coords[:, 0] = 10.0 * np.arange(len(atoms))
    coords[_index(atoms, 20, "SG")] = coords[_index(atoms, 10, "SG")] + [0.0, sg_sg, 0.0]
    assert MuPotentialBuilder.find_disulfides(atoms, coords) == ([(10, 20)] if bridged else [])


@pytest.fixture(scope="module")
def crambin() -> md.Trajectory:
    return md.load(str(CRAMBIN_PDB))


def _bridges(structure: md.Trajectory) -> list[tuple[int, int]]:
    ff = MCPUForceField(structure)
    return MuPotentialBuilder.find_disulfides(ff.ordered_atom_list, ff.coords[0] * 10.0)


def test_the_bridges_of_real_structures(crambin: md.Trajectory, heavy: md.Trajectory) -> None:
    """Crambin's three; none in chignolin, which has no cysteine, or in actin,
    whose two closest SG atoms (CYS 219 and 259 in the file) are 3.75 A apart."""
    assert _bridges(crambin) == CRAMBIN_BRIDGES
    assert _bridges(heavy) == []
    actin = md.load(str(TEST_PDB))
    assert _bridges(actin.atom_slice(actin.topology.select("not element H"))) == []


def test_crambins_bridges_leave_the_clash_check(crambin: md.Trajectory) -> None:
    """Six pairs are at most three bonds apart across each bridge: SG-SG',
    CB-SG', SG-CB', CA-SG', SG-CA' and CB-CB'. mcpu08's rules already leave out
    SG-SG' (CYS SG pairs far apart in sequence are contact-only), so the
    bridges take 15 more pairs out of the clash check at
    clash_bond_exclusion=3, and no contact changes."""
    ff = MCPUForceField(crambin)
    atoms = ff.ordered_atom_list
    shape = (ff.n_atoms, ff.n_atoms)

    def masks(exclusion: int, disulfides=()) -> tuple[np.ndarray, np.ndarray]:
        contact, clash = MuPotentialBuilder.build_topology_masks(
            atoms, ff.n_atoms, clash_bond_exclusion=exclusion, disulfides=disulfides)
        return np.reshape(contact, shape), np.reshape(clash, shape)

    contact0, clash0 = masks(0)
    assert np.array_equal(masks(0, CRAMBIN_BRIDGES)[1], clash0)  # not used at 0
    _, clash3 = masks(3)
    contact3, clash3_bridged = masks(3, CRAMBIN_BRIDGES)
    templates = MuPotentialBuilder.bond_distances(atoms, 3) <= 3
    across = (MuPotentialBuilder.bond_distances(atoms, 3, CRAMBIN_BRIDGES) <= 3) & ~templates
    assert np.count_nonzero(np.triu(across, 1)) == 18
    assert np.count_nonzero(np.triu(across & (clash3 == 1), 1)) == 15
    assert np.all(clash3_bridged[across] == 0)
    assert np.array_equal(clash3_bridged[~across], clash3[~across])
    assert np.array_equal(contact3, contact0)


def test_crambin_has_no_start_overlap_across_its_bridges(crambin: md.Trajectory, caplog) -> None:
    """At a hard core of 0.85 radius sums, CB and SG across each bridge
    (2.98-3.07 A apart, against 3.10 A) are inside it: without the bridges as
    bonds, these six pairs were crambin's only start overlaps under
    clash_bond_exclusion=3, which a penalised relaxation would push apart."""
    def start_overlap_pairs(**options) -> int:
        ff = MCPUForceField(crambin, **options)
        context = mcpu_core.Context(ff.create_system(crambin.topology))
        context.set_positions((ff.coords[0] * 10.0).T.astype(np.float32))
        context.calculate_total_energy(-1)
        assert not context.has_steric_clash()
        return int(context.mu_potential.start_overlap_pairs)

    with caplog.at_level(logging.INFO, logger="pymcpu.forcefields.mcpu"):
        assert start_overlap_pairs(
            clash_alpha=0.85, clash_bond_exclusion=3, clash_start_exemption=False) == 0
    assert any("counts 3 disulfide bridge(s)" in r.getMessage() for r in caplog.records)
    assert start_overlap_pairs(clash_alpha=0.85, clash_bond_exclusion=3) == 0
