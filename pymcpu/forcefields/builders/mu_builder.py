"""Builder for the Mu (pairwise atom-atom contact) potential.

Assembles the C++ ``mcpu_core.MuPotential`` energy term: loads the MCPU08
atom-type/radius lookup table and the raw contact-energy matrix, computes the
coordinate-independent clash/contact eligibility masks consumed by the C++
engine, and includes a Python port of the relevant parts of legacy MCPU's
``pdb_util.h`` atom-classification logic that those computations depend on.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from pymcpu import mcpu_core
import pandas as pd

if TYPE_CHECKING:  # avoid a circular import -- pymcpu.forcefields.mcpu imports this module
    from pymcpu.forcefields.mcpu import MCPUAtom

N_ATOM_TYPES = 84 # MCPU08 has 84 atom types

# Set True to print the largest contact cutoff once per process (a developer
# sanity check against the neighbour-grid cell size). Off for normal runs.
_MU_BUILDER_VERBOSE = False
_MAX_CONTACT_PRINTED = False

#: Covalent bonds by atom name, for ``clash_bond_exclusion``: the backbone's
#: (with either name of a C-terminal oxygen), and each residue's side chain.
#: CA-CB (every residue but glycine) and the peptide bond C(i)-N(i+1) are
#: added when the graph is built. Proline's ring closes through CD-N. ILE's
#: delta carbon is CD1 in mcpu08's template and CD in some others.
_BACKBONE_BONDS = (("N", "CA"), ("CA", "C"), ("C", "O"), ("C", "OXT"), ("C", "OCT"))
_SIDECHAIN_BONDS = {
    "ALA": (),
    "ARG": (("CB", "CG"), ("CG", "CD"), ("CD", "NE"), ("NE", "CZ"), ("CZ", "NH1"), ("CZ", "NH2")),
    "ASN": (("CB", "CG"), ("CG", "OD1"), ("CG", "ND2")),
    "ASP": (("CB", "CG"), ("CG", "OD1"), ("CG", "OD2")),
    "CYS": (("CB", "SG"),),
    "GLN": (("CB", "CG"), ("CG", "CD"), ("CD", "OE1"), ("CD", "NE2")),
    "GLU": (("CB", "CG"), ("CG", "CD"), ("CD", "OE1"), ("CD", "OE2")),
    "GLY": (),
    "HIS": (("CB", "CG"), ("CG", "ND1"), ("CG", "CD2"), ("ND1", "CE1"), ("CD2", "NE2"),
            ("CE1", "NE2")),
    "ILE": (("CB", "CG1"), ("CB", "CG2"), ("CG1", "CD1"), ("CG1", "CD")),
    "LEU": (("CB", "CG"), ("CG", "CD1"), ("CG", "CD2")),
    "LYS": (("CB", "CG"), ("CG", "CD"), ("CD", "CE"), ("CE", "NZ")),
    "MET": (("CB", "CG"), ("CG", "SD"), ("SD", "CE")),
    "PHE": (("CB", "CG"), ("CG", "CD1"), ("CG", "CD2"), ("CD1", "CE1"), ("CD2", "CE2"),
            ("CE1", "CZ"), ("CE2", "CZ")),
    "PRO": (("CB", "CG"), ("CG", "CD"), ("CD", "N")),
    "SER": (("CB", "OG"),),
    "THR": (("CB", "OG1"), ("CB", "CG2")),
    "TRP": (("CB", "CG"), ("CG", "CD1"), ("CG", "CD2"), ("CD1", "NE1"), ("NE1", "CE2"),
            ("CD2", "CE2"), ("CD2", "CE3"), ("CE2", "CZ2"), ("CE3", "CZ3"), ("CZ2", "CH2"),
            ("CZ3", "CH2")),
    "TYR": (("CB", "CG"), ("CG", "CD1"), ("CG", "CD2"), ("CD1", "CE1"), ("CD2", "CE2"),
            ("CE1", "CZ"), ("CE2", "CZ"), ("CZ", "OH")),
    "VAL": (("CB", "CG1"), ("CB", "CG2")),
}
#: A disulfide bridge, for ``clash_bond_exclusion``: two CYS SG atoms closer
#: than this (Angstrom) in the structure the force field is built from. An
#: S-S bond is about 2.05 A long.
DISULFIDE_SG_CUTOFF_A = 2.5


class MuPotentialBuilder:
    """Stateless namespace of classmethods that build the Mu contact potential.

    Never instantiated -- callers use the classmethods directly (see
    ``pymcpu.forcefields.mcpu.MCPUForceField``, which drives the full
    pipeline: ``load_atom_types``/``load_parameters`` to read the on-disk
    atom-type table and MCPU08 energy matrix, ``build_topology_masks`` to
    derive the coordinate-independent per-atom-pair eligibility masks the C++
    engine turns into its pair-flag table, and ``build`` to assemble all of the
    above into a ready ``mcpu_core.MuPotential``.
    """

    # legacy pdb_util.h IsSidechainAtom(): backbone-named atoms are never
    # "sidechain" for clash/contact ELIGIBILITY purposes, decided by name
    # alone. GLY's CA is ordinary backbone here, like every other CA; it
    # differs only in its Mu-potential *type* value.
    _ELIGIBILITY_BACKBONE_NAMES = frozenset({'N', 'CA', 'C', 'O', 'OCT', 'OXT'})

    @classmethod
    def _is_sidechain_for_eligibility(cls, atom: MCPUAtom) -> bool:
        return atom.name not in cls._ELIGIBILITY_BACKBONE_NAMES

    @classmethod
    def load_atom_types(cls, filepath: str) -> dict[tuple[str, str], tuple[int, float]]:
        """
        Parses the atom types CSV into a fast O(1) lookup dictionary.
        This should only be called once during initialization.
        """
        atom_type_df = pd.read_csv(filepath)
        lookup = {
            (row.residue, row.atom): (int(row.type), float(row.radius))
            for row in atom_type_df.itertuples(index=False)
        }
        # The engine keeps one Mu entry per pair of atom types, so a type
        # whose atoms had different radii would get whichever pair it saw
        # last, depending on atom order.
        first_with_type: dict[int, tuple[str, str, float]] = {}
        for (residue, atom), (atom_type, radius) in lookup.items():
            seen = first_with_type.setdefault(atom_type, (residue, atom, radius))
            if seen[2] != radius:
                raise ValueError(
                    f"{filepath}: atom type {atom_type} has radius {seen[2]} for "
                    f"{seen[0]} {seen[1]} but {radius} for {residue} {atom}; every "
                    "atom of a type must have the same radius"
                )
        return lookup

    @classmethod
    def load_parameters(cls, filepath: str) -> np.ndarray:
        """
        PHASE 1: Read and shape the binary Mu Potential data.
        """
        raw_params = np.fromfile(filepath, dtype=np.float32)
        expected = N_ATOM_TYPES * N_ATOM_TYPES
        if raw_params.size != expected:
            raise ValueError(
                f"mu_potentials.bin: expected {expected} floats "
                f"({N_ATOM_TYPES}x{N_ATOM_TYPES}), got {raw_params.size}"
            )
        matrix = raw_params.reshape((N_ATOM_TYPES, N_ATOM_TYPES))
        # One energy per unordered type pair: the engine stores (a, b) and
        # (b, a) as the same entry.
        if not np.all(np.isfinite(matrix)):
            raise ValueError(f"mu_potentials.bin: {filepath} contains a non-finite energy")
        if not np.array_equal(matrix, matrix.T):
            a, b = np.argwhere(matrix != matrix.T)[0]
            raise ValueError(
                f"mu_potentials.bin: {filepath} is not symmetric: energy({a}, {b}) = "
                f"{matrix[a, b]} but energy({b}, {a}) = {matrix[b, a]}"
            )
        return matrix

    @classmethod
    def find_disulfides(
        cls,
        ordered_atom_list: list[MCPUAtom],
        coords: np.ndarray,
        cutoff: float = DISULFIDE_SG_CUTOFF_A,
    ) -> list[tuple[int, int]]:
        """The disulfide bridges of a structure: ``(i, j)`` residue indices,
        ``i < j``, sorted, of every two CYS residues whose SG atoms are closer
        than ``cutoff`` Angstrom. ``coords`` is ``(n, 3)``, in Angstrom, in
        the order of ``ordered_atom_list``.

        They are read from coordinates because a topology may lack them:
        mdtraj's PDB reader adds an S-S bond only from a CONECT record.
        """
        sg = [(k, int(a.residue_index)) for k, a in enumerate(ordered_atom_list)
              if a.residue_name == "CYS" and a.name == "SG"]
        xyz = np.asarray(coords, dtype=np.float64)[[k for k, _ in sg]].reshape(-1, 3)
        bridges = []
        for p in range(len(sg)):
            for q in range(p + 1, len(sg)):
                if np.linalg.norm(xyz[p] - xyz[q]) < cutoff:
                    bridges.append((min(sg[p][1], sg[q][1]), max(sg[p][1], sg[q][1])))
        return sorted(bridges)

    @classmethod
    def bond_distances(
        cls,
        ordered_atom_list: list[MCPUAtom],
        max_bonds: int,
        disulfides: Sequence[tuple[int, int]] = (),
    ) -> np.ndarray:
        """``(n, n)`` int16 array: how many covalent bonds apart each pair of
        atoms is, or ``max_bonds + 1`` when farther (and for every pair with
        an explicit amide hydrogen, ``'H'``, which no mask lets clash).

        The bonds come from residue templates by atom name
        (``_BACKBONE_BONDS``, ``_SIDECHAIN_BONDS``, CA-CB, and C(i)-N(i+1)
        between consecutive residue indices), never from coordinates, so a
        structure gives the same graph in any conformation; plus an SG-SG
        bond for each pair of residue indices in ``disulfides`` (see
        :meth:`find_disulfides`). An atom is one node per (residue, name): a
        glycine CA listed in two slots is one atom, 0 bonds from itself.

        Raises ValueError for a residue without a bond template, an atom
        that no template bond reaches, or a disulfide that does not join the
        SG atoms of two CYS residues.
        """
        n = len(ordered_atom_list)
        far = max_bonds + 1
        node_of: dict[tuple[int, str], int] = {}
        atom_node = np.full(n, -1, dtype=np.int64)
        residue_name: dict[int, str] = {}
        for k, atom in enumerate(ordered_atom_list):
            if atom.name == "H":
                continue
            key = (int(atom.residue_index), atom.name)
            atom_node[k] = node_of.setdefault(key, len(node_of))
            residue_name[int(atom.residue_index)] = atom.residue_name
        neighbours: list[list[int]] = [[] for _ in node_of]

        def link(r1: int, name1: str, r2: int, name2: str) -> None:
            u, v = node_of.get((r1, name1)), node_of.get((r2, name2))
            if u is not None and v is not None:
                neighbours[u].append(v)
                neighbours[v].append(u)

        for r, name in residue_name.items():
            if name not in _SIDECHAIN_BONDS:
                raise ValueError(
                    f"clash_bond_exclusion needs the covalent bonds of every residue, and "
                    f"residue {r} ({name}) has no bond template; known: "
                    f"{sorted(_SIDECHAIN_BONDS)}"
                )
            bonds = _BACKBONE_BONDS + (() if name == "GLY" else (("CA", "CB"),))
            for a1, a2 in bonds + _SIDECHAIN_BONDS[name]:
                link(r, a1, r, a2)
            if r + 1 in residue_name:
                link(r, "C", r + 1, "N")
        for pair in disulfides:
            if len(set(pair)) != 2 or any(
                    residue_name.get(r) != "CYS" or (r, "SG") not in node_of for r in pair):
                raise ValueError(
                    f"a disulfide bridge joins the SG atoms of two CYS residues, and "
                    f"{tuple(pair)} ({', '.join(str(residue_name.get(r)) for r in pair)}) "
                    "does not"
                )
            link(pair[0], "SG", pair[1], "SG")
        lonely = [f"{atom} of {residue_name[r]} {r}" for (r, atom), u in node_of.items()
                  if not neighbours[u]]
        if lonely:
            raise ValueError(
                "clash_bond_exclusion needs the covalent bonds of every atom, and no "
                f"template bond reaches {', '.join(lonely[:10])}"
            )

        m = len(node_of)
        dist = np.full((m, m), far, dtype=np.int16)
        for source in range(m):
            dist[source, source] = 0
            queue = deque([source])
            while queue:
                u = queue.popleft()
                if dist[source, u] >= max_bonds:
                    continue
                for v in neighbours[u]:
                    if dist[source, v] > dist[source, u] + 1:
                        dist[source, v] = dist[source, u] + 1
                        queue.append(v)
        out = np.full((n, n), far, dtype=np.int16)
        typed = atom_node >= 0
        nodes = atom_node[typed]
        out[np.ix_(typed, typed)] = dist[np.ix_(nodes, nodes)]
        return out

    @classmethod
    def _exclude_bonded_pairs(
        cls,
        ordered_atom_list: list[MCPUAtom],
        contact_mask: np.ndarray,
        clash_mask: np.ndarray,
        max_bonds: int,
        disulfides: Sequence[tuple[int, int]] = (),
    ) -> int:
        """Take every pair at most ``max_bonds`` covalent bonds apart out of
        ``clash_mask`` (in place), and return how many pairs that changed.

        Refuses (ValueError) if a pair that close through the residue
        templates' bonds is contact-eligible: only the clash check is to
        change, and the contact mask is left alone. A pair that close only
        through a bridge in ``disulfides`` may be contact-eligible, as a
        bridge can join residues far apart in sequence; it keeps its
        contacts, as mcpu08's rules keep those of CYS SG-SG pairs.
        """
        distances = cls.bond_distances(ordered_atom_list, max_bonds)
        near = distances <= max_bonds
        np.fill_diagonal(near, False)
        both = near & (contact_mask != 0)
        if np.any(both):
            i, j = (int(k) for k in np.argwhere(both)[0])
            a, b = ordered_atom_list[i], ordered_atom_list[j]
            raise ValueError(
                f"clash_bond_exclusion={max_bonds}: atoms {i} ({a.residue_name} "
                f"{a.residue_index} {a.name}) and {j} ({b.residue_name} {b.residue_index} "
                f"{b.name}) are {int(distances[i, j])} bonds apart and contact-eligible; "
                "only the clash check may change, so this is refused"
            )
        if len(disulfides):
            near = cls.bond_distances(ordered_atom_list, max_bonds, disulfides) <= max_bonds
            np.fill_diagonal(near, False)
        changed = int(np.count_nonzero(np.triu(near & (clash_mask != 0), 1)))
        clash_mask[near] = 0
        return changed

    @classmethod
    def build_topology_masks(
        cls,
        ordered_atom_list: list[MCPUAtom],
        n_atoms: int,
        skip_local_contact_range: int = 4,
        clash_bond_exclusion: int = 0,
        disulfides: Sequence[tuple[int, int]] = (),
    ) -> tuple[list[int], list[int]]:
        """Build the static (topology-only, coordinate-independent) per-atom-pair
        clash/contact masks the C++ MuPotential bakes into its pair-flag table.

        ``clash_mask[i,j] == 1`` means the pair is eligible to be scored as a
        hard-core clash (subject to the runtime distance check in the engine);
        ``contact_mask[i,j] == 1`` means the pair is eligible to be scored as a
        long-range mu-potential contact. A pair can be eligible for at most one
        of the two. Both masks are symmetric; the diagonal (i==j) is never
        evaluated by the engine and its value here is incidental.

        Special-cased categories, in the order they're checked below (each is
        reverse-engineered from legacy MCPU behaviour, not derived):

        * H atoms: zeroed out of both masks entirely.
        * ``res_diff == 0`` (same residue): BB-BB and SC-SC pairs never clash
          (covalently close / already accounted for elsewhere). A BB-SC pair
          is exempted from clash only for the direct peptide-adjacent
          backbone/CB bond (C/N/CA vs CB), for any pair within a PRO residue
          (ring geometry makes the generic BB/SC clash rule inapplicable), or
          for a backbone CA paired with a sidechain "G*"-named atom.
        * ``res_diff == 1`` (adjacent residues): the peptide-bond exemption is
          the N of the second (higher-index) residue clashing with the C/CA of
          the first -- except when the second residue is PRO, whose CD ring
          closure changes which atoms are bonded-adjacent, so the PRO-CD
          exemption is checked first and short-circuits the generic rule.
          Any other adjacent-residue heavy-atom pair, and any pair touching a
          sidechain atom, does clash.
        * ``1 < res_diff < skip_local_contact_range`` (near but not adjacent):
          clash-eligible only, never contact-eligible -- local density here is
          governed by the triplet/backbone potentials, not mu contacts.
        * ``res_diff > skip_local_contact_range`` (long range): CYS SG-SG
          pairs are contact-only (never clash), modeling a potential
          disulfide-like attractive interaction instead of a hard collision.
          All other long-range pairs are clash-eligible, and additionally
          contact-eligible unless both atoms are backbone (BB-BB long-range
          pairs are excluded from the mu contact term).

        ``clash_bond_exclusion`` 0 (the default) applies these rules alone.
        A positive value also takes out of the clash check every pair at most
        that many covalent bonds apart (see :meth:`bond_distances`), as
        molecular force fields leave bonded neighbours out of their
        non-bonded terms, with each disulfide in ``disulfides`` (residue index
        pairs, from :meth:`find_disulfides`) as a bond. The contact mask is
        not changed. ``disulfides`` is not used at 0.
        """
        clash_mask = np.ones((n_atoms, n_atoms), dtype=np.int8)
        contact_mask = np.zeros((n_atoms, n_atoms), dtype=np.int8)
        for i in range(n_atoms):
            atom_i = ordered_atom_list[i]
            if atom_i.name == 'H':
                clash_mask[i, :] = 0
                clash_mask[:, i] = 0
                contact_mask[i, :] = 0
                contact_mask[:, i] = 0
                continue
            for j in range(i + 1, n_atoms):
                atom_j = ordered_atom_list[j]
                if atom_j.name == 'H':
                    clash_mask[j, :] = 0
                    clash_mask[:, j] = 0
                    contact_mask[j, :] = 0
                    contact_mask[:, j] = 0
                    continue
                res_diff = abs(atom_i.residue_index - atom_j.residue_index)
                check_clash = 1
                check_contact = 0
                sc_i = cls._is_sidechain_for_eligibility(atom_i)
                sc_j = cls._is_sidechain_for_eligibility(atom_j)
                if res_diff == 0:
                    # Same residue: BB-BB / SC-SC never clash. A BB-SC pair is
                    # exempted only for the direct C/N/CA-CB bond, for any pair
                    # within PRO (ring geometry), or for CA paired with a
                    # sidechain "G*" atom.
                    if sc_i == sc_j:
                        check_clash = 0
                    else:
                        s_atom = atom_i if sc_i else atom_j
                        b_atom = atom_j if sc_i else atom_i
                        if b_atom.name in ['C', 'N', 'CA'] and s_atom.name == 'CB':
                            check_clash = 0
                        elif atom_i.residue_name == 'PRO':
                            check_clash = 0
                        elif b_atom.name == 'CA' and s_atom.name.startswith('G'):
                            # No side-chain atom name starts with 'G' (CG
                            # does not), so CA-CG and other two-bond pairs
                            # stay clash-eligible here; clash_bond_exclusion
                            # takes them out.
                            check_clash = 0

                elif res_diff == 1:
                    # Adjacent residues: the peptide bond (second residue's N
                    # to the first residue's C/CA) is exempted from clash,
                    # except when the second residue is PRO -- its ring
                    # closure via CD changes the bonded-adjacent atom, so that
                    # exemption is checked first and short-circuits this one.
                    i_first = atom_i.residue_index < atom_j.residue_index
                    first = atom_i if i_first else atom_j
                    second = atom_j if i_first else atom_i
                    first_sc = sc_i if i_first else sc_j
                    second_sc = sc_j if i_first else sc_i
                    is_pro_cd = (second.name == 'CD' and second.residue_name == 'PRO')
                    if is_pro_cd and first.name in ['C', 'CA']:
                        check_clash = 0
                    elif first.name == 'N' or second.name not in ['CA', 'N'] or first_sc or second_sc:
                        check_clash = 1
                    else:
                        check_clash = 0

                elif res_diff <= skip_local_contact_range:
                    # Near but not adjacent residues: clash-eligible only;
                    # local density here is handled by the triplet/backbone
                    # potentials, not the mu contact term. Legacy's
                    # CheckCorrelation (init.h) uses <= here, so res_diff ==
                    # skip_local_contact_range (e.g. the canonical i,i+4
                    # alpha-helix spacing) must land in this "near" bucket
                    # too, not fall through to the long-range contact branch.
                    check_clash = 1
                    check_contact = 0

                else:
                    # Long range: CYS SG-SG is contact-only (models a
                    # disulfide-like attraction, not a hard collision).
                    # Everything else is clash-eligible, and additionally
                    # contact-eligible unless both atoms are backbone.
                    if atom_i.residue_name == 'CYS' and atom_j.residue_name == 'CYS' and atom_i.name == 'SG' and atom_j.name == 'SG':
                        check_clash = 0
                        check_contact = 1
                    else:
                        check_clash = 1
                        check_contact = 1
                        if not sc_i and not sc_j:
                            check_contact = 0

                # Symmetric assignment
                clash_mask[i, j] = clash_mask[j, i] = check_clash
                contact_mask[i, j] = contact_mask[j, i] = check_contact

        if clash_bond_exclusion > 0:
            cls._exclude_bonded_pairs(
                ordered_atom_list, contact_mask, clash_mask, clash_bond_exclusion,
                disulfides)
        return contact_mask.flatten().tolist(), clash_mask.flatten().tolist()

    @classmethod
    def build(
        cls,
        atom_list: list[MCPUAtom],
        atom_to_residue: list[int],
        mu_potential_matrix: np.ndarray,
        atom_type_lookup: dict[tuple[str, str], tuple[int, float]],
        lambda_val: float = 1.8,
        alpha_val: float = 0.75,
        hard_core_alpha: float | None = None,
    ) -> mcpu_core.MuPotential:
        """Build the C++ Mu term for ``atom_list``.

        A contact-eligible pair is in contact closer than ``lambda_val *
        alpha_val`` times the sum of its atoms' radii (1.35 for mcpu08), and a
        clash-eligible pair may come no closer than ``hard_core_alpha`` times
        it, its hard core. ``hard_core_alpha`` None (the default) takes
        ``alpha_val``, as mcpu08 always has; another value moves the hard core
        alone and leaves the contact cutoff where it was.
        """
        if hard_core_alpha is None:
            hard_core_alpha = alpha_val
        atom_types = []
        atom_radii = []
        
        # 1. Assign atom types and radii
        for atom in atom_list:
            a_name = atom.name
            if atom.residue_name == 'GLY' and a_name == 'CA':
                r_name = 'GLY'
            else:
                r_name = 'XXX' if a_name in ['N', 'CA', 'C', 'O', 'OCT', 'OXT'] else atom.residue_name
                
            try:
                if a_name != 'H':
                    a_type, a_radius = atom_type_lookup[(r_name, a_name)]
                    atom_types.append(a_type)
                    atom_radii.append(a_radius)
                else:
                    atom_types.append(-1)
                    atom_radii.append(0.0)
            except KeyError:
                raise ValueError(f"Atom type not found for residue {r_name} and atom {a_name}")

        types_arr = np.array(atom_types, dtype=np.int32)
        radii_arr = np.array(atom_radii, dtype=np.float32)

        # 2. Vectorized initialization of physical matrices
        rad_sum = radii_arr[:, np.newaxis] + radii_arr[np.newaxis, :]
        hard_sq_mat = (hard_core_alpha * rad_sum) ** 2
        dist_sq_mat = (lambda_val * alpha_val * rad_sum) ** 2

        global _MAX_CONTACT_PRINTED
        if _MU_BUILDER_VERBOSE and not _MAX_CONTACT_PRINTED:
            print("Maximum contact distance (squared):", np.max(dist_sq_mat))
            _MAX_CONTACT_PRINTED = True
        
        # Safe indexing (handle -1 for H safely by setting dummy energies to 0)
        safe_types = np.where(types_arr < 0, 0, types_arr)
        energy_mat = mu_potential_matrix[np.ix_(safe_types, safe_types)]
        if np.any(types_arr < 0):
            energy_mat[types_arr < 0, :] = 0.0
            energy_mat[:, types_arr < 0] = 0.0

        # 3. Return the base C++ object (Step 1 of the Hybrid Approach)
        from pymcpu import mcpu_core
        return mcpu_core.MuPotential(
            energy_mat,
            dist_sq_mat,
            hard_sq_mat,
            atom_types,
            list(atom_to_residue)
        )