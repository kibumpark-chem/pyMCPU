"""Builder for the Mu (pairwise atom-atom contact) potential.

Assembles the C++ ``mcpu_core.MuPotential`` energy term: loads the MCPU08
atom-type/radius lookup table and the raw contact-energy matrix, computes the
coordinate-independent Layer-1 clash/contact eligibility masks and per-atom
metadata consumed by the C++ engine, and includes a Python port of the
relevant parts of legacy MCPU's ``pdb_util.h`` atom-classification logic that
those computations depend on.
"""

from __future__ import annotations

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


class MuPotentialBuilder:
    """Stateless namespace of classmethods that build the Mu contact potential.

    Never instantiated -- callers use the classmethods directly (see
    ``pymcpu.forcefields.mcpu.MCPUForceField``, which drives the full
    pipeline: ``load_atom_types``/``load_parameters`` to read the on-disk
    atom-type table and MCPU08 energy matrix, ``build_topology_masks`` and
    ``layer1_atom_meta`` to derive the coordinate-independent per-atom-pair
    eligibility masks and per-atom role metadata consumed by the C++ engine's
    Layer-1 gating, and ``build`` to assemble all of the above into a ready
    ``mcpu_core.MuPotential``.
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
    def build_topology_masks(
        cls,
        ordered_atom_list: list[MCPUAtom],
        n_atoms: int,
        skip_local_contact_range: int = 4
    ) -> tuple[list[int], list[int]]:
        """Build the static (topology-only, coordinate-independent) per-atom-pair
        clash/contact masks consumed by the C++ MuPotential's Layer-1 gating.

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

        return contact_mask.flatten().tolist(), clash_mask.flatten().tolist()

    # MuAtomRole / MuResClass enums — must match MuPotential.h
    _ROLE_OTHER = 0
    _ROLE_H = 1
    _ROLE_N = 2
    _ROLE_CA = 3
    _ROLE_C = 4
    _ROLE_O = 5
    _ROLE_CB = 6
    _ROLE_CD = 7
    _ROLE_SG = 8
    _ROLE_GX = 9
    _RES_OTHER = 0
    _RES_PRO = 1
    _RES_CYS = 2

    @classmethod
    def layer1_atom_meta(
        cls, ordered_atom_list: list[MCPUAtom]
    ) -> tuple[list[int], list[int], list[int], list[int]]:
        """Per-atom Layer 1 arrays for MuPotential.set_topology_atom_meta.

        Returns (res_index, is_sidechain, atom_role, res_class).
        """
        res_index = [int(a.residue_index) for a in ordered_atom_list]
        is_sidechain = [1 if cls._is_sidechain_for_eligibility(a) else 0 for a in ordered_atom_list]
        atom_role: list[int] = []
        res_class: list[int] = []
        for atom in ordered_atom_list:
            name = atom.name
            if name == "H":
                role = cls._ROLE_H
            elif name == "N":
                role = cls._ROLE_N
            elif name == "CA":
                role = cls._ROLE_CA
            elif name == "C":
                role = cls._ROLE_C
            elif name in ("O", "OCT", "OXT"):
                role = cls._ROLE_O
            elif name == "CB":
                role = cls._ROLE_CB
            elif name == "CD":
                role = cls._ROLE_CD
            elif name == "SG":
                role = cls._ROLE_SG
            elif name.startswith("G"):
                role = cls._ROLE_GX
            else:
                role = cls._ROLE_OTHER
            atom_role.append(role)
            if atom.residue_name == "PRO":
                res_class.append(cls._RES_PRO)
            elif atom.residue_name == "CYS":
                res_class.append(cls._RES_CYS)
            else:
                res_class.append(cls._RES_OTHER)
        return res_index, is_sidechain, atom_role, res_class

    @classmethod
    def build(
        cls,
        atom_list: list[MCPUAtom],
        atom_to_residue: list[int],
        mu_potential_matrix: np.ndarray,
        atom_type_lookup: dict[tuple[str, str], tuple[int, float]],
        lambda_val: float = 1.8,
        alpha_val: float = 0.75
    ) -> mcpu_core.MuPotential:
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
        hard_sq_mat = (alpha_val * rad_sum) ** 2
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