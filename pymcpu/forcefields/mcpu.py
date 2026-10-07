"""MCPU force field: builds the C++ mcpu_core ``System`` from an MDTraj trajectory.

``MCPUForceField`` loads the MCPU parameter set, reorders atoms into the
global layout the C++ engine expects (all backbone N/CA/C, then all
backbone O, then all sidechains), infers or defers amide-hydrogen
placement, and derives the per-residue block bookkeeping (``BlockIndices``,
``DownstreamCache``) the engine needs. :meth:`MCPUForceField.create_system`
then delegates to the potential builders in ``pymcpu.forcefields.builders``
to assemble a ready-to-use ``mcpu_core.System`` (Mu, backbone/sidechain
triplet, hydrogen-bond, and aromatic potentials).
"""

from __future__ import annotations

import json
import numpy as np
import mdtraj as md
from dataclasses import dataclass
from pathlib import Path
import logging

from pymcpu import mcpu_core
from pymcpu.mcpu_core import DownstreamCache
from pymcpu.params import ParamsError, ensure_params, optional_files, required_files


from .base import BaseForceField

# Import your delegate builders
from .builders.mu_builder import MuPotentialBuilder
from .builders.triplet_builder import TripletPotentialBuilder, SidechainTripletBuilder
from .builders.hbond_builder import AMINO_INDEX, HydrogenBondBuilder
from .builders.aromatic_builder import AromaticPotentialBuilder
from .builders.rotamer_builder import RotamerLibraryBuilder
from .builders.rama_mixture_builder import RamaMixtureLibraryBuilder

logger = logging.getLogger(__name__)

_MU_LAMBDA_DEFAULT = 1.8
_MU_ALPHA_DEFAULT  = 0.75
_DEFAULT_PARAM_SET = "mcpu08"

@dataclass
class MCPUAtom:
    """One engine atom: the topology atom it came from, its name and residue.

    ``original_index`` is the atom's index in the topology, or -1 for an
    explicit amide hydrogen, which the topology does not have.
    """

    original_index: int
    name: str
    residue_name: str
    residue_index: int


class MCPUForceField(BaseForceField):
    """The MCPU force field: all-atom, with the five MCPU knowledge-based potentials.

    Build it once from a structure; it loads the parameters, checks the
    structure and puts the atoms in the engine's order. Then build systems
    with :meth:`create_system`. :attr:`inverse_mapping` maps the engine's atom
    order back to the topology's, for reporters.
    """

    def __init__(self,
             trajectory: md.Trajectory,
             param_dir: str | None = None,
             param_set: str = _DEFAULT_PARAM_SET,
             virtual_amide_h: bool = True,
             compute_dssp: bool = False,
             dssp_coil_state: str = "C",
             allow_provisional_rama: bool = False) -> None:
        """
        Load the parameters, check the structure and lay out its atoms.

        By default no amide hydrogens are added: the hydrogen-bond potential
        places virtual ones from the backbone (N, CA and the previous C)
        whenever it evaluates, so they always match the current structure.

        Parameters
        ----------
        trajectory
            The structure, with heavy atoms only (see :meth:`prepare_trajectory`).
        param_dir
            Explicit parameter root (``constants/`` + ``mcpu_params/``).
            If omitted, resolves via :func:`pymcpu.params.ensure_params`.
        param_set
            Registry set name used when ``param_dir`` is omitted (default ``mcpu08``).
        virtual_amide_h
            If True (default), use virtual amide hydrogens as above. If False,
            add explicit amide hydrogens to the engine's atoms instead.
        compute_dssp
            If True, compute per-residue secondary structure via
            ``mdtraj.compute_dssp(trajectory, simplified=True)`` (frame 0) and feed
            it to the H-bond potential's secondary-structure gates. Default False:
            every residue counts as coil (``'C'``).
        dssp_coil_state
            Legacy secstr state to map DSSP's simplified ``'C'``/``'NA'`` codes onto
            (only used when ``compute_dssp=True``). Default ``'C'``; legacy's other
            coil-like state ``'L'`` ("confident predicted coil") has no DSSP
            equivalent and is never inferred automatically.
        allow_provisional_rama
            If True, permit loading rama-mixture categories marked
            ``"provisional"`` in ``rama_mixture.json`` (default False:
            raises rather than silently using an unvalidated fit).
        """
        if param_dir is None:
            try:
                param_dir = ensure_params(param_set)
            except ParamsError as exc:
                raise FileNotFoundError(str(exc)) from exc
        self.param_dir = Path(param_dir)
        self.param_set = param_set
        self.virtual_amide_h = bool(virtual_amide_h)
        self.compute_dssp = bool(compute_dssp)
        self.dssp_coil_state = dssp_coil_state
        self.allow_provisional_rama = bool(allow_provisional_rama)
        self._load_parameters()
        #: The topology the engine actually simulates. Identical in shape to
        #: the input here, unlike KORPForceField's -- but exposed by both so a
        #: caller can write trajectories without knowing which it has.
        self.output_topology = trajectory.topology
        self._canonicalize_residue_names(trajectory.topology)
        self._validate_topology(trajectory.topology)
        self.secondary_structure = self._compute_secondary_structure(trajectory)
        self._order_atoms(trajectory.topology)
        self.coords = self._infer_hydrogens(trajectory)
        self._initialize_attributes(trajectory.topology)

    @classmethod
    def prepare_trajectory(cls, trajectory: md.Trajectory) -> md.Trajectory:
        """Return the trajectory without hydrogens; MCPU uses heavy atoms only.

        A trajectory that has no hydrogens comes back unchanged.
        """
        return trajectory.atom_slice(
            trajectory.topology.select("not element H"))

    def _infer_hydrogens(self, trajectory: md.Trajectory) -> np.ndarray:
        """
        Optionally append explicit amide H atoms (virtual_amide_h=False escape hatch).

        Geometry matches legacy CheckHBond(): H = N - normalize((CA-N)+(Cprev-N)),
        with N–H = 0.1 nm (= 1.0 Å).
        When virtual_amide_h=True (default), only reorder heavy atoms — no H segment.
        """
        coords = trajectory.xyz
        ordered_coords = coords[:, self.ordered_indices, :]
        count_non_pro = 0

        for res_idx in range(1, trajectory.topology.n_residues):
            residue = trajectory.topology.residue(res_idx)
            if residue.name == "PRO":
                continue
            count_non_pro += 1

        if self.virtual_amide_h:
            self.total_h_atoms = 0
            return ordered_coords

        h_count = 0
        for res_idx in range(1, trajectory.topology.n_residues):
            residue = trajectory.topology.residue(res_idx)
            if residue.name == "PRO":
                continue

            c_idx = trajectory.topology.select(f"resid {res_idx-1} and name C")[0]
            n_idx = trajectory.topology.select(f"resid {res_idx} and name N")[0]
            ca_idx = trajectory.topology.select(f"resid {res_idx} and name CA")[0]

            c_pos = coords[:, c_idx, :]
            n_pos = coords[:, n_idx, :]
            ca_pos = coords[:, ca_idx, :]

            # Legacy: normalize(sum of unnormalized vectors), not sum of unit vectors.
            bisect_vector = (c_pos - n_pos) + (ca_pos - n_pos)
            bisect_vector_norm = bisect_vector / np.linalg.norm(
                bisect_vector, axis=1, keepdims=True
            )
            h_pos = n_pos - bisect_vector_norm * 0.1
            ordered_coords = np.concatenate(
                (ordered_coords, h_pos[:, np.newaxis, :]), axis=1
            )

            self.ordered_atom_list.append(
                MCPUAtom(
                    original_index=-1,
                    name="H",
                    residue_name=residue.name,
                    residue_index=residue.index,
                )
            )
            h_count += 1

        self.total_h_atoms = h_count
        assert h_count == count_non_pro
        return ordered_coords

    def _compute_secondary_structure(self, trajectory: md.Trajectory) -> str:
        """Per-residue legacy secstr string (H/E/C), residue-index order.

        Returns "" when ``compute_dssp`` is off -- ``System.secondary_structure``
        treats an empty string as all-'C', so this is byte-identical to today
        when the feature isn't requested.
        """
        if not self.compute_dssp:
            return ""
        codes = md.compute_dssp(trajectory[0], simplified=True)[0]
        return "".join(c if c in ("H", "E") else self.dssp_coil_state for c in codes)

    def _validate_topology(self, topology: md.Topology) -> None:
        """Raise early if topology contains residues MCPU cannot score."""
        unknown = sorted({
            res.name for res in topology.residues
            if res.name not in self.ff_template
        })
        if not unknown:
            return
        hints = []
        if "MSE" in unknown:
            hints.append(
                "MSE (selenomethionine) is not a rename: its SD is replaced by "
                "SE, so both the atom name and the atom type differ from MET. "
                "To model it as methionine, rename the residue to MET *and* its "
                "SE atom to SD, accepting sulfur parameters for selenium."
            )
        if {"SEP", "TPO", "PTR"} & set(unknown):
            hints.append(
                "Phosphorylated residues (SEP/TPO/PTR) carry extra P/O atoms "
                "with no MCPU atom type and cannot be represented."
            )
        raise ValueError(
            "Topology contains residues MCPU has no parameters for: "
            f"{unknown}. Known residues: {sorted(self.ff_template)}. "
            "Protonation-state variants (HSD/HSE/HSP/HID/HIE/HIP, CYX/CYM, "
            "ASH/GLH/LYN/ARN) are renamed automatically; anything else must be "
            "resolved before building the system."
            + ("" if not hints else " " + " ".join(hints))
        )

    def _load_parameters(self) -> None:
        root = Path(self.param_dir)
        # Driven by the registry `layout` block, NOT a local list: this loader
        # and ensure_params' completeness check must agree by construction, or
        # ensure_params can succeed and then this raises FileNotFoundError.
        required_paths = {
            role: root / rel
            for role, rel in required_files(self.param_set).items()
        }
        # The knowledge-based (rama-mixture) library is OPTIONAL: the move it
        # serves defaults to pivot_rama_probability = 0.0, so a parameter set
        # without constants/rama_mixture.json is perfectly valid and must load.
        # Making it required would break every pre-existing param dir (e.g. the
        # smoothed-triplet sets under p0.4) for a move that is off by default.
        # .get(), not []: optional_files() returns {} for a registry set with
        # no `optional` block (pymcpu/params.py returns
        # `dict(_layout(set_name).get("optional") or {})`), so indexing raised
        # KeyError: 'rama mixture' from this constructor. That is precisely the
        # first thing an outside developer hits when they add their own
        # parameter set declaring only `required`/`tables`/`constants` -- the
        # workflow this project wants to support. The shipped mcpu08 set does
        # declare it, so the bug was invisible in CI.
        _optional = optional_files(self.param_set)
        _rama_rel = _optional.get("rama mixture")
        rama_mixture_path = (root / _rama_rel) if _rama_rel else None
        for name, path in required_paths.items():
            if not path.exists():
                raise FileNotFoundError(
                    f"MCPU parameter file not found: {name}\n"
                    f"Expected at: {path}\n"
                    f"Hint: export MCPU_PARAMS_DIR or run `mcpu materialize-params` "
                    f"(see pymcpu.params.ensure_params)."
                )
        with open(required_paths["amino acids template"], "r") as f:
            self.ff_template = json.load(f)
        self.atom_type_lookup = MuPotentialBuilder.load_atom_types(
            str(required_paths["atom types"])
        )
        self.mu_energies = MuPotentialBuilder.load_parameters(
            str(required_paths["mu potentials"])
        )
        self.bb_triplet = TripletPotentialBuilder.load_parameters(
            str(required_paths["bb triplet"])
        )
        self.sc_triplet = SidechainTripletBuilder.load_parameters(
            str(required_paths["sc triplet"])
        )
        self.hbond = HydrogenBondBuilder.load_parameters(
            str(required_paths["hbond"])
        )
        self.hbond_seq_dep = HydrogenBondBuilder.load_seq_dep_parameters(
            str(required_paths["hbond seq_dep"])
        )
        self.aromatic = AromaticPotentialBuilder.load_parameters(
            str(required_paths["aromatic"])
        )
        self.rotamer_lib_raw = RotamerLibraryBuilder.load_parameters(
            str(required_paths["rotamer library"])
        )
        self.rama_mixture_raw = (
            RamaMixtureLibraryBuilder.load_parameters(
                str(rama_mixture_path),
                allow_provisional=self.allow_provisional_rama,
            )
            if rama_mixture_path is not None and rama_mixture_path.is_file()
            else None
        )
        if self.rama_mixture_raw is None:
            logger.info(
                "No %s; the knowledge-based backbone pivot will be unavailable "
                "(set pivot_rama_probability > 0 only with a param set that has it).",
                rama_mixture_path.name
                if rama_mixture_path is not None
                else "rama mixture (not declared by this parameter set)",
            )

    def _initialize_attributes(self, topology: md.Topology) -> None:
        # --- From ordered_indices initialize per atom traits ---
        self.atom_to_res = [atom.residue_index for atom in self.ordered_atom_list]
        self.n_atoms = len(self.ordered_atom_list)
        self.n_res = len(set(self.atom_to_res))

        # --- Build BlockIndices for each residue based on the new global atom order ---
        self.blocks = [mcpu_core.BlockIndices() for _ in range(self.n_res)]
        for atom_idx, atom in enumerate(self.ordered_atom_list):
            res_idx = atom.residue_index
            if atom.name == "N":
                self.blocks[res_idx].bb_start = atom_idx
            elif atom.name == "C":
                self.blocks[res_idx].c_start = atom_idx
            elif atom.name == "O":
                self.blocks[res_idx].o_start = atom_idx
            elif atom.name == "CB":
                # GLY has no CB, so its block keeps sc_start = -1: no
                # sidechain atoms. sc_count and first_sc_of_residue below
                # handle that.
                self.blocks[res_idx].sc_start = atom_idx
            elif atom.name == "H":
                self.blocks[res_idx].h_start = atom_idx

        # Mark amide donors (non-PRO, residue index >= 1). Used when H is virtual.
        self.is_proline = [0] * self.n_res
        # legacy pdb_util.h GetAminoNumber() index, for HBondPotential's seq_hb table.
        # Unrecognized residue names (e.g. LNK) fall back to index 0 (ALA); their
        # H-bond energy is unaffected in practice since linker residues are excluded
        # from energy evaluation elsewhere.
        self.amino_index = [0] * self.n_res
        for res_idx, residue in enumerate(topology.residues):
            if residue.name == "PRO":
                self.is_proline[res_idx] = 1
            if res_idx >= 1 and residue.name != "PRO":
                self.blocks[res_idx].amide_donor = True
            if self.blocks[res_idx].h_start >= 0:
                self.blocks[res_idx].amide_donor = True
            self.amino_index[res_idx] = AMINO_INDEX.get(residue.name, 0)

        # Per-residue chi-atom-index table: chi_atom_indices[res_idx][k] =
        # [i1, i2, i3, i4] (engine atom indices for chi k's 4 defining atoms),
        # or [-1, -1, -1, -1] for k >= that residue's ntorsions. Built from
        # real per-residue atom-NAME topology (standard_amino_acids.json's
        # "chi_atoms", transcribed from legacy's amino_torsion.data) rather
        # than assuming array-position order matches the true dihedral
        # chain -- positional offsets get this wrong for any branched
        # sidechain (e.g. ILE's chi2 is CA-CB-CG1-CD1, not CA-CB-CG1-CG2,
        # the 3rd stored sidechain atom).
        atoms_by_residue: list[dict[str, int]] = [dict() for _ in range(self.n_res)]
        for atom_idx, atom in enumerate(self.ordered_atom_list):
            atoms_by_residue[atom.residue_index][atom.name] = atom_idx

        self.chi_atom_indices: list[list[list[int]]] = []
        for res_idx, residue in enumerate(topology.residues):
            chi_rows = self.ff_template.get(residue.name, {}).get("chi_atoms", [])
            names_to_idx = atoms_by_residue[res_idx]
            rows = []
            for k in range(4):
                if k < len(chi_rows):
                    rows.append([names_to_idx[name] for name in chi_rows[k]])
                else:
                    rows.append([-1, -1, -1, -1])
            self.chi_atom_indices.append(rows)

        # Per-residue chi-moved-atom-range table (rotamer-library sidechain
        # move only): chi_moved_atom_ranges[res_idx][k] = [lo, hi) -- the
        # half-open engine-index range of atoms distal to (rotated by)
        # residue res_idx's k-th chi bond, or [-1, -1] for k >= ntorsions.
        # Sourced from standard_amino_acids.json's "chi_moved_atoms"
        # (transcribed from legacy's amino_torsion.data section 3). This is
        # NOT a positional "[sc_start+k+1, end)" formula -- e.g. ILE's chi2
        # moves only CD1 (a single atom that sits after CG2 in storage
        # order, not "everything from CG2 onward") -- so each chi's set is
        # resolved independently from real atom names, with contiguity
        # asserted here as the safety net for the assumption that this
        # named atom set forms a contiguous run under the engine's current
        # atom ordering.
        self.chi_moved_atom_ranges: list[list[list[int]]] = []
        for res_idx, residue in enumerate(topology.residues):
            moved_rows = self.ff_template.get(residue.name, {}).get("chi_moved_atoms", [])
            names_to_idx = atoms_by_residue[res_idx]
            ranges = []
            for k in range(4):
                if k < len(moved_rows):
                    indices = [names_to_idx[name] for name in moved_rows[k]]
                    lo, hi = min(indices), max(indices) + 1
                    if hi - lo != len(indices):
                        raise ValueError(
                            f"{residue.name} (residue {res_idx}) chi{k}'s moved-atom "
                            f"set {moved_rows[k]} is not contiguous in engine atom "
                            f"order (resolved indices={sorted(indices)}); the "
                            f"rotamer-library sidechain move's per-chi cascading "
                            f"rotation requires a contiguous range."
                        )
                    ranges.append([lo, hi])
                else:
                    ranges.append([-1, -1])
            self.chi_moved_atom_ranges.append(ranges)

        # Build downstream cache
        # (store the global indices of the first atom in each block for quick access in C++)
        sc_end = self.total_bb_atoms + self.total_o_atoms + self.total_sc_atoms
        first_sc_of_residue = []
        first_o_of_residue = []
        first_h_of_residue = []
        for res_idx in range(self.n_res):
            block = self.blocks[res_idx]
            if block.bb_start == -1:
                raise ValueError(f"Backbone start index not set for residue {res_idx}. Check if N, CA, C atoms are correctly identified and ordered.")
            if block.o_start == -1:
                raise ValueError(f"Oxygen start index not set for residue {res_idx}. Check if O atoms are correctly identified and ordered.")
            else:
                first_o_of_residue.append(block.o_start)
            if block.sc_start == -1:
                next_valid_sc = sc_end
                for lookahead_idx in range(res_idx + 1, self.n_res):
                    if self.blocks[lookahead_idx].sc_start != -1:
                        next_valid_sc = self.blocks[lookahead_idx].sc_start
                        break
                first_sc_of_residue.append(next_valid_sc)
            else:
                first_sc_of_residue.append(block.sc_start)
            if block.h_start == -1:
                next_valid_h = self.n_atoms
                for lookahead_idx in range(res_idx + 1, self.n_res):
                    if self.blocks[lookahead_idx].h_start != -1:
                        next_valid_h = self.blocks[lookahead_idx].h_start
                        break
                first_h_of_residue.append(next_valid_h)
            else:
                first_h_of_residue.append(block.h_start)

        # Contiguous SC atom counts (required for atom reorder / residue-local
        # moves): a residue's sidechain runs up to where the next one starts.
        for res_idx, block in enumerate(self.blocks):
            if block.sc_start >= 0:
                next_sc = first_sc_of_residue[res_idx + 1] if res_idx + 1 < self.n_res else sc_end
                block.sc_count = next_sc - block.sc_start

        self.downstream = DownstreamCache()
        self.downstream.first_sc_of_residue = np.array(first_sc_of_residue, dtype=np.int32)
        self.downstream.first_o_of_residue = np.array(first_o_of_residue, dtype=np.int32)
        self.downstream.first_h_of_residue = np.array(first_h_of_residue, dtype=np.int32)

    def create_system(self, topology: md.Topology) -> mcpu_core.System:
        """
        Build a System with the five MCPU potentials for this structure.

        Pass the topology the force field was built from. Call it once for
        each simulation; every call returns a new System.
        """
        system = mcpu_core.System(self.n_atoms, self.n_res)
        system.set_block_indices(self.blocks)
        system.atom_to_residue = self.atom_to_res
        # Index ff_template directly rather than via .get(name, {}).get(...,0).
        # The old default could not fire -- _validate_topology() (called from
        # __init__) already rejects any residue name absent from ff_template --
        # but it read as though unknown names were tolerated, when silently
        # treating one as zero-chi would now DROP it from the sidechain-torsion
        # energy entirely (that term skips zero-chi residues, matching legacy
        # sctenergy()). Direct indexing states the invariant instead of hiding it.
        ntorsions = [
            self.ff_template[topology.residue(i).name]["ntorsions"]
            for i in range(self.n_res)
        ]
        logger.debug("Number of torsions per residue:")
        logger.debug(ntorsions)
        system.set_torsions_per_residue(ntorsions)
        system.set_chi_atom_indices(self.chi_atom_indices)
        system.set_chi_moved_atom_ranges(self.chi_moved_atom_ranges)
        system.set_rotamer_library(RotamerLibraryBuilder.build(self.rotamer_lib_raw))
        if self.rama_mixture_raw is not None:
            system.set_rama_mixture_library(
                RamaMixtureLibraryBuilder.build(self.rama_mixture_raw)
            )
        system.set_atom_counts(self.total_bb_atoms, self.total_o_atoms, self.total_sc_atoms, self.total_h_atoms)
        system.set_virtual_amide_h(self.virtual_amide_h)
        system.set_downstream_cache(self.downstream)
        system.set_is_proline([int(x) for x in self.is_proline])
        system.set_amino_index([int(x) for x in self.amino_index])
        system.set_secondary_structure(self.secondary_structure)
        # The loop-closure (KIC) move takes its bond lengths, bond angles and
        # omegas from the START structure -- the exact float32 coordinates every
        # replica is positioned with ((coords[0] * 10).T as float32) -- never
        # from the moving chain. Built from the PDB, so replica exchange and
        # checkpoint restores cannot change it. KIC refuses to run without it.
        system.set_kic_reference((self.coords[0] * 10.0).T.astype(np.float32))

        # --- 3. Build Forces via Delegate Builders ---
        # (Pass ordered_atom_list instead of topology.atoms so builders use the new order)
        mu_potential = MuPotentialBuilder.build(
            atom_list=self.ordered_atom_list, 
            atom_to_residue=self.atom_to_res,
            mu_potential_matrix=self.mu_energies,
            atom_type_lookup=self.atom_type_lookup,
            lambda_val=_MU_LAMBDA_DEFAULT,
            alpha_val=_MU_ALPHA_DEFAULT
        )
        topo_contact, topo_clash = MuPotentialBuilder.build_topology_masks(
            self.ordered_atom_list,
            self.n_atoms
        )
        mu_potential.cache_necessary_data(
            topo_contact, 
            topo_clash, 
            self.coords[0].T * 10.0  # Pass coordinates in Angstroms
        )
        mu_potential.set_energy_group(1)
        mu_potential.set_name("mu")
        system.add_potential(mu_potential)
        
        bb_potential = TripletPotentialBuilder.build(
            atom_list=self.ordered_atom_list,
            reshaped_params=self.bb_triplet
        )
        bb_potential.set_energy_group(2)
        bb_potential.set_name("backbone_torsion")
        system.add_potential(bb_potential)

        sc_potential = SidechainTripletBuilder.build(
            atom_list=self.ordered_atom_list,
            reshaped_params=self.sc_triplet
        )
        sc_potential.set_energy_group(3)
        sc_potential.set_name("sidechain_torsion")
        system.add_potential(sc_potential)

        hbond_potential = HydrogenBondBuilder.build(
            raw_params=self.hbond,
            seq_dep_params=self.hbond_seq_dep,
        )
        hbond_potential.set_energy_group(4)
        hbond_potential.set_name("hydrogen_bond")
        system.add_potential(hbond_potential)

        # One ring entry per aromatic residue (PHE / TRP; TYR excluded to match
        # the reference aromatic_noTYR table). The three engine-internal atom
        # indices define the ring centroid and plane normal:
        #   index 0 = CG, indices 1/2 = the two outer ring atoms.
        RING_ATOMS = {
            "PHE": ("CG", "CE1", "CE2"),
            "TRP": ("CG", "CZ2", "CZ3"),
        }
        rings_by_residue: dict[int, dict[str, int]] = {}
        residue_name_by_index: dict[int, str] = {}
        for i, atom in enumerate(self.ordered_atom_list):
            ring_names = RING_ATOMS.get(atom.residue_name)
            if ring_names is None or atom.name not in ring_names:
                continue
            rings_by_residue.setdefault(atom.residue_index, {})[atom.name] = i
            residue_name_by_index[atom.residue_index] = atom.residue_name

        aromatic_atom_indices = []
        for res_idx in sorted(rings_by_residue):
            ring_names = RING_ATOMS[residue_name_by_index[res_idx]]
            found = rings_by_residue[res_idx]
            missing = [name for name in ring_names if name not in found]
            if missing:
                logger.warning(
                    "Skipping aromatic residue %d (%s): missing ring atoms %s",
                    res_idx, residue_name_by_index[res_idx], missing,
                )
                continue
            aromatic_atom_indices.append(
                np.array([found[name] for name in ring_names], dtype=np.int32)
            )

        aromatic_potential = AromaticPotentialBuilder.build(
            aromatic_atom_indices=aromatic_atom_indices,
            reshaped_params=self.aromatic
        )
        aromatic_potential.set_energy_group(5)
        aromatic_potential.set_name("aromatic")
        system.add_potential(aromatic_potential)
        
        return system
    
    def _order_atoms(self, topology: md.Topology) -> None:
        """Internal helper to map MDTraj atoms to MCPU's required global order:
           All (N, CA, C), then All (O), then All Sidechains.
        """
        bb_indices = []
        o_indices = []
        sc_indices = []

        for residue in topology.residues:
            expected_order = self.ff_template.get(residue.name).get("atoms")
            if not expected_order:
                raise ValueError(f"ForceField does not recognize residue: {residue.name}")

            current_atoms = {atom.name: atom for atom in residue.atoms}
            
            # 1. All Backbone: N, CA, C
            for atom_name in ["N", "CA", "C"]:
                if atom_name in current_atoms:
                    bb_indices.append(current_atoms[atom_name].index)
            
            # 2. All Backbone Oxygen: O (and terminal oxygens)
            for atom_name in ["O", "OXT", "OCT"]:
                if atom_name in current_atoms:
                    o_indices.append(current_atoms[atom_name].index)
            
            # 3. All Sidechains (preserving template order for the specific residue)
            for atom_name in expected_order:
                if atom_name not in ["N", "CA", "C", "O", "OXT", "OCT"]:
                    if atom_name in current_atoms:
                        sc_indices.append(current_atoms[atom_name].index)
        self.total_bb_atoms = len(bb_indices)
        self.total_o_atoms = len(o_indices)
        self.total_sc_atoms = len(sc_indices)

        self.ordered_indices = bb_indices + o_indices + sc_indices

        self.ordered_atom_list = []
        for idx in self.ordered_indices:
            atom = topology.atom(idx)
            self.ordered_atom_list.append(
                MCPUAtom(
                    original_index=atom.index,
                    name=atom.name,
                    residue_name=atom.residue.name,
                    residue_index=atom.residue.index,
                )
            )
    
    @property
    def inverse_mapping(self) -> list[int]:
        """
        Topology atom index of each engine atom, in engine order.

        -1 for an explicit amide hydrogen, which the topology does not have;
        the XTC reporter skips those. Pass this to a reporter to write frames
        in topology order. ``xyz[inverse_mapping]`` puts a frame from the
        topology's order into the engine's.
        """
        return [atom.original_index for atom in self.ordered_atom_list]
        