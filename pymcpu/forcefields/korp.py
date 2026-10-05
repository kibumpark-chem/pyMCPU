"""KORP: a backbone-only force field built on the 6D orientational potential.

KORP scores residue **pairs** from a local frame built on each residue's own
N, CA and C, with the pair coordinate being CA-CA. It reads no sidechain atom
at all, so rather than carry sidechains along unused this force field drops
them: the engine sees N, CA, C and O only.

Why O is kept even though KORP never reads it
---------------------------------------------
Purely so the existing move machinery and segment bookkeeping work unchanged.
The engine's atom layout is ``[N,CA,C][O][sidechains]``, and pivot, KIC and the
``DownstreamCache`` are all written against that shape. Keeping O costs one
atom per residue and avoids special-casing any of them.

Two consequences worth knowing before you run this
--------------------------------------------------
* **The trajectory you get back is not the trajectory you put in.** Sidechain
  atoms are gone, so an XTC written from this force field must be loaded
  against :attr:`KORPForceField.output_topology`, not against your input PDB's
  topology.
* **Sidechain moves must be switched off.** These residues have no chi angles,
  so every sidechain proposal would return without proposing anything. Call
  ``integrator.set_move_weights(pivot, kic, 0.0)``; the engine raises rather
  than silently discarding that share of the run.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import mdtraj as md
import numpy as np

from pymcpu import mcpu_core
from pymcpu.forcefields.base import BaseForceField
from pymcpu.forcefields.builders.korp_builder import KorpPotentialBuilder
from pymcpu.forcefields.korp_map import KORP_RESIDUE_ORDER, KorpMapError, load_korp_map

logger = logging.getLogger(__name__)

__all__ = ["KORPForceField"]

#: Energy groups. 7 is the last one the engine still times
#: (``Context::energy_delta_ns_`` is ``[8]``), so the expensive term takes it.
#: Group 4 is deliberately avoided: ``EnergyWeights::weight_for_group``
#: silently multiplies it by 2.0. Both default to an outer weight of 1.0.
KORP_ENERGY_GROUP = 7
STERIC_ENERGY_GROUP = 8

#: mdtraj's ``backbone`` keyword excludes OXT, so the atom names are explicit.
_BACKBONE_SELECTION = (
    "protein and (name N or name CA or name C "
    "or name O or name OXT or name OCT)"
)

_FRAME_ATOMS = ("N", "CA", "C")

#: Protonation-state spellings that are the same molecule in heavy atoms.
#: Only the ones that matter for a backbone-only model -- nothing here changes
#: N, CA, C or O, so the rename is exact.
_RESIDUE_ALIASES = {
    "HSD": "HIS", "HSE": "HIS", "HSP": "HIS",
    "HID": "HIS", "HIE": "HIS", "HIP": "HIS",
    "CYX": "CYS", "CYM": "CYS",
    "ASH": "ASP", "GLH": "GLU", "LYN": "LYS", "ARN": "ARG",
    "HISD": "HIS", "HISE": "HIS",
}


def _keep_source_identity(source: md.Topology, kept, target: md.Topology) -> None:
    """Put back the chain IDs and residue numbers ``Topology.subset`` loses.

    ``target`` is ``source.subset(kept)`` or the topology of
    ``atom_slice(kept)``. mdtraj (1.10.3 at least) rebuilds every chain with
    no ID and replaces a residue number of 0 with the residue's index. KORP
    keys sequence separation on both: without the IDs every chain reads as
    ``' '``, so two chains are numbering-checked and scored as one, and the
    steric guard excuses cross-chain contacts as bonded neighbours.

    ``subset`` keeps atoms in ``source`` order and drops only empty residues
    and chains, so target atom k is source atom ``unique(kept)[k]``, and the
    first atom of each target chain or residue names its source.
    """
    src = np.unique(np.asarray(kept, dtype=int))
    for chain in target.chains:
        chain.chain_id = source.atom(int(src[next(chain.atoms).index])).residue.chain.chain_id
    for residue in target.residues:
        residue.resSeq = source.atom(int(src[next(residue.atoms).index])).residue.resSeq


class KORPForceField(BaseForceField):
    """Backbone-only KORP force field.

    Parameters
    ----------
    trajectory
        Any protein structure. Side chains and hydrogens are removed here.
    map_path
        The ``korp6Dv1.bin`` energy map, which pyMCPU does not ship (it is
        316 MiB). Looked for here, then at ``$KORP_MAP_PATH``, then in
        ``$MCPU_PARAMS_DIR``, then in the cache.
    steric_guard
        Install the CA-CA excluded-volume filter. On by default and you almost
        certainly want it: KORP has no hard-core repulsion, so without it a
        chain will collapse through itself during MC.
    min_separation
        The steric guard checks only CA pairs at least this many residues
        apart in sequence (default 3).
    min_distance
        The smallest CA-CA distance the steric guard allows, in Å (default
        3.2). The closest such contact in the native structures tested was
        3.53 Å, so the guard does not reject native structures.
    strict_residue_numbering
        Raise if residue numbers do not increase within a chain (default
        True). KORP takes sequence separation from these numbers, so
        out-of-order numbering would silently change the energy.
    """

    def __init__(
        self,
        trajectory: md.Trajectory,
        *,
        map_path: str | Path | None = None,
        steric_guard: bool = True,
        min_separation: int = 3,
        min_distance: float = 3.2,
        strict_residue_numbering: bool = True,
    ) -> None:
        self.map_path = self._resolve_map_path(map_path)
        self.korp_map = load_korp_map(self.map_path)
        self.steric_guard = bool(steric_guard)
        self.min_separation = int(min_separation)
        self.min_distance = float(min_distance)

        self._canonicalize_residue_names(trajectory.topology)
        sliced = self._slice_backbone(trajectory)

        self._collect_residues(sliced)
        if strict_residue_numbering:
            self._validate_residue_numbering()
        self._build_layout(sliced)
        # Must follow _build_layout: it needs ordered_indices.
        self._build_output_topology(sliced)
        if self.steric_guard:
            self._check_initial_sterics()

    # ------------------------------------------------------------------ setup

    @staticmethod
    def _resolve_map_path(explicit) -> Path:
        candidates = []
        if explicit is not None:
            candidates.append(Path(explicit))
        env = os.environ.get("KORP_MAP_PATH")
        if env:
            candidates.append(Path(env))
        params_dir = os.environ.get("MCPU_PARAMS_DIR")
        if params_dir:
            candidates.append(Path(params_dir) / "korp6Dv1.bin")
        cache = os.environ.get("MCPU_CACHE_DIR")
        cache_root = Path(cache) if cache else Path.home() / ".cache" / "pymcpu"
        candidates.append(cache_root / "korp" / "Korp6Dv1" / "korp6Dv1.bin")

        for path in candidates:
            if path.is_file():
                return path

        searched = "\n  ".join(str(c) for c in candidates)
        raise KorpMapError(
            "KORP energy map not found.\n\n"
            "pyMCPU does not ship korp6Dv1.bin: at 316 MiB it is well over\n"
            "PyPI's per-file limit, so it is obtained once by whoever runs the\n"
            "simulation.\n\n"
            "  Download Korp6Dv1.txz from https://chaconlab.org/modeling/korp\n"
            "  tar xJf Korp6Dv1.txz\n"
            "  export KORP_MAP_PATH=$PWD/Korp6Dv1/korp6Dv1.bin\n\n"
            "Expected: korp6Dv1.bin, 331777205 bytes, sha256\n"
            "  8c586500f80ad31f297652e050d391702a01927ee58d598017650ef2e0fbf971\n\n"
            f"Searched:\n  {searched}\n\n"
            "Please cite Lopez-Blanco JR & Chacon P, Bioinformatics 2019, "
            "35(17):3013-3019."
        )

    @staticmethod
    def _canonicalize_residue_names(topology: md.Topology) -> None:
        renamed: dict[str, str] = {}
        for residue in topology.residues:
            canon = _RESIDUE_ALIASES.get(residue.name)
            if canon is not None and canon != residue.name:
                renamed[residue.name] = canon
                residue.name = canon
        if renamed:
            logger.warning(
                "Renamed protonation-state variants to their standard "
                "equivalents: %s. These differ only in hydrogens, which this "
                "backbone-only force field does not represent.",
                ", ".join(f"{k}->{v}" for k, v in sorted(renamed.items())),
            )

    @staticmethod
    def _slice_backbone(trajectory: md.Trajectory) -> md.Trajectory:
        selection = trajectory.topology.select(_BACKBONE_SELECTION)
        if selection.size == 0:
            raise ValueError(
                "no backbone atoms found; KORPForceField needs protein "
                "residues with N, CA and C"
            )
        sliced = trajectory.atom_slice(selection)
        _keep_source_identity(trajectory.topology, selection, sliced.topology)
        return sliced

    def _collect_residues(self, sliced: md.Trajectory) -> None:
        """Gather per-residue identity and check every frame can be built."""
        self.res_names, self.res_seq, self.chain_ids = [], [], []
        self._frame_atoms, self._o_atoms = [], []

        missing, unknown = [], set()
        for residue in sliced.topology.residues:
            atoms = {atom.name: atom.index for atom in residue.atoms}
            absent = [name for name in _FRAME_ATOMS if name not in atoms]
            if absent:
                missing.append(f"{residue.name}{residue.resSeq} (missing {absent})")
                continue
            if residue.name not in KORP_RESIDUE_ORDER:
                unknown.add(residue.name)
                continue
            oxygen = next(
                (atoms[n] for n in ("O", "OXT", "OCT") if n in atoms), None)
            if oxygen is None:
                missing.append(f"{residue.name}{residue.resSeq} (missing O)")
                continue

            self._frame_atoms.append([atoms[n] for n in _FRAME_ATOMS])
            self._o_atoms.append(oxygen)
            self.res_names.append(residue.name)
            self.res_seq.append(int(residue.resSeq))
            self.chain_ids.append(residue.chain.chain_id or " ")

        if missing:
            # Raise rather than skip. KORP has no redundancy -- a residue
            # without all of N, CA and C has no frame at all -- so quietly
            # dropping one changes the energy with nothing to show for it.
            raise ValueError(
                "these residues cannot be scored by KORP because their "
                f"backbone is incomplete: {missing[:10]}"
                + (f" (and {len(missing) - 10} more)" if len(missing) > 10 else "")
            )
        if unknown:
            raise ValueError(
                f"KORP is defined for the 20 standard residues only; found "
                f"{sorted(unknown)}"
            )
        if len(self.res_names) < 3:
            raise ValueError(
                f"need at least 3 scorable residues, found {len(self.res_names)}"
            )
        chains = list(dict.fromkeys(self.chain_ids))
        if len(chains) > 1:
            # The energy terms read chain identity, but the moves do not: the
            # engine has one continuous backbone, so a pivot carries every later
            # chain with it and KIC keeps the gap between chains as a bond.
            logger.warning(
                "KORPForceField: %d chains (%s). Energies treat them as separate "
                "chains, but the moves treat them as one bonded backbone, so the "
                "chains cannot move independently. Use multi-chain input for "
                "scoring, or sample one chain.",
                len(chains), ", ".join(repr(c) for c in chains),
            )

    def _validate_residue_numbering(self) -> None:
        """KORP takes sequence separation from PDB numbering, not array order.

        A chain whose residue numbers run backwards or repeat would have its
        local/non-local split computed from meaningless separations, and the
        energy would differ from the reference implementation on the same
        coordinates with nothing to indicate why.
        """
        previous: dict[str, int] = {}
        for name, seq, chain in zip(self.res_names, self.res_seq, self.chain_ids):
            last = previous.get(chain)
            if last is not None and seq <= last:
                raise ValueError(
                    f"residue numbering is not increasing within chain "
                    f"{chain!r}: {name}{seq} follows residue {last}. KORP "
                    f"derives sequence separation from these numbers, so this "
                    f"would silently change the energy. Renumber the "
                    f"structure, or pass strict_residue_numbering=False if the "
                    f"numbering really is intentional."
                )
            previous[chain] = seq

    def _build_layout(self, sliced: md.Trajectory) -> None:
        """Order atoms into ``[N,CA,C][O]`` with an empty sidechain segment."""
        n_res = len(self.res_names)
        self.n_res = n_res
        self.total_bb_atoms = 3 * n_res
        self.total_o_atoms = n_res
        self.total_sc_atoms = 0
        self.total_h_atoms = 0
        self.n_atoms = self.total_bb_atoms + self.total_o_atoms

        # ordered_indices[engine index] -> index in the sliced topology.
        self.ordered_indices = [i for frame in self._frame_atoms for i in frame]
        self.ordered_indices.extend(self._o_atoms)

        self.atom_to_res = []
        for r in range(n_res):
            self.atom_to_res.extend([r, r, r])
        self.atom_to_res.extend(range(n_res))

        self.n_atom_index = [3 * r for r in range(n_res)]
        self.ca_atom_index = [3 * r + 1 for r in range(n_res)]
        self.c_atom_index = [3 * r + 2 for r in range(n_res)]

        self.blocks = []
        for r in range(n_res):
            block = mcpu_core.BlockIndices()
            block.bb_start = 3 * r
            block.c_start = 3 * r + 2
            block.o_start = self.total_bb_atoms + r
            # Must be -1, not a valid index with a zero count: the KIC move has
            # a fallback that, given sc_start >= 0 and sc_count <= 0, builds a
            # span reaching into other residues' atoms and transforms them. It
            # stays in bounds, so the only symptom is wrong geometry.
            block.sc_start = -1
            block.sc_count = 0
            block.h_start = -1
            self.blocks.append(block)

        self.downstream = mcpu_core.DownstreamCache()
        # Pivot rotates [first_sc_of_residue[r], sc_end) without checking, and
        # with no sidechain segment that interval must be empty -- so this is
        # the end of the O segment, which is also n_atoms. Anything smaller
        # rotates part of the O segment a second time.
        self.downstream.first_sc_of_residue = np.full(n_res, self.n_atoms, dtype=np.int32)
        self.downstream.first_o_of_residue = np.asarray(
            [self.total_bb_atoms + r for r in range(n_res)], dtype=np.int32)
        self.downstream.first_h_of_residue = np.full(n_res, self.n_atoms, dtype=np.int32)

        # Coordinates in nm, engine order -- same convention as MCPUForceField,
        # so callers keep doing `coords[0] * 10.0` to get Angstrom.
        self.coords = sliced.xyz[:, self.ordered_indices, :]

    def _build_output_topology(self, sliced: md.Trajectory) -> None:
        """Topology for exactly the atoms the engine holds, plus the mapping.

        `_BACKBONE_SELECTION` admits OXT/OCT on purpose, so a residue with no
        plain O can still supply one (see `_collect_residues`). When a terminus
        carries BOTH, only one is chosen and the other stays in `sliced` with
        no engine slot -- `ordered_indices` has 4*n_res entries but the sliced
        topology has more. Left alone that makes `inverse_mapping` sparse, and
        the XTC reporter sizes its output from `max(mapping) + 1` and zero-fills
        the rest. Two symptoms, both bad, depending on where the unused atom
        sorts: an extra atom written at the origin in a file that loads cleanly
        (CLN025, whose TYR10 is ordered O, OXT, N, CA), or a file with fewer
        atoms than this topology that will not load at all (NuG2). Restricting
        the topology to the engine's own atoms makes the mapping dense by
        construction, so neither can happen.

        `used` is SORTED because `Topology.subset()` reorders a non-monotonic
        index array (and warns). `ordered_indices` is [N,CA,C]*n then [O]*n,
        which is not monotonic, so engine order is carried by `_output_index`
        -- never by the subset itself. Subsetting on `ordered_indices` directly
        would silently return a sorted topology and a fresh mismatch.
        ``subset`` also drops chain IDs, which are put back.
        """
        used = sorted(self.ordered_indices)
        remap = {old: new for new, old in enumerate(used)}
        self.output_topology = sliced.topology.subset(used)
        _keep_source_identity(sliced.topology, used, self.output_topology)
        self._output_index = [remap[i] for i in self.ordered_indices]

    def _check_initial_sterics(self) -> None:
        """Fail at construction if the input already violates the guard.

        Without this the run starts, the guard returns its sentinel on the
        very first full energy evaluation, and the user gets a mid-run
        StericClashError instead of a clear statement that the structure was
        already clashing before a single move was made.
        """
        ca = self.coords[0][self.ca_atom_index] * 10.0   # nm -> Angstrom
        seq = np.asarray(self.res_seq)
        chain = np.asarray(self.chain_ids)

        distances = np.linalg.norm(ca[:, None, :] - ca[None, :, :], axis=-1)
        separation = np.abs(seq[:, None] - seq[None, :])
        same_chain = chain[:, None] == chain[None, :]
        checked = (~same_chain) | (separation >= self.min_separation)
        np.fill_diagonal(checked, False)

        # The engine judges a whole state against a floor STATE_CLASH_BUFFER_A
        # (0.001 A) below min_distance; moves are tested against min_distance.
        floor = self.min_distance - mcpu_core.STATE_CLASH_BUFFER_A
        violations = checked & (distances < floor)
        if not violations.any():
            return
        i, j = np.argwhere(violations)[0]
        worst = distances[violations].min()
        raise ValueError(
            f"the input structure already violates the CA-CA steric guard: "
            f"{int(violations.sum()) // 2} pair(s) closer than "
            f"{floor:.3f} A, the closest at {worst:.3f} A "
            f"(e.g. {self.res_names[i]}{self.res_seq[i]} and "
            f"{self.res_names[j]}{self.res_seq[j]}). Monte Carlo cannot start "
            f"from a state the guard rejects. Either fix the structure, lower "
            f"min_distance, or pass steric_guard=False -- but note that "
            f"without the guard KORP alone will let the chain collapse through "
            f"itself."
        )

    # ------------------------------------------------------------------- API

    @property
    def inverse_mapping(self) -> list[int]:
        """Index in :attr:`output_topology` of each engine atom, in engine order.

        Every engine atom has one, so there are no -1 entries. Pass this to a
        reporter, and load what it writes against :attr:`output_topology`.
        """
        return list(self._output_index)

    def create_system(self, topology: md.Topology) -> "mcpu_core.System":
        del topology  # the layout is fixed at construction, from the slice
        system = mcpu_core.System(self.n_atoms, self.n_res)
        system.set_atom_counts(
            self.total_bb_atoms, self.total_o_atoms,
            self.total_sc_atoms, self.total_h_atoms)
        system.set_block_indices(self.blocks)
        system.atom_to_residue = self.atom_to_res
        system.set_torsions_per_residue([0] * self.n_res)
        system.set_downstream_cache(self.downstream)
        # KIC's closure targets (bond lengths, angles, omegas) come from the
        # START structure, measured once here from the coordinates every run
        # is positioned with -- see MCPUForceField.create_system.
        system.set_kic_reference((self.coords[0] * 10.0).T.astype(np.float32))
        # The moves keep proline phi fixed only for residues flagged here: the
        # pivot resamples a phi pivot at a proline, and KIC skips any window
        # whose phi it would change at one. Unflagged (as this used to be),
        # every System.is_proline is False, and on chignolin 10k KIC moves
        # turned proline phi by 47 deg, 10k pivots by 5.6 deg.
        system.set_is_proline([int(name == "PRO") for name in self.res_names])
        # No rotamer or rama library: both are default-constructed on System and
        # the moves that would read them return before they get that far.

        engine_map = KorpPotentialBuilder.build_map(self.korp_map)
        # Kept for callers that inspect it. Lifetime no longer depends on this:
        # the engine map owns a reference to its table (see the binding), which
        # matters because this attribute holds only the LATEST map.
        self._engine_map = engine_map

        if self.steric_guard:
            # Added FIRST: System::evaluateDeltaEnergy walks potentials in
            # insertion order and stops at the first hard rejection, so a
            # clashing proposal never pays for KORP's 16 A pair loop.
            guard = KorpPotentialBuilder.build_steric_guard(
                ca_atom=self.ca_atom_index,
                res_seq=self.res_seq,
                chain_ids=self.chain_ids,
                min_separation=self.min_separation,
                min_distance=self.min_distance,
            )
            guard.set_energy_group(STERIC_ENERGY_GROUP)
            guard.set_name("calpha_excluded_volume")
            system.add_potential(guard)

        potential = KorpPotentialBuilder.build_pair_potential(
            engine_map,
            n_atom=self.n_atom_index,
            ca_atom=self.ca_atom_index,
            c_atom=self.c_atom_index,
            res_names=self.res_names,
            res_seq=self.res_seq,
            chain_ids=self.chain_ids,
        )
        potential.set_energy_group(KORP_ENERGY_GROUP)
        potential.set_name("korp_6d")
        system.add_potential(potential)
        #: Kept so callers can reach the term after the system is built -- the
        #: rigid-skip toggle in particular, which tests use to check the
        #: incremental path against an enumeration that assumes nothing.
        self.pair_potential = potential
        return system

    def apply_energy_weights(self, context) -> None:
        """Set the weights of the KORP terms (groups 7 and 8) to 1.

        They already start at 1, so this only makes sure no weight meant for
        an MCPU term applies to them. Call it after creating the simulation.
        """
        context.set_energy_weight(KORP_ENERGY_GROUP, 1.0)
        context.set_energy_weight(STERIC_ENERGY_GROUP, 1.0)
