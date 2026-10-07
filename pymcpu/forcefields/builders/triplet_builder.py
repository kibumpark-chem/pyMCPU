"""Builders for the two consecutive-triplet knowledge-based potentials used by
the MCPU force field: the backbone triplet potential (``E_trp``, Yang et al.
2007 Eq. 3) and the sidechain-torsion triplet potential. Both map a cached
legacy binary parameter table keyed by (residue-i, residue-i+1, residue-i+2)
type, then slice out the flat per-window parameters for a given ordered atom
list's sequence and hand them to the corresponding ``mcpu_core`` force.

The tables are memory-mapped read-only, not read: the sidechain table is
633 MiB, and a mapping lets every process on a node share one copy through
the page cache. Each process touches only the windows of its own sequence.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pymcpu import mcpu_core
from pymcpu.forcefields.builders.hbond_builder import AMINO_INDEX

if TYPE_CHECKING:  # avoid importing the engine just to use this module
    from pymcpu.forcefields.mcpu import MCPUAtom

# Same legacy pdb_util.h GetAminoNumber() alphabetical order as
# hbond_builder.AMINO_INDEX (verified identical for all 20 residues) --
# shared here rather than re-derived, so the triplet and H-bond potentials'
# residue indexing can't silently diverge.
RES_TO_INT = dict(AMINO_INDEX)

def _map_table(filepath: str, shape: tuple[int, ...]) -> np.ndarray:
    """Map a float32 table read-only and view it with ``shape``."""
    return np.memmap(filepath, dtype=np.float32, mode="r").reshape(shape)

def _build_sequence_params(
    atom_list: list[MCPUAtom],
    reshaped_params: np.ndarray
) -> tuple[int, np.ndarray]:
    """
    Shared helper: extracts residue sequence from atom list,
    maps to integer types, and slices parameter blocks.
    Returns (n_pos, flat_params_array).
    """
    residue_map = {}
    for atom in atom_list:
        if atom.residue_index not in residue_map:
            residue_map[atom.residue_index] = atom.residue_name

    sequence = [residue_map[i] for i in sorted(residue_map.keys())]
    n_pos = len(sequence) - 2
    if n_pos < 1:
        return 0, np.array([], dtype=np.float32)

    try:
        type_seq = [RES_TO_INT[res] for res in sequence]
    except KeyError as e:
        raise ValueError(f"Unrecognized residue: {e}") from e

    params = [reshaped_params[type_seq[i], type_seq[i+1], type_seq[i+2]]
              for i in range(n_pos)]
    return n_pos, np.concatenate(params)

class TripletPotentialBuilder:
    """Builds the backbone triplet potential (``E_trp``, Yang et al. 2007 Eq. 3),
    which scores each three-consecutive-residue window of the chain on
    backbone virtual-bond/dihedral geometry (pCA, bCA, phi, psi bins).
    ``load_parameters`` maps the legacy ``triplet_potentials.bin`` table
    to ``(20, 20, 20, 1296)``; ``build`` slices out the flat per-window
    parameters for an atom list's sequence and wraps them in an
    ``mcpu_core.TripletPotential`` force.
    """

    BB_DIM = 6
    BB_DIM_RES = 20
    BLOCK_SIZE = BB_DIM ** 4  # 1296

    @classmethod
    def load_parameters(cls, filepath: str) -> np.ndarray:
        """Map the binary table read-only. Called once per force field."""
        return _map_table(filepath, (cls.BB_DIM_RES, cls.BB_DIM_RES, cls.BB_DIM_RES, cls.BLOCK_SIZE))

    @classmethod
    def build(
        cls,
        atom_list: list[MCPUAtom],
        reshaped_params: np.ndarray
    ) -> mcpu_core.TripletPotential:
        _, flat_params = _build_sequence_params(atom_list, reshaped_params)
        return mcpu_core.TripletPotential(flat_params)

class SidechainTripletBuilder:
    """Builds the sidechain-torsion triplet potential, which scores each
    three-consecutive-residue window on the middle residue's chi1-chi4
    dihedral bins. ``load_parameters`` maps the legacy
    ``sidechain_triplet_potentials.bin`` table to ``(20, 20, 20, 20736)``;
    ``build`` slices out the flat per-window parameters for an atom list's
    sequence and wraps them in an ``mcpu_core.SidechainTripletPotential`` force.
    """

    SC_DIM = 12
    SC_DIM_RES = 20
    BLOCK_SIZE = SC_DIM ** 4

    @classmethod
    def load_parameters(cls, filepath: str) -> np.ndarray:
        """Map the binary table read-only. Called once per force field."""
        return _map_table(filepath, (cls.SC_DIM_RES, cls.SC_DIM_RES, cls.SC_DIM_RES, cls.BLOCK_SIZE))

    @classmethod
    def build(
        cls,
        atom_list: list[MCPUAtom],
        reshaped_params: np.ndarray
    ) -> mcpu_core.SidechainTripletPotential:
        _, flat_params = _build_sequence_params(atom_list, reshaped_params)
        return mcpu_core.SidechainTripletPotential(flat_params)
