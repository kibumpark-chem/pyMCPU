"""Builder for the MCPU hydrogen-bond force term.

Loads the legacy MCPU08 binary parameter files -- the main geometry-dependent
H-bond energy table and the sequence-dependent (residue-pair) scaling table --
and assembles them into the C++ ``mcpu_core.HBondPotential`` used by
``MCPUForceField``. Also exposes ``AMINO_ORDER``/``AMINO_INDEX``, the fixed
alphabetical amino-acid ordering (matching legacy ``pdb_util.h``
``GetAminoNumber()``) used to index the sequence-dependent scaling table.
"""

from __future__ import annotations

import os
import numpy as np
from pymcpu import mcpu_core

# legacy pdb_util.h GetAminoNumber() -- exact alphabetical order. Indexes the
# sequence-dependent H-bond scaling table (seq_hb[3][20][20] in legacy hbonds.h).
AMINO_ORDER = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
]
AMINO_INDEX = {name: i for i, name in enumerate(AMINO_ORDER)}


class HydrogenBondBuilder:
    """Loads and validates the binary H-bond parameter files and builds the
    C++ ``mcpu_core.HBondPotential`` from them.

    All members are classmethods; the class is never instantiated.
    ``load_parameters`` and ``load_seq_dep_parameters`` each perform a
    one-time read + shape validation of a legacy MCPU08 binary file (flat
    ``float32`` arrays, checked against the expected element counts derived
    from ``HBOND_DIM``/``N_SS_TYPES``/``N_AMINO``); ``build`` then hands the
    two validated arrays straight to ``mcpu_core.HBondPotential``.
    """

    HBOND_DIM = 9
    N_SS_TYPES = 3
    TOTAL_ELEMENTS = N_SS_TYPES * (HBOND_DIM ** 6)
    N_AMINO = 20
    SEQ_DEP_TOTAL_ELEMENTS = N_SS_TYPES * N_AMINO * N_AMINO

    @classmethod
    def load_parameters(cls, filepath: str) -> np.ndarray:
        """
        PHASE 1: Read and validate the binary HBond data ONCE.
        """
        if not os.path.exists(filepath):
            raise RuntimeError(f"Critical Error: Could not find HBond Potential file at: {filepath}")

        # Read the binary file directly into a 1D NumPy array
        try:
            raw_params = np.fromfile(filepath, dtype=np.float32)
            if raw_params.size == 0:
                raise ValueError(f"HBond parameter file is empty: {filepath}")
        except OSError as e:
            raise FileNotFoundError(
                f"HBond parameter file not found: {filepath}"
            ) from e
        # Safety Check: Ensure the binary file isn't corrupted
        if raw_params.size != cls.TOTAL_ELEMENTS:
            raise ValueError(
                f"Shape mismatch in {filepath}. "
                f"Expected {cls.TOTAL_ELEMENTS} floats, got {raw_params.size}."
            )
            
        return raw_params

    @classmethod
    def load_seq_dep_parameters(cls, filepath: str) -> np.ndarray:
        """PHASE 1: Read and validate the binary seq-dependent-scaling data ONCE."""
        if not os.path.exists(filepath):
            raise RuntimeError(f"Critical Error: Could not find HBond seq-dep file at: {filepath}")
        try:
            raw_params = np.fromfile(filepath, dtype=np.float32)
            if raw_params.size == 0:
                raise ValueError(f"HBond seq-dep parameter file is empty: {filepath}")
        except OSError as e:
            raise FileNotFoundError(
                f"HBond seq-dep parameter file not found: {filepath}"
            ) from e
        if raw_params.size != cls.SEQ_DEP_TOTAL_ELEMENTS:
            raise ValueError(
                f"Shape mismatch in {filepath}. "
                f"Expected {cls.SEQ_DEP_TOTAL_ELEMENTS} floats, got {raw_params.size}."
            )
        return raw_params

    @classmethod
    def build(cls, raw_params: np.ndarray, seq_dep_params: np.ndarray) -> mcpu_core.HBondPotential:
        """Construct the C++ HBondPotential from pre-validated parameter arrays."""
        return mcpu_core.HBondPotential(raw_params, seq_dep_params)