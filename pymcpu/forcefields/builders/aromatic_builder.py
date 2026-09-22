"""Builder for the C++ ``AromaticPotential`` ring-orientation force.

Loads the flat, per-plane-angle-bin energy table from a legacy MCPU binary
aromatic data file, then assembles per-residue aromatic-ring atom indices
together with that parameter table into the native ``mcpu_core`` potential.
"""

from __future__ import annotations

import os
import numpy as np
from pymcpu import mcpu_core


class AromaticPotentialBuilder:
    """Two-phase builder for the aromatic (ring-orientation) potential.

    Phase 1 (:meth:`load_parameters`) reads and validates the binary
    per-angle-bin energy table from disk once, up front. Phase 2
    (:meth:`build`) is called per-structure to construct the
    ``mcpu_core.AromaticPotential`` from that cached parameter array and the
    aromatic-ring atom indices for the current topology. See
    ``MCPUForceField`` for how the two phases are wired together.
    """

    # Plane-angle bins (0-90 deg, 10 deg each). Matches the reference
    # `short aromatic_E[9]` table from MCPU's aromatic potential.
    NUM_ANGLE_BINS = 9

    @classmethod
    def load_parameters(cls, filepath: str) -> np.ndarray:
        """
        PHASE 1: Read and validate the binary aromatic data ONCE.

        The file is expected to hold ``NUM_ANGLE_BINS`` float32 energies, one
        per 10-degree ring-plane-angle bin.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"Aromatic parameter file not found: {filepath}"
            )

        raw_params = np.fromfile(filepath, dtype=np.float32)
        if raw_params.size == 0:
            raise ValueError(f"Aromatic parameter file is empty: {filepath}")
        if raw_params.size != cls.NUM_ANGLE_BINS:
            raise ValueError(
                f"Shape mismatch in {filepath}. "
                f"Expected {cls.NUM_ANGLE_BINS} floats, got {raw_params.size}."
            )

        return raw_params

    @classmethod
    def build(
        cls,
        aromatic_atom_indices: list[np.ndarray],
        reshaped_params: np.ndarray,
    ) -> mcpu_core.AromaticPotential:
        """Construct the C++ AromaticPotential from ring indices and the flat
        per-angle-bin parameter array."""
        params = np.ascontiguousarray(reshaped_params, dtype=np.float32).ravel()
        return mcpu_core.AromaticPotential(aromatic_atom_indices, params)
