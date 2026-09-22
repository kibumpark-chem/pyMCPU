"""Builder for the knowledge-based backbone (rama-mixture) pivot move's
data table.

Parses a consolidated JSON file of per-amino-acid wrapped-bivariate-normal
(phi, psi) mixtures -- fit offline against a large pooled sample, by a
conversion script that is not part of this distribution -- into per-residue
weight/mean/covariance
arrays and assembles them into the C++ ``mcpu_core.RamaMixtureLibrary`` used
by ``MCIntegrator::apply_rama_pivot_at``. This is proposal data, not an
energy term -- wired onto ``System`` via ``set_rama_mixture_library``, not
registered as a ``Force``/``add_potential()``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np

from pymcpu import mcpu_core
from pymcpu.forcefields.builders.hbond_builder import AMINO_INDEX

# Sanity bounds on K (mixture component count) -- matches the range actually
# swept by mode_a_k_selection_cv.py's grid ({2,3,4,5,6,7,8,10,12}); a value
# outside this almost certainly indicates a corrupted/mis-generated file
# rather than a legitimate fit.
_MIN_COMPONENTS = 1
_MAX_COMPONENTS = 12

# Weights are already-renormalized EM output (not a re-derived percentage
# column like the rotamer library's), so a looser sum-to-1 tolerance than
# RotamerLibraryBuilder's is appropriate.
_WEIGHT_SUM_TOL = 1e-4

# Covariance positive-definiteness check tolerance (eigenvalues must exceed
# this to count as "positive" rather than a floating-point zero/near-zero).
_PD_EIGENVALUE_FLOOR = 1e-12


@dataclass
class ParsedRamaMixtureResidue:
    """One residue category's parsed (phi, psi) mixture, ready for
    RamaMixtureLibrary."""

    weights: list[float]
    means: list[list[float]]  # [K][2] (phi, psi), radians
    covariances: list[list[list[float]]]  # [K][2][2], radians^2
    status: str  # "confirmed" | "provisional"


@dataclass
class ParsedRamaMixture:
    """A fully parsed+validated rama-mixture file: the fitter's periodic-sum
    convention (n_wrap) plus each registered residue category. Kept as one
    object (rather than a bare dict + a separate n_wrap parameter) so
    build() can never be called with an n_wrap that doesn't actually match
    what the file's components were fit under."""

    n_wrap: int
    residues: dict[str, ParsedRamaMixtureResidue]


class RamaMixtureLibraryBuilder:
    """Loads and validates the consolidated rama-mixture JSON file, builds
    the C++ ``mcpu_core.RamaMixtureLibrary`` from it.

    All members are classmethods; the class is never instantiated.
    ``load_parameters`` performs a one-time JSON parse + validation (called
    once per ``MCPUForceField``); ``build`` then hands the validated
    per-residue rows to ``mcpu_core.RamaMixtureLibrary``.
    """

    @classmethod
    def load_parameters(
        cls, filepath: str, *, allow_provisional: bool = False
    ) -> ParsedRamaMixture:
        """PHASE 1: Parse and validate the rama-mixture JSON file ONCE."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Rama-mixture parameter file not found: {filepath}")

        with open(filepath, "r") as f:
            data = json.load(f)

        residues_raw = data.get("residues")
        if not residues_raw:
            raise ValueError(f"Rama-mixture file has no 'residues' entries: {filepath}")

        n_wrap = data.get("source", {}).get("n_wrap")
        if n_wrap is None:
            raise ValueError(f"{filepath}: missing source.n_wrap")

        parsed: dict[str, ParsedRamaMixtureResidue] = {}
        for residue, entry in residues_raw.items():
            if residue not in AMINO_INDEX:
                raise ValueError(f"Unrecognized residue in {filepath}: {residue}")

            weights = list(entry["weights"])
            means = [list(m) for m in entry["means"]]
            covariances = [[list(row) for row in cov] for cov in entry["covariances"]]
            status = entry.get("status", "confirmed")

            k = len(weights)
            if not (_MIN_COMPONENTS <= k <= _MAX_COMPONENTS):
                raise ValueError(
                    f"{filepath}: residue {residue} has {k} mixture components, "
                    f"expected {_MIN_COMPONENTS}-{_MAX_COMPONENTS}"
                )
            if len(means) != k or len(covariances) != k:
                raise ValueError(
                    f"{filepath}: residue {residue} weights/means/covariances "
                    f"length mismatch (K={k}, len(means)={len(means)}, "
                    f"len(covariances)={len(covariances)})"
                )

            total_weight = sum(weights)
            if any(w < 0 for w in weights) or not np.isclose(
                total_weight, 1.0, atol=_WEIGHT_SUM_TOL
            ):
                raise ValueError(
                    f"{filepath}: residue {residue} weights must be non-negative "
                    f"and sum to 1 (got sum={total_weight})"
                )

            for k_idx, cov in enumerate(covariances):
                cov_arr = np.asarray(cov, dtype=float)
                if cov_arr.shape != (2, 2):
                    raise ValueError(
                        f"{filepath}: residue {residue} component {k_idx} covariance "
                        f"must be 2x2, got shape {cov_arr.shape}"
                    )
                if abs(cov_arr[0, 1] - cov_arr[1, 0]) > 1e-6:
                    raise ValueError(
                        f"{filepath}: residue {residue} component {k_idx} covariance "
                        "is not symmetric"
                    )
                eigvals = np.linalg.eigvalsh(cov_arr)
                if np.any(eigvals <= _PD_EIGENVALUE_FLOOR):
                    raise ValueError(
                        f"{filepath}: residue {residue} component {k_idx} covariance "
                        f"is not positive-definite (eigenvalues={eigvals})"
                    )

            if status == "provisional" and not allow_provisional:
                raise ValueError(
                    f"{filepath}: residue {residue} is PROVISIONAL -- pass "
                    "allow_provisional=True to load it anyway"
                )

            parsed[residue] = ParsedRamaMixtureResidue(
                weights=weights, means=means, covariances=covariances, status=status
            )

        return ParsedRamaMixture(n_wrap=int(n_wrap), residues=parsed)

    @classmethod
    def build(cls, parsed: ParsedRamaMixture) -> "mcpu_core.RamaMixtureLibrary":
        """PHASE 2: Construct the C++ RamaMixtureLibrary from pre-validated rows."""
        lib = mcpu_core.RamaMixtureLibrary(n_wrap=parsed.n_wrap)
        for residue, entry in parsed.residues.items():
            covariances_flat = [
                [cov[0][0], cov[0][1], cov[1][1]] for cov in entry.covariances
            ]
            lib.add_residue_type(
                AMINO_INDEX[residue], entry.weights, entry.means, covariances_flat
            )
        return lib
