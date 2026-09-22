"""Builder for the discrete rotamer-library sidechain move's data table.

Parses the legacy Dunbrack backbone-independent rotamer library
(``bbind02.May.lib``) into per-residue weight/mean/sigma arrays and
assembles them into the C++ ``mcpu_core.RotamerLibrary`` used by the
rotamer-library sidechain move (``sidechain_move_mode="rotamer_library"``,
see ``MCIntegrator::apply_rotamer_at``). This is proposal data, not an
energy term -- wired onto ``System`` via ``set_rotamer_library``, not
registered as a ``Force``/``add_potential()``.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

from pymcpu import mcpu_core
from pymcpu.forcefields.builders.hbond_builder import AMINO_INDEX

# Minimum sigma (radians) after degree->radian conversion -- pure insurance
# against a divide-by-zero/degenerate row in log_mixture_density; the real
# library's smallest observed SD is 5 degrees, so this floor never touches
# real data.
_MIN_SIGMA_RAD = math.radians(1.0)

# bbind02.May.lib is 19 whitespace-separated columns, no header:
#   0: residue name
#   1-4: chi1-4 rotamer bin index (opaque/informational -- not used here;
#        e.g. PRO's two rows share identical bin codes and don't uniquely
#        key anything)
#   5-6: observation counts (unused)
#   7: marginal probability (%) of this rotamer row -- the sampling weight
#   8: SE of column 7 (unused)
#   9-10: a conditional probability + SE (unused)
#   11,13,15,17: chi1-4 mean angle, degrees
#   12,14,16,18: chi1-4 angular standard deviation, degrees
_NUM_COLUMNS = 19
_PROB_COLUMN = 7
_FIRST_MEAN_COLUMN = 11


@dataclass
class ParsedRotamerRows:
    """One residue type's parsed rotamer table, ready for RotamerLibrary."""

    weights: list[float]
    means: list[list[float]]   # [row][chi], radians
    sigmas: list[list[float]]  # [row][chi], radians


class RotamerLibraryBuilder:
    """Loads and validates ``bbind02.May.lib``, builds the C++
    ``mcpu_core.RotamerLibrary`` from it.

    All members are classmethods; the class is never instantiated.
    ``load_parameters`` performs a one-time text parse + validation (called
    once per ``MCPUForceField``); ``build`` then hands the validated
    per-residue rows to ``mcpu_core.RotamerLibrary``.
    """

    @classmethod
    def load_parameters(cls, filepath: str) -> dict[str, ParsedRotamerRows]:
        """PHASE 1: Parse and validate the rotamer library text file ONCE."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Rotamer library file not found: {filepath}")

        raw_rows: dict[str, list[list[str]]] = {}
        with open(filepath, "r") as f:
            for line_num, line in enumerate(f, start=1):
                tokens = line.split()
                if not tokens:
                    continue  # tolerate trailing blank lines
                if len(tokens) != _NUM_COLUMNS:
                    raise ValueError(
                        f"{filepath}:{line_num}: expected {_NUM_COLUMNS} "
                        f"columns, got {len(tokens)}: {line!r}"
                    )
                raw_rows.setdefault(tokens[0], []).append(tokens)

        if not raw_rows:
            raise ValueError(f"Rotamer library file is empty: {filepath}")

        parsed: dict[str, ParsedRotamerRows] = {}
        for residue, rows in raw_rows.items():
            if residue not in AMINO_INDEX:
                raise ValueError(f"Unrecognized residue in {filepath}: {residue}")

            # Column 7 is a percentage that does not necessarily sum to
            # exactly 100 across a residue's rows (verified: e.g. 99.95 for
            # ARG) -- always renormalize, never assume an exact total.
            weights = [float(row[_PROB_COLUMN]) for row in rows]
            total = sum(weights)
            if total <= 0:
                raise ValueError(
                    f"{filepath}: residue {residue} has non-positive total "
                    f"rotamer-row probability ({total})"
                )
            weights = [w / total for w in weights]

            means: list[list[float]] = []
            sigmas: list[list[float]] = []
            for row in rows:
                row_means = []
                row_sigmas = []
                for k in range(4):
                    col = _FIRST_MEAN_COLUMN + 2 * k
                    mean_deg = float(row[col])
                    sigma_deg = float(row[col + 1])
                    row_means.append(math.radians(mean_deg))
                    row_sigmas.append(max(math.radians(sigma_deg), _MIN_SIGMA_RAD))
                means.append(row_means)
                sigmas.append(row_sigmas)

            parsed[residue] = ParsedRotamerRows(weights=weights, means=means, sigmas=sigmas)

        return parsed

    @classmethod
    def build(cls, parsed: dict[str, ParsedRotamerRows]) -> "mcpu_core.RotamerLibrary":
        """PHASE 2: Construct the C++ RotamerLibrary from pre-validated rows."""
        lib = mcpu_core.RotamerLibrary()
        for residue, rows in parsed.items():
            lib.add_residue_type(AMINO_INDEX[residue], rows.weights, rows.means, rows.sigmas)
        return lib
