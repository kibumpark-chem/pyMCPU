"""Builders for the rotamer-library sidechain move's proposal tables.

The move (``sidechain_move_mode="rotamer_library"``, see
``MCIntegrator::apply_rotamer_at``) proposes chi angles from a table held by
the System (``System.set_rotamer_library``). It is proposal data, not an
energy term: with the Hastings ratio both tables below target the same
ensemble, but ``dunbrack2002`` samples only part of it.
:func:`load_rotamer_library` builds one of
:data:`pymcpu.config.VALID_ROTAMER_LIBRARIES` from Dunbrack's 2002
backbone-independent library (``bbind02.May.lib`` of the force field's
parameter set):

``dunbrack2002_sym`` (the default)
    The 2002 rows without those of zero weight. Each chi is drawn from a
    normal truncated to one period around the row mean, and the two-fold
    symmetric chi (ASP chi2, GLU chi3, PHE chi2, TYR chi2) get period 180
    degrees and a random 180 degree image, so that both halves of those chi
    are proposed. The Hastings term uses exactly the density of that draw.
``dunbrack2002``
    The rows as legacy MCPU draws them: an untruncated normal per chi, and the
    symmetric chi only on the half the rows cover, corrected by the Gaussian
    mixture density at the nearest image. A residue whose symmetric chi
    starts in the other half can stay frozen for the whole run. pyMCPU's
    proposal before ``dunbrack2002_sym``, kept for comparisons with legacy
    MCPU.
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pymcpu import mcpu_core
from pymcpu.config import DEFAULT_ROTAMER_LIBRARY, normalize_rotamer_library
from pymcpu.forcefields.builders.hbond_builder import AMINO_INDEX

#: Bumped whenever a change here changes a dunbrack2002_sym table; part of the
#: fingerprint that checkpoints record.
_TRUNCATED_TABLE_BUILDER_VERSION = 1

_CONSTANTS = Path(__file__).resolve().parents[2] / "data" / "params" / "mcpu08" / "constants"
_DUNBRACK2002_FILE = _CONSTANTS / "bbind02.May.lib"

#: Two-fold symmetric chi (0-based index): period pi, random 180 degree image.
SYMMETRIC_CHI = {"ASP": 1, "GLU": 2, "PHE": 1, "TYR": 1}

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
    """Loads and validates ``bbind02.May.lib``, and builds the ``dunbrack2002``
    row tables from it.

    All members are classmethods; the class is never instantiated.
    ``load_parameters`` parses and validates the text file; ``build`` hands the
    validated per-residue rows to ``mcpu_core.RotamerLibrary``. Use
    :func:`load_rotamer_library`, which builds either library once per
    process.
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


def load_rotamer_library(
    name: str = DEFAULT_ROTAMER_LIBRARY, *, bbind_path: str | os.PathLike | None = None
) -> "mcpu_core.RotamerLibrary":
    """Build rotamer library ``name``, ``"dunbrack2002_sym"`` (the default)
    or ``"dunbrack2002"``, from ``bbind_path`` (default: the shipped mcpu08
    parameter set's file). The module docstring and the Rotamer proposal
    physics note describe the two.

    Built once per process and argument pair; attach it with
    ``system.set_rotamer_library(...)``, which copies it. Do not modify the
    returned object.
    """
    name = normalize_rotamer_library(name)
    return _load(name, str(Path(bbind_path or _DUNBRACK2002_FILE).resolve()))


@functools.lru_cache(maxsize=None)
def _load(name: str, path: str) -> "mcpu_core.RotamerLibrary":
    parsed = RotamerLibraryBuilder.load_parameters(path)
    md5 = hashlib.md5(Path(path).read_bytes()).hexdigest()
    if name == "dunbrack2002":
        lib = RotamerLibraryBuilder.build(parsed)
        lib.fingerprint = f"dunbrack2002 md5={md5}"
        return lib
    # dunbrack2002_sym: the rows without those of zero weight (never drawn,
    # and nothing in the density), on the chi each residue type has.
    ntorsions = _ntorsions()
    lib = mcpu_core.RotamerLibrary()
    for residue, rows in parsed.items():
        n = ntorsions.get(residue, 0)
        if residue == "PRO" or n == 0:
            continue
        keep = [k for k, w in enumerate(rows.weights) if w > 0.0]
        periods = np.full(n, 2.0 * math.pi)
        if residue in SYMMETRIC_CHI:
            periods[SYMMETRIC_CHI[residue]] = math.pi
        lib.add_truncated_table(
            AMINO_INDEX[residue],
            np.array([rows.weights[k] for k in keep]),
            np.array([rows.means[k][:n] for k in keep]),
            np.array([rows.sigmas[k][:n] for k in keep]),
            periods,
        )
    lib.fingerprint = f"dunbrack2002_sym md5={md5} builder={_TRUNCATED_TABLE_BUILDER_VERSION}"
    return lib


@functools.lru_cache(maxsize=1)
def _ntorsions() -> dict[str, int]:
    """Chi angles per residue type, as the mcpu08 force field moves them."""
    path = _CONSTANTS / "standard_amino_acids.json"
    return {res: int(v["ntorsions"]) for res, v in json.loads(path.read_text()).items()}
