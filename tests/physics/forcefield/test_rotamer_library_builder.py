"""``RotamerLibraryBuilder`` parsing/validation unit tests.

Pure builder-logic tests against the real vendored ``bbind02.May.lib``
(Dunbrack backbone-independent rotamer library) -- no C++ ``Context``/PDB
involved, mirroring ``test_mu_builder_gly_eligibility.py``'s "exercise the
builder directly" style.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from pymcpu.forcefields.builders.hbond_builder import AMINO_INDEX
from pymcpu.forcefields.builders.rotamer_builder import RotamerLibraryBuilder

BBIND_LIB_PATH = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "pymcpu"
    / "parameters"
    / "pretrained"
    / "mcpu08"
    / "constants"
    / "bbind02.May.lib"
)

# Actual row counts per residue, read directly from the file (NOT assumed
# from a 3**ntorsions formula -- ASN and GLN have extra bins on an
# amide-flip chi, verified by direct inspection: ASN's chi2 column ranges
# 1-6 (18 rows, not 9); GLN's chi3 column ranges 1-4 (36 rows, not 27)).
EXPECTED_ROW_COUNTS = {
    "ARG": 81,
    "ASN": 18,
    "ASP": 9,
    "CYS": 3,
    "GLN": 36,
    "GLU": 27,
    "HIS": 9,
    "ILE": 9,
    "LEU": 9,
    "LYS": 81,
    "MET": 27,
    "PHE": 6,
    "PRO": 2,
    "SER": 3,
    "THR": 3,
    "TRP": 9,
    "TYR": 6,
    "VAL": 3,
}


@pytest.fixture(scope="module")
def parsed():
    return RotamerLibraryBuilder.load_parameters(str(BBIND_LIB_PATH))


def test_row_counts_match_file_not_power_of_three_formula(parsed) -> None:
    assert set(parsed) == set(EXPECTED_ROW_COUNTS)
    for residue, expected_k in EXPECTED_ROW_COUNTS.items():
        assert len(parsed[residue].weights) == expected_k, residue
        assert len(parsed[residue].means) == expected_k, residue
        assert len(parsed[residue].sigmas) == expected_k, residue


def test_gly_ala_absent() -> None:
    """No-chi residues have no rotamer rows in the library."""
    parsed = RotamerLibraryBuilder.load_parameters(str(BBIND_LIB_PATH))
    assert "GLY" not in parsed
    assert "ALA" not in parsed


def test_weights_renormalized_to_sum_to_one(parsed) -> None:
    """Column 8's raw percentages do not sum to exactly 100 for every
    residue (e.g. ARG sums to 99.95) -- load_parameters must always
    renormalize, never assume an exact total."""
    for residue, rows in parsed.items():
        total = sum(rows.weights)
        assert total == pytest.approx(1.0, abs=1e-6), residue


def test_means_and_sigmas_are_radians_above_floor(parsed) -> None:
    """Degrees->radians conversion happened (values fit in [-pi, pi], not
    [-180, 180]) and every sigma respects the defensive 1-degree floor."""
    floor = math.radians(1.0) - 1e-9
    for residue, rows in parsed.items():
        for row_means, row_sigmas in zip(rows.means, rows.sigmas):
            for m in row_means:
                assert -math.pi - 1e-6 <= m <= math.pi + 1e-6, (residue, m)
            for s in row_sigmas:
                assert s >= floor, (residue, s)


def test_pro_parses_despite_degenerate_bin_codes(parsed) -> None:
    """PRO's two library rows share identical chi-bin index codes (ring
    pucker states, not independent chi rotamers) -- the parser must not key
    off those columns and must still parse both rows successfully."""
    assert len(parsed["PRO"].weights) == 2
    assert sum(parsed["PRO"].weights) == pytest.approx(1.0, abs=1e-6)


def test_build_constructs_populated_c_library(parsed) -> None:
    lib = RotamerLibraryBuilder.build(parsed)
    for residue, expected_k in EXPECTED_ROW_COUNTS.items():
        assert lib.num_rows(AMINO_INDEX[residue]) == expected_k


def test_missing_file_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        RotamerLibraryBuilder.load_parameters(str(tmp_path / "does_not_exist.lib"))


def test_malformed_column_count_raises(tmp_path) -> None:
    bad_file = tmp_path / "bad.lib"
    bad_file.write_text("ARG\t1\t1\t1\t1\n")  # far fewer than 19 columns
    with pytest.raises(ValueError, match="19 columns"):
        RotamerLibraryBuilder.load_parameters(str(bad_file))


def test_unrecognized_residue_raises(tmp_path) -> None:
    bad_file = tmp_path / "bad.lib"
    # 19 columns, but an unrecognized residue name.
    bad_file.write_text(
        "XYZ\t1\t1\t1\t1\t1\t1\t50.0\t1.0\t50.0\t1.0\t0\t10\t0\t10\t0\t10\t0\t10\n"
    )
    with pytest.raises(ValueError, match="Unrecognized residue"):
        RotamerLibraryBuilder.load_parameters(str(bad_file))
