"""Per-chi "moved atom" topology regression tests (rotamer-library move).

The rotamer-library sidechain move (``MCIntegrator::apply_rotamer_at``)
needs, for each chi bond, the set of atoms that rotate with it -- distinct
from ``chi_atoms`` (the 4 atoms that merely *define* the dihedral). This is
sourced from legacy's ``amino_torsion.data`` Section 3
(``-#atoms affected by torsion- -all atoms affected-``), transcribed into
``standard_amino_acids.json``'s ``chi_moved_atoms`` field, exactly mirroring
how ``chi_atoms`` was already transcribed from Section 2 (see
``test_chi_topology.py``).

This is NOT a simple "drop the front atom each chi" suffix rule: ILE's chi2
moves only ``CD1`` -- a single atom that sits *after* ``CG2`` in storage
order, not "everything from CG2 onward". Each chi's moved-atom set is
resolved independently per residue from real atom names
(``MCPUForceField._initialize_attributes``), with contiguity in engine atom
order asserted at build time as the safety net for the rotation-cascade
implementation, which reuses the existing contiguous-range rotation
primitive rather than a generic arbitrary-atom-subset one.
"""

from __future__ import annotations

import json
from pathlib import Path

import mdtraj as md
import pytest

from pymcpu.forcefields.mcpu import MCPUForceField
from tests.fixtures.context_builders import resolve_test_pdb

AMINO_TORSION_DATA = (
    Path(__file__).resolve().parents[3]
    / "examples"
    / "actin"
    / "legacy"
    / "config_files"
    / "amino_torsion.data"
)
STANDARD_AMINO_ACIDS_JSON = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "pymcpu"
    / "parameters"
    / "pretrained"
    / "mcpu08"
    / "constants"
    / "standard_amino_acids.json"
)


def _parse_amino_torsion_section3(path: Path) -> dict[str, dict[int, list[str]]]:
    """Parse amino_torsion.data's Section 3 (the per-chi moved-atom table).

    Independent, from-scratch parser -- deliberately not shared code with
    whatever transcribed the JSON, so this test can catch drift.
    """
    lines = path.read_text().splitlines()
    star_indices = [i for i, line in enumerate(lines) if line.strip() == "*"]
    sec3_start = star_indices[2] + 1
    sec3_end = star_indices[3] if len(star_indices) > 3 else len(lines)

    moved_by_res: dict[str, dict[int, list[str]]] = {}
    for line in lines[sec3_start:sec3_end]:
        line = line.strip()
        if not line or line.startswith("!!"):
            continue
        tokens = line.split()
        res, chi_idx, n_atoms = tokens[0], int(tokens[1]), int(tokens[2])
        atoms = tokens[3:]
        assert len(atoms) == n_atoms, f"malformed section-3 line: {line!r}"
        moved_by_res.setdefault(res, {})[chi_idx] = atoms

    return moved_by_res


def test_json_chi_moved_atoms_matches_legacy_amino_torsion_data() -> None:
    """standard_amino_acids.json's "chi_moved_atoms" must exactly match
    legacy's amino_torsion.data Section 3, for every residue with at least
    one chi -- the source of truth this table depends on."""
    expected = _parse_amino_torsion_section3(AMINO_TORSION_DATA)
    actual = json.loads(STANDARD_AMINO_ACIDS_JSON.read_text())

    for res, entry in actual.items():
        n_tors = entry["ntorsions"]
        if n_tors == 0:
            assert not entry.get("chi_moved_atoms"), (
                f"{res} has ntorsions=0 but nonempty chi_moved_atoms"
            )
            continue
        assert res in expected, f"no legacy section-3 rows found for {res}"
        expected_rows = [expected[res][k] for k in range(n_tors)]
        assert entry["chi_moved_atoms"] == expected_rows, (
            f"{res}: JSON chi_moved_atoms {entry['chi_moved_atoms']} != legacy "
            f"amino_torsion.data {expected_rows}"
        )


def test_chi_moved_atoms_are_nested_subsets() -> None:
    """chi_{k+1}'s moved-atom set must be a subset of chi_k's (a more
    proximal chi's rotation carries every more-distal chi's atoms along with
    it) -- a distinct property from range-contiguity, checked purely as
    atom-name sets."""
    actual = json.loads(STANDARD_AMINO_ACIDS_JSON.read_text())
    for res, entry in actual.items():
        moved = entry.get("chi_moved_atoms", [])
        for k in range(len(moved) - 1):
            outer, inner = set(moved[k]), set(moved[k + 1])
            assert inner <= outer, (
                f"{res}: chi{k + 1} moved-atom set {inner} is not a subset of "
                f"chi{k}'s {outer}"
            )


def test_ile_chi2_moved_atoms_is_cd1_only_not_suffix() -> None:
    """Regression pin for the branched-sidechain edge case: ILE's chi2 bond
    (CB-CG1 axis) moves only CD1 -- NOT {CG2, CD1} -- since CG2 hangs off CB
    on the fixed side of that rotation. Guards against a future
    "simplification" back to a naive positional-suffix formula that would
    silently move CG2 too."""
    actual = json.loads(STANDARD_AMINO_ACIDS_JSON.read_text())
    assert actual["ILE"]["chi_moved_atoms"][1] == ["CD1"]
    assert set(actual["ILE"]["chi_moved_atoms"][1]) != {"CG2", "CD1"}


@pytest.fixture(scope="module")
def acta_forcefield() -> MCPUForceField:
    pdb_path = resolve_test_pdb()
    traj = md.load(str(pdb_path))
    heavy = traj.atom_slice(traj.topology.select("not element H"))
    return MCPUForceField(heavy)


def test_chi_moved_ranges_are_contiguous_for_every_residue(
    acta_forcefield: MCPUForceField,
) -> None:
    """Runtime confirmation that the Python-side contiguity assertion in
    _initialize_attributes actually ran (would have raised otherwise) and
    that every valid chi slot resolved to a well-formed non-empty [lo, hi)
    range (or the [-1, -1] sentinel for k >= that residue's ntorsions)."""
    ranges = acta_forcefield.chi_moved_atom_ranges
    assert len(ranges) == acta_forcefield.n_res
    for res_idx, res_ranges in enumerate(ranges):
        assert len(res_ranges) == 4
        for lo, hi in res_ranges:
            if lo == -1:
                assert hi == -1
            else:
                assert hi > lo >= 0, (
                    f"residue {res_idx}: invalid chi-moved range [{lo}, {hi})"
                )


def test_chi_moved_ranges_match_ntorsions_slot_count(
    acta_forcefield: MCPUForceField,
) -> None:
    """Exactly ntorsions[r] slots are populated (non-sentinel) per residue,
    matching chi_atom_indices' own k < ntorsions convention."""
    for res_idx, res_ranges in enumerate(acta_forcefield.chi_moved_atom_ranges):
        n_tors = sum(1 for lo, _ in res_ranges if lo != -1)
        # Populated slots must be a prefix (k=0..n_tors-1), matching how
        # chi_atom_indices/chi_moved_atoms are always built up to ntorsions.
        for k, (lo, hi) in enumerate(res_ranges):
            if k < n_tors:
                assert lo != -1, f"residue {res_idx} chi{k}: unexpected sentinel"
            else:
                assert lo == -1 and hi == -1, (
                    f"residue {res_idx} chi{k}: expected sentinel, got [{lo}, {hi})"
                )
