"""Sidechain chi-angle topology regression tests.

``computeTorsions()``/``recompute_sidechain_torsion()`` used to compute chi
angles POSITIONALLY (the Nth stored sidechain atom in array order = the Nth
atom in the real dihedral chain). This is wrong for any branched sidechain:
ILE's real chi2 is CA-CB-CG1-CD1, but the positional formula read
CA-CB-CG1-CG2 (CG2 is a branch off CB, not on the CG1->CD1 chain) -- an
improper angle, not chi2. It was also inconsistent across
``atom_reorder_mode`` settings (confirmed for TYR) since the DFS-based
reorder doesn't preserve the positional-chi assumption uniformly.

Fixed by resolving each residue's chi atoms by NAME (from
``standard_amino_acids.json``'s ``chi_atoms``, transcribed from legacy's
``amino_torsion.data``) into a real per-residue atom-index table
(``System.get_chi_atom_indices()``), correctly remapped under atom permutation.

These tests are independent of legacy MCPU or any hardcoded numeric
target -- correctness is checked against mdtraj's own dihedral computation
(ground truth, computed directly from the same PDB coordinates) and via
self-consistency (reorder-mode invariance), not against a frozen baseline.
"""

from __future__ import annotations

import json
from pathlib import Path

import mdtraj as md
import numpy as np
import pytest

from pymcpu import mcpu_core
from pymcpu.forcefields.mcpu import MCPUForceField
from tests.fixtures.context_builders import build_raw_context, resolve_test_pdb

AMINO_TORSION_DATA = (
    Path(__file__).resolve().parents[3]
    / "examples"
    / "actin"
    / "legacy"
    / "config_files"
    / "amino_torsion.data"
)
def _standard_amino_acids_json() -> Path:
    """Resolve the shipped amino-acid template through the params API.

    Deliberately not a hardcoded path: this used to point at
    ``pymcpu/parameters/...``, which was a symlink into the raw dev tree. That
    symlink was removed because scikit-build-core dereferenced it into the
    wheel (1.36 GB of tables). Going through ``params_path`` means the test
    works whichever source answers -- dev tree, in-wheel archive, or a
    pre-staged ``MCPU_PARAMS_DIR``.
    """
    from pymcpu.params import params_path, required_files

    return params_path("mcpu08", required_files("mcpu08")["amino acids template"])


def _parse_amino_torsion_chi_atoms(path: Path) -> dict[str, list[list[str]]]:
    """Parse amino_torsion.data's Section 2 (the per-chi atom-name table),
    the same table transcribed into standard_amino_acids.json's "chi_atoms".
    Independent, from-scratch parser -- deliberately not shared code with
    whatever originally generated the JSON, so this test can catch drift.
    """
    lines = path.read_text().splitlines()
    star_indices = [i for i, line in enumerate(lines) if line.strip() == "*"]
    sec2_start, sec2_end = star_indices[1] + 1, star_indices[2]

    chi_by_res: dict[str, dict[int, list[str]]] = {}
    for line in lines[sec2_start:sec2_end]:
        line = line.strip()
        if not line or line.startswith("!!"):
            continue
        res, chi_idx, a1, a2, a3, a4, offset = line.split()
        assert offset == "0", f"unexpected nonzero offset: {line!r}"
        chi_by_res.setdefault(res, {})[int(chi_idx)] = [a1, a2, a3, a4]

    return {
        res: [rows[i] for i in range(len(rows))]
        for res, rows in chi_by_res.items()
    }


def test_json_chi_atoms_matches_legacy_amino_torsion_data() -> None:
    """standard_amino_acids.json's "chi_atoms" must exactly match legacy's
    amino_torsion.data, for every residue that has at least one chi -- this
    is the source of truth the fix depends on, and it must never silently
    drift from it."""
    expected = _parse_amino_torsion_chi_atoms(AMINO_TORSION_DATA)
    actual = json.loads(_standard_amino_acids_json().read_text())

    for res, entry in actual.items():
        n_tors = entry["ntorsions"]
        if n_tors == 0:
            assert not entry.get("chi_atoms"), f"{res} has ntorsions=0 but nonempty chi_atoms"
            continue
        assert res in expected, f"no legacy chi rows found for {res}"
        assert entry["chi_atoms"] == expected[res], (
            f"{res}: JSON chi_atoms {entry['chi_atoms']} != legacy "
            f"amino_torsion.data {expected[res]}"
        )


@pytest.fixture(scope="module")
def acta_traj() -> md.Trajectory:
    pdb_path = resolve_test_pdb()
    traj = md.load(str(pdb_path))
    return traj.atom_slice(traj.topology.select("not element H"))


def _engine_chi_angles(traj: md.Trajectory, param_dir: str | None = None) -> list[list[float]]:
    kwargs = {"param_dir": param_dir} if param_dir else {}
    ff = MCPUForceField(traj, **kwargs)
    system = ff.create_system(traj.topology)
    ctx = mcpu_core.Context(system)
    coords = (np.asarray(ff.coords[0], dtype=float) * 10.0).T.astype("float32")
    ctx.set_positions(coords)
    state = ctx.get_state()
    return [list(r.chi_angles) for r in state.sidechain_torsions]


def test_ile_chi2_matches_ground_truth_ca_cb_cg1_cd1(acta_traj: md.Trajectory) -> None:
    """ILE's chi2 must equal the true CA-CB-CG1-CD1 dihedral (computed
    independently via mdtraj, no MCPU code involved), and must NOT equal the
    old positional (CA-CB-CG1-CG2) improper angle -- a regression guard
    against reintroducing the bug."""
    ile_residues = [r.index for r in acta_traj.topology.residues if r.name == "ILE"]
    assert ile_residues, "test PDB must contain ILE residues for this test to be meaningful"

    chi_angles = _engine_chi_angles(acta_traj)

    for res_idx in ile_residues:
        residue = acta_traj.topology.residue(res_idx)
        by_name = {a.name: a.index for a in residue.atoms}
        true_quad = [by_name["CA"], by_name["CB"], by_name["CG1"], by_name["CD1"]]
        wrong_quad = [by_name["CA"], by_name["CB"], by_name["CG1"], by_name["CG2"]]
        true_chi2 = float(md.compute_dihedrals(acta_traj, [true_quad])[0][0])
        wrong_chi2 = float(md.compute_dihedrals(acta_traj, [wrong_quad])[0][0])
        engine_chi2 = chi_angles[res_idx][1]

        def ang_diff(a: float, b: float) -> float:
            d = abs(a - b)
            return min(d, abs(d - 2 * np.pi))

        assert ang_diff(engine_chi2, true_chi2) < 1e-4, (
            f"res {res_idx}: engine chi2={engine_chi2:.4f} != true CD1-based "
            f"chi2={true_chi2:.4f}"
        )
        assert ang_diff(engine_chi2, wrong_chi2) > 1e-3, (
            f"res {res_idx}: engine chi2 matches the OLD (wrong, CG2-based) "
            "value -- the positional-chi bug appears to have regressed"
        )


def test_chi_angles_invariant_across_atom_reorder_mode(acta_traj: md.Trajectory) -> None:
    """Chi angles must be identical regardless of atom_reorder_mode -- the
    whole point of resolving chi atoms by NAME instead of array position is
    that reordering the array can no longer change which physical atoms a
    chi dihedral is computed from. Checked across every residue (not just
    the originally-reported TYR case)."""
    per_mode: dict[str, list[list[float]]] = {}
    for mode in ("off", "init_only"):
        ctx, topology = build_raw_context(reorder=mode)
        state = ctx.get_state()
        per_mode[mode] = [list(r.chi_angles) for r in state.sidechain_torsions]

    n_res = len(per_mode["off"])
    assert n_res == len(per_mode["init_only"])
    for res_idx in range(n_res):
        a, b = per_mode["off"][res_idx], per_mode["init_only"][res_idx]
        for k, (x, y) in enumerate(zip(a, b)):
            assert x == pytest.approx(y, abs=1e-6), (
                f"res {res_idx} ({topology.residue(res_idx).name}) chi[{k}]: "
                f"off={x} init_only={y}"
            )


def test_tyr_chi2_uses_cd2_not_cd1(acta_traj: md.Trajectory) -> None:
    """Legacy's own chi2 definition for TYR is CA-CB-CG-CD2 (not the IUPAC
    CD1) -- confirmed directly from amino_torsion.data. Pin this specific,
    easy-to-get-backwards convention."""
    tyr_residues = [r.index for r in acta_traj.topology.residues if r.name == "TYR"]
    assert tyr_residues

    chi_angles = _engine_chi_angles(acta_traj)
    for res_idx in tyr_residues:
        residue = acta_traj.topology.residue(res_idx)
        by_name = {a.name: a.index for a in residue.atoms}
        true_quad = [by_name["CA"], by_name["CB"], by_name["CG"], by_name["CD2"]]
        true_chi2 = float(md.compute_dihedrals(acta_traj, [true_quad])[0][0])
        engine_chi2 = chi_angles[res_idx][1]
        d = abs(engine_chi2 - true_chi2)
        assert min(d, abs(d - 2 * np.pi)) < 1e-4
