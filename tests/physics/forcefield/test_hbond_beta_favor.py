"""The hydrogen-bond sheet factor, ``beta_favor``: the factor on every pair
more than four residues apart.

mcpu08 defaults it to 2.0; legacy MCPU compiled 3.0 (``define.h``), which the
legacy-parity tests pass explicitly. The engine's energy is checked here
against an independent NumPy evaluation of the rule, and the incremental
energy against a full recompute at a value other than the default.
"""
from __future__ import annotations

import inspect
import json
from pathlib import Path

import mdtraj as md
import numpy as np
import pytest

from pymcpu import mcpu_core
from pymcpu.config import EngineSpec, load_config_auto
from pymcpu.forcefields.builders.hbond_builder import AMINO_INDEX, HydrogenBondBuilder
from pymcpu.forcefields.mcpu import MCPUForceField
from pymcpu.sampling import EngineSession
from tests.fixtures.context_builders import ATOL, resolve_test_pdb
from tests.legacy_parity.helpers.legacy_hbond_oracle import _angle, _dihedral_from_points

HBOND_GROUP = 4
HBOND_WEIGHT = 1.35 * 2.0  # outer weight x RDTHREE_CON
DIM = 9


def _heavy(pdb: Path | str) -> md.Trajectory:
    traj = md.load(str(pdb))
    return traj.atom_slice(traj.topology.select("not element H"))


def _context(forcefield: MCPUForceField, heavy: md.Trajectory) -> mcpu_core.Context:
    context = mcpu_core.Context(forcefield.create_system(heavy.topology))
    context.set_positions((forcefield.coords[0] * 10.0).T.astype(np.float32))
    context.calculate_total_energy(-1)
    return context


def _hbond_term(system) -> mcpu_core.HBondPotential:
    (term,) = [p for p in system.get_potentials() if isinstance(p, mcpu_core.HBondPotential)]
    return term


def _amide_h(n: np.ndarray, ca: np.ndarray, c_prev: np.ndarray) -> np.ndarray:
    """The engine's virtual amide H (virtual_amide_h.h): 1 A from N, opposite
    the sum of the N->CA and N->C(i-1) bond vectors."""
    v = (ca - n) + (c_prev - n)
    return n - v / np.linalg.norm(v)


def _reference_sums(heavy: md.Trajectory, table: np.ndarray,
                    seq_dep: np.ndarray) -> tuple[float, float]:
    """Weighted group-4 energy evaluated pair by pair in float64, every residue
    coil ('C'), virtual amide hydrogens, split into the pairs four apart and
    the pairs more than four apart, before any sheet factor."""
    top = heavy.topology
    xyz = heavy.xyz[0].astype(np.float64) * 10.0
    n = top.n_residues
    atoms = [{a.name: xyz[a.index] for a in top.residue(r).atoms} for r in range(n)]
    names = [top.residue(r).name for r in range(n)]
    bin_size = np.radians(20.0)

    def a_pca(n1, ca1, c1, n2, ca2, c2):
        return _angle(np.cross(n1 - ca1, c1 - ca1), np.cross(n2 - ca2, c2 - ca2))

    def a_bca(n1, ca1, c1, n2, ca2, c2):
        def bis(p, c, q):
            a, b = p - c, q - c
            s = a / np.linalg.norm(a) + b / np.linalg.norm(b)
            return s / np.linalg.norm(s)
        return _angle(bis(n1, ca1, c1), bis(n2, ca2, c2))

    deg = np.degrees(1.0)
    sums = {"near": 0.0, "far": 0.0}
    for i in range(1, n - 1):
        if names[i] == "PRO":
            continue
        d_n, d_ca, d_c = atoms[i]["N"], atoms[i]["CA"], atoms[i]["C"]
        p_n, p_ca, p_c = atoms[i - 1]["N"], atoms[i - 1]["CA"], atoms[i - 1]["C"]
        nx_ca = atoms[i + 1]["CA"]
        h = _amide_h(d_n, d_ca, p_c)
        for j in range(1, n - 1):
            diff = abs(i - j)
            if diff < 4:
                continue
            a_n, a_ca, a_c, a_o = atoms[j]["N"], atoms[j]["CA"], atoms[j]["C"], atoms[j]["O"]
            an_n, an_ca, an_c = atoms[j + 1]["N"], atoms[j + 1]["CA"], atoms[j + 1]["C"]
            ap_ca = atoms[j - 1]["CA"]
            if np.sum((h - a_o) ** 2) >= 2.5 ** 2:
                continue
            min1 = min(np.sum((p_ca - a_ca) ** 2), np.sum((p_ca - an_ca) ** 2))
            min2 = min(np.sum((d_ca - an_ca) ** 2), np.sum((d_ca - a_ca) ** 2))
            min3 = min(min1, min2)
            if diff == 4:
                if min1 > 5.8 ** 2 or min2 > 5.8 ** 2 or min3 > 5.5 ** 2:
                    continue
                cls = 0
            else:
                if min1 > 6.0 ** 2 or min2 > 6.0 ** 2 or min3 > 5.4 ** 2:
                    continue
                if _dihedral_from_points(p_c, d_n, d_ca, d_c) * deg + 180.0 > 150.0:
                    continue
                if 30.0 < _dihedral_from_points(p_n, p_ca, p_c, d_n) * deg + 180.0 < 210.0:
                    continue
                if _dihedral_from_points(a_c, an_n, an_ca, an_c) * deg + 180.0 > 150.0:
                    continue
                if 30.0 < _dihedral_from_points(a_n, a_ca, a_c, an_n) * deg + 180.0 < 210.0:
                    continue
                cls = 1 if _angle(nx_ca - p_ca, an_ca - ap_ca) < np.pi / 2 else 2
            angles = (
                a_pca(d_n, d_ca, d_c, a_n, a_ca, a_c), a_bca(d_n, d_ca, d_c, a_n, a_ca, a_c),
                a_pca(p_n, p_ca, p_c, an_n, an_ca, an_c), a_bca(p_n, p_ca, p_c, an_n, an_ca, an_c),
                a_pca(p_c, d_n, d_ca, a_ca, a_c, an_n), a_bca(p_c, d_n, d_ca, a_ca, a_c, an_n),
            )
            idx = (cls, *(min(int(a / bin_size), DIM - 1) for a in angles))
            e = float(table[idx]) * float(
                seq_dep[cls, AMINO_INDEX.get(names[i], 0), AMINO_INDEX.get(names[j], 0)])
            sums["near" if diff == 4 else "far"] += e
    return sums["near"] / 1000.0 * HBOND_WEIGHT, sums["far"] / 1000.0 * HBOND_WEIGHT


@pytest.fixture(scope="module")
def heavy() -> md.Trajectory:
    """Actin: hydrogen bonds four apart and more than four apart."""
    return _heavy(resolve_test_pdb())


@pytest.fixture(scope="module")
def reference(heavy) -> tuple[float, float]:
    ff = MCPUForceField(heavy)
    table = np.asarray(ff.hbond, dtype=np.float64).reshape((3,) + (DIM,) * 6)
    seq_dep = np.asarray(ff.hbond_seq_dep, dtype=np.float64).reshape(3, 20, 20)
    return _reference_sums(heavy, table, seq_dep)


def test_the_default_is_two_in_cpp_and_python(heavy) -> None:
    ff = MCPUForceField(heavy)
    assert ff.hbond_beta_favor == 2.0
    # None takes the parameter set's own factor (constants/paramset.json),
    # which mcpu08 does not set: test_paramset_manifest.py.
    assert inspect.signature(MCPUForceField).parameters["hbond_beta_favor"].default is None
    assert inspect.signature(HydrogenBondBuilder.build).parameters["beta_favor"].default == 2.0
    assert mcpu_core.HBondPotential(ff.hbond, ff.hbond_seq_dep).beta_favor == 2.0
    assert _hbond_term(ff.create_system(heavy.topology)).beta_favor == 2.0


def test_the_factor_reads_back(heavy) -> None:
    ff = MCPUForceField(heavy, hbond_beta_favor=3.0)
    assert ff.hbond_beta_favor == 3.0
    assert _hbond_term(ff.create_system(heavy.topology)).beta_favor == 3.0
    assert mcpu_core.HBondPotential(ff.hbond, ff.hbond_seq_dep, beta_favor=2.5).beta_favor == 2.5


def test_the_factor_is_keyword_only(heavy) -> None:
    ff = MCPUForceField(heavy)
    with pytest.raises(TypeError):
        mcpu_core.HBondPotential(ff.hbond, ff.hbond_seq_dep, 3.0)


@pytest.mark.parametrize("beta_favor", [1.0, 2.0, 3.0])
def test_the_factor_applies_to_pairs_more_than_four_apart_only(
    heavy, reference, beta_favor
) -> None:
    near, far = reference
    assert near < -50.0 and far < -10.0  # both kinds of pair are present
    ff = MCPUForceField(heavy, hbond_beta_favor=beta_favor)
    got = _context(ff, heavy).calculate_total_energy(HBOND_GROUP)
    assert got == pytest.approx(near + beta_favor * far, abs=1e-3)


def test_the_incremental_energy_matches_a_full_recompute(heavy) -> None:
    context = _context(MCPUForceField(heavy, hbond_beta_favor=3.0), heavy)
    context.set_hbond_ledger_check(True)
    integ = mcpu_core.Integrator(temperature=300.0, step_size_rad=0.05)
    integ.set_seed(2024)
    integ.verify_physics_consistency(context, num_steps=30, atol=ATOL)
    integ.run(context, 300)
    checks, mismatches = context.hbond_ledger_check_counts()
    assert checks > 0
    assert mismatches == 0


# 1e-46 and 1e300 are 0 and inf in the engine's float32.
@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf"), 1e-46, 1e300])
def test_a_factor_that_is_not_a_finite_number_above_zero_is_refused(heavy, bad) -> None:
    ff = MCPUForceField(heavy)
    with pytest.raises(ValueError, match="beta_favor must be a finite number above 0"):
        mcpu_core.HBondPotential(ff.hbond, ff.hbond_seq_dep, beta_favor=bad)
    with pytest.raises(ValueError, match="hbond_beta_favor must be a finite number above 0"):
        MCPUForceField(heavy, hbond_beta_favor=bad)


@pytest.mark.parametrize("bad", [True, "two", "3"])
def test_the_force_field_refuses_a_factor_that_is_not_a_number(heavy, bad) -> None:
    with pytest.raises(ValueError, match="hbond_beta_favor must be a finite number above 0"):
        MCPUForceField(heavy, hbond_beta_favor=bad)


# --- reaching it from a config ---

@pytest.mark.parametrize("name", ["run.yaml", "run.json"])
def test_a_config_reaches_the_hydrogen_bond_term(chignolin_pdb_path, tmp_path, name) -> None:
    path = tmp_path / name
    if name.endswith(".yaml"):
        path.write_text(
            f"pdb: {chignolin_pdb_path}\ntemperatures: [0.6]\n"
            "forcefield_options: {hbond_beta_favor: 3.0}\n"
        )
    else:
        path.write_text(json.dumps(
            {"mode": "folding", "pdb": str(chignolin_pdb_path),
             "forcefield_options": {"hbond_beta_favor": 3.0}}
        ))
    cfg = load_config_auto(path)
    assert cfg.forcefield_options == {"hbond_beta_favor": 3.0}
    session = EngineSession(EngineSpec.from_simulation_config(cfg))
    assert _hbond_term(session._ensure_sim().system).beta_favor == 3.0


def test_the_runners_build_the_factor_they_are_given(chignolin_pdb_path, tmp_path) -> None:
    from pymcpu.sampling.folding import FoldingRunner
    from pymcpu.sampling.replica_exchange import ReplicaExchange

    options = {"hbond_beta_favor": 3.0}
    folding = FoldingRunner(chignolin_pdb_path, output_dir=tmp_path / "fold",
                            forcefield_options=options, verbose=False)
    rex = ReplicaExchange(chignolin_pdb_path, temperatures=[0.5, 0.6], n_targets=[0.0],
                          k_bias=0.0, output_dir=tmp_path / "rex",
                          forcefield_options=options)
    for runner in (folding, rex):
        assert runner.forcefield.hbond_beta_favor == 3.0
        assert _hbond_term(runner.system).beta_favor == 3.0


def test_a_bad_factor_in_a_config_is_refused_when_the_force_field_is_built(
    chignolin_pdb_path,
) -> None:
    spec = EngineSpec(pdb=chignolin_pdb_path, forcefield_options={"hbond_beta_favor": -2})
    with pytest.raises(ValueError, match="hbond_beta_favor must be a finite number above 0"):
        EngineSession(spec)._ensure_sim()
