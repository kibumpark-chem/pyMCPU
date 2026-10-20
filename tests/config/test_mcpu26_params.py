"""The mcpu26 parameter set: what it ships and how it loads.

mcpu26 (legacy form) is mcpu08's energy function with tables re-estimated
from PISCES 2026 and the fitted weights folded into them. Its tables decode
to exactly the files the builder wrote (their md5s are pinned below, from the
builder's ``MD5SUMS``), its constants are byte copies of mcpu08's, and its
``constants/paramset.json`` gives its sheet factor and the hard core its
weights were fitted with, which a force field takes when it is given none.
Its tables live in ``mcpu26_params/``, so a folder of
mcpu08's given for it fails loudly, as one of its own given for mcpu08 does.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mdtraj as md
import numpy as np
import pytest

from pymcpu import mcpu_core, params
from pymcpu.forcefields import load_forcefield
from pymcpu.forcefields.mcpu import MCPUForceField
from pymcpu.paramcodec import decode_table, read_header
from pymcpu.params import (
    PARAMSET_FORMAT,
    PARAMSET_MANIFEST,
    ParamsError,
    constants_files,
    ensure_params,
    required_files,
    table_layout,
)
from pymcpu.runners import default_example_pdb

_PACKAGE = Path(params.__file__).resolve().parent / "data" / "params"
_ARCHIVE = _PACKAGE / "mcpu26" / "tables.npz"

#: md5 of each decoded table: the builder's MD5SUMS of the parameter folder
#: mcpu26_v1b_w5 (2026-10-19), from which the set was taken.
TABLE_MD5 = {
    "mu_potentials": "2bf7bdfe322c6da22de6f22636ec7ea3",
    "triplet_potentials": "c9d42d2bffbc204c72442f686bae5f5f",
    "sidechain_triplet_potentials": "2b7c345d21f6361bc154dedddca5b11f",
    "hydrogen_bond_potentials": "051d88e6c836ffde310474a5339e337d",
    "hbond_seq_dep": "b43739f44ba93b409783d2dcf0186071",
    "aromatic_potentials": "4cd37da33b716f3e105a668524f6c399",
}
#: The fitted sheet factor: float32(3 x hbfar / hb4) of the final weights.
BETA_FAVOR = 2.3993940353393555
#: The hard core the weights were fitted with: clash_alpha, clash_bond_exclusion.
HARD_CORE = (0.85, 3)
#: w_new / w_legacy, already multiplied into each term's table.
FOLDED_WEIGHTS = {
    "mu": 1.0577709144050786,
    "bb_torsion": 0.8199679176379738,
    "sc_torsion": 0.18420013147238207,
    "hbond": 1.1580308223754334,
    "aromatic": 0.8627578878628941,
}


@pytest.fixture(scope="module")
def heavy() -> md.Trajectory:
    traj = md.load(str(default_example_pdb()))
    return traj.atom_slice(traj.topology.select("not element H"))


def _shipped_manifest() -> dict:
    return json.loads((_PACKAGE / "mcpu26" / PARAMSET_MANIFEST).read_text(encoding="utf-8"))


def _hbond_term(system) -> mcpu_core.HBondPotential:
    (term,) = [p for p in system.get_potentials() if isinstance(p, mcpu_core.HBondPotential)]
    return term


# --- what ships ---

@pytest.mark.parametrize(
    "name",
    [name for name in TABLE_MD5 if name != "sidechain_triplet_potentials"],
)
def test_the_tables_decode_to_the_builders_files(name: str) -> None:
    decoded = decode_table(_ARCHIVE, name, verify=True)
    assert hashlib.md5(decoded.tobytes()).hexdigest() == TABLE_MD5[name]


@pytest.mark.slow
def test_the_large_table_decodes_to_the_builders_file() -> None:
    decoded = decode_table(_ARCHIVE, "sidechain_triplet_potentials", verify=True)
    assert hashlib.md5(decoded.tobytes()).hexdigest() == TABLE_MD5["sidechain_triplet_potentials"]


def test_the_archive_holds_the_six_tables_of_the_layout() -> None:
    header = read_header(_ARCHIVE)
    assert header["set"] == "mcpu26"
    assert set(header["tables"]) == set(TABLE_MD5) == set(table_layout("mcpu26"))
    assert all(rel.startswith("mcpu26_params/") for rel in table_layout("mcpu26").values())


def test_the_manifest_records_the_fit() -> None:
    manifest = _shipped_manifest()
    assert manifest["format"] == PARAMSET_FORMAT
    assert manifest["name"] == "mcpu26"
    assert manifest["hbond_beta_favor"] == BETA_FAVOR
    assert (manifest["clash_alpha"], manifest["clash_bond_exclusion"]) == HARD_CORE
    assert manifest["folded_weights"] == FOLDED_WEIGHTS
    # The md5 of each table, by its path in the set's own layout, so a
    # force field built from the set compares every table with the builder's.
    assert manifest["provenance"]["md5"] == {
        table_layout("mcpu26")[name]: md5 for name, md5 in TABLE_MD5.items()
    }


def test_the_constants_are_mcpu08s() -> None:
    """``rotamer_builder`` reads mcpu08's packaged template and rotamer library
    whatever the set is, so mcpu26 must ship byte copies of them; the other
    constants are mcpu08's too."""
    assert constants_files("mcpu26") == constants_files("mcpu08")
    for rel in constants_files("mcpu26"):
        assert (_PACKAGE / "mcpu26" / rel).read_bytes() == (_PACKAGE / "mcpu08" / rel).read_bytes(), rel


# --- how it loads ---

def test_a_force_field_takes_the_sets_sheet_factor(heavy) -> None:
    ff = MCPUForceField(heavy, param_set="mcpu26")
    assert ff.param_set == "mcpu26"
    assert ff.paramset_manifest["name"] == "mcpu26"
    assert ff.hbond_beta_favor == BETA_FAVOR
    assert _hbond_term(ff.create_system(heavy.topology)).beta_favor == np.float32(BETA_FAVOR)
    assert ff.param_signature == params.param_signature(ensure_params("mcpu26"), "mcpu26")
    # An explicit factor wins, as for any set.
    assert MCPUForceField(heavy, param_set="mcpu26", hbond_beta_favor=3.0).hbond_beta_favor == 3.0


def test_load_forcefield_takes_the_sets_sheet_factor() -> None:
    pdb = default_example_pdb()
    assert load_forcefield(pdb, param_set="mcpu26").hbond_beta_favor == BETA_FAVOR
    assert load_forcefield(pdb, param_set="mcpu26",
                           options={"hbond_beta_favor": 2.5}).hbond_beta_favor == 2.5
    # mcpu08 keeps its own: no manifest, 2.0.
    assert load_forcefield(pdb).hbond_beta_favor == 2.0


def _hard_core(ff) -> tuple[float, int, bool]:
    return ff.clash_alpha, ff.clash_bond_exclusion, ff.clash_start_exemption


def test_a_force_field_takes_the_hard_core_the_set_was_fitted_with(heavy) -> None:
    pdb = default_example_pdb()
    assert _hard_core(MCPUForceField(heavy, param_set="mcpu26")) == (*HARD_CORE, True)
    assert _hard_core(load_forcefield(pdb, param_set="mcpu26")) == (*HARD_CORE, True)
    # Values given explicitly win, each on its own.
    assert _hard_core(load_forcefield(pdb, param_set="mcpu26", options={
        "clash_alpha": 0.75, "clash_bond_exclusion": 0})) == (0.75, 0, True)
    assert _hard_core(MCPUForceField(heavy, param_set="mcpu26", clash_alpha=0.8)) == (0.8, 3, True)
    assert _hard_core(MCPUForceField(heavy, param_set="mcpu26",
                                     clash_start_exemption=False)) == (*HARD_CORE, False)
    # mcpu08 has no manifest: its own hard core.
    assert _hard_core(MCPUForceField(heavy)) == (0.75, 0, True)


def test_mcpu26_scores_with_its_own_tables(heavy) -> None:
    """The structure's contact energy differs between the two sets: a force
    field of mcpu26 reads mcpu26's tables, not mcpu08's."""
    def energies(ff):
        context = mcpu_core.Context(ff.create_system(heavy.topology))
        context.set_positions((ff.coords[0] * 10.0).T.astype(np.float32))
        context.calculate_total_energy(-1)
        return [context.calculate_total_energy(group) for group in range(1, 6)]

    assert energies(MCPUForceField(heavy, param_set="mcpu26"))[0] != energies(MCPUForceField(heavy))[0]


# --- a folder of the other set fails loudly ---

def _linked(root: Path, set_name: str) -> Path:
    """Links to every file of ``set_name`` that its layout names."""
    src = ensure_params(set_name)
    for rel in params.signed_files(set_name):
        if (src / rel).is_file():
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            (root / rel).symlink_to(src / rel)
    return root


def test_an_mcpu08_params_dir_given_for_mcpu26_fails_loudly(heavy, tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("MCPU_PARAMS_DIR", raising=False)
    mcpu08 = _linked(tmp_path / "mcpu08", "mcpu08")
    monkeypatch.setenv("MCPU_PARAMS_DIR", str(mcpu08))
    with pytest.raises(ParamsError) as refused:
        ensure_params("mcpu26")
    message = str(refused.value)
    for rel in table_layout("mcpu26").values():
        assert rel in message
    assert "laid out as 'mcpu26', containing constants/ and mcpu26_params/" in message
    with pytest.raises(FileNotFoundError, match="mcpu26_params/mu_potentials.bin"):
        MCPUForceField(heavy, param_set="mcpu26")
    # Given as param_dir, the force field names the first file it lacks.
    with pytest.raises(FileNotFoundError, match=r"Expected at: .*/mcpu26_params/mu_potentials\.bin"):
        MCPUForceField(heavy, param_set="mcpu26", param_dir=str(mcpu08))


def test_an_mcpu26_params_dir_given_for_mcpu08_fails_loudly(heavy, tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("MCPU_PARAMS_DIR", raising=False)
    mcpu26 = _linked(tmp_path / "mcpu26", "mcpu26")
    monkeypatch.setenv("MCPU_PARAMS_DIR", str(mcpu26))
    with pytest.raises(ParamsError) as refused:
        ensure_params("mcpu08")
    for rel in table_layout("mcpu08").values():
        assert rel in str(refused.value)
    # Given as param_dir, its manifest names its set before any table is read.
    with pytest.raises(ValueError, match="describes parameter set 'mcpu26', but the folder was "
                                         "loaded as parameter set 'mcpu08'"):
        MCPUForceField(heavy, param_dir=str(mcpu26))


def test_an_mcpu26_folder_without_its_manifest_is_incomplete(heavy, tmp_path, monkeypatch) -> None:
    """Without its manifest the set would load with mcpu08's sheet factor, so
    the layout requires it."""
    monkeypatch.delenv("MCPU_PARAMS_DIR", raising=False)
    folder = _linked(tmp_path / "f", "mcpu26")
    (folder / PARAMSET_MANIFEST).unlink()
    monkeypatch.setenv("MCPU_PARAMS_DIR", str(folder))
    with pytest.raises(ParamsError, match=r"Missing:\n  - constants/paramset\.json"):
        ensure_params("mcpu26")
    with pytest.raises(FileNotFoundError, match="parameter set manifest"):
        MCPUForceField(heavy, param_set="mcpu26", param_dir=str(folder))
    assert required_files("mcpu26")["parameter set manifest"] == PARAMSET_MANIFEST
