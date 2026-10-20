"""A parameter set's own defaults: ``constants/paramset.json``.

A parameter folder may carry a manifest (format ``pymcpu-paramset/1``) that
names its set and can give its hydrogen-bond sheet factor. ``MCPUForceField``
uses that factor when it is not given one (``hbond_beta_favor=None``, the
default), and 2.0 when the folder has no manifest, as mcpu08 has none; a value
given explicitly always wins. Its hard core, ``clash_alpha`` and
``clash_bond_exclusion``, resolves the same way, else to mcpu08's 0.75 and 0.
``folded_weights`` is a record only. The
manifest is checked strictly: unknown or repeated keys, values of the wrong
kind, a name other than the set the folder is loaded as, and listed files
whose md5 has changed are refused.

The folders here link mcpu08's files and are loaded as mcpu08, so their
manifests name mcpu08, and each check compares a build from a manifest with a
build given the same factor explicitly.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import mdtraj as md
import numpy as np
import pytest

from pymcpu import checkpointing, mcpu_core, params
from pymcpu.checkpointing import load_checkpoint
from pymcpu.config import EngineSpec, load_config_auto
from pymcpu.forcefields import load_forcefield
from pymcpu.forcefields.mcpu import MCPUForceField
from pymcpu.params import PARAMSET_FORMAT, PARAMSET_MANIFEST, load_paramset_manifest
from pymcpu.sampling import EngineSession
from pymcpu.sampling.folding import FoldingRunner
from tests.fixtures.context_builders import resolve_test_pdb
from tests.helpers.param_folders import linked_param_folder, register_set_like_mcpu08

HBOND_GROUP = 4
#: A small table to list in provenance.md5 and change: nine float32 energies.
TABLE = "mcpu_params/aromatic_potentials.bin"
MD5_ZERO = "0" * 32


def _heavy(pdb: Path | str) -> md.Trajectory:
    traj = md.load(str(pdb))
    return traj.atom_slice(traj.topology.select("not element H"))


@pytest.fixture(scope="module")
def heavy() -> md.Trajectory:
    """Actin, which has hydrogen bonds more than four apart, so the factor
    shows in its energy; chignolin's native structure has none."""
    return _heavy(resolve_test_pdb())


@pytest.fixture(scope="module")
def small(chignolin_pdb_path) -> md.Trajectory:
    """Chignolin, for the checks that need no energy."""
    return _heavy(chignolin_pdb_path)


@pytest.fixture(scope="module")
def explicit_energies(heavy):
    """Energies of actin built from mcpu08 with the factor given explicitly,
    one evaluation per factor (each takes a few seconds)."""
    cache: dict[float, dict[int, float]] = {}

    def energies(factor: float) -> dict[int, float]:
        if factor not in cache:
            built, cache[factor] = _evaluate(MCPUForceField(heavy, hbond_beta_favor=factor), heavy)
            assert built == factor
        return cache[factor]

    return energies


def _manifest(**fields: Any) -> dict[str, Any]:
    """A manifest for a folder loaded as mcpu08, the default param_set."""
    return {"format": PARAMSET_FORMAT, "name": "mcpu08", **fields}


def _folder(root: Path, manifest: dict[str, Any] | str | None = None) -> Path:
    """mcpu08's files linked into ``root``, plus a manifest (a dict, or the
    file's exact text) unless it is None."""
    folder = linked_param_folder(root)
    if manifest is not None:
        text = manifest if isinstance(manifest, str) else json.dumps(manifest)
        (folder / PARAMSET_MANIFEST).write_text(text)
    return folder


def _own_copy(folder: Path, rel: str) -> Path:
    """Replace the link at ``rel`` with a file of its own, which a test may change."""
    path = folder / rel
    data = path.read_bytes()
    path.unlink()
    path.write_bytes(data)
    return path


def _hbond_term(system) -> mcpu_core.HBondPotential:
    (term,) = [p for p in system.get_potentials() if isinstance(p, mcpu_core.HBondPotential)]
    return term


def _evaluate(ff: MCPUForceField, traj: md.Trajectory) -> tuple[float, dict[int, float]]:
    """The factor of the H-bond term in a new system from ``ff``, and each
    energy group's energy of the structure in that system."""
    system = ff.create_system(traj.topology)
    factor = _hbond_term(system).beta_favor
    context = mcpu_core.Context(system)
    context.set_positions((ff.coords[0] * 10.0).T.astype(np.float32))
    context.calculate_total_energy(-1)
    return factor, {group: context.calculate_total_energy(group) for group in range(1, 6)}


# --- which factor a force field uses ---

def test_without_a_manifest_the_factor_is_two_and_mcpu08_is_unchanged(
    heavy, explicit_energies, tmp_path
) -> None:
    default = MCPUForceField(heavy)
    assert default.paramset_manifest is None
    assert default.hbond_beta_favor == 2.0
    assert _evaluate(default, heavy) == (2.0, explicit_energies(2.0))
    # 0.1.0's weighted H-bond energy of the actin native at 2.0 (f8dca3d).
    assert explicit_energies(2.0)[HBOND_GROUP] == pytest.approx(-149.283, abs=5e-4)
    # The same files in another folder, still without a manifest.
    linked = MCPUForceField(heavy, param_dir=str(_folder(tmp_path / "plain")))
    assert linked.paramset_manifest is None
    assert linked.hbond_beta_favor == 2.0
    assert _evaluate(linked, heavy) == (2.0, explicit_energies(2.0))


def test_the_manifest_factor_gives_the_energies_of_an_explicit_one(
    heavy, explicit_energies, tmp_path
) -> None:
    folder = _folder(tmp_path / "three", _manifest(hbond_beta_favor=3))
    ff = MCPUForceField(heavy, param_dir=str(folder))
    assert ff.paramset_manifest == _manifest(hbond_beta_favor=3.0)
    assert ff.hbond_beta_favor == 3.0 and isinstance(ff.hbond_beta_favor, float)
    factor, energies = _evaluate(ff, heavy)
    assert factor == 3.0
    assert energies == explicit_energies(3.0)
    # The factor shows: only the H-bond term differs from 2.0.
    assert energies[HBOND_GROUP] < explicit_energies(2.0)[HBOND_GROUP] < 0.0
    assert {g: e for g, e in energies.items() if g != HBOND_GROUP} == {
        g: e for g, e in explicit_energies(2.0).items() if g != HBOND_GROUP}


@pytest.mark.parametrize("explicit", [2.5, 2.0])
def test_an_explicit_factor_wins_over_the_manifest(
    heavy, explicit_energies, tmp_path, explicit
) -> None:
    folder = _folder(tmp_path / "three", _manifest(hbond_beta_favor=3.0))
    ff = MCPUForceField(heavy, param_dir=str(folder), hbond_beta_favor=explicit)
    assert ff.hbond_beta_favor == explicit
    assert _evaluate(ff, heavy) == (explicit, explicit_energies(explicit))


def test_a_manifest_without_a_factor_leaves_two(small, tmp_path) -> None:
    folder = _folder(tmp_path / "named", _manifest(description="d"))
    ff = MCPUForceField(small, param_dir=str(folder))
    assert ff.paramset_manifest == _manifest(description="d")
    assert ff.hbond_beta_favor == 2.0


def test_folded_weights_are_a_record_only(heavy, explicit_energies, tmp_path) -> None:
    weights = {"mu": 1.25, "bb_torsion": 0.5, "sc_torsion": 2.0, "hbond": 0.75, "aromatic": 3}
    folder = _folder(tmp_path / "w", _manifest(folded_weights=weights))
    ff = MCPUForceField(heavy, param_dir=str(folder))
    assert ff.paramset_manifest["folded_weights"] == {**weights, "aromatic": 3.0}
    assert _evaluate(ff, heavy) == (2.0, explicit_energies(2.0))


def test_a_folded_weight_may_be_zero(small, tmp_path) -> None:
    """A fit with weights of 0 or above (NNLS) can switch a term off: its
    table is multiplied by 0, and the manifest records that 0. The sheet
    factor, which the engine applies, must stay above 0."""
    weights = {"mu": 0, "hbond": 0.0, "sc_torsion": 1e-46}
    folder = _folder(tmp_path / "z", _manifest(folded_weights=weights))
    recorded = {"mu": 0.0, "hbond": 0.0, "sc_torsion": 1e-46}
    assert load_paramset_manifest(folder)["folded_weights"] == recorded
    ff = MCPUForceField(small, param_dir=str(folder))
    assert ff.paramset_manifest["folded_weights"] == recorded
    assert ff.hbond_beta_favor == 2.0


def test_a_config_takes_the_sets_factor_unless_it_gives_one(chignolin_pdb_path, tmp_path) -> None:
    folder = _folder(tmp_path / "three", _manifest(hbond_beta_favor=3.0))
    assert load_forcefield(chignolin_pdb_path, param_dir=folder).hbond_beta_favor == 3.0
    assert load_forcefield(chignolin_pdb_path, options={"hbond_beta_favor": 2.5},
                           param_dir=folder).hbond_beta_favor == 2.5
    for options, expected in (("", 3.0), ("forcefield_options: {hbond_beta_favor: 2.5}\n", 2.5)):
        config = tmp_path / "run.yaml"
        config.write_text(
            f"pdb: {chignolin_pdb_path}\ntemperatures: [0.6]\nparam_dir: {folder}\n{options}")
        session = EngineSession(EngineSpec.from_simulation_config(load_config_auto(config)))
        assert _hbond_term(session._ensure_sim().system).beta_favor == expected


def test_a_checkpoint_records_the_factor_the_run_used(chignolin_pdb_path, tmp_path) -> None:
    folder = _folder(tmp_path / "three", _manifest(hbond_beta_favor=3.0))

    def folding(resume: bool = False, **options: float) -> FoldingRunner:
        return FoldingRunner(
            chignolin_pdb_path, output_dir=str(tmp_path / "out"),
            checkpoint_dir=str(tmp_path / "ck"), seed=5, report_interval=5, steps_per_cycle=5,
            checkpoint_interval=1, resume=resume, param_dir=str(folder),
            forcefield_options=options or None, verbose=False,
        )

    folding().run(n_cycles=2)
    assert load_checkpoint(tmp_path / "ck" / "last.chk")["hbond_beta_favor"] == 3.0
    folding(resume=True, hbond_beta_favor=3.0).run(n_cycles=3)  # the same factor, given explicitly
    refused = "written with hbond_beta_favor 3.0, but this run uses 2.0"
    with pytest.raises(ValueError, match=refused):
        folding(resume=True, hbond_beta_favor=2.0).run(n_cycles=4)
    # The checkpoint's signature covers the manifest: a changed one is another set of files.
    (folder / PARAMSET_MANIFEST).write_text(json.dumps(_manifest(hbond_beta_favor=2.5)))
    with pytest.raises(ValueError, match="a table or constants file differs"):
        folding(resume=True).run(n_cycles=4)


# --- the hard core it gives ---

def _start_overlap_pairs(ff: MCPUForceField, traj: md.Trajectory) -> int:
    """Pairs inside their hard core in the start structure, as the system's
    contact term counts them: they depend on clash_alpha (chignolin: 8 at
    0.75, 15 at 0.85) and on clash_bond_exclusion (0 at 3)."""
    system = ff.create_system(traj.topology)
    (mu,) = [p for p in system.get_potentials() if isinstance(p, mcpu_core.MuPotential)]
    return mu.start_overlap_pairs


def test_without_a_manifest_the_hard_core_is_mcpu08s(small) -> None:
    ff = MCPUForceField(small)
    assert (ff.clash_alpha, ff.clash_bond_exclusion) == (0.75, 0)
    assert _start_overlap_pairs(ff, small) == 8


@pytest.mark.parametrize("given, expected, pairs", [
    ({"clash_alpha": 0.85}, (0.85, 0), 15),
    ({"clash_bond_exclusion": 3}, (0.75, 3), 0),
    ({"clash_alpha": 0.85, "clash_bond_exclusion": 3}, (0.85, 3), 0),
], ids=["alpha", "bonds", "both"])
def test_the_manifest_gives_the_hard_core(small, tmp_path, given, expected, pairs) -> None:
    """A force field given no hard core takes the set's, each option on its
    own, and builds the system an explicit one would."""
    ff = MCPUForceField(small, param_dir=str(_folder(tmp_path / "hc", _manifest(**given))))
    assert ff.paramset_manifest == _manifest(**given)
    assert (ff.clash_alpha, ff.clash_bond_exclusion) == expected
    assert type(ff.clash_alpha) is float and type(ff.clash_bond_exclusion) is int
    explicit = MCPUForceField(small, clash_alpha=expected[0], clash_bond_exclusion=expected[1])
    assert _start_overlap_pairs(ff, small) == _start_overlap_pairs(explicit, small) == pairs


@pytest.mark.parametrize("given", [
    {"clash_alpha": 0.75}, {"clash_bond_exclusion": 0},
    {"clash_alpha": 0.8, "clash_bond_exclusion": 2},
])
def test_an_explicit_hard_core_wins_over_the_manifest(small, tmp_path, given) -> None:
    folder = _folder(tmp_path / "hc", _manifest(clash_alpha=0.85, clash_bond_exclusion=3))
    ff = MCPUForceField(small, param_dir=str(folder), **given)
    expected = {"clash_alpha": 0.85, "clash_bond_exclusion": 3, **given}
    assert (ff.clash_alpha, ff.clash_bond_exclusion) == (
        expected["clash_alpha"], expected["clash_bond_exclusion"])


def test_a_config_takes_the_sets_hard_core_unless_it_gives_one(chignolin_pdb_path, tmp_path) -> None:
    folder = _folder(tmp_path / "hc", _manifest(clash_alpha=0.85, clash_bond_exclusion=3))
    for options, expected in (
        (None, (0.85, 3)),
        ({"clash_alpha": None, "clash_bond_exclusion": None}, (0.85, 3)),  # null asks for the set's
        ({"clash_alpha": 0.75, "clash_bond_exclusion": 0}, (0.75, 0)),
    ):
        ff = load_forcefield(chignolin_pdb_path, options=options, param_dir=folder)
        assert (ff.clash_alpha, ff.clash_bond_exclusion) == expected
    config = tmp_path / "run.yaml"
    config.write_text(f"pdb: {chignolin_pdb_path}\ntemperatures: [0.6]\nparam_dir: {folder}\n")
    session = EngineSession(EngineSpec.from_simulation_config(load_config_auto(config)))
    built = session._ensure_forcefield()
    assert (built.clash_alpha, built.clash_bond_exclusion) == (0.85, 3)


def test_a_checkpoint_records_the_hard_core_the_run_used(chignolin_pdb_path, tmp_path) -> None:
    folder = _folder(tmp_path / "hc", _manifest(clash_alpha=0.85, clash_bond_exclusion=3))

    def folding(resume: bool = False, **options) -> FoldingRunner:
        return FoldingRunner(
            chignolin_pdb_path, output_dir=str(tmp_path / "out"),
            checkpoint_dir=str(tmp_path / "ck"), seed=5, report_interval=5, steps_per_cycle=5,
            checkpoint_interval=1, resume=resume, param_dir=str(folder),
            forcefield_options=options or None, verbose=False,
        )

    folding().run(n_cycles=2)
    saved = load_checkpoint(tmp_path / "ck" / "last.chk")
    assert (saved["clash_alpha"], saved["clash_bond_exclusion"], saved["clash_start_exemption"]) == (
        0.85, 3, True)
    assert saved["format_version"] == 4  # not mcpu08's hard core
    folding(resume=True, clash_alpha=0.85, clash_bond_exclusion=3).run(n_cycles=3)  # given explicitly
    with pytest.raises(ValueError, match=re.escape(
        "written with clash_alpha 0.85, clash_bond_exclusion 3, but this run uses "
        "clash_alpha 0.75, clash_bond_exclusion 0; resume it with forcefield_options: "
        "{clash_alpha: 0.85, clash_bond_exclusion: 3}"
    )):
        folding(resume=True, clash_alpha=0.75, clash_bond_exclusion=0).run(n_cycles=4)


def test_the_manifest_keeps_the_force_fields_bounds() -> None:
    """``pymcpu.params`` checks a manifest's hard core by the force field's
    own bounds, which it cannot import (the force field imports it)."""
    from pymcpu.forcefields import mcpu as mcpu_ff

    assert params._CLASH_ALPHA_LIMIT == mcpu_ff._CLASH_ALPHA_LIMIT == 1.35
    assert params._MAX_CLASH_BOND_EXCLUSION == mcpu_ff._MAX_CLASH_BOND_EXCLUSION == 3
    assert checkpointing.LEGACY_CLASH_OPTIONS == {
        "clash_alpha": mcpu_ff._MU_ALPHA_DEFAULT, "clash_bond_exclusion": 0,
        "clash_start_exemption": True}


# --- the set it names ---

def test_a_manifest_naming_another_set_is_refused(small, tmp_path, monkeypatch) -> None:
    """A folder's tables and defaults are those of the set its manifest
    names, so a force field of another set refuses it, before any file the
    manifest lists is read. Read without a set to compare, as a script may,
    the manifest loads; a force field of the set it names builds."""
    monkeypatch.setenv("MCPU_CACHE_DIR", str(tmp_path / "cache"))
    folder = _folder(tmp_path / "f")
    md5 = hashlib.md5(_own_copy(folder, TABLE).read_bytes()).hexdigest()
    (folder / PARAMSET_MANIFEST).write_text(json.dumps(_manifest(
        name="other_set", hbond_beta_favor=3.0, provenance={"md5": {TABLE: md5}})))
    reads: list[str] = []
    hash_file = params._hash_file
    monkeypatch.setattr(params, "_hash_file",
                        lambda path: reads.append(Path(path).name) or hash_file(path))
    refused = re.escape(
        f"{folder / PARAMSET_MANIFEST} describes parameter set 'other_set', but the folder "
        "was loaded as parameter set 'mcpu08'. A folder's tables and defaults belong to the "
        "set its manifest names: load it with param_set 'other_set' (param_set: other_set "
        "in a config), or give a folder of 'mcpu08'")
    with pytest.raises(ValueError, match=refused):
        MCPUForceField(small, param_dir=str(folder))
    with pytest.raises(ValueError, match=refused):
        load_paramset_manifest(folder, set_name="mcpu08")
    with pytest.raises(ValueError, match=refused):
        load_forcefield(small, param_dir=folder)
    assert reads == []  # refused before the listed table was read
    assert load_paramset_manifest(folder)["name"] == "other_set"
    assert load_paramset_manifest(folder, set_name="other_set")["hbond_beta_favor"] == 3.0
    name = register_set_like_mcpu08(monkeypatch, "other_set")
    ff = MCPUForceField(small, param_set=name, param_dir=str(folder))
    assert ff.paramset_manifest["name"] == "other_set"
    assert ff.hbond_beta_favor == 3.0


def test_the_name_must_match_exactly(small, tmp_path) -> None:
    for other in ("MCPU08", "mcpu08 ", "mcpu"):
        folder = _folder(tmp_path / repr(other), _manifest(name=other))
        with pytest.raises(ValueError, match=f"describes parameter set {re.escape(repr(other))}"):
            MCPUForceField(small, param_dir=str(folder))


# --- what a manifest may hold ---

# The values MCPUForceField refuses for its own hbond_beta_favor, and null.
# 1e-46 and 1e300 are 0 and inf in the engine's float32.
BAD_FACTORS = [0.0, -1.0, float("nan"), float("inf"), float("-inf"), 1e-46, 1e300,
               True, "3", None]


@pytest.mark.parametrize("bad", BAD_FACTORS, ids=repr)
def test_a_factor_the_force_field_would_refuse_is_refused(small, tmp_path, bad) -> None:
    folder = _folder(tmp_path / "f", _manifest(hbond_beta_favor=bad))
    message = r"paramset\.json: 'hbond_beta_favor' must be a finite number above 0, got "
    with pytest.raises(ValueError, match=message):
        load_paramset_manifest(folder)
    with pytest.raises(ValueError, match=message):
        MCPUForceField(small, param_dir=str(folder))
    if bad is not None:  # None, as an argument, asks for the set's factor
        with pytest.raises(ValueError, match="hbond_beta_favor must be a finite number above 0"):
            MCPUForceField(small, hbond_beta_favor=bad)


MALFORMED = {
    "unknown key": (_manifest(colour="blue"), r"unknown keys \['colour'\]"),
    "other format": (_manifest(format="pymcpu-paramset/2"),
                     "'format' must be 'pymcpu-paramset/1', got 'pymcpu-paramset/2'"),
    "mcpu26x format": (_manifest(format="pymcpu-forcefield/1"), "'format' must be"),
    "no format": ({"name": "a"}, r"missing \['format'\]"),
    "no name": ({"format": PARAMSET_FORMAT}, r"missing \['name'\]"),
    "empty name": (_manifest(name=" "), "'name' must be a non-empty string"),
    "name not text": (_manifest(name=5), "'name' must be a non-empty string"),
    "description not text": (_manifest(description=3), "'description' must be a string"),
    **{f"clash_alpha {bad!r}": (_manifest(clash_alpha=bad),
                                r"'clash_alpha' must be a number above 0 and below 1\.35")
       for bad in (0, -0.5, 1.35, 1.5, float("nan"), float("inf"), True, "0.85", None)},
    **{f"clash_bond_exclusion {bad!r}": (_manifest(clash_bond_exclusion=bad),
                                         "'clash_bond_exclusion' must be a whole number from 0 to 3")
       for bad in (-1, 4, 3.0, True, "3", None)},
    "factor beyond a float": (_manifest(hbond_beta_favor=10 ** 400),
                              "'hbond_beta_favor' must be a finite number above 0"),
    "weights not an object": (_manifest(folded_weights=[1.0]),
                              "'folded_weights' must be an object"),
    "unknown weight": (_manifest(folded_weights={"solvation": 1.0}),
                       r"unknown 'folded_weights' terms \['solvation'\]"),
    "weight negative": (_manifest(folded_weights={"mu": -0.5}),
                        "'folded_weights.mu' must be a finite number, 0 or above, got -0.5"),
    "weight nan": (_manifest(folded_weights={"hbond": float("nan")}),
                   "'folded_weights.hbond' must be a finite number, 0 or above"),
    "weight inf": (_manifest(folded_weights={"aromatic": float("inf")}),
                   "'folded_weights.aromatic' must be a finite number, 0 or above"),
    "weight beyond float32": (_manifest(folded_weights={"bb_torsion": 1e300}),
                              "'folded_weights.bb_torsion' must be a finite number, 0 or above"),
    "weight text": (_manifest(folded_weights={"sc_torsion": "1"}),
                    "'folded_weights.sc_torsion' must be a finite number, 0 or above"),
    "weight true": (_manifest(folded_weights={"sc_torsion": True}),
                    "'folded_weights.sc_torsion' must be a finite number, 0 or above"),
    "provenance not an object": (_manifest(provenance="by hand"), "'provenance' must be an object"),
    "md5 not an object": (_manifest(provenance={"md5": [TABLE]}), "'provenance.md5' must map"),
    "md5 not text": (_manifest(provenance={"md5": {TABLE: 5}}), "'provenance.md5' must map"),
    "md5 absolute path": (_manifest(provenance={"md5": {"/etc/hosts": MD5_ZERO}}),
                          "must be relative to the folder and stay inside it"),
    "md5 outside path": (_manifest(provenance={"md5": {"../x.bin": MD5_ZERO}}),
                         "must be relative to the folder and stay inside it"),
    "md5 of itself": (_manifest(provenance={"md5": {PARAMSET_MANIFEST: MD5_ZERO}}),
                      "cannot list the manifest itself"),
    "md5 not hex": (_manifest(provenance={"md5": {TABLE: "abc"}}), "which is not an md5"),
    "md5 of a missing file": (_manifest(provenance={"md5": {"mcpu_params/gone.bin": MD5_ZERO}}),
                              "lists mcpu_params/gone.bin, which is not a file in"),
    "md5 mismatch": (_manifest(provenance={"md5": {TABLE: MD5_ZERO}}),
                     f"{TABLE} has md5 [0-9a-f]{{32}}, but .* records {MD5_ZERO}: the file "
                     "changed"),
    "key twice": ('{"format": "pymcpu-paramset/1", "name": "a", "name": "b"}',
                  r"keys given more than once: \['name'\]"),
    "nested key twice": ('{"format": "pymcpu-paramset/1", "name": "a", '
                         '"folded_weights": {"mu": 1, "mu": 2}}',
                         r"keys given more than once: \['mu'\]"),
    "not JSON": ('{"format": "pymcpu-paramset/1", "name": ', "is not valid JSON"),
    "not an object": ("[]", "must hold a JSON object, not list"),
}


@pytest.mark.parametrize("case", list(MALFORMED))
def test_a_malformed_manifest_is_refused(small, tmp_path, case) -> None:
    manifest, message = MALFORMED[case]
    folder = _folder(tmp_path / "f", manifest)
    with pytest.raises(ValueError, match=message):
        load_paramset_manifest(folder)
    with pytest.raises(ValueError, match=message):
        MCPUForceField(small, param_dir=str(folder))


def test_a_manifest_that_is_not_a_readable_file_is_refused(tmp_path) -> None:
    folder = _folder(tmp_path / "f")
    (folder / PARAMSET_MANIFEST).mkdir()
    with pytest.raises(ValueError, match="exists but is not a file that can be read"):
        load_paramset_manifest(folder)
    (folder / PARAMSET_MANIFEST).rmdir()
    (folder / PARAMSET_MANIFEST).symlink_to(tmp_path / "missing.json")  # a broken link
    with pytest.raises(ValueError, match="exists but is not a file that can be read"):
        load_paramset_manifest(folder)


def test_provenance_md5s_are_checked_when_the_set_is_loaded(small, tmp_path) -> None:
    folder = _folder(tmp_path / "f")
    table = _own_copy(folder, TABLE)
    md5 = hashlib.md5(table.read_bytes()).hexdigest()
    atom_types = hashlib.md5((folder / "constants/atom_types.csv").read_bytes()).hexdigest()
    provenance = {"builder": "test",
                  "md5": {TABLE: md5.upper(), "constants/atom_types.csv": atom_types}}
    (folder / PARAMSET_MANIFEST).write_text(json.dumps(_manifest(provenance=provenance)))
    ff = MCPUForceField(small, param_dir=str(folder))  # an upper-case digest matches too
    assert ff.paramset_manifest["provenance"] == provenance

    values = np.fromfile(table, dtype=np.float32)
    values[0] += 1.0
    table.unlink()
    values.tofile(table)
    with pytest.raises(ValueError, match=f"has md5 [0-9a-f]{{32}}, but .* records {md5}: the file "
                                         "changed after the parameter set was written"):
        MCPUForceField(small, param_dir=str(folder))


def test_each_listed_file_is_hashed_once_until_it_changes(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MCPU_CACHE_DIR", str(tmp_path / "cache"))
    reads: list[str] = []
    hash_file = params._hash_file
    monkeypatch.setattr(params, "_hash_file",
                        lambda path: reads.append(Path(path).name) or hash_file(path))
    folder = tmp_path / "small"
    (folder / "constants").mkdir(parents=True)
    listed = folder / "constants" / "atom_types.csv"
    listed.write_text("a,b\n")
    md5 = {"constants/atom_types.csv": hashlib.md5(b"a,b\n").hexdigest()}
    (folder / PARAMSET_MANIFEST).write_text(json.dumps(_manifest(provenance={"md5": md5})))
    load_paramset_manifest(folder)
    load_paramset_manifest(folder)
    assert reads == ["atom_types.csv"]
    listed.write_text("a,b,c\n")
    with pytest.raises(ValueError, match="the file changed"):
        load_paramset_manifest(folder)
    assert reads[1:] == ["atom_types.csv"]


def test_provenance_can_be_compared_later(small, tmp_path, monkeypatch) -> None:
    """``verify_provenance=False`` builds a force field without reading the
    files the manifest lists md5s for (MPI replica exchange does so on every
    rank, and compares on rank 0 alone); ``check_provenance`` compares them.
    What the manifest lists is still checked when it is read."""
    monkeypatch.setenv("MCPU_CACHE_DIR", str(tmp_path / "cache"))
    folder = _folder(tmp_path / "f")
    _own_copy(folder, TABLE)
    (folder / PARAMSET_MANIFEST).write_text(json.dumps(
        _manifest(provenance={"md5": {TABLE: MD5_ZERO}})))
    reads: list[str] = []
    hash_file = params._hash_file
    monkeypatch.setattr(params, "_hash_file",
                        lambda path: reads.append(Path(path).name) or hash_file(path))

    ff = MCPUForceField(small, param_dir=str(folder), verify_provenance=False)
    assert reads == []
    with pytest.raises(ValueError, match=f"records {MD5_ZERO}: the file changed"):
        ff.check_provenance()
    assert reads == ["aromatic_potentials.bin"]
    with pytest.raises(ValueError, match=f"records {MD5_ZERO}: the file changed"):
        MCPUForceField(small, param_dir=str(folder))
    (folder / PARAMSET_MANIFEST).write_text(json.dumps(
        _manifest(provenance={"md5": {"mcpu_params/gone.bin": MD5_ZERO}})))
    with pytest.raises(ValueError, match="lists mcpu_params/gone.bin, which is not a file"):
        MCPUForceField(small, param_dir=str(folder), verify_provenance=False)
    # A set without listed md5s has nothing to compare.
    MCPUForceField(small, verify_provenance=False).check_provenance()
