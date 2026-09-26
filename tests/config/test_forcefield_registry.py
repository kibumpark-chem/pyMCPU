"""Selecting a force field by name, from a config file.

Before this existed, every call site constructed `MCPUForceField` directly, so
a second force field was reachable only by hand-assembling objects. The tests
that matter most here are the *back-compat* ones: a config that does not
mention a force field must still mean exactly what it meant before.
"""

from __future__ import annotations

import warnings

import mdtraj as md
import pytest

from pymcpu.config import EngineSpec, SimulationConfig, config_from_dict
from pymcpu.forcefields import (
    available_forcefields,
    build_forcefield,
    get_forcefield,
    register_forcefield,
)
from pymcpu.forcefields.base import BaseForceField
from pymcpu.forcefields.korp import KORPForceField
from pymcpu.forcefields.mcpu import MCPUForceField
from pymcpu.runners import default_example_pdb

EXAMPLE_PDB = str(default_example_pdb())


@pytest.fixture(scope="module")
def example_traj():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return md.load(EXAMPLE_PDB)


def test_both_force_fields_are_registered():
    assert set(available_forcefields()) >= {"korp", "mcpu", "mcpu08"}
    assert get_forcefield("mcpu08") is MCPUForceField
    assert get_forcefield("mcpu") is MCPUForceField     # convenience alias
    assert get_forcefield("korp") is KORPForceField


def test_an_unknown_name_lists_the_known_ones():
    with pytest.raises(ValueError, match=r"unknown force field 'nope'"):
        get_forcefield("nope")


def test_registering_a_non_forcefield_is_refused():
    class NotAForceField:
        pass

    with pytest.raises(TypeError, match="BaseForceField"):
        register_forcefield("bogus", NotAForceField)


def test_re_registering_the_same_class_is_fine_but_a_conflict_is_not():
    register_forcefield("mcpu08", MCPUForceField)        # idempotent
    with pytest.raises(ValueError, match="already registered"):
        register_forcefield("mcpu08", KORPForceField)


def test_unknown_options_name_the_force_field_that_refused_them():
    """A typo in a YAML file must not be silently dropped.

    Passing ``None`` for the trajectory is the point: the option check has to
    happen before anything touches the structure, so a bad key is reported
    without paying to load and slice a PDB first.
    """
    with pytest.raises(ValueError, match=r"'mcpu08' does not accept \['map_path'\]"):
        build_forcefield("mcpu08", None, {"map_path": "/nowhere"})
    with pytest.raises(ValueError, match=r"'korp' does not accept \['param_set'\]"):
        build_forcefield("korp", None, {"param_set": "mcpu_v1"})


def test_mcpu_prepares_by_dropping_hydrogens(example_traj):
    """Each force field declares its own preprocessing.

    The factory must not need to know that MCPU wants heavy atoms while KORP
    slices itself -- otherwise adding a third would mean editing the factory.
    """
    prepared = MCPUForceField.prepare_trajectory(example_traj)
    assert prepared.n_atoms <= example_traj.n_atoms
    assert not any(a.element.symbol == "H" for a in prepared.topology.atoms)
    # Idempotent, so call sites that already sliced are unaffected.
    assert MCPUForceField.prepare_trajectory(prepared).n_atoms == prepared.n_atoms


def test_the_base_class_prepares_nothing_by_default(example_traj):
    assert BaseForceField.prepare_trajectory(example_traj) is example_traj


def test_building_mcpu_through_the_factory_matches_direct_construction(example_traj):
    direct = MCPUForceField(MCPUForceField.prepare_trajectory(example_traj))
    vianame = build_forcefield("mcpu08", example_traj)
    assert type(vianame) is MCPUForceField
    assert vianame.n_atoms == direct.n_atoms
    assert vianame.inverse_mapping == direct.inverse_mapping


# --- back-compat: a config that says nothing must mean what it always meant ---

def test_the_default_force_field_is_mcpu08():
    assert SimulationConfig(mode="folding", pdb="x").forcefield == "mcpu08"
    assert SimulationConfig(mode="folding", pdb="x").forcefield_options == {}
    assert EngineSpec(pdb=EXAMPLE_PDB).forcefield == "mcpu08"


def test_a_config_without_the_key_is_unchanged():
    cfg = config_from_dict({"mode": "folding", "pdb": EXAMPLE_PDB})
    assert cfg.forcefield == "mcpu08"
    assert EngineSpec.from_simulation_config(cfg).forcefield == "mcpu08"


def test_the_key_round_trips_through_the_json_loader():
    cfg = config_from_dict({
        "mode": "folding",
        "pdb": EXAMPLE_PDB,
        "forcefield": "korp",
        "forcefield_options": {"min_distance": 3.0},
    })
    assert cfg.forcefield == "korp"
    assert cfg.forcefield_options == {"min_distance": 3.0}
    spec = EngineSpec.from_simulation_config(cfg)
    assert spec.forcefield == "korp"
    assert spec.forcefield_options == {"min_distance": 3.0}


def test_an_unknown_name_is_rejected_by_the_config_not_at_build_time():
    with pytest.raises(ValueError, match="unknown force field"):
        EngineSpec(pdb=EXAMPLE_PDB, forcefield="nope")
