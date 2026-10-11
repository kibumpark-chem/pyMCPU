"""Selecting a force field by name, from a config file.

Before this existed, every call site constructed `MCPUForceField` directly, so
a second force field was reachable only by hand-assembling objects. The tests
that matter most here are the *back-compat* ones: a config that does not
mention a force field must still mean exactly what it meant before.

pyMCPU ships one force field, so the paths only another force field reaches
are tested with a stub registered for the duration of one test.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import mdtraj as md
import numpy as np
import pytest

import pymcpu.forcefields as forcefields
from pymcpu.checkpointing import checkpoint_forcefield_error
from pymcpu.config import (
    EngineSpec,
    SimulationConfig,
    check_move_weights,
    config_from_dict,
    load_yaml_config,
)
from pymcpu.forcefields import (
    available_forcefields,
    build_forcefield,
    get_forcefield,
    load_forcefield,
    register_forcefield,
)
from pymcpu.forcefields.base import BaseForceField
from pymcpu.forcefields.mcpu import MCPUForceField
from pymcpu.runners import default_example_pdb
from pymcpu.trajectory_utils import trajectory_topology_path

EXAMPLE_PDB = str(default_example_pdb())


@pytest.fixture(scope="module")
def example_traj():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return md.load(EXAMPLE_PDB)


def test_mcpu_is_registered_under_both_names():
    assert set(available_forcefields()) >= {"mcpu", "mcpu08"}
    assert get_forcefield("mcpu08") is MCPUForceField
    assert get_forcefield("mcpu") is MCPUForceField     # convenience alias


def test_an_unknown_name_lists_the_known_ones():
    with pytest.raises(ValueError,
                       match=r"unknown force field 'nope'; known: \[.*'mcpu08'.*\]"):
        get_forcefield("nope")


def test_registering_a_non_forcefield_is_refused():
    class NotAForceField:
        pass

    with pytest.raises(TypeError, match="BaseForceField"):
        register_forcefield("bogus", NotAForceField)


def test_re_registering_the_same_class_is_fine_but_a_conflict_is_not():
    class OtherForceField(BaseForceField):
        pass

    register_forcefield("mcpu08", MCPUForceField)        # idempotent
    with pytest.raises(ValueError, match="already registered"):
        register_forcefield("mcpu08", OtherForceField)


def test_unknown_options_name_the_force_field_that_refused_them():
    """A typo in a YAML file must not be silently dropped.

    Passing ``None`` for the trajectory is the point: the option check has to
    happen before anything touches the structure, so a bad key is reported
    without paying to load and slice a PDB first.
    """
    with pytest.raises(ValueError, match=r"'mcpu08' does not accept \['param_sett'\]"):
        build_forcefield("mcpu08", None, {"param_sett": "mcpu08"})


def test_mcpu_prepares_by_dropping_hydrogens(example_traj):
    """Each force field declares its own preprocessing.

    The factory must not need to know that MCPU wants heavy atoms --
    otherwise adding another force field would mean editing the factory.
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
        "forcefield": "mcpu",
        "forcefield_options": {"virtual_amide_h": False},
    })
    assert cfg.forcefield == "mcpu"
    assert cfg.forcefield_options == {"virtual_amide_h": False}
    spec = EngineSpec.from_simulation_config(cfg)
    assert spec.forcefield == "mcpu"
    assert spec.forcefield_options == {"virtual_amide_h": False}


def test_an_unknown_name_is_rejected_by_the_config_not_at_build_time():
    with pytest.raises(ValueError, match="unknown force field"):
        EngineSpec(pdb=EXAMPLE_PDB, forcefield="nope")


def test_an_unknown_name_is_rejected_when_a_simulation_config_is_built():
    with pytest.raises(ValueError, match="unknown force field 'nope'"):
        SimulationConfig(mode="folding", pdb="x", forcefield="nope")


def test_an_unknown_name_in_a_yaml_file_is_rejected_when_it_loads(tmp_path):
    path = tmp_path / "run.yaml"
    path.write_text("pdb: x.pdb\ntemperatures: [0.5]\nforcefield: amber\n")
    with pytest.raises(ValueError, match="unknown force field 'amber'"):
        load_yaml_config(path)


# --- a force field other than MCPU ---

class _BackboneStub(BaseForceField):
    """Simulates the backbone heavy atoms only, so it has fewer atoms than
    MCPU. Engine atom k is topology atom k - 1, and engine atom 0 the last
    one. Unlike a reversal, that order is not its own inverse, so the
    trajectory topology test below fails if the mapping is applied the wrong
    way round. No test here builds a System from it."""

    @classmethod
    def prepare_trajectory(cls, trajectory):
        return trajectory.atom_slice(trajectory.topology.select("name N CA C O"))

    def __init__(self, trajectory, scale: float = 1.0):
        self.scale = scale
        self.output_topology = trajectory.topology
        n = trajectory.n_atoms
        self.inverse_mapping = [n - 1, *range(n - 1)]
        self.coords = trajectory.xyz[:, self.inverse_mapping, :]

    def create_system(self, topology):
        raise NotImplementedError


@pytest.fixture
def stub_forcefield(monkeypatch):
    """Register the stub as ``"stub"`` for one test.

    The registry has no unregister, so the stub goes into a copy of it that
    is swapped back afterwards; later tests see only the shipped names.
    """
    monkeypatch.setattr(forcefields, "_REGISTRY", dict(forcefields._REGISTRY))
    register_forcefield("stub", _BackboneStub)
    return "stub"


def test_mcpu_settings_are_refused_for_another_force_field(stub_forcefield):
    """param_set, param_dir and compute_dssp are top-level config fields that
    only MCPU reads; set for another force field they are a mistake to
    report, not to drop. The structure is never read: the refusal comes
    first."""
    unread = "never-read.pdb"
    refused = "param_set/param_dir configure the MCPU force field"
    with pytest.raises(ValueError, match=refused):
        load_forcefield(unread, stub_forcefield, param_dir="/params")
    with pytest.raises(ValueError, match=refused):
        load_forcefield(unread, stub_forcefield, param_set="mcpu_other")
    with pytest.raises(ValueError, match="compute_dssp applies to the MCPU force field, not 'stub'"):
        load_forcefield(unread, stub_forcefield, compute_dssp=True)


def test_another_force_field_gets_its_own_options_and_none_of_mcpus(
    stub_forcefield, example_traj
):
    ff = load_forcefield(example_traj, stub_forcefield, {"scale": 2.0})
    assert type(ff) is _BackboneStub
    assert ff.scale == 2.0


def test_a_force_field_with_fewer_atoms_writes_its_own_trajectory_topology(
    stub_forcefield, example_traj, tmp_path
):
    """XTC frames are read against the force field's output topology. MCPU's
    is the input's heavy atoms, so the input PDB serves; a force field with
    fewer atoms gets a topology file holding its starting coordinates, each
    engine atom at its topology index."""
    ff = load_forcefield(example_traj, stub_forcefield)
    out = tmp_path / "run" / "rex_topology.pdb"

    # MPI ranks other than the one that writes it only take the path.
    assert trajectory_topology_path(ff, EXAMPLE_PDB, EXAMPLE_PDB, out, write=False) == str(out)
    assert not out.exists()

    assert trajectory_topology_path(ff, EXAMPLE_PDB, EXAMPLE_PDB, out) == str(out)
    written = md.load(str(out))
    backbone = _BackboneStub.prepare_trajectory(example_traj)
    assert [str(a) for a in written.topology.atoms] == [str(a) for a in backbone.topology.atoms]
    np.testing.assert_allclose(written.xyz[0], backbone.xyz[0], atol=1e-4)


def test_a_checkpoint_resumes_only_under_the_force_field_that_wrote_it(stub_forcefield):
    # A checkpoint without the field predates it and was written by mcpu08;
    # the two names of one force field match.
    assert checkpoint_forcefield_error("", "mcpu08") is None
    assert checkpoint_forcefield_error(None, "mcpu") is None
    assert checkpoint_forcefield_error("mcpu", "mcpu08") is None

    message = checkpoint_forcefield_error("", stub_forcefield, "run/last.chk")
    assert message.startswith("checkpoint run/last.chk was written by a 'mcpu08' run")
    assert "builds forcefield 'stub'" in message
    # Nor does a force field this pyMCPU does not register.
    assert "written by a 'retired' run" in checkpoint_forcefield_error("retired", "mcpu08")


# --- a force field without sidechain atoms ---

def test_sidechain_moves_are_refused_when_there_are_no_sidechain_atoms():
    """An all-glycine chain has none, under MCPU too. A nonzero sidechain
    weight is refused before a run opens its output files, with the fix in
    the message (the engine would refuse it only at its first step); a zero
    weight passes."""
    no_sidechains = SimpleNamespace(total_sc_atoms=0)
    with pytest.raises(ValueError, match=r"'mcpu08' has no sidechains.*\[pivot, kic, 0\.0\]"):
        check_move_weights(no_sidechains, [0.5, 0.25, 0.25], "mcpu08")
    check_move_weights(no_sidechains, [0.5, 0.5, 0.0], "mcpu08")
    check_move_weights(SimpleNamespace(total_sc_atoms=5), [0.5, 0.25, 0.25], "mcpu08")
