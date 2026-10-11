"""Round-trip for ``sidechain_move_mode`` on IntegratorConfig.

Mirrors ``test_contact_atom_mode_config.py``'s structure: a valid-value
round-trip, an invalid-value rejection, and a default-preserved check --
across BOTH config loaders, since (unlike ``contact_atom_mode``, only ever
exercised through the YAML loader in that file) ``IntegratorConfig`` is
reachable from both ``config_from_dict`` (JSON, via the generic
``_merge_dataclass`` pass-through) and ``yaml_dict_to_config`` (which
hand-extracts each ``IntegratorConfig`` field and therefore needed an
explicit new line for this field -- this test is what would catch that line
being forgotten).

``rotamer_library`` is the default (matches ``MCIntegrator``'s C++ default
and ``FoldingRunner``/``EngineSpec``'s Python defaults) -- ``continuous`` is
the explicitly-opt-in alternative, and the round-trip uses it: a loader
that ignored the key would still hand back the default.
"""

from __future__ import annotations


import pytest

from pymcpu.config import config_from_dict, yaml_dict_to_config
from pymcpu.config import EngineSpec
from pymcpu.runners import default_example_pdb

_BASE_YAML = {
    "pdb": "examples/data/1uao.pdb",
    "temperatures": [0.6],
    "num_cycles": 1,
    "mc_replica_steps": 10,
}

_BASE_JSON = {
    "mode": "folding",
    "pdb": "examples/data/1uao.pdb",
}


def test_sidechain_move_mode_defaults_to_rotamer_library_yaml() -> None:
    cfg = yaml_dict_to_config(dict(_BASE_YAML))
    assert cfg.integrator.sidechain_move_mode == "rotamer_library"


def test_sidechain_move_mode_defaults_to_rotamer_library_json() -> None:
    cfg = config_from_dict(dict(_BASE_JSON))
    assert cfg.integrator.sidechain_move_mode == "rotamer_library"


def test_sidechain_move_mode_continuous_roundtrips_yaml() -> None:
    cfg = yaml_dict_to_config({**_BASE_YAML, "sidechain_move_mode": "continuous"})
    assert cfg.integrator.sidechain_move_mode == "continuous"


def test_sidechain_move_mode_continuous_roundtrips_json() -> None:
    cfg = config_from_dict(
        {**_BASE_JSON, "integrator": {"sidechain_move_mode": "continuous"}}
    )
    assert cfg.integrator.sidechain_move_mode == "continuous"


def test_invalid_sidechain_move_mode_raises_yaml() -> None:
    with pytest.raises(ValueError, match="sidechain_move_mode"):
        yaml_dict_to_config({**_BASE_YAML, "sidechain_move_mode": "discrete"})


# Resolved from the installed package, not the repo layout: this file is
# about every config entry point agreeing, and it should keep working when
# the suite runs against a wheel.
_TEST_PDB = str(default_example_pdb())
_BASE_SPEC_KWARGS = dict(pdb=_TEST_PDB)


def test_enginespec_sidechain_move_mode_defaults_to_rotamer_library() -> None:
    assert EngineSpec(**_BASE_SPEC_KWARGS).sidechain_move_mode == "rotamer_library"


def test_enginespec_sidechain_move_mode_continuous_roundtrips() -> None:
    spec = EngineSpec(**_BASE_SPEC_KWARGS, sidechain_move_mode="continuous")
    assert spec.sidechain_move_mode == "continuous"


def test_enginespec_invalid_sidechain_move_mode_raises() -> None:
    with pytest.raises(ValueError, match="sidechain_move_mode"):
        EngineSpec(**_BASE_SPEC_KWARGS, sidechain_move_mode="discrete")
