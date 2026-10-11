"""A YAML config that leaves out the move settings gets the engine's defaults.

The YAML loader used to fill a missing ``pivot_rama_probability`` with 0.05,
although the engine, ``IntegratorConfig``, JSON configs and every Python
signature default to 0.0 (the rama pivot is opt-in). Anything built from such
a config -- an engine via ``EngineSpec.from_simulation_config``, and folding
or replica exchange once their move settings were wired through -- would have
run rama pivots at 5% without the config saying so.
"""

from __future__ import annotations

import pytest

from pymcpu import mcpu_core
from pymcpu.config import EngineSpec, normalize_move_settings, yaml_dict_to_config
from pymcpu.runners import default_example_pdb

_PDB = str(default_example_pdb())  # EngineSpec checks that the file exists


@pytest.mark.parametrize("temperatures", [[0.6], [0.5, 0.6]], ids=["folding", "remd"])
def test_yaml_move_defaults_match_a_bare_integrator(temperatures: list[float]) -> None:
    cfg = yaml_dict_to_config({"pdb": _PDB, "temperatures": temperatures})
    bare = mcpu_core.Integrator(temperatures[0])

    assert cfg.integrator.pivot_rama_probability == bare.pivot_rama_probability() == 0.0
    assert cfg.integrator.pivot_rama_schedule is None
    assert cfg.integrator.sidechain_move_mode == bare.sidechain_move_mode()
    assert cfg.integrator.move_weights == pytest.approx(bare.move_weights())
    assert cfg.integrator.step_size_rad == pytest.approx(bare.backbone_step_size_rad())
    assert cfg.integrator.kic_step_size_rad == pytest.approx(bare.kic_step_size_rad())

    assert EngineSpec.from_simulation_config(cfg).pivot_rama_probability == 0.0


def test_an_explicit_yaml_rama_probability_is_kept() -> None:
    cfg = yaml_dict_to_config(
        {"pdb": _PDB, "temperatures": [0.6], "pivot_rama_probability": 0.3}
    )
    assert cfg.integrator.pivot_rama_probability == pytest.approx(0.3)


def test_normalize_move_settings_defaults_to_no_rama_pivots() -> None:
    assert normalize_move_settings(pivot_rama_probability=None)["pivot_rama_probability"] == 0.0
