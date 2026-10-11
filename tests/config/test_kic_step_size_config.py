"""``kic_step_size_rad``: the KIC driver's own width.

The KIC driver used to draw its angle from the pivot's distribution, so
``step_size_rad`` set both. ``kic_step_size_rad`` now sets the driver and
``step_size_rad`` the pivot only. The default is pi/6 (30 deg): at 0.1 rad,
the width the two shared before, KIC never moved the psi of CLN025's proline
out of its basin. Every entry point takes it: the Python classes, YAML and
JSON configs, ``EngineSpec`` and the engine session built from it.
"""
from __future__ import annotations

import math

import pytest

from pymcpu import mcpu_core
from pymcpu.config import (
    DEFAULT_KIC_STEP_SIZE_RAD,
    EngineSpec,
    IntegratorConfig,
    config_from_dict,
    normalize_kic_step_size_rad,
    normalize_move_settings,
    yaml_dict_to_config,
)
from pymcpu.runners import default_example_pdb
from pymcpu.sampling.engine_session import EngineSession

_PDB = str(default_example_pdb())


def test_the_default_is_the_engines() -> None:
    assert DEFAULT_KIC_STEP_SIZE_RAD == math.pi / 6
    assert mcpu_core.Integrator(0.6).kic_step_size_rad() == pytest.approx(DEFAULT_KIC_STEP_SIZE_RAD)
    assert IntegratorConfig().kic_step_size_rad == DEFAULT_KIC_STEP_SIZE_RAD
    assert normalize_move_settings()["kic_step_size_rad"] == DEFAULT_KIC_STEP_SIZE_RAD
    assert normalize_kic_step_size_rad(None) == DEFAULT_KIC_STEP_SIZE_RAD


def test_the_integrator_keeps_the_two_widths_apart() -> None:
    integ = mcpu_core.Integrator(0.6, step_size_rad=0.05, kic_step_size_rad=0.4)
    assert integ.backbone_step_size_rad() == pytest.approx(0.05)
    assert integ.kic_step_size_rad() == pytest.approx(0.4)
    integ.set_kic_step_size_rad(0.2)
    assert integ.kic_step_size_rad() == pytest.approx(0.2)
    assert integ.backbone_step_size_rad() == pytest.approx(0.05)


@pytest.mark.parametrize("temperatures", [[0.6], [0.5, 0.6]], ids=["folding", "remd"])
def test_a_yaml_key_sets_it(temperatures: list[float]) -> None:
    cfg = yaml_dict_to_config(
        {"pdb": _PDB, "temperatures": temperatures, "kic_step_size_rad": 0.3}
    )
    assert cfg.integrator.kic_step_size_rad == pytest.approx(0.3)
    assert cfg.integrator.step_size_rad == pytest.approx(0.1)  # the pivot keeps its own
    assert EngineSpec.from_simulation_config(cfg).kic_step_size_rad == pytest.approx(0.3)


def test_a_json_integrator_block_sets_it() -> None:
    cfg = config_from_dict(
        {"mode": "folding", "pdb": _PDB, "integrator": {"kic_step_size_rad": 0.3}}
    )
    assert cfg.integrator.kic_step_size_rad == pytest.approx(0.3)


@pytest.mark.parametrize("bad", [0.0, -0.1, math.inf, math.nan, True])
def test_a_width_that_is_not_positive_is_refused(bad) -> None:
    with pytest.raises(ValueError, match="kic_step_size_rad"):
        normalize_kic_step_size_rad(bad)
    with pytest.raises(ValueError, match="kic_step_size_rad"):
        yaml_dict_to_config({"pdb": _PDB, "temperatures": [0.6], "kic_step_size_rad": bad})
    with pytest.raises(ValueError, match="kic_step_size_rad"):
        IntegratorConfig(kic_step_size_rad=bad)
    with pytest.raises(ValueError, match="kic_step_size_rad"):
        EngineSpec(pdb=_PDB, kic_step_size_rad=bad)
    if not isinstance(bad, bool):
        with pytest.raises(ValueError, match="kic_step_size_rad"):
            mcpu_core.Integrator(0.6, kic_step_size_rad=bad)
        with pytest.raises(ValueError, match="kic_step_size_rad"):
            mcpu_core.Integrator(0.6).set_kic_step_size_rad(bad)


def test_the_engine_session_builds_its_integrator_with_it(engine_spec_factory) -> None:
    spec = engine_spec_factory(step_size_rad=0.05, kic_step_size_rad=0.3)
    integ = EngineSession(spec)._ensure_sim().integrator
    assert integ.kic_step_size_rad() == pytest.approx(0.3)
    assert integ.backbone_step_size_rad() == pytest.approx(0.05)


def test_the_folding_runner_builds_its_integrator_with_it(tmp_path, monkeypatch) -> None:
    from pymcpu.sampling.folding import FoldingRunner

    monkeypatch.chdir(tmp_path)
    runner = FoldingRunner(
        _PDB, step_size_rad=0.05, kic_step_size_rad=0.3, output_dir=tmp_path / "fold"
    )
    assert runner.simulation.integrator.kic_step_size_rad() == pytest.approx(0.3)
    assert runner.simulation.integrator.backbone_step_size_rad() == pytest.approx(0.05)
    with pytest.raises(ValueError, match="kic_step_size_rad"):
        FoldingRunner(_PDB, kic_step_size_rad=0.0, output_dir=tmp_path / "bad")
