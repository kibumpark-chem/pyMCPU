"""Tests for the GROMACS-style YAML config bridge: ``pymcpu.utils.yaml_parser``
(``load_yaml``, ``config_from_yaml``, ``simulation_from_yaml``) and
``pymcpu.config.yaml_dict_to_config``.

Two shipped YAML files anchor the "real file" tests (``inputs/template.yaml``
and ``examples/gromacs_style/example_input.yaml``); the schema-logic tests
(temperature-ladder expansion, q/n-target mutual exclusion, required/unknown
field handling) use inline dicts/tmp_path files instead, since that logic
doesn't depend on either shipped file's contents.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pymcpu import PACKAGE_ROOT
from pymcpu.config import yaml_dict_to_config
from pymcpu.utils.yaml_parser import config_from_yaml, load_yaml, simulation_from_yaml

REPO_ROOT = Path(PACKAGE_ROOT).parent
TEMPLATE_YAML = REPO_ROOT / "inputs" / "template.yaml"
EXAMPLE_YAML = REPO_ROOT / "examples" / "gromacs_style" / "example_input.yaml"


class TestShippedYamlFiles:
    """Parsing of the two YAML files shipped in the repo."""

    def test_load_yaml_template(self) -> None:
        cfg = load_yaml(TEMPLATE_YAML)
        assert isinstance(cfg, dict)
        assert "pdb" in cfg
        assert "temp_min" in cfg

    def test_load_yaml_example(self) -> None:
        cfg = load_yaml(EXAMPLE_YAML)
        assert cfg["pdb"] == "examples/data/1uao.pdb"  # mirrors example_input.yaml's pdb field
        assert cfg["num_cycles"] == 5  # mirrors example_input.yaml's num_cycles field

    def test_config_from_yaml_template(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(REPO_ROOT)
        cfg = config_from_yaml(TEMPLATE_YAML)
        # template.yaml has n_temps:11 > 1 -> mode inference rule picks replica_exchange_2d
        assert cfg.mode == "replica_exchange_2d"
        assert cfg.integrator.seed == 0  # mirrors template.yaml's seed field

    def test_config_from_yaml_example(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(REPO_ROOT)
        cfg = config_from_yaml(EXAMPLE_YAML)
        # example_input.yaml has n_temps:2 > 1 -> replica_exchange_2d
        assert cfg.mode == "replica_exchange_2d"
        assert cfg.integrator.seed == 42  # mirrors example_input.yaml's seed field
        assert cfg.replica_exchange is not None
        assert cfg.replica_exchange.q_targets == [0.0, 0.5]  # mirrors example_input.yaml's q_targets
        # example_input.yaml sets q_targets, not n_targets -> native_contact_targets stays None
        assert cfg.replica_exchange.native_contact_targets is None

    def test_simulation_from_yaml_dryrun(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``simulation_from_yaml`` returns a runnable handle without running anything."""
        monkeypatch.chdir(REPO_ROOT)
        handle = simulation_from_yaml(EXAMPLE_YAML, output_dir="./test_out")
        assert hasattr(handle, "describe")
        assert hasattr(handle, "run")
        assert handle.config.outputs.output_dir == "./test_out"  # output_dir override applied


class TestYamlSchemaLogic:
    """Field-mapping/expansion/validation logic, independent of any shipped file."""

    def test_n_targets_expand_temperature_ladder(self) -> None:
        cfg = yaml_dict_to_config(
            {
                "pdb": "dummy.pdb",
                "temp_min": 0.4,
                "temp_step": 0.025,
                "n_temps": 25,
                "n_targets": [0.0, 5.0],
                "k_bias": 0.02,
                "num_cycles": 10,
                "mc_replica_steps": 100,
            }
        )
        assert cfg.mode == "replica_exchange_2d"
        assert cfg.replica_exchange is not None
        # ladder length is a direct algebraic consequence of n_temps, not a derived physics value
        assert len(cfg.replica_exchange.temperatures) == 25
        assert cfg.replica_exchange.native_contact_targets == [0.0, 5.0]
        assert cfg.replica_exchange.q_targets is None

    def test_rejects_both_n_and_q_targets(self) -> None:
        with pytest.raises(ValueError, match="not both"):
            yaml_dict_to_config(
                {
                    "pdb": "dummy.pdb",
                    "temp_min": 0.4,
                    "n_temps": 2,
                    "n_targets": [0.0, 5.0],
                    "q_targets": [0.0, 0.5],
                    "num_cycles": 1,
                    "mc_replica_steps": 10,
                }
            )

    def test_missing_required_field_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text("num_cycles: 10\n")  # no 'pdb' field
        with pytest.raises(ValueError, match="required fields"):
            load_yaml(bad)

    def test_unknown_field_warns(self, tmp_path: Path) -> None:
        f = tmp_path / "warn.yaml"
        f.write_text("pdb: test.pdb\ncompletely_unknown_field: 42\n")
        with pytest.warns(UserWarning, match="Unknown fields"):
            load_yaml(f)
