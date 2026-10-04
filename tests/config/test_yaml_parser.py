"""Tests for the GROMACS-style YAML config bridge: ``pymcpu.utils.yaml_parser``
(``load_yaml``, ``config_from_yaml``, ``simulation_from_yaml``) and
``pymcpu.config.yaml_dict_to_config``.

Two shipped YAML files anchor the "real file" tests (``examples/configs/template.yaml``
and ``examples/gromacs_style/example_input.yaml``); the schema-logic tests
(temperature-ladder expansion, q/n-target mutual exclusion, required/unknown
field handling) use inline dicts/tmp_path files instead, since that logic
doesn't depend on either shipped file's contents.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from pymcpu import PACKAGE_ROOT
from pymcpu.config import replica_grid_dims, yaml_dict_to_config
from pymcpu.utils.yaml_parser import config_from_yaml, load_yaml, simulation_from_yaml

REPO_ROOT = Path(PACKAGE_ROOT).parent
TEMPLATE_YAML = REPO_ROOT / "examples" / "configs" / "template.yaml"
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

    def test_unknown_field_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "unknown.yaml"
        f.write_text("pdb: test.pdb\ncompletely_unknown_field: 42\n")
        with pytest.raises(ValueError, match="Unknown key in .*unknown.yaml: 'completely_unknown_field'"):
            load_yaml(f)

    def test_misspelled_key_names_the_closest_known_key(self) -> None:
        with pytest.raises(ValueError, match=r"'num_cylces' \(did you mean 'num_cycles'\?\)"):
            yaml_dict_to_config({"pdb": "dummy.pdb", "temperatures": [0.5], "num_cylces": 10})

    def test_unknown_keys_at_both_levels_are_reported_together(self) -> None:
        with pytest.raises(ValueError) as error:
            yaml_dict_to_config(
                {
                    "pdb": "dummy.pdb",
                    "temperatures": [0.5],
                    "num_cylces": 10,
                    "checkpointing": {"checkpoint_intrval": 5},
                }
            )
        assert "'num_cylces' (did you mean 'num_cycles'?)" in str(error.value)
        assert (
            "'checkpointing.checkpoint_intrval' (did you mean 'checkpoint_interval'?)"
            in str(error.value)
        )

    @pytest.mark.parametrize(
        "key, value, hint",
        [
            ("temperature", 0.5, "write 'temperatures: [T]'"),
            ("integrator", {"steps": 5}, "a block of the JSON schema"),
            ("checkpoint", {"checkpoint_interval": 5}, "the YAML block is called 'checkpointing'"),
            ("report_interval", 5, "the YAML name is 'log_interval'"),
        ],
    )
    def test_keys_from_the_json_schema_say_what_to_write(
        self, key: str, value: object, hint: str
    ) -> None:
        with pytest.raises(ValueError, match="Unknown key") as error:
            yaml_dict_to_config({"pdb": "dummy.pdb", "temperatures": [0.5], key: value})
        assert hint in str(error.value)

    def test_retired_keys_load_with_a_warning(self) -> None:
        # output_layout and mode were accepted and never read; existing
        # configs carry them, so they warn rather than fail.
        with pytest.warns(UserWarning) as warned:
            cfg = yaml_dict_to_config(
                {
                    "pdb": "dummy.pdb",
                    "temperatures": [0.4, 0.5],
                    "output_layout": "standard",
                    "mode": "replica_exchange",
                }
            )
        assert cfg.mode == "replica_exchange_2d"
        messages = [str(w.message) for w in warned]
        assert any("'output_layout'" in m and "has no effect" in m for m in messages)
        assert any("'mode'" in m and "has no effect" in m for m in messages)

    def test_checkpointing_must_be_a_mapping(self) -> None:
        # `checkpointing: false` used to be ignored, leaving checkpointing on.
        with pytest.raises(ValueError, match="must be a mapping"):
            yaml_dict_to_config({"pdb": "dummy.pdb", "temperatures": [0.5], "checkpointing": False})

    @pytest.mark.parametrize(
        "windows", [{}, {"q_targets": [0.0, 0.5, 1.0]}, {"n_targets": [0.0, 5.0]}]
    )
    def test_replica_grid_dims_matches_the_config(self, windows: dict) -> None:
        data = {"pdb": "dummy.pdb", "temperatures": [0.4, 0.5], **windows}
        rex = yaml_dict_to_config(data).replica_exchange
        assert rex is not None
        n_windows = len(rex.q_targets or rex.native_contact_targets or [None])
        n_temps = len(rex.temperatures)
        assert replica_grid_dims(data) == (n_temps * n_windows, n_temps)

    def test_replica_grid_dims_ignores_n_q_windows(self) -> None:
        # The YAML loader never read n_q_windows, so the run makes one window.
        data = {"pdb": "dummy.pdb", "temperatures": [0.4, 0.5], "n_q_windows": 3}
        assert replica_grid_dims(data) == (2, 2)


@pytest.mark.parametrize(
    "extra, ok",
    [("q_targets: [0.0, 0.5]\n", True), ("q_target: [0.0, 0.5]\n", False)],
)
def test_submit_script_checks_the_keys_before_submitting(
    tmp_path: Path, extra: str, ok: bool
) -> None:
    script = REPO_ROOT / "scripts" / "submit.sh"
    if shutil.which("bash") is None or not script.is_file():
        pytest.skip("needs bash and a source checkout")
    # submit.sh runs `python`; make that this interpreter.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "python").write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    (bin_dir / "python").chmod(0o755)
    env = {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"}
    config = tmp_path / "remd.yaml"
    config.write_text("pdb: examples/data/1uao.pdb\ntemperatures: [0.4, 0.5]\n" + extra)

    result = subprocess.run(
        ["bash", str(script), str(config), "--dry-run"],
        capture_output=True, text=True, env=env, cwd=REPO_ROOT,
    )

    if ok:
        assert result.returncode == 0, result.stderr
        assert "sbatch --ntasks=4" in result.stdout
    else:
        assert result.returncode != 0
        assert "'q_target' (did you mean 'q_targets'?)" in result.stderr
        assert "sbatch" not in result.stdout
