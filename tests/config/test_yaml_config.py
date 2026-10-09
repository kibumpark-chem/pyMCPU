"""Tests for the flat YAML config schema: ``pymcpu.config.load_yaml_config``
and ``pymcpu.config.yaml_dict_to_config``.

The shipped ``examples/configs/template.yaml`` anchors the "real file" test;
the schema-logic tests (temperature-ladder expansion, q/n-target mutual
exclusion, required/unknown field handling) use inline dicts/tmp_path files
instead, since that logic doesn't depend on the shipped file's contents.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from pymcpu import PACKAGE_ROOT
from pymcpu.config import load_yaml_config, replica_grid_dims, yaml_dict_to_config

REPO_ROOT = Path(PACKAGE_ROOT).parent
TEMPLATE_YAML = REPO_ROOT / "examples" / "configs" / "template.yaml"


class TestShippedYamlFiles:
    """Parsing of the YAML template shipped in the repo."""

    def test_load_yaml_config_template(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(REPO_ROOT)
        cfg = load_yaml_config(TEMPLATE_YAML)
        # template.yaml has n_temps:11 > 1 -> mode inference rule picks replica_exchange_2d
        assert cfg.mode == "replica_exchange_2d"
        assert cfg.integrator.seed == 0  # mirrors template.yaml's seed field


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
        with pytest.raises(ValueError, match="requires 'pdb'"):
            load_yaml_config(bad)

    def test_unknown_field_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "unknown.yaml"
        f.write_text("pdb: test.pdb\ncompletely_unknown_field: 42\n")
        with pytest.raises(ValueError, match="Unknown key in .*unknown.yaml: 'completely_unknown_field'"):
            load_yaml_config(f)

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


def _yaml(**keys) -> object:
    return yaml_dict_to_config({"pdb": "dummy.pdb", **keys})


class TestUmbrellaDefaults:
    """Without targets a replica exchange config gets no umbrella; with
    targets and no k_bias, the umbrella strength stays 1.0."""

    def test_temperatures_only_gets_no_umbrella(self) -> None:
        # It used to get k_bias 1.0 and one window at N = 0, which pulled every
        # replica toward unfolded structures.
        rex = _yaml(temperatures=[0.4, 0.5]).replica_exchange
        assert not rex.has_targets()
        assert rex.effective_k_bias() == 0.0

    @pytest.mark.parametrize("targets", [{"n_targets": [0, 5]}, {"q_targets": [0.2, 0.8]}])
    def test_targets_without_k_bias_keep_the_default(self, targets: dict) -> None:
        assert _yaml(temperatures=[0.4, 0.5], **targets).replica_exchange.effective_k_bias() == 1.0

    def test_an_explicit_k_bias_applies_without_targets(self) -> None:
        assert _yaml(temperatures=[0.4, 0.5], k_bias=0.3).replica_exchange.effective_k_bias() == 0.3

    def test_an_empty_target_list_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="'n_targets' is empty"):
            _yaml(temperatures=[0.4, 0.5], n_targets=[])


class TestOneTemperatureWithTargets:
    """One temperature with targets is umbrella sampling at that temperature;
    it used to run folding and drop the targets without a message."""

    def test_runs_replica_exchange_over_the_windows(self) -> None:
        data = {"pdb": "dummy.pdb", "temperatures": [0.5], "n_targets": [0, 5, 10], "k_bias": 0.5}
        cfg = yaml_dict_to_config(data)
        assert cfg.mode == "replica_exchange_2d"
        assert cfg.replica_exchange.temperatures == [0.5]
        assert cfg.replica_exchange.native_contact_targets == [0.0, 5.0, 10.0]
        assert cfg.replica_exchange.effective_k_bias() == 0.5
        assert replica_grid_dims(data) == (3, 1)

    def test_q_targets_count_too(self) -> None:
        assert _yaml(temp_min=0.5, q_targets=[0.3]).mode == "replica_exchange_2d"

    def test_no_targets_still_runs_folding(self) -> None:
        cfg = _yaml(temperatures=[0.5])
        assert cfg.mode == "folding"
        assert cfg.replica_exchange is None

    def test_the_mode_warning_names_the_rule(self) -> None:
        with pytest.warns(UserWarning, match="more than one temperature or sets targets"):
            _yaml(temperatures=[0.5], mode="folding")


class TestFoldingRunLength:
    """In a folding config ``steps`` is the total number of MC steps; it used
    to count per cycle, so ``steps: N`` ran 10 N steps."""

    def test_steps_is_the_total(self) -> None:
        cfg = _yaml(temperatures=[0.5], steps=5000)
        assert cfg.integrator.steps == 5000
        assert cfg.folding.steps_per_cycle == 1000
        assert cfg.integrator.report_interval == 1000  # log_interval: one cycle

    def test_short_runs_are_one_cycle(self) -> None:
        cfg = _yaml(temperatures=[0.5], steps=300)
        assert (cfg.integrator.steps, cfg.folding.steps_per_cycle) == (300, 300)

    @pytest.mark.parametrize(
        "keys, total, per_cycle",
        [
            ({"steps": 6000, "mc_replica_steps": 500}, 6000, 500),
            ({"steps": 6000, "num_cycles": 4}, 6000, 1500),
            ({"steps": 6000, "num_cycles": 12, "mc_replica_steps": 500}, 6000, 500),
            ({"num_cycles": 3, "mc_replica_steps": 200}, 600, 200),
            ({}, 10_000, 1000),
        ],
    )
    def test_cycle_length(self, keys: dict, total: int, per_cycle: int) -> None:
        cfg = _yaml(temperatures=[0.5], **keys)
        assert (cfg.integrator.steps, cfg.folding.steps_per_cycle) == (total, per_cycle)

    def test_log_interval_stays_independent(self) -> None:
        cfg = _yaml(temperatures=[0.5], steps=6000, log_interval=100)
        assert (cfg.integrator.report_interval, cfg.folding.steps_per_cycle) == (100, 1000)

    @pytest.mark.parametrize(
        "keys, message",
        [
            ({"steps": 6000, "num_cycles": 4, "mc_replica_steps": 500}, "must equal num_cycles x"),
            ({"steps": 6001, "num_cycles": 4}, "must be a multiple of num_cycles"),
            ({"steps": 0}, "steps must be a positive whole number"),
        ],
    )
    def test_inconsistent_settings_are_rejected(self, keys: dict, message: str) -> None:
        with pytest.raises(ValueError, match=message):
            _yaml(temperatures=[0.5], **keys)

    def test_replica_exchange_keeps_steps_per_cycle(self) -> None:
        rex = _yaml(temperatures=[0.4, 0.5], steps=200, num_cycles=3).replica_exchange
        assert (rex.steps_per_cycle, rex.cycles) == (200, 3)


class TestFoldingSettings:
    def test_early_stop_is_off_by_default(self) -> None:
        assert _yaml(temperatures=[0.5]).folding.q_threshold is None

    def test_early_stop_keys(self) -> None:
        folding = _yaml(temperatures=[0.5], q_threshold=0.8, convergence_window=4).folding
        assert (folding.q_threshold, folding.convergence_window) == (0.8, 4)

    @pytest.mark.parametrize("value", [1.1, 0, True])
    def test_q_threshold_must_be_a_fraction(self, value) -> None:
        with pytest.raises(ValueError, match="q_threshold is a fraction"):
            _yaml(temperatures=[0.5], q_threshold=value)

    def test_early_stop_keys_are_refused_in_replica_exchange(self) -> None:
        with pytest.raises(ValueError, match="only applies to a folding run"):
            _yaml(temperatures=[0.4, 0.5], q_threshold=0.8)

    def test_native_contact_keys_reach_a_folding_run(self) -> None:
        folding = _yaml(
            temperatures=[0.5], contact_cutoff=8.0, min_seq_sep=3, contact_atom_mode="cb"
        ).folding
        assert (folding.contact_cutoff, folding.min_seq_sep, folding.contact_atom_mode) == (
            8.0, 3, "cb"
        )
