"""CLI entry point smoke tests: ``pymcpu.cli.build_parser``/``main``.

Covers argparse subcommand wiring and exit-code/stdout contract for
``mcpu version`` and ``mcpu validate`` only -- no simulation is run. Config
*content* parsing correctness lives in ``test_config_schema.py``; this file
only checks that the CLI plumbs a config path through to
``pymcpu.config.load_config_auto`` and reports success/failure the way a
shell caller (or CI) would rely on (rc=0/1, a human-readable stdout line).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pymcpu import PACKAGE_ROOT
from pymcpu.cli import build_parser, main

REPO_ROOT = Path(PACKAGE_ROOT).parent
FOLDING_CONFIG = REPO_ROOT / "examples" / "configs" / "folding.json"


def test_mcpu_version(capsys: pytest.CaptureFixture[str]) -> None:
    # rc=0 is the CLI's own success contract (Unix convention), not derived from anything.
    assert main(["version"]) == 0
    out = capsys.readouterr().out.strip()
    assert out  # exact version string is pymcpu.__version__'s concern, not the CLI's


def test_parser_recognizes_validate_and_run_subcommands() -> None:
    parser = build_parser()
    args = parser.parse_args(["validate", str(FOLDING_CONFIG)])
    assert args.command == "validate"
    args2 = parser.parse_args(["run", "config.json"])
    assert args2.command == "run"


def test_validate_example_config_reports_ok(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(REPO_ROOT)  # folding.json's pdb path is repo-relative
    rc = main(["validate", str(FOLDING_CONFIG)])
    assert rc == 0
    # "OK" is _cmd_validate's own literal success marker (pymcpu/cli.py), not a physics value
    assert "OK" in capsys.readouterr().out


def test_validate_missing_file_returns_error(tmp_path: Path) -> None:
    rc = main(["validate", str(tmp_path / "missing.json")])
    assert rc == 1  # rc=1 is the CLI's own failure contract


def test_validate_reports_a_missing_pdb(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = tmp_path / "missing_pdb.yaml"
    config.write_text("pdb: no_such_structure.pdb\ntemperatures: [0.5]\n")
    assert main(["validate", str(config)]) == 1
    assert "pdb 'no_such_structure.pdb': not found" in capsys.readouterr().err


def test_validate_reports_a_missing_reference_pdb(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(REPO_ROOT)
    config = tmp_path / "missing_reference.yaml"
    config.write_text(
        "pdb: examples/data/1uao.pdb\n"
        "reference_pdb: no_such_native.pdb\n"
        "temperatures: [0.4, 0.5]\n"
    )
    assert main(["validate", str(config)]) == 1
    assert "reference_pdb 'no_such_native.pdb': not found" in capsys.readouterr().err


def test_validate_checks_the_reference_pdb_of_a_folding_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A folding run counts Q against reference_pdb (for its early stop and
    # its checkpoints), so validate checks it there too.
    monkeypatch.chdir(REPO_ROOT)
    config = tmp_path / "folding.yaml"
    config.write_text(
        "pdb: examples/data/1uao.pdb\n"
        "reference_pdb: no_such_native.pdb\n"
        "temperatures: [0.5]\n"
    )
    assert main(["validate", str(config)]) == 1
    assert "reference_pdb 'no_such_native.pdb': not found" in capsys.readouterr().err


def test_validate_says_when_checkpointing_is_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(REPO_ROOT)
    config = tmp_path / "off.yaml"
    config.write_text(
        "pdb: examples/data/1uao.pdb\n"
        "temperatures: [0.5]\n"
        "checkpointing:\n"
        "  enabled: false\n"
    )
    assert main(["validate", str(config)]) == 0
    out = capsys.readouterr().out
    assert "checkpointing off" in out
    assert "checkpoint_dir=" not in out


def test_validate_reports_a_pdb_that_is_a_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(REPO_ROOT)
    config = tmp_path / "directory_pdb.yaml"
    config.write_text("pdb: examples/data\ntemperatures: [0.5]\n")
    assert main(["validate", str(config)]) == 1
    assert "is not a file" in capsys.readouterr().err


def test_validate_reports_a_bad_value_in_one_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # A number where the loader expects a list raises TypeError, which used to
    # escape as a traceback.
    config = tmp_path / "scalar_temperatures.yaml"
    config.write_text("pdb: examples/data/1uao.pdb\ntemperatures: 0.5\n")
    assert main(["validate", str(config)]) == 1
    assert "config validation failed: TypeError" in capsys.readouterr().err


def _config_with_checkpointing(tmp_path: Path) -> Path:
    config = tmp_path / "run.yaml"
    config.write_text(
        "pdb: examples/data/1uao.pdb\n"
        "temperatures: [0.5, 0.6]\n"
        "checkpointing:\n"
        "  checkpoint_dir: mine\n"
        "  checkpoint_interval: 7\n"
        "  keep_last_n: 2\n"
    )
    return config


def _checkpoint_settings(checkpoint) -> tuple:
    return (
        checkpoint.checkpoint_dir,
        checkpoint.checkpoint_interval,
        checkpoint.keep_last_n,
        checkpoint.resume,
    )


# A checkpoint flag left out keeps the config's setting; a flag given wins.
_CHECKPOINT_FLAG_CASES = [
    ([], ("mine", 7, 2, False)),
    (["--checkpoint-interval", "3", "--resume"], ("mine", 3, 2, True)),
]


@pytest.mark.parametrize("command", ["run", "validate"])
@pytest.mark.parametrize("flags, expected", _CHECKPOINT_FLAG_CASES)
def test_checkpoint_flags_override_only_what_they_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, command: str, flags: list, expected: tuple
) -> None:
    import pymcpu.config

    monkeypatch.chdir(REPO_ROOT)
    loaded = []
    load = pymcpu.config.load_config_auto
    monkeypatch.setattr(
        pymcpu.config, "load_config_auto", lambda path: loaded.append(load(path)) or loaded[-1]
    )
    monkeypatch.setattr("pymcpu.runners.run_from_config", lambda cfg, **kwargs: None)

    assert main([command, str(_config_with_checkpointing(tmp_path)), *flags]) == 0
    assert _checkpoint_settings(loaded[0].checkpoint) == expected
