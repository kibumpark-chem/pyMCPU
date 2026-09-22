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
