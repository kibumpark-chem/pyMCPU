"""Characterization + regression tests for scripts/run_mcpu_replica_exchange.py.

This script drives the production REMD jobs on the cluster
(``python scripts/run_mcpu_replica_exchange.py --mpi -c <yaml>``). It takes a
config file only and goes through ``pymcpu.runners.run_from_config``, the
same path as ``mcpu run``; these tests pin its dispatch behavior and,
separately, run it for real end-to-end against a tiny config.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_mcpu_replica_exchange.py"
EXAMPLE_PDB = REPO_ROOT / "examples" / "data" / "1uao.pdb"


def _load_script_module() -> ModuleType:
    """scripts/ isn't a package, so load the module directly by path."""
    spec = importlib.util.spec_from_file_location("run_mcpu_replica_exchange_script", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def script() -> ModuleType:
    return _load_script_module()


def _write_config(tmp_path: Path, **overrides) -> Path:
    cfg: dict = {
        "mode": "replica_exchange_2d",
        "pdb": str(EXAMPLE_PDB),
        "integrator": {"seed": 42},
        "replica_exchange": {
            "temperatures": [0.5, 0.6],
            "native_contact_targets": [0, 5],
            "k_native_contacts": 0.0,
            "cycles": 2,
            "steps_per_cycle": 5,
            "log_interval": 5,
        },
        "outputs": {"output_dir": str(tmp_path / "out"), "prefix": "rex"},
    }
    cfg.update(overrides)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg))
    return path


def test_config_path_calls_run_from_config_without_mpi(
    script: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _write_config(tmp_path)
    called = MagicMock(return_value=None)
    monkeypatch.setattr(script, "run_from_config", called)
    monkeypatch.setattr(
        sys, "argv", ["run_mcpu_replica_exchange.py", "-c", str(config_path)]
    )

    script.main()

    called.assert_called_once()
    cfg = called.call_args.args[0]
    assert cfg.mode == "replica_exchange_2d"
    assert called.call_args.kwargs["comm"] is None
    assert called.call_args.kwargs["verbose"] is True


def test_config_path_builds_comm_when_mpi_flag_set(
    script: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _write_config(tmp_path)
    called = MagicMock(return_value=None)
    monkeypatch.setattr(script, "run_from_config", called)
    monkeypatch.setattr(
        sys, "argv", ["run_mcpu_replica_exchange.py", "--mpi", "-c", str(config_path)]
    )

    fake_mpi = MagicMock()
    fake_mpi.COMM_WORLD = "fake-comm-world"
    monkeypatch.setitem(sys.modules, "mpi4py", MagicMock(MPI=fake_mpi))

    script.main()

    called.assert_called_once()
    assert called.call_args.kwargs["comm"] == "fake-comm-world"


def test_config_path_applies_hdf5_and_checkpoint_cli_overrides(
    script: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _write_config(tmp_path)
    called = MagicMock(return_value=None)
    monkeypatch.setattr(script, "run_from_config", called)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_mcpu_replica_exchange.py",
            "-c", str(config_path),
            "--hdf5", "custom.h5",
            "--checkpoint-interval", "7",
            "--resume",
        ],
    )

    script.main()

    cfg = called.call_args.args[0]
    assert cfg.outputs.hdf5 == "custom.h5"
    assert cfg.checkpoint.checkpoint_interval == 7
    assert cfg.checkpoint.resume is True


def test_config_path_keeps_the_config_checkpoint_settings_without_flags(
    script: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A checkpoint flag left out keeps the config's setting."""
    config_path = _write_config(
        tmp_path, checkpoint={"checkpoint_interval": 7, "keep_last_n": 2}
    )
    called = MagicMock(return_value=None)
    monkeypatch.setattr(script, "run_from_config", called)
    monkeypatch.setattr(sys, "argv", ["run_mcpu_replica_exchange.py", "-c", str(config_path)])

    script.main()

    checkpoint = called.call_args.args[0].checkpoint
    assert (checkpoint.checkpoint_interval, checkpoint.keep_last_n) == (7, 2)


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["--pdb", str(EXAMPLE_PDB), "--temp-min", "0.4", "--n-temps", "2"],
    ],
    ids=["no-config", "old-flags"],
)
def test_requires_a_config_file(
    script: ModuleType, monkeypatch: pytest.MonkeyPatch, argv: list[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["run_mcpu_replica_exchange.py", *argv])
    with pytest.raises(SystemExit) as exit_info:
        script.parse_args()
    assert exit_info.value.code == 2


def test_config_path_rejects_config_missing_replica_exchange(
    script: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "folding.json"
    path.write_text(
        json.dumps(
            {
                "mode": "folding",
                "pdb": str(EXAMPLE_PDB),
                "integrator": {"seed": 42},
                "outputs": {"output_dir": str(tmp_path / "out")},
            }
        )
    )
    monkeypatch.setattr(sys, "argv", ["run_mcpu_replica_exchange.py", "-c", str(path)])

    with pytest.raises(ValueError, match="replica exchange"):
        script.main()


@pytest.mark.slow
def test_config_path_smoke_run_serial(
    script: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real end-to-end run (tiny cycles/steps, no mocking) through the
    rewritten config path -- the smoke test called for before pointing real
    production submission.sh jobs at this script."""
    config_path = _write_config(tmp_path)
    monkeypatch.setattr(sys, "argv", ["run_mcpu_replica_exchange.py", "-c", str(config_path)])

    script.main()

    out_dir = tmp_path / "out"
    produced = {p.name for p in out_dir.iterdir()}
    assert any(name.endswith(".csv") for name in produced)


@pytest.mark.slow
def test_config_path_smoke_run_mpi_with_mock_comm(
    script: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mock_comm_rank0
) -> None:
    """Real end-to-end run through the ``--mpi`` branch of the rewritten
    config path, using a mock single-rank comm (no real mpirun required) --
    this is the actual production invocation shape
    (``--mpi -c <yaml>``, per submission.sh)."""
    import pymcpu.sampling.mpi_replica_exchange as mpi_re_module

    if mpi_re_module.MPI is None:  # pragma: no cover - only when libmpi is missing
        fake_mpi = MagicMock()
        fake_mpi.TAG_UB = 32767
        fake_mpi.LOR = object()
        monkeypatch.setattr(mpi_re_module, "MPI", fake_mpi)
        monkeypatch.setattr(mpi_re_module, "_require_mpi", lambda: fake_mpi)

    config_path = _write_config(tmp_path)
    monkeypatch.setattr(
        sys, "argv", ["run_mcpu_replica_exchange.py", "--mpi", "-c", str(config_path)]
    )

    fake_mpi4py_mpi = MagicMock()
    fake_mpi4py_mpi.COMM_WORLD = mock_comm_rank0
    monkeypatch.setitem(sys.modules, "mpi4py", MagicMock(MPI=fake_mpi4py_mpi))

    script.main()

    out_dir = tmp_path / "out"
    produced = {p.name for p in out_dir.iterdir()}
    assert any(name.endswith(".csv") for name in produced)
