"""Lean replica-exchange logging: no legacy MCPU equivalent exists for this.

The REMD orchestration layer (``pymcpu.sampling.replica_exchange``) defaults
to compact logging -- no ``*_exchange.csv``/``*_state.csv`` unless explicitly
requested -- because those files scale with cycles x replica-pairs and get
large fast. These tests check the logging *plumbing* (which files appear,
their contents, JSON stats formatting, YAML config defaults), not physics:
legacy MCPU's C/MPI code has no "lean logging" concept to compare against,
so this belongs in integration/, not legacy_parity/.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pymcpu.sampling.replica_exchange import write_rex_stats


def test_write_rex_stats_compact(tmp_path: Path) -> None:
    path = write_rex_stats(
        tmp_path / "rex_stats.json",
        n_temp_accepts=8,
        n_temp_attempts=10,
        n_q_accepts=3,
        n_q_attempts=4,
        cycles_completed=5,
    )
    data = json.loads(path.read_text())
    assert data["temperature"]["attempts"] == 10
    assert data["temperature"]["accepted"] == 8
    assert data["temperature"]["rate"] == pytest.approx(0.8)  # 8/10
    assert data["Q"]["rate"] == pytest.approx(0.75)  # 3/4
    assert data["cycles_completed"] == 5


def test_serial_rex_default_no_exchange_or_state_csv(tmp_path: Path, chignolin_pdb_path: str) -> None:
    pytest.importorskip("mdtraj")
    from pymcpu.sampling.replica_exchange import ReplicaExchange

    out = tmp_path / "out"
    out.mkdir()
    rex = ReplicaExchange(
        str(chignolin_pdb_path),
        temperatures=[0.5, 0.6],
        n_targets=[0.0, 2.0],
        k_bias=0.0,
        log_interval=5,
        output_prefix="rex",
        output_dir=out,
        seed=0,
        exchange_log="none",
        state_log_interval=0,
        log_walker_in_data_csv=True,
    )
    summary = rex.run(2, 5, verbose=False, write_logs=True, checkpoint_dir=None)
    assert summary.exchange_log is None
    assert summary.state_log is None
    assert summary.rex_stats_path is not None
    assert summary.rex_stats_path.is_file()
    assert not (out / "rex_exchange.csv").exists()
    assert not (out / "rex_state.csv").exists()

    stats = json.loads(summary.rex_stats_path.read_text())
    assert stats["temperature"]["attempts"] == summary.n_temp_attempts
    assert stats["temperature"]["accepted"] == summary.n_temp_accepts

    # walker_id column must be present on a per-walker data CSV even with
    # exchange/state logging off -- that column is controlled separately by
    # log_walker_in_data_csv.
    data_files = list(out.glob("rex_*_data.csv"))
    assert data_files
    header = data_files[0].read_text().splitlines()[0]
    assert header.split(",")[-1] == "walker_id"
    rows = data_files[0].read_text().strip().splitlines()[1:]
    assert rows
    wid = int(rows[0].split(",")[-1])
    assert wid >= 0


def test_serial_rex_exchange_log_all(tmp_path: Path, chignolin_pdb_path: str) -> None:
    pytest.importorskip("mdtraj")
    from pymcpu.sampling.replica_exchange import ReplicaExchange

    out = tmp_path / "out"
    out.mkdir()
    rex = ReplicaExchange(
        str(chignolin_pdb_path),
        temperatures=[0.5, 0.6],
        n_targets=[0.0],
        k_bias=0.0,
        log_interval=10,
        output_prefix="rex",
        output_dir=out,
        seed=1,
        exchange_log="all",
        state_log_interval=1,
        log_walker_in_data_csv=True,
    )
    summary = rex.run(2, 5, verbose=False, write_logs=True, checkpoint_dir=None)
    assert summary.exchange_log is not None and summary.exchange_log.is_file()
    assert summary.state_log is not None and summary.state_log.is_file()

    exchange_rows = summary.exchange_log.read_text().strip().splitlines()[1:]
    # Every attempted exchange (accepted or not) writes exactly one row, so
    # the row count must equal total attempts the run itself reports --
    # ties the check to the run's own accounting instead of a hardcoded
    # number derived from internal exchange-scheduling details.
    expected_rows = summary.n_temp_attempts + summary.n_q_attempts
    assert len(exchange_rows) == expected_rows


def test_yaml_lean_logging_defaults() -> None:
    from pymcpu.config import yaml_dict_to_config

    cfg = yaml_dict_to_config(
        {
            "pdb": "dummy.pdb",
            "temperatures": [0.5, 0.6],
            "n_targets": [0, 1],
            "num_cycles": 2,
            "mc_replica_steps": 5,
        }
    )
    assert cfg.replica_exchange is not None
    # Defaults verified against pymcpu/config.py's ReplicaExchangeConfig
    # dataclass fields, not copied from the old test unread.
    assert cfg.replica_exchange.exchange_log == "none"
    assert cfg.replica_exchange.state_log_interval == 0
    assert cfg.replica_exchange.log_walker_in_data_csv is True
