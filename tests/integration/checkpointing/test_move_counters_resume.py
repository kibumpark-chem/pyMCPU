"""Move counters survive a checkpoint resume.

The integrator's accept/attempt counters are cumulative, and the energy CSV
writes them as running totals. Checkpoints used to save only the RNG state,
so a resumed run built a fresh integrator whose counters started at 0 while
the CSV kept appending: the move columns dropped back to zero mid-file.

Each test compares an interrupted-and-resumed run with an uninterrupted run
of the same seed. Both checkpoint at every cycle, because reading the RNG
state for a checkpoint also resets the cached normal draw -- two runs only
share a stream if they checkpoint at the same points.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

pytest.importorskip("mdtraj")

from pymcpu.checkpointing import (  # noqa: E402
    get_integrator_move_counters,
    set_integrator_move_counters,
)
from pymcpu.sampling.folding import FoldingRunner  # noqa: E402
from pymcpu.sampling.replica_exchange import ReplicaExchange  # noqa: E402


def _move_columns(csv_path: Path) -> list[dict[str, int]]:
    with csv_path.open() as fh:
        rows = list(csv.DictReader(fh))
    return [
        {k: int(v) for k, v in row.items() if k.endswith(("_accepted", "_attempted"))}
        for row in rows
    ]


def _assert_never_decreases(rows: list[dict[str, int]]) -> None:
    for prev, cur in zip(rows, rows[1:]):
        for col in cur:
            assert cur[col] >= prev[col], f"{col} dropped from {prev[col]} to {cur[col]}"


def _folding(pdb: str, out: Path, ckpt: Path, *, resume: bool = False) -> FoldingRunner:
    return FoldingRunner(
        pdb,
        output_dir=str(out),
        checkpoint_dir=str(ckpt),
        seed=11,
        report_interval=5,
        steps_per_cycle=5,
        checkpoint_interval=1,
        resume=resume,
        verbose=False,
    )


def test_folding_resume_continues_the_counters(chignolin_pdb_path: str, tmp_path: Path) -> None:
    straight = _folding(chignolin_pdb_path, tmp_path / "a", tmp_path / "ck_a")
    straight.run(n_cycles=6)
    expected = straight.simulation.integrator.get_move_counters()

    first = _folding(chignolin_pdb_path, tmp_path / "b", tmp_path / "ck_b")
    first.run(n_cycles=3)
    resumed = _folding(chignolin_pdb_path, tmp_path / "b", tmp_path / "ck_b", resume=True)
    resumed.run(n_cycles=6)

    assert resumed.simulation.integrator.get_move_counters() == expected
    rows = _move_columns(tmp_path / "b" / "folding_data.csv")
    _assert_never_decreases(rows)
    assert rows == _move_columns(tmp_path / "a" / "folding_data.csv")


def _rex(pdb: str, out: Path, ckpt: Path) -> ReplicaExchange:
    return ReplicaExchange(
        pdb,
        temperatures=[0.5, 0.6],
        n_targets=[0.0],
        k_bias=0.0,
        log_interval=5,
        output_prefix="rex",
        output_dir=out,
        seed=7,
        checkpoint_dir=ckpt,
        checkpoint_interval=1,
    )


def test_remd_resume_continues_the_counters(chignolin_pdb_path: str, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    straight = _rex(chignolin_pdb_path, tmp_path / "a", tmp_path / "ck_a")
    straight.run(4, 5, verbose=False, checkpoint_dir=tmp_path / "ck_a", checkpoint_interval=1)
    expected = get_integrator_move_counters(straight.replicas)

    first = _rex(chignolin_pdb_path, tmp_path / "b", tmp_path / "ck_b")
    first.run(2, 5, verbose=False, checkpoint_dir=tmp_path / "ck_b", checkpoint_interval=1)
    resumed = _rex(chignolin_pdb_path, tmp_path / "b", tmp_path / "ck_b")
    resumed.run(
        4, 5, verbose=False, checkpoint_dir=tmp_path / "ck_b", checkpoint_interval=1,
        resume=tmp_path / "ck_b" / "last.chk",
    )

    assert get_integrator_move_counters(resumed.replicas) == expected
    for f in sorted((tmp_path / "b").glob("rex_*_data.csv")):
        _assert_never_decreases(_move_columns(f))


def test_counters_from_an_older_checkpoint_are_optional(chignolin_pdb_path: str, tmp_path: Path) -> None:
    """A checkpoint written before counters were saved has none: nothing is
    restored and the replica keeps counting from where it is."""
    runner = _folding(chignolin_pdb_path, tmp_path / "o", tmp_path / "ck")
    runner.run(n_cycles=1)
    before = runner.simulation.integrator.get_move_counters()
    set_integrator_move_counters(runner.replicas, [])
    set_integrator_move_counters(runner.replicas, [{}])
    assert runner.simulation.integrator.get_move_counters() == before
