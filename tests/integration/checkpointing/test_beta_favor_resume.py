"""A checkpoint resumes only with the hydrogen-bond sheet factor it was
written with.

The factor (``forcefield_options: {hbond_beta_favor: ...}``) is part of the
energy function, so resuming with another value would change the ensemble in
the middle of a run. Checkpoints record it; one written before it was
recorded came from a build that used legacy MCPU's 3.0. The MPI driver's
case is in ``tests/integration/mpi/test_mpi_checkpointing.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mdtraj")

from pymcpu.checkpointing import (  # noqa: E402
    checkpoint_beta_favor_error,
    load_checkpoint,
    save_checkpoint,
)
from pymcpu.sampling.folding import FoldingRunner  # noqa: E402
from pymcpu.sampling.replica_exchange import ReplicaExchange  # noqa: E402


def test_the_check_compares_the_saved_factor_with_the_runs() -> None:
    assert checkpoint_beta_favor_error(2.0, 2.0) is None
    assert checkpoint_beta_favor_error(None, 3.0) is None  # written before it was recorded
    assert checkpoint_beta_favor_error(2.5, None) is None  # a force field without the factor
    message = checkpoint_beta_favor_error(2.5, 2.0, "run/last.chk")
    assert message.startswith(
        "checkpoint run/last.chk was written with hbond_beta_favor 2.5, but this run uses 2.0;")
    assert "forcefield_options: {hbond_beta_favor: 2.5}" in message
    assert "hbond_beta_favor 3.0, but this run uses 2.0" in checkpoint_beta_favor_error(None, 2.0)
    # The engine holds the factor as float32: values it cannot tell apart match, and the
    # value the advice prints resumes the run.
    assert checkpoint_beta_favor_error(4 / 3, np.float32(4 / 3)) is None
    assert "{hbond_beta_favor: 1.3333334}" in checkpoint_beta_favor_error(4 / 3, 2.0)
    assert checkpoint_beta_favor_error(4 / 3, 1.3333334) is None


def _folding(pdb: str, root: Path, *, resume: bool = False, **options) -> FoldingRunner:
    return FoldingRunner(
        pdb, output_dir=str(root / "out"), checkpoint_dir=str(root / "ck"), seed=5,
        report_interval=5, steps_per_cycle=5, checkpoint_interval=1, resume=resume,
        forcefield_options=options or None, verbose=False,
    )


def _attempted(runner: FoldingRunner) -> int:
    integ = runner.simulation.integrator
    return integ.get_bb_attempted() + integ.get_kic_attempted() + integ.get_sc_attempted()


def test_folding_records_the_factor_and_refuses_another(
    chignolin_pdb_path: str, tmp_path: Path
) -> None:
    _folding(chignolin_pdb_path, tmp_path).run(n_cycles=2)
    assert load_checkpoint(tmp_path / "ck" / "last.chk")["hbond_beta_favor"] == 2.0

    other = _folding(chignolin_pdb_path, tmp_path, resume=True, hbond_beta_favor=3.0)
    with pytest.raises(ValueError, match="written with hbond_beta_favor 2.0, but this run uses 3.0"):
        other.run(n_cycles=4)
    assert _attempted(other) == 0

    same = _folding(chignolin_pdb_path, tmp_path, resume=True)
    same.run(n_cycles=4)
    assert _attempted(same) == 4 * 5  # the counters continue from the checkpoint


def test_an_old_folding_checkpoint_resumes_with_the_legacy_factor(
    chignolin_pdb_path: str, tmp_path: Path
) -> None:
    _folding(chignolin_pdb_path, tmp_path, hbond_beta_favor=3.0).run(n_cycles=2)
    for chk in (tmp_path / "ck").glob("*.chk"):
        state = load_checkpoint(chk)
        assert state.pop("hbond_beta_favor") == 3.0  # saved by this version...
        save_checkpoint(state, chk.parent, filename=chk.name)  # ...and now removed

    default = _folding(chignolin_pdb_path, tmp_path, resume=True)
    with pytest.raises(ValueError, match=r"forcefield_options: \{hbond_beta_favor: 3.0\}"):
        default.run(n_cycles=4)

    legacy = _folding(chignolin_pdb_path, tmp_path, resume=True, hbond_beta_favor=3.0)
    legacy.run(n_cycles=4)
    assert _attempted(legacy) == 4 * 5


def test_serial_replica_exchange_refuses_another_factor(
    chignolin_pdb_path: str, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)

    def rex(**options) -> ReplicaExchange:
        return ReplicaExchange(
            chignolin_pdb_path, temperatures=[0.5, 0.6], n_targets=[0.0], k_bias=0.0,
            log_interval=5, output_prefix="rex", output_dir=tmp_path / "out", seed=3,
            checkpoint_dir=tmp_path / "ck", checkpoint_interval=1,
            forcefield_options=options or None,
        )

    rex().run(1, 5, verbose=False, checkpoint_dir=tmp_path / "ck", checkpoint_interval=1)
    assert load_checkpoint(tmp_path / "ck" / "last.chk")["hbond_beta_favor"] == 2.0
    with pytest.raises(ValueError, match="written with hbond_beta_favor 2.0, but this run uses 3.0"):
        rex(hbond_beta_favor=3.0).run(2, 5, verbose=False, checkpoint_dir=tmp_path / "ck",
                                      checkpoint_interval=1, resume=tmp_path / "ck" / "last.chk")
