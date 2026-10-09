"""``ReplicaExchange``-level checkpoint integration: coordinate/cycle/walker
restore and partial resume.

Unlike ``test_checkpoint_io.py`` (plain dicts/arrays through
``save_checkpoint``/``load_checkpoint``), these tests build a real serial
``ReplicaExchange`` over a small reference PDB and run genuine MC cycles --
they are slower, integration-flavored tests of the resume *orchestration*
(does the right coordinate/step/walker state come back, does resuming run
only the remaining cycles), not of MC physics itself. No legacy MCPU
equivalent exists for any of this.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import pymcpu

REPO_ROOT = Path(pymcpu.PACKAGE_ROOT).parent
TINY_PDB = REPO_ROOT / "examples" / "data" / "1uao.pdb"

pytestmark = pytest.mark.skipif(not TINY_PDB.is_file(), reason="example PDB missing")


def test_rex_checkpoint_restores_coords_and_cycle(tmp_path: Path) -> None:
    from pymcpu.sampling.replica_exchange import ReplicaExchange, get_coords

    out = tmp_path / "out"
    ckpt = tmp_path / "ckpt"
    rex = ReplicaExchange(
        str(TINY_PDB),
        temperatures=[0.5, 0.6],
        n_targets=[0.0],
        k_bias=0.0,
        log_interval=10,
        output_prefix="rex",
        output_dir=out,
        seed=7,
        checkpoint_dir=ckpt,
        checkpoint_interval=1,
        keep_last_n=3,
    )
    rex.run(2, 2, verbose=False, write_logs=False, checkpoint_dir=ckpt, checkpoint_interval=1)
    assert (ckpt / "last.chk").is_file()
    assert rex.cycle == 2

    coords_before = [get_coords(r.simulation.context).copy() for r in rex.replicas]
    walkers_before = rex._walker_at_state.copy()

    # Diverge state on every replica, then restore from the checkpoint.
    for rep in rex.replicas:
        c = get_coords(rep.simulation.context)
        rep.simulation.context.set_positions(c + 1.0)
        rep.simulation.current_step = 0
    rex._cycle = 0
    rex._walker_at_state[:] = 0

    rex.load_checkpoint(ckpt)
    assert rex.cycle == 2
    assert rex._walker_at_state.tolist() == walkers_before.tolist()
    for i, rep in enumerate(rex.replicas):
        np.testing.assert_allclose(
            get_coords(rep.simulation.context), coords_before[i], rtol=0, atol=1e-5
        )
        # 2 cycles * 2 MC steps/cycle from the rex.run() call above.
        assert rep.simulation.current_step == 2 * 2


def test_rex_resume_runs_only_remaining_cycles(tmp_path: Path) -> None:
    from pymcpu.sampling.replica_exchange import ReplicaExchange

    out = tmp_path / "out"
    ckpt = tmp_path / "ckpt"
    rex = ReplicaExchange(
        str(TINY_PDB),
        temperatures=[0.55],
        n_targets=[0.0],
        k_bias=0.0,
        log_interval=10,
        output_prefix="rex",
        output_dir=out,
        seed=3,
        checkpoint_dir=ckpt,
        checkpoint_interval=1,
    )
    rex.run(2, 1, verbose=False, write_logs=False)
    assert rex.cycle == 2

    summary = rex.run(
        4,
        1,
        verbose=False,
        write_logs=False,
        resume=ckpt / "last.chk",
    )
    assert rex.cycle == 4
    # A single temperature has no temperature-dimension exchange to attempt.
    assert summary.n_temp_attempts == 0
