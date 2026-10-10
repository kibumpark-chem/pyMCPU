"""A checkpoint written before move counters were saved still resumes.

The integrator's accept/attempt counters are cumulative, and the energy CSV
writes them as running totals. Checkpoints used to save only the RNG state,
so a resumed run built a fresh integrator whose counters started at 0 while
the CSV kept appending: the move columns dropped back to zero mid-file.
Checkpoints now carry the counters, and ``test_resume_matches_uninterrupted.py``
checks that a resumed run's CSVs and final counters match an uninterrupted
run. This file keeps the old-checkpoint case.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("mdtraj")

from pymcpu.checkpointing import load_checkpoint, save_checkpoint  # noqa: E402
from pymcpu.sampling.folding import FoldingRunner  # noqa: E402


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


def test_a_checkpoint_without_counters_still_resumes(chignolin_pdb_path: str, tmp_path: Path) -> None:
    """A checkpoint written before counters were saved has none. It still
    loads, and its replica counts from 0 again, as every resume used to."""
    first = _folding(chignolin_pdb_path, tmp_path / "o", tmp_path / "ck")
    first.run(n_cycles=2)
    for chk in (tmp_path / "ck").glob("*.chk"):
        state = load_checkpoint(chk)
        assert state.pop("integrator_move_counters")  # saved by this version...
        save_checkpoint(state, chk.parent, filename=chk.name)  # ...and now removed

    resumed = _folding(chignolin_pdb_path, tmp_path / "o", tmp_path / "ck", resume=True)
    resumed.run(n_cycles=4)
    integ = resumed.simulation.integrator
    # Only the 2 cycles after the resume (5 steps each) were counted.
    assert integ.get_bb_attempted() + integ.get_kic_attempted() + integ.get_sc_attempted() == 2 * 5
