"""A resumed run caches the chi angles of its coordinates at both chain ends.

A resume places the saved coordinates with ``Context.set_positions``, which
measures every residue's chi angles from them, the first and the last
residue's included (``tests/physics/test_chain_end_chi.py``). Each runner here
turns sidechains (both ends among them) before its checkpoints, and a new
runner, whose own start placement cached the start structure's chi, resumes
from them. Crambin (PDB 3NIR) starts with THR and ends with ASN.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mdtraj")

import pymcpu  # noqa: E402
from pymcpu.sampling.folding import FoldingRunner  # noqa: E402
from pymcpu.sampling.replica_exchange import ReplicaExchange, get_coords  # noqa: E402
from tests.helpers.chi_cache import angle_gap, chi_from_coordinates, stale_chi  # noqa: E402

CRAMBIN_PDB = (
    Path(pymcpu.PACKAGE_ROOT).parent / "tests" / "physics" / "forcefield" / "data" / "3nir_heavy.pdb"
)
SIDECHAINS_ONLY = (0.0, 0.0, 1.0)


def _ends_turned(forcefield, start: np.ndarray, now: np.ndarray) -> bool:
    a, b = chi_from_coordinates(forcefield, start), chi_from_coordinates(forcefield, now)
    last = len(a) - 1
    return bool(angle_gap(a[0, :1], b[0, :1]).min() > 0.01
                and angle_gap(a[last, :2], b[last, :2]).min() > 0.01)


def _folding(root: Path) -> FoldingRunner:
    return FoldingRunner(
        str(CRAMBIN_PDB), output_dir=str(root / "out"), checkpoint_dir=str(root / "ck"), seed=5,
        temperature=1.0, move_weights=SIDECHAINS_ONLY, report_interval=1000,
        steps_per_cycle=1000, checkpoint_interval=1, verbose=False,
    )


def test_a_folding_resume_caches_every_residues_chi(tmp_path: Path) -> None:
    first = _folding(tmp_path)
    start = (first.forcefield.coords[0] * 10.0).T
    first.run(n_cycles=3)
    saved = get_coords(first.simulation.context)
    assert _ends_turned(first.forcefield, start, saved)

    second = _folding(tmp_path)
    second.load_checkpoint(tmp_path / "ck" / "last.chk")
    ctx = second.simulation.context
    np.testing.assert_array_equal(get_coords(ctx), saved)
    assert stale_chi(ctx, second.forcefield) == []


def _replica_exchange(root: Path) -> ReplicaExchange:
    return ReplicaExchange(
        str(CRAMBIN_PDB), temperatures=[0.95, 1.0], n_targets=[0.0], k_bias=0.0,
        log_interval=1000, output_prefix="rex", output_dir=root / "out", seed=7,
        move_weights=SIDECHAINS_ONLY, checkpoint_dir=root / "ck", checkpoint_interval=1,
        keep_last_n=2,
    )


def test_a_replica_exchange_run_and_resume_cache_every_residues_chi(tmp_path: Path) -> None:
    """Each cycle ends with the exchange, so the replicas' caches are read just
    after the swaps; then a new run resumes from the checkpoint."""
    rex = _replica_exchange(tmp_path)
    start = (rex.forcefield.coords[0] * 10.0).T
    summary = rex.run(4, 1000, write_logs=False, verbose=False,
                      checkpoint_dir=tmp_path / "ck", checkpoint_interval=1)
    assert summary.n_temp_accepts > 0
    saved = [get_coords(rep.simulation.context) for rep in rex.replicas]
    assert all(_ends_turned(rex.forcefield, start, c) for c in saved)
    for rep in rex.replicas:
        assert stale_chi(rep.simulation.context, rex.forcefield) == []

    resumed = _replica_exchange(tmp_path)
    resumed.load_checkpoint(tmp_path / "ck")
    for rep, coords in zip(resumed.replicas, saved):
        np.testing.assert_array_equal(get_coords(rep.simulation.context), coords)
        assert stale_chi(rep.simulation.context, resumed.forcefield) == []
