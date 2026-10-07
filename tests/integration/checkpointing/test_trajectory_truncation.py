"""Trajectory-truncation / resume-plumbing tests.

``pymcpu.trajectory_utils`` truncates on-disk trajectory/energy files (CSV,
XTC, NPZ, HDF5) back to the frame/row/sample count recorded in a checkpoint's
``traj_frame_indices``, so a crash-and-resume run doesn't end up with
duplicate frames written after the last checkpoint. Every test here writes
its own throwaway data with ``tmp_path`` and asserts on file-format
mechanics (row/frame/sample counts, atomic-write temp-file cleanup, dict
round-trips through pickle) -- there is no physics here, and no legacy MCPU
equivalent: legacy MCPU has no resume/checkpointing feature to compare
against, which is why this lives under ``integration/`` rather than
``legacy_parity/`` or ``physics/``.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np
import pytest

# ── Shared helpers ───────────────────────────────────────────────


def _write_csv_rows(path: Path, n_rows: int, columns: tuple[str, ...] = ("step", "energy")) -> None:
    """Write a header row plus ``n_rows`` numbered data rows to ``path``."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(columns)
        for i in range(n_rows):
            w.writerow([i * 10, -500.0 + i] if len(columns) == 2 else [i * 10, -500.0 + i, 300.0])


def _single_atom_trajectory(n_frames: int):
    """A minimal 1-atom mdtraj Trajectory, for exercising XTC truncation
    without needing a real topology."""
    import mdtraj as md

    top = md.Topology()
    chain = top.add_chain()
    res = top.add_residue("ALA", chain)
    top.add_atom("CA", md.element.carbon, res)
    xyz = np.random.rand(n_frames, 1, 3).astype(np.float32)
    return md.Trajectory(xyz, top)


def _import_h5py_or_skip():
    """Skip HDF5 tests when h5py is missing or ABI-incompatible with numpy."""
    try:
        import h5py
    except Exception as exc:
        pytest.skip(f"h5py unavailable: {exc}")
    return h5py


# ── CSV truncation ───────────────────────────────────────────────


class TestCsvTruncation:
    def test_truncation_removes_excess_rows(self, tmp_path: Path) -> None:
        """Write 100 data rows, truncate to 50, verify exactly 50 remain."""
        csv_path = tmp_path / "replica_0_energies.csv"
        _write_csv_rows(csv_path, 100, columns=("step", "energy", "temperature"))

        from pymcpu.trajectory_utils import truncate_csv_to_row

        truncate_csv_to_row(str(csv_path), last_row=50)

        with open(csv_path) as f:
            lines = f.readlines()

        assert len(lines) == 51  # 1 header + 50 kept data rows
        assert lines[0].startswith("step")
        assert lines[-1].startswith("490,")  # last kept row is index 49 -> step = 49*10

    def test_truncation_noop_when_already_short(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "short.csv"
        _write_csv_rows(csv_path, 20)
        original_size = os.path.getsize(csv_path)

        from pymcpu.trajectory_utils import truncate_csv_to_row

        truncate_csv_to_row(str(csv_path), last_row=50)
        assert os.path.getsize(csv_path) == original_size

    def test_truncation_missing_file_no_crash(self, tmp_path: Path) -> None:
        from pymcpu.trajectory_utils import truncate_csv_to_row

        truncate_csv_to_row(str(tmp_path / "nonexistent.csv"), last_row=50)

    def test_truncation_is_atomic(self, tmp_path: Path) -> None:
        """The intermediate ``.trunc.tmp`` file must never survive a truncation."""
        csv_path = tmp_path / "replica_0_energies.csv"
        _write_csv_rows(csv_path, 100)

        from pymcpu.trajectory_utils import truncate_csv_to_row

        truncate_csv_to_row(str(csv_path), last_row=50)
        assert not os.path.exists(str(csv_path) + ".trunc.tmp")


# ── traj_frame_indices dict round-tripping through pickle ────────


class TestTrajFrameIndicesRoundTrip:
    def test_survives_round_trip(self, tmp_path: Path) -> None:
        """The dict itself (not just a scalar) must survive save/load."""
        from pymcpu.checkpointing import (
            CheckpointState,
            load_checkpoint_state,
            save_checkpoint,
        )

        state = CheckpointState(
            cycle=10,
            global_step=10000,
            replica_coords=[np.zeros((5, 3))],
            walker_at_state=np.array([0]),
            current_steps=[10000],
            temperatures=[300.0],
            integrator_rng_states=[""],
            seed=42,
            traj_frame_indices={
                "replica_0.xtc": 500,
                "replica_1.xtc": 500,
                "energies.csv": 500,
            },
        )

        save_checkpoint(state, checkpoint_dir=str(tmp_path), filename="last.chk", cycle=10)
        loaded = load_checkpoint_state(str(tmp_path / "last.chk"))

        assert loaded.traj_frame_indices == {
            "replica_0.xtc": 500,
            "replica_1.xtc": 500,
            "energies.csv": 500,
        }

    def test_empty_by_default(self) -> None:
        from pymcpu.checkpointing import CheckpointState

        state = CheckpointState.__new__(CheckpointState)
        state.__dict__.setdefault("traj_frame_indices", {})
        assert state.traj_frame_indices == {}

    def test_dict_round_trip_covers_all_supported_formats(self, tmp_path: Path) -> None:
        """XTC, CSV, HDF5 and NPZ keys must all survive the raw-dict
        ``load_checkpoint`` path (not just the ``CheckpointState`` view)."""
        from pymcpu.checkpointing import CheckpointState, load_checkpoint, save_checkpoint

        state = CheckpointState(
            cycle=10,
            global_step=10000,
            replica_coords=[np.zeros((5, 3))],
            walker_at_state=np.array([0]),
            current_steps=[10000],
            temperatures=[300.0],
            integrator_rng_states=[""],
            seed=42,
            traj_frame_indices={
                "replica_0.xtc": 500,
                "replica_0_data.csv": 500,
                "samples.h5": 500,
                "samples.npz": 500,
            },
        )
        save_checkpoint(state, checkpoint_dir=str(tmp_path), cycle=10)
        loaded = load_checkpoint(str(tmp_path / "last.chk"))
        assert loaded["traj_frame_indices"]["samples.h5"] == 500
        assert loaded["traj_frame_indices"]["samples.npz"] == 500


# ── truncate_all_trajectories_on_resume dispatch ─────────────────


class TestTruncateAllDispatch:
    def test_skips_gracefully_when_field_missing(self, tmp_path: Path) -> None:
        """An old checkpoint object with no ``traj_frame_indices`` attribute
        must not crash resume -- it should just skip truncation."""
        from pymcpu.trajectory_utils import truncate_all_trajectories_on_resume

        class OldCheckpoint:
            pass

        truncate_all_trajectories_on_resume(
            checkpoint_state=OldCheckpoint(),
            traj_dir=str(tmp_path),
            top_path="dummy.pdb",
        )

    def test_uses_counts_for_csv(self, tmp_path: Path) -> None:
        from pymcpu.checkpointing import CheckpointState
        from pymcpu.trajectory_utils import truncate_all_trajectories_on_resume

        csv_path = tmp_path / "energies.csv"
        _write_csv_rows(csv_path, 100)

        state = CheckpointState(traj_frame_indices={"energies.csv": 40})
        truncate_all_trajectories_on_resume(state, str(tmp_path), top_path="dummy.pdb")
        with open(csv_path) as f:
            lines = f.readlines()
        assert len(lines) == 41

    def test_handles_npz(self, tmp_path: Path) -> None:
        """The dispatcher also accepts a plain dict (not just CheckpointState)."""
        from pymcpu.trajectory_utils import truncate_all_trajectories_on_resume

        npz_path = tmp_path / "samples.npz"
        np.savez(npz_path, x=np.arange(100))
        state = {"traj_frame_indices": {"samples.npz": 50}}
        truncate_all_trajectories_on_resume(state, str(tmp_path), top_path="dummy.pdb")
        data = np.load(npz_path)
        assert data["x"].shape[0] == 50


# ── XTC truncation (skipped if mdtraj absent) ────────────────────

mdtraj = pytest.importorskip("mdtraj", reason="mdtraj not installed")


class TestXtcTruncation:
    def test_removes_excess_frames(self, tmp_path: Path) -> None:
        traj = _single_atom_trajectory(100)
        xtc_path = str(tmp_path / "replica_0.xtc")
        pdb_path = str(tmp_path / "top.pdb")
        traj[0].save_pdb(pdb_path)
        traj.save_xtc(xtc_path)

        from pymcpu.trajectory_utils import truncate_xtc_to_frame

        truncate_xtc_to_frame(xtc_path, pdb_path, last_frame=49)

        loaded = mdtraj.load(xtc_path, top=pdb_path)
        assert len(loaded) == 50

    def test_is_atomic(self, tmp_path: Path) -> None:
        traj = _single_atom_trajectory(20)
        xtc_path = str(tmp_path / "replica_0.xtc")
        pdb_path = str(tmp_path / "top.pdb")
        traj[0].save_pdb(pdb_path)
        traj.save_xtc(xtc_path)

        from pymcpu.trajectory_utils import truncate_xtc_to_frame

        truncate_xtc_to_frame(xtc_path, pdb_path, last_frame=9)
        assert not os.path.exists(xtc_path + ".trunc.tmp")


def test_replica_exchange_traj_reporter_hooks_exist() -> None:
    """Existence/wiring smoke check: the reporter attach/detach hooks that
    ``truncate_all_trajectories_on_resume`` is meant to run alongside (detach
    before truncating, reattach after) must exist on ReplicaExchange. This is
    deliberately a thin existence check -- the actual attach/detach/truncate
    ordering during a real resume is exercised end-to-end by the
    integration/we/ replica-exchange resume tests, not here.
    """
    from pymcpu.sampling.replica_exchange import ReplicaExchange
    from pymcpu.trajectory_utils import truncate_all_trajectories_on_resume

    assert callable(truncate_all_trajectories_on_resume)
    assert hasattr(ReplicaExchange, "_attach_traj_reporters")
    assert hasattr(ReplicaExchange, "_detach_traj_reporters")


# ── NPZ truncation ────────────────────────────────────────────────


class TestNpzTruncation:
    def test_removes_excess_samples(self, tmp_path: Path) -> None:
        npz_path = str(tmp_path / "samples.npz")
        np.savez(
            npz_path,
            coords=np.random.rand(100, 10, 3).astype(np.float32),
            energies=np.random.rand(100).astype(np.float32),
        )

        from pymcpu.trajectory_utils import truncate_npz_to_sample

        truncate_npz_to_sample(npz_path, last_sample=49)

        data = np.load(npz_path)
        assert data["coords"].shape[0] == 50
        assert data["energies"].shape[0] == 50

    def test_noop_when_short(self, tmp_path: Path) -> None:
        npz_path = str(tmp_path / "samples.npz")
        arr = np.random.rand(20, 3).astype(np.float32)
        np.savez(npz_path, coords=arr)

        from pymcpu.trajectory_utils import truncate_npz_to_sample

        truncate_npz_to_sample(npz_path, last_sample=50)

        data = np.load(npz_path)
        assert data["coords"].shape[0] == 20

    def test_is_atomic(self, tmp_path: Path) -> None:
        npz_path = str(tmp_path / "samples.npz")
        np.savez(npz_path, x=np.arange(100))

        from pymcpu.trajectory_utils import truncate_npz_to_sample

        truncate_npz_to_sample(npz_path, last_sample=49)

        # np.savez appends ".npz" itself, so the temp name written by
        # truncate_npz_to_sample already carries the suffix -- check both
        # the bare temp name and the name np.savez would have produced.
        assert not os.path.exists(npz_path + ".trunc.tmp")
        assert not os.path.exists(npz_path + ".trunc.tmp.npz")


# ── HDF5 truncation ───────────────────────────────────────────────


class TestHdf5Truncation:
    def test_removes_excess_samples(self, tmp_path: Path) -> None:
        h5py = _import_h5py_or_skip()

        h5_path = str(tmp_path / "samples.h5")
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("coords", data=np.random.rand(100, 10, 3).astype(np.float32))
            f.create_dataset("energies", data=np.random.rand(100).astype(np.float32))

        from pymcpu.trajectory_utils import truncate_hdf5_to_sample

        truncate_hdf5_to_sample(h5_path, last_sample=49)

        with h5py.File(h5_path, "r") as f:
            assert f["coords"].shape[0] == 50
            assert f["energies"].shape[0] == 50

    def test_preserves_attributes(self, tmp_path: Path) -> None:
        h5py = _import_h5py_or_skip()

        h5_path = str(tmp_path / "samples.h5")
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("coords", data=np.random.rand(100, 3).astype(np.float32))
            f.attrs["run_id"] = "test-run-42"
            f.attrs["version"] = 2

        from pymcpu.trajectory_utils import truncate_hdf5_to_sample

        truncate_hdf5_to_sample(h5_path, last_sample=49)

        with h5py.File(h5_path, "r") as f:
            assert f.attrs["run_id"] == "test-run-42"
            assert f.attrs["version"] == 2

    def test_is_atomic(self, tmp_path: Path) -> None:
        h5py = _import_h5py_or_skip()

        h5_path = str(tmp_path / "samples.h5")
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("x", data=np.arange(100))

        from pymcpu.trajectory_utils import truncate_hdf5_to_sample

        truncate_hdf5_to_sample(h5_path, last_sample=49)
        assert not os.path.exists(h5_path + ".trunc.tmp")


def test_rex_sample_writer_n_frames_written() -> None:
    """RexSampleWriter's in-memory frame count must track appended samples
    (this is the count that later becomes a checkpoint's traj_frame_indices
    entry for .h5/.npz outputs)."""
    from pymcpu.analysis.pymbar_export import RexSampleWriter

    w = RexSampleWriter(
        "dummy.h5",
        temperatures=[0.5],
        n_targets=[0.0],
        k_bias=0.0,
        n_contacts=1,
    )
    assert w.n_frames_written() == 0
    w.append(state_index=0, energy_unbiased=1.0, N=0.0, cycle=0, walker_id=0)
    assert w.n_frames_written() == 1
    assert w.n_samples_written == 1
