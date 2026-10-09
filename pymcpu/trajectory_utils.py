"""
trajectory_utils.py
Utilities for truncating and rejoining trajectory files on checkpoint resume.
Supports XTC, CSV energy logs, per-cycle CSV logs, HDF5 (via h5py), and NPZ.
DCD truncation is not yet implemented (see the warning in
truncate_all_trajectories_on_resume()).
"""

from __future__ import annotations

import logging
import os
import struct
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _fsync_path(path: str) -> None:
    """fsync a file already written and closed by a library that doesn't
    hand us its file descriptor (h5py.File, np.savez*),
    so the os.replace() that follows is preceded by a durable write —
    matching the fsync-before-replace guarantee of the atomic-write helpers
    elsewhere in this package (checkpointing.py, we/state.py). Also makes
    a file just cut in place (os.truncate) durable."""
    fd = os.open(path, os.O_RDWR)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


_XTC_MAGIC = 1995


def _xtc_frame_offsets(xtc_path: str) -> list[int]:
    """
    Byte offsets of the complete frames in an XTC file, followed by the end
    of the last one; a partly written frame at the end is not counted.

    Reads only the frame headers (big-endian XDR): magic, natoms, step,
    time, a 3x3 box and natoms again (56 bytes). Up to 9 atoms, 3 * natoms
    floats follow; otherwise precision, minint[3], maxint[3], smallidx and
    a byte count, then that many bytes padded to a multiple of 4. A header
    with a negative atom or byte count, which only a corrupted file has,
    ends the scan there.
    """
    offsets = [0]
    size = os.path.getsize(xtc_path)
    with open(xtc_path, "rb") as f:
        while True:
            start = offsets[-1]
            f.seek(start)
            head = f.read(92)
            if len(head) < 56:
                break
            magic, natoms = struct.unpack(">ii", head[:8])
            if magic != _XTC_MAGIC or natoms < 0:
                break
            if natoms <= 9:
                end = start + 56 + 12 * natoms
            elif len(head) == 92:
                (nbytes,) = struct.unpack(">i", head[88:92])
                if nbytes < 0:
                    break
                end = start + 92 + (nbytes + 3) // 4 * 4
            else:
                break
            if end > size:
                break
            offsets.append(end)
    return offsets


def truncate_xtc_to_frame(xtc_path: str, top_path: str, last_frame: int) -> None:
    """
    Truncate xtc_path so it contains exactly (last_frame + 1) frames
    (frames are 0-indexed, so last_frame=49 keeps frames 0..49).

    The file is cut at the end of the last frame kept, so the frames kept
    stay byte for byte as written, step numbers included. A partly written
    frame at the end of the file, as a killed job can leave, is cut too.
    Cutting a file in place cannot corrupt the frames it keeps.

    Args:
        xtc_path:   Full path to the XTC file to truncate.
        top_path:   Unused: the frame boundaries are read from the XTC
                    itself. Kept so callers need not change.
        last_frame: Index of the last frame to KEEP (0-indexed).
                    Frames after this index are discarded.
    """
    if not os.path.exists(xtc_path):
        logger.warning(
            f"[Trajectory] XTC not found, skipping truncation: {xtc_path}"
        )
        return

    offsets = _xtc_frame_offsets(xtc_path)
    n_frames = len(offsets) - 1
    size = os.path.getsize(xtc_path)
    if n_frames == 0 and size > 0:
        logger.error(f"[Trajectory] No complete XTC frame in {xtc_path}; left as is")
        return

    if n_frames < last_frame + 1:
        logger.warning(
            f"[Trajectory] {xtc_path} has {n_frames} complete frames, "
            f"fewer than the {last_frame + 1} to keep"
        )
    target = min(last_frame + 1, n_frames)  # number of frames to keep
    if offsets[target] >= size:
        logger.info(
            f"[Trajectory] No truncation needed: {xtc_path} "
            f"has {n_frames} frames, target is {last_frame + 1}"
        )
        return

    os.truncate(xtc_path, offsets[target])
    _fsync_path(xtc_path)
    logger.info(
        f"[Trajectory] Truncated {xtc_path}: "
        f"{n_frames} → {target} frames"
    )


def truncate_csv_to_row(csv_path: str, last_row: int) -> None:
    """
    Truncate a CSV energy/log file so it contains the header line
    plus exactly last_row data rows.

    last_row=50 means lines[0] (header) + lines[1..50] are kept.
    Uses atomic overwrite.

    Args:
        csv_path: Full path to the CSV file.
        last_row: Number of DATA rows to keep (header not counted).
    """
    if not os.path.exists(csv_path):
        logger.warning(
            f"[Trajectory] CSV not found, skipping truncation: {csv_path}"
        )
        return

    try:
        with open(csv_path, "r") as f:
            lines = f.readlines()
    except Exception as e:
        logger.error(f"[Trajectory] Could not read {csv_path}: {e}")
        return

    # lines[0] = header, lines[1:] = data rows
    total_data_rows = len(lines) - 1
    if last_row >= total_data_rows:
        logger.info(
            f"[Trajectory] No truncation needed: {csv_path} "
            f"has {total_data_rows} data rows, target is {last_row}"
        )
        return

    keep = lines[: last_row + 1]  # header + last_row data rows
    tmp_path = csv_path + ".trunc.tmp"

    try:
        with open(tmp_path, "w") as f:
            f.writelines(keep)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, csv_path)
        logger.info(
            f"[Trajectory] Truncated {csv_path}: "
            f"{total_data_rows} → {last_row} data rows"
        )
    except Exception as e:
        logger.error(f"[Trajectory] CSV truncation write failed: {e}")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def truncate_csv_to_cycle(csv_path: str, n_cycles: int) -> None:
    """
    Truncate a per-cycle log (a CSV with a ``cycle`` column, such as the
    replica exchange and state logs) to its header and the rows of cycles
    before n_cycles.

    A run stopped between checkpoints has logged cycles past its last
    checkpoint, and the resumed run logs them again. Rows are in cycle
    order; a partly written last row is cut too. The rows kept are left
    byte for byte as written (the csv module ends these lines with CRLF).

    Args:
        csv_path: Full path to the CSV log.
        n_cycles: Cycles to keep: the checkpoint's cycle count.
    """
    if not os.path.exists(csv_path):
        return
    with open(csv_path, "rb") as f:
        lines = f.readlines()
    if not lines:
        return
    header = lines[0].decode().strip().split(",")
    if "cycle" not in header:
        logger.warning(f"[Trajectory] No cycle column, skipping truncation: {csv_path}")
        return
    col = header.index("cycle")

    keep = len(lines[0])
    for line in lines[1:]:
        if not line.endswith(b"\n") or int(line.split(b",")[col]) >= n_cycles:
            break
        keep += len(line)
    if keep < os.path.getsize(csv_path):
        os.truncate(csv_path, keep)
        _fsync_path(csv_path)
        logger.info(f"[Trajectory] Truncated {csv_path} to cycles before {n_cycles}")


def truncate_hdf5_to_sample(h5_path: str, last_sample: int) -> None:
    """
    Truncate an HDF5 file so it contains exactly (last_sample + 1)
    samples (0-indexed). Uses atomic overwrite.

    Requires h5py. Logs a warning and returns cleanly if h5py is
    not installed — never raises on missing dependency.

    Args:
        h5_path:     Full path to the HDF5 file.
        last_sample: Index of the last sample to keep (0-indexed).
    """
    if not os.path.exists(h5_path):
        logger.warning(
            f"[Trajectory] HDF5 not found, skipping: {h5_path}"
        )
        return
    try:
        import h5py
    except ImportError:
        logger.warning(
            "[Trajectory] h5py not installed — "
            "cannot truncate HDF5. pip install h5py"
        )
        return

    target = last_sample + 1
    tmp_path = h5_path + ".trunc.tmp"

    try:
        with h5py.File(h5_path, "r") as src:
            with h5py.File(tmp_path, "w") as dst:
                for key in src.keys():
                    data = src[key]
                    # Only truncate datasets with a sample dimension
                    if hasattr(data, "shape") and len(data.shape) > 0:
                        n = data.shape[0]
                        if target < n:
                            dst.create_dataset(key, data=data[:target])
                        else:
                            src.copy(key, dst)
                    else:
                        src.copy(key, dst)
                # Copy all attributes from root
                for attr_key, attr_val in src.attrs.items():
                    dst.attrs[attr_key] = attr_val

        _fsync_path(tmp_path)
        os.replace(tmp_path, h5_path)
        logger.info(
            f"[Trajectory] Truncated HDF5: {h5_path} to {target} samples"
        )

    except Exception as e:
        logger.error(f"[Trajectory] HDF5 truncation failed: {e}")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def truncate_npz_to_sample(npz_path: str, last_sample: int) -> None:
    """
    Truncate an NPZ file so it contains exactly (last_sample + 1)
    samples along axis 0 for every array. Uses atomic overwrite.

    Args:
        npz_path:    Full path to the NPZ file.
        last_sample: Index of the last sample to keep (0-indexed).
    """
    if not os.path.exists(npz_path):
        logger.warning(
            f"[Trajectory] NPZ not found, skipping: {npz_path}"
        )
        return

    import numpy as np

    target = last_sample + 1
    # Use an explicit .npz suffix so np.savez writes exactly this path.
    tmp_path = npz_path + ".trunc.tmp.npz"

    try:
        data = np.load(npz_path, allow_pickle=False)
        truncated = {}
        for key in data.files:
            arr = data[key]
            if arr.ndim > 0 and arr.shape[0] > target:
                truncated[key] = arr[:target]
            else:
                truncated[key] = arr
        data.close()

        np.savez_compressed(tmp_path, **truncated)
        _fsync_path(tmp_path)
        os.replace(tmp_path, npz_path)
        logger.info(
            f"[Trajectory] Truncated NPZ: {npz_path} to {target} samples"
        )

    except Exception as e:
        logger.error(f"[Trajectory] NPZ truncation failed: {e}")
        for p in [tmp_path, npz_path + ".trunc.tmp", npz_path + ".trunc.tmp.npz"]:
            if os.path.exists(p):
                os.remove(p)
        raise


def truncate_all_trajectories_on_resume(
    checkpoint_state: Any,
    traj_dir: str,
    top_path: str,
) -> None:
    """
    Call this AFTER load_checkpoint() and BEFORE the cycle loop in run().

    Reads traj_frame_indices from the checkpoint and truncates every
    trajectory file back to the frame count that existed at save time.
    This removes any frames written after the checkpoint (e.g. during
    a run that crashed after saving but before the next checkpoint).

    Args:
        checkpoint_state: A CheckpointState (or mapping/object) with
                          traj_frame_indices populated. Values are treated as
                          **counts** (n_frames / n_csv_rows / n_samples).
                          XTC/HDF5/NPZ truncation converts count → last index
                          (count - 1).
        traj_dir:         Directory where trajectory files live.
        top_path:         Unused (XTC frame boundaries are read from the file).
    """
    indices = None
    if hasattr(checkpoint_state, "traj_frame_indices"):
        indices = checkpoint_state.traj_frame_indices
    elif isinstance(checkpoint_state, dict):
        indices = checkpoint_state.get("traj_frame_indices")

    if not indices:
        logger.warning(
            "[Trajectory] checkpoint_state has no traj_frame_indices. "
            "Trajectory truncation skipped. "
            "This checkpoint was saved before traj_frame_indices was added."
        )
        return

    for fname, count in indices.items():
        full_path = os.path.join(traj_dir, fname)
        n = int(count)

        if fname.endswith(".xtc"):
            # Stored value is frame COUNT; truncate keeps indices [0, n).
            if n <= 0:
                logger.info(
                    "[Trajectory] Skipping XTC truncation for empty count: %s",
                    fname,
                )
                continue
            truncate_xtc_to_frame(full_path, top_path, last_frame=n - 1)

        elif fname.endswith(".csv"):
            truncate_csv_to_row(full_path, last_row=n)

        elif fname.endswith(".dcd"):
            logger.warning(
                f"[Trajectory] DCD truncation not yet implemented: {fname}. "
                "Delete manually or implement truncate_dcd_to_frame()."
            )

        elif fname.endswith(".h5") or fname.endswith(".hdf5"):
            if n <= 0:
                continue
            truncate_hdf5_to_sample(full_path, last_sample=n - 1)

        elif fname.endswith(".npz"):
            if n <= 0:
                continue
            truncate_npz_to_sample(full_path, last_sample=n - 1)

        else:
            logger.warning(
                f"[Trajectory] Unknown format, skipping truncation: {fname}"
            )


def trajectory_topology_path(
    forcefield: Any,
    input_pdb: str | Path,
    default: str | Path,
    out_path: str | Path,
    *,
    write: bool = True,
) -> str:
    """Topology file to read this run's XTC trajectories against.

    The XTC reporters write the force field's ``output_topology``. For MCPU
    that is the input's heavy atoms, so ``default`` (an input PDB) is
    returned unchanged. A force field that simulates fewer atoms gets its
    topology and starting coordinates written to ``out_path``, which is
    returned; read its trajectories against that file.
    ``write=False`` returns the path without writing, for MPI ranks other
    than the one that writes it.
    """
    import mdtraj as md

    top = forcefield.output_topology
    heavy = md.load_topology(str(input_pdb)).select("not element H").size
    if top.n_atoms == heavy:
        return str(default)
    if not write:
        return str(out_path)
    xyz = np.zeros((1, top.n_atoms, 3), dtype=np.float32)
    xyz[0, np.asarray(forcefield.inverse_mapping)] = np.asarray(forcefield.coords[0])
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    md.Trajectory(xyz, top).save_pdb(str(out_path))
    return str(out_path)
