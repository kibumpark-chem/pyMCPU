"""Low-level checkpoint save/load mechanics: ``pymcpu.checkpointing``.

Covers the atomic-write / prune / round-trip primitives used by every
checkpointing caller (serial and MPI replica exchange, folding runs) --
none of this is physics, and there is no legacy MCPU equivalent to compare
against (the legacy C/MPI code has no checkpoint format at all), so this
lives under ``tests/integration/checkpointing`` rather than
``tests/physics`` or ``tests/legacy_parity``.

Replica-exchange-level checkpoint restore (coordinates, walker state, cycle
continuity) is exercised separately in ``test_replica_exchange_resume.py``,
since that requires a real ``ReplicaExchange`` + PDB rather than plain
dicts/arrays.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pymcpu.checkpointing import (
    CHECKPOINT_FORMAT_VERSION,
    find_latest_checkpoint,
    load_checkpoint,
    prune_cycle_checkpoints,
    restore_numpy_rng,
    save_checkpoint,
    serialize_numpy_rng,
)


def test_checkpoint_round_trips_state_dict_and_rng(tmp_path: Path) -> None:
    """Nested arrays, scalars, and a NumPy RNG state must survive a save/load cycle."""
    model = {"w": np.arange(12, dtype=np.float64).reshape(3, 4)}
    opt = {"m": np.ones(4), "v": np.zeros(4)}
    state = {
        "format_version": 1,
        "global_step": 42,
        "cycle": 7,
        "model_state_dict": model,
        "optimizer_state_dict": opt,
        "best_metric": 0.91,
        "config": {"lr": 1e-3},
        "rng_state": serialize_numpy_rng(np.random.default_rng(123)),
    }

    path = save_checkpoint(state, tmp_path, filename="checkpoint.chk")
    assert path.is_file()

    # Mutate the "live" objects in place to prove load_checkpoint returns an
    # independent copy read back from disk, not the same in-memory arrays.
    model["w"] += 99.0
    opt["m"] *= 0.0

    loaded = load_checkpoint(path)
    assert loaded["global_step"] == 42
    assert loaded["best_metric"] == pytest.approx(0.91)
    np.testing.assert_array_equal(loaded["model_state_dict"]["w"], np.arange(12).reshape(3, 4))
    np.testing.assert_array_equal(loaded["optimizer_state_dict"]["m"], np.ones(4))

    rng = np.random.default_rng(999)
    restore_numpy_rng(rng, loaded["rng_state"])
    ref = np.random.default_rng(123)
    # Restoring bit-generator state must reproduce the exact draw sequence,
    # not just "a" random number -- this is the whole point of RNG checkpointing.
    assert rng.random() == pytest.approx(ref.random())


def test_checkpoint_atomic_write_leaves_no_corrupt_file_on_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crash mid-``pickle.dump`` must not leave a partially-written ``last.chk``."""
    import pymcpu.checkpointing as ckpt

    def boom(obj, file, protocol=None):
        file.write(b"CORRUPT_PARTIAL")
        raise OSError("simulated disk full")

    monkeypatch.setattr(ckpt.pickle, "dump", boom)

    with pytest.raises(OSError, match="simulated disk full"):
        save_checkpoint({"cycle": 1}, tmp_path, filename="last.chk")

    final = tmp_path / "last.chk"
    # save_checkpoint writes to a temp file first and os.replace()s only on
    # success -- the crash must happen before that swap.
    assert not final.exists(), "atomic write must not leave a corrupted final file"
    assert find_latest_checkpoint(tmp_path) is None


def test_a_newer_format_version_is_rejected(tmp_path: Path) -> None:
    path = save_checkpoint(
        {"cycle": 1, "format_version": CHECKPOINT_FORMAT_VERSION + 1}, tmp_path
    )
    with pytest.raises(ValueError, match="newer than supported"):
        load_checkpoint(path)


def test_a_version_1_checkpoint_still_loads(tmp_path: Path) -> None:
    """Version 1 files load; resuming one is refused only if its atom count
    differs from the system's (see test_atom_layout_guard.py)."""
    path = save_checkpoint({"cycle": 3, "format_version": 1}, tmp_path)
    assert load_checkpoint(path)["cycle"] == 3


def test_load_checkpoint_missing_path_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        load_checkpoint(tmp_path / "does_not_exist.chk")


def test_prune_cycle_checkpoints_keeps_only_newest_n(tmp_path: Path) -> None:
    for cycle in range(1, 6):
        save_checkpoint(
            {"cycle": cycle, "format_version": 1},
            tmp_path,
            filename=f"checkpoint_cycle_{cycle:06d}.chk",
            keep_last_n=None,
        )
    prune_cycle_checkpoints(tmp_path, keep_last_n=2)
    remaining = sorted(p.name for p in tmp_path.glob("checkpoint_cycle_*.chk"))
    # Tautological given prune's documented contract (keep newest N by cycle
    # number) and the 5 files this test itself created -- confirms the two
    # highest-numbered cycles (4, 5) survive, not e.g. the two oldest.
    assert remaining == ["checkpoint_cycle_000004.chk", "checkpoint_cycle_000005.chk"]


def test_checkpoint_resume_continues_from_saved_step(tmp_path: Path) -> None:
    """Simulate a tiny MC loop: save at step 3, diverge, reload, and confirm
    the loaded state reflects step 3 -- not the post-checkpoint divergence."""
    global_step = 0
    weights = np.array([1.0, 2.0, 3.0])
    for step in range(1, 6):
        global_step = step
        weights = weights + 0.1
        if step == 3:
            save_checkpoint(
                {
                    "format_version": 1,
                    "global_step": global_step,
                    "cycle": global_step,
                    "model_state_dict": {"w": weights.copy()},
                    "optimizer_state_dict": {"step": global_step},
                },
                tmp_path,
                filename="last.chk",
            )

    weights += 100.0  # diverge after checkpoint; loaded state must not see this
    global_step = 0

    loaded = load_checkpoint(tmp_path / "last.chk")
    global_step = int(loaded["global_step"])
    weights = loaded["model_state_dict"]["w"]
    assert global_step == 3
    # weights after 3 loop iterations of +0.1 starting from [1,2,3]; pure
    # arithmetic consequence of this test's own loop, not an external baseline.
    np.testing.assert_allclose(weights, np.array([1.0, 2.0, 3.0]) + 0.3)
