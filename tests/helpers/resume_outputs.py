"""Compare the outputs of a resumed run with those of an uninterrupted one.

A run that is stopped and resumed (once, twice, or after being killed between
checkpoints) must leave exactly what the same run left when it went straight
through: every output file byte for byte, and the same final checkpoint.
Shared by the resume tests of all three runners.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

#: Final-checkpoint fields that must match. Keys a runner does not write are
#: absent on both sides and compare equal.
CHECKPOINT_KEYS = (
    "cycle",
    "global_step",
    "replica_coords",
    "replica_frame_offsets",
    "current_steps",
    "integrator_rng_states",
    "integrator_move_counters",
    "walker_at_state",
    "exchange_rng",
    "exchange_counts",
    "traj_frame_indices",
)


def _difference(name: str, got: bytes, want: bytes) -> str:
    """Say how a resumed run's file differs from the uninterrupted run's."""
    if name.endswith(".json"):
        return f"{json.loads(got)} after resume, {json.loads(want)} uninterrupted"
    if name.endswith(".csv"):
        return f"{len(got.splitlines())} lines after resume, {len(want.splitlines())} uninterrupted"
    first = next((i for i, (x, y) in enumerate(zip(got, want)) if x != y), min(len(got), len(want)))
    return f"{len(got)} vs {len(want)} bytes, first difference at byte {first}"


def _plain(value: Any) -> str:
    """JSON text of a checkpoint value (arrays as lists, dict keys sorted)."""
    return json.dumps(value, default=lambda o: np.asarray(o).tolist(), sort_keys=True)


def assert_same_outputs(
    uninterrupted: Path,
    resumed: Path,
    uninterrupted_checkpoint: Path,
    resumed_checkpoint: Path,
) -> None:
    """Every file in the output directories, and the two checkpoints, match."""
    from pymcpu.checkpointing import load_checkpoint

    want = {p.name: p.read_bytes() for p in Path(uninterrupted).iterdir() if p.is_file()}
    got = {p.name: p.read_bytes() for p in Path(resumed).iterdir() if p.is_file()}
    assert sorted(got) == sorted(want)
    for name in sorted(want):
        assert got[name] == want[name], f"{name}: {_difference(name, got[name], want[name])}"

    a = load_checkpoint(uninterrupted_checkpoint)
    b = load_checkpoint(resumed_checkpoint)
    for key in CHECKPOINT_KEYS:
        assert _plain(b.get(key)) == _plain(a.get(key)), f"checkpoint {key} differs"
