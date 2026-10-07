"""Atomic checkpoint save/load for pyMCPU Monte Carlo / replica exchange.

This is an MC simulation package (not PyTorch training). Checkpoints are
serialized with pickle so we can store nested NumPy RNG state, coordinate
arrays, and metadata in one file. Writes use a temp file + ``os.replace``
so a crash mid-write cannot leave a truncated ``last.chk``.
"""

from __future__ import annotations

import logging
import os
import pickle
import re
import tempfile
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# 2: the MCPU layout stores each glycine CA once. Version 1 files from a
# protein with glycine have one extra atom per glycine and are rejected on
# resume by checkpoint_layout_error.
CHECKPOINT_FORMAT_VERSION = 2
DEFAULT_FILENAME = "last.chk"
_CYCLE_NAME_RE = re.compile(r"^checkpoint_cycle_(\d+)\.chk$")


@dataclass
class CheckpointConfig:
    """Checkpointing and resume settings."""

    checkpoint_dir: str | None = "checkpoints"
    checkpoint_interval: int = 50
    resume: str | bool | None = False
    keep_last_n: int | None = 3
    enabled: bool = True

    def __post_init__(self) -> None:
        if int(self.checkpoint_interval) > 500:
            warnings.warn(
                f"checkpoint_interval={self.checkpoint_interval} is very high. "
                "You may lose significant progress on a crash. "
                "Recommended: 50-100 for most runs.",
                UserWarning,
                stacklevel=2,
            )

    def resolved_resume_path(self) -> str | None:
        """Normalize ``resume`` to a path string, or ``None`` if not resuming."""
        if self.resume is False or self.resume is None or self.resume == "":
            return None
        if self.resume is True:
            if not self.checkpoint_dir:
                raise ValueError("resume=True requires checkpoint_dir")
            return str(self.checkpoint_dir)
        return str(self.resume)


@dataclass
class CheckpointState:
    """Serializable RE checkpoint payload (pickled as a plain dict)."""

    cycle: int = 0
    global_step: int = 0
    seed: int = 0
    format_version: int = CHECKPOINT_FORMAT_VERSION
    kind: str = "replica_exchange"
    pdb_path: str = ""
    reference_pdb: str = ""
    #: Registered name of the force field the run was built with. Empty in
    #: checkpoints written before it was recorded; those runs could only
    #: build mcpu08, so that is what an empty value means on resume.
    forcefield: str = ""
    temperatures: Any = None
    n_targets: Any = None
    k_bias: float = 0.0
    contact_cutoff: float = 6.0
    min_seq_sep: int = 4
    contact_atom_mode: str = "ca"
    native_contact_pairs: list[list[int]] | None = None
    fixed_residues: list[int] = field(default_factory=list)
    linker_residues: list[int] = field(default_factory=list)
    linker_energy_mode: str = "ignore_all"
    walker_at_state: Any = None
    replica_coords: list[Any] = field(default_factory=list)
    current_steps: list[int] = field(default_factory=list)
    exchange_rng: Any = None
    integrator_rng_states: list[Any] = field(default_factory=list)
    #: Per replica, the integrator's cumulative move counters
    #: (Integrator.get_move_counters). Empty in checkpoints written before
    #: counters were saved; those replicas count from 0 again on resume.
    integrator_move_counters: list[Any] = field(default_factory=list)
    n_replicas: int = 0
    # basename -> frame/row count at checkpoint time
    traj_frame_indices: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CheckpointState":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        kwargs = {k: v for k, v in data.items() if k in known}
        kwargs.setdefault("traj_frame_indices", {})
        return cls(**kwargs)


@dataclass
class FoldingCheckpointState(CheckpointState):
    """
    Extends CheckpointState with folding-specific fields.

    All folding fields are Optional so that:
      - Old base CheckpointState pickles load without error.
      - Fields not used by a particular folding setup are simply None.
      - FoldingCheckpointState is a drop-in wherever CheckpointState
        is expected (Liskov substitution).

    Note: the current FoldingRunner is single-temperature MC (no Q/RMSD
    tracking, funnel bias, basins, or convergence criteria yet). Those
    optional fields are reserved for future folding analytics.
    """

    # Identifies this as a folding checkpoint when loaded generically
    checkpoint_type: str = "folding"

    # ── Folding progress (reserved; not computed by FoldingRunner yet) ──
    native_contacts_fraction: Any | None = None
    rmsd_to_native: Any | None = None

    # ── Folding bias / funnel (reserved) ────────────────────────────────
    folding_bias_params: dict[str, Any] | None = None

    # ── Basin / cluster tracking (reserved) ─────────────────────────────
    basin_assignments: Any | None = None

    # ── Convergence history / event log (initialized empty on runner) ───
    convergence_history: list[Any] = field(default_factory=list)
    folding_events: list[Any] = field(default_factory=list)


def checkpoint_cycle_filename(cycle: int) -> str:
    return f"checkpoint_cycle_{int(cycle):06d}.chk"


def _coerce_checkpoint_mapping(state: dict[str, Any] | CheckpointState) -> dict[str, Any]:
    if isinstance(state, CheckpointState):
        return state.to_dict()
    if isinstance(state, dict):
        return state
    raise TypeError(
        f"checkpoint state must be a dict or CheckpointState, got {type(state)!r}"
    )


def save_checkpoint(
    state: dict[str, Any] | CheckpointState,
    checkpoint_dir: str | Path,
    filename: str = DEFAULT_FILENAME,
    *,
    is_best: bool = False,
    keep_last_n: int | None = None,
    cycle: int | None = None,
) -> Path:
    """Safely save a checkpoint using an atomic write.

    Parameters
    ----------
    state
        Serializable mapping or :class:`CheckpointState`.
    checkpoint_dir
        Directory created if missing.
    filename
        Target basename inside ``checkpoint_dir``.
    is_best
        If True, also copy to ``best.chk``.
    keep_last_n
        If set, retain only the newest N ``checkpoint_cycle_*.chk`` files
        (``last.chk`` / ``best.chk`` are always kept).
    cycle
        Optional cycle override written into the payload.
    """
    state = _coerce_checkpoint_mapping(state)
    if cycle is not None:
        state = dict(state)
        state["cycle"] = int(cycle)
        state.setdefault("global_step", int(cycle))

    checkpoint_dir = Path(checkpoint_dir)
    try:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.error("Cannot create checkpoint directory %s: %s", checkpoint_dir, exc)
        raise

    filepath = checkpoint_dir / filename
    # Write beside the final path so replace stays on the same filesystem.
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{filepath.name}.",
        suffix=".tmp",
        dir=str(checkpoint_dir),
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as fh:
            pickle.dump(state, fh, protocol=pickle.HIGHEST_PROTOCOL)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, filepath)
        logger.info(
            "Checkpoint saved: %s (keys=%s)",
            filepath,
            sorted(state.keys()),
        )
    except Exception as exc:
        logger.error("Failed to save checkpoint %s: %s", filepath, exc)
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise

    if is_best:
        best_path = checkpoint_dir / "best.chk"
        _atomic_copy(filepath, best_path)
        logger.info("Best checkpoint updated: %s", best_path)

    if keep_last_n is not None:
        prune_cycle_checkpoints(checkpoint_dir, keep_last_n=int(keep_last_n))

    return filepath


def load_checkpoint(checkpoint_path: str | Path) -> dict[str, Any]:
    """Load a checkpoint with clear errors if the path is missing/corrupt."""
    path = Path(checkpoint_path)
    if path.is_dir():
        candidate = path / DEFAULT_FILENAME
        if not candidate.is_file():
            raise FileNotFoundError(
                f"Checkpoint directory {path} has no {DEFAULT_FILENAME}"
            )
        path = candidate

    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    logger.info("Loading checkpoint: %s", path)
    try:
        with path.open("rb") as fh:
            checkpoint = pickle.load(fh)
    except Exception as exc:
        raise RuntimeError(f"Failed to load checkpoint {path}: {exc}") from exc

    if isinstance(checkpoint, CheckpointState):
        checkpoint = checkpoint.to_dict()

    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"Checkpoint {path} did not contain a dict (got {type(checkpoint)!r})"
        )

    # Back-compat for pickles written before traj_frame_indices existed.
    checkpoint.setdefault("traj_frame_indices", {})

    version = checkpoint.get("format_version")
    if version is not None and int(version) > CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"Checkpoint format_version={version} is newer than supported "
            f"{CHECKPOINT_FORMAT_VERSION}"
        )

    logger.info(
        "Resumed from cycle %s | global_step %s | keys=%s",
        checkpoint.get("cycle", "?"),
        checkpoint.get("global_step", "?"),
        sorted(checkpoint.keys()),
    )
    return checkpoint


def load_checkpoint_state(checkpoint_path: str | Path) -> CheckpointState:
    """Load a checkpoint and return a :class:`CheckpointState` view."""
    return CheckpointState.from_dict(load_checkpoint(checkpoint_path))


def find_latest_checkpoint(checkpoint_dir: str | Path) -> Path | None:
    """Return ``last.chk`` if present, else the highest ``checkpoint_cycle_*.chk``."""
    d = Path(checkpoint_dir)
    if not d.is_dir():
        return None
    last = d / DEFAULT_FILENAME
    if last.is_file():
        return last
    cycles: list[tuple[int, Path]] = []
    for p in d.glob("checkpoint_cycle_*.chk"):
        m = _CYCLE_NAME_RE.match(p.name)
        if m:
            cycles.append((int(m.group(1)), p))
    if not cycles:
        return None
    cycles.sort(key=lambda t: t[0])
    return cycles[-1][1]


def prune_cycle_checkpoints(checkpoint_dir: str | Path, *, keep_last_n: int) -> None:
    """Delete older ``checkpoint_cycle_*.chk`` files; keep the newest N."""
    if keep_last_n < 0:
        raise ValueError(f"keep_last_n must be >= 0, got {keep_last_n}")
    d = Path(checkpoint_dir)
    cycles: list[tuple[int, Path]] = []
    for p in d.glob("checkpoint_cycle_*.chk"):
        m = _CYCLE_NAME_RE.match(p.name)
        if m:
            cycles.append((int(m.group(1)), p))
    cycles.sort(key=lambda t: t[0])
    # keep_last_n == 0 means drop all numbered cycle files
    to_delete = cycles if keep_last_n == 0 else cycles[:-keep_last_n]
    for _, path in to_delete:
        try:
            path.unlink()
            logger.info("Pruned old checkpoint: %s", path)
        except OSError as exc:
            logger.warning("Could not prune %s: %s", path, exc)


def serialize_numpy_rng(rng: np.random.Generator) -> dict[str, Any]:
    """Capture a NumPy Generator bit-generator state for exact resume."""
    return {
        "bit_generator": type(rng.bit_generator).__name__,
        "state": rng.bit_generator.state,
    }


def restore_numpy_rng(rng: np.random.Generator, payload: dict[str, Any]) -> None:
    """Restore NumPy Generator state previously saved by ``serialize_numpy_rng``."""
    expected = type(rng.bit_generator).__name__
    got = payload.get("bit_generator")
    if got is not None and got != expected:
        logger.warning(
            "RNG bit_generator mismatch: checkpoint=%s current=%s", got, expected
        )
    rng.bit_generator.state = payload["state"]


def _replica_integrator(replica: Any) -> Any:
    """Resolve C++ Integrator from a Replica, Simulation, or Integrator object."""
    if hasattr(replica, "get_rng_state") and hasattr(replica, "set_rng_state"):
        return replica
    integ = getattr(replica, "integrator", None)
    if integ is not None:
        return integ
    sim = getattr(replica, "simulation", None)
    if sim is not None:
        return sim.integrator
    raise TypeError(
        f"Cannot resolve integrator from {type(replica)!r}; "
        "expected Replica, Simulation, or Integrator"
    )


def get_integrator_rng_states(replicas: list[Any]) -> list[str]:
    """Serialize each replica's C++ mt19937 state (empty string if unavailable)."""
    states: list[str] = []
    for replica in replicas:
        try:
            states.append(str(_replica_integrator(replica).get_rng_state()))
        except AttributeError:
            states.append("")
    return states


def set_integrator_rng_states(replicas: list[Any], states: list[Any]) -> None:
    """Restore C++ mt19937 states previously returned by ``get_integrator_rng_states``."""
    warned = False
    for replica, state in zip(replicas, states):
        if not state:
            continue
        try:
            _replica_integrator(replica).set_rng_state(state)
        except AttributeError:
            if not warned:
                logger.warning(
                    "mcpu_core lacks set_rng_state — MC stream not restored. "
                    "Rebuild with C++20 toolchain to activate exact RNG restore."
                )
                warned = True


def checkpoint_forcefield_error(
    saved: str | None, current: str, source: Any = None
) -> str | None:
    """Why a checkpoint written with force field ``saved`` cannot resume a run
    built with ``current``, or None.

    Checked before the atom layout, whose message would otherwise blame an
    older pyMCPU. Aliases of one force field ("mcpu" and "mcpu08") match.
    """
    from pymcpu.forcefields import get_forcefield

    saved_name = saved or "mcpu08"
    try:
        same = get_forcefield(saved_name) is get_forcefield(current)
    except ValueError:
        same = False
    if same:
        return None
    label = f"checkpoint {source}" if source else "the checkpoint"
    return (
        f"{label} was written by a {saved_name!r} run, but this run builds "
        f"forcefield {current!r}; resume it with the force field it was "
        "started with, or start a new run without resume"
    )


def checkpoint_layout_error(
    replica_coords: list[Any], n_atoms: int, source: Any = None
) -> str | None:
    """Why ``replica_coords`` cannot be loaded into a system of ``n_atoms``, or None.

    Coordinates are stored per atom slot in the engine's layout. A checkpoint
    from a version with a different layout has the wrong number of columns,
    and loading it would shift every atom after the first difference. For
    example, format version 1 gave every glycine CA a second slot.
    """
    for i, coords in enumerate(replica_coords or []):
        if coords is None:
            continue
        arr = np.asarray(coords)
        if arr.ndim != 2:
            continue
        found = arr.shape[1] if arr.shape[0] == 3 else arr.shape[0]
        if found != n_atoms:
            label = f"checkpoint {source}" if source else "the checkpoint"
            return (
                f"{label} holds coordinates for {found} atoms "
                f"(replica {i}), but this system has {n_atoms}. It was written "
                "with a different atom layout, for example by an older pyMCPU, "
                "and cannot be resumed; start the run again from its input "
                "structure."
            )
    return None


def get_integrator_move_counters(replicas: list[Any]) -> list[dict[str, int]]:
    """Each replica's cumulative move counters, for a checkpoint.

    Without them a resumed run restarts every counter at 0 while its energy
    CSV keeps appending, so the cumulative move columns drop back mid-file.
    """
    return [dict(_replica_integrator(replica).get_move_counters()) for replica in replicas]


def set_integrator_move_counters(replicas: list[Any], counters: list[Any]) -> None:
    """Restore counters saved by :func:`get_integrator_move_counters`.

    Replicas with no saved counters (an older checkpoint) are left as they are.
    """
    for replica, saved in zip(replicas, counters):
        if saved:
            _replica_integrator(replica).set_move_counters(dict(saved))


def _atomic_copy(src: Path, dst: Path) -> None:
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{dst.name}.",
        suffix=".tmp",
        dir=str(dst.parent),
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as out_fh, src.open("rb") as in_fh:
            while True:
                chunk = in_fh.read(1024 * 1024)
                if not chunk:
                    break
                out_fh.write(chunk)
            out_fh.flush()
            os.fsync(out_fh.fileno())
        os.replace(tmp_path, dst)
    except Exception:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise
