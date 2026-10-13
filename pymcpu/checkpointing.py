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
    #: The force field's hydrogen-bond sheet factor (``hbond_beta_favor``);
    #: None for a force field without one, and in checkpoints written before
    #: it was recorded, whose builds all used legacy MCPU's 3.0.
    hbond_beta_favor: float | None = None
    #: What the run's rotamer library was (RotamerLibrary.fingerprint: name,
    #: data checksum, builder constants). Empty in checkpoints written before
    #: it was recorded; those runs could only use dunbrack2002.
    rotamer_library: str = ""
    #: The parameter set the force field was built from
    #: (MCPUForceField.param_set). Empty for a force field without one, and
    #: in checkpoints written before it was recorded, which count as mcpu08.
    #: Keyword-only, like the next two, so the fields before and after them
    #: keep the positional order they had before these were added.
    param_set: str = field(default="", kw_only=True)
    #: sha256 of the parameter files the force field read
    #: (MCPUForceField.param_signature); empty when param_set is.
    param_signature: str = field(default="", kw_only=True)
    #: The folder those files were read from, as its real path (links
    #: resolved). Only reported, never required: the same files elsewhere
    #: resume the run.
    param_dir: str = field(default="", kw_only=True)
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
    #: Per replica, the engine frame offset (Context.frame_offset, float64,
    #: shape (3,)) its coordinates were saved from; set_positions(coords,
    #: frame_offset=...) restores the engine coordinates bit for bit. The
    #: offset moves with the chain (Context.recenter), so a fresh Context's
    #: own may differ. Empty in checkpoints written before it was saved;
    #: those restore in the fresh Context's frame, as before.
    replica_frame_offsets: list[Any] = field(default_factory=list)
    current_steps: list[int] = field(default_factory=list)
    exchange_rng: Any = None
    integrator_rng_states: list[Any] = field(default_factory=list)
    #: Per replica, the integrator's cumulative move counters
    #: (Integrator.get_move_counters). Empty in checkpoints written before
    #: counters were saved; those replicas count from 0 again on resume.
    integrator_move_counters: list[Any] = field(default_factory=list)
    #: Replica exchange attempts and acceptances so far, under the
    #: write_rex_stats names (n_temp_attempts, n_temp_accepts, n_q_attempts,
    #: n_q_accepts). Empty in checkpoints written before they were saved;
    #: those runs count from 0 again on resume.
    exchange_counts: dict[str, int] = field(default_factory=dict)
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

    All folding fields are optional, so a base CheckpointState pickle loads
    as one, and a FoldingCheckpointState works wherever a CheckpointState is
    expected. Fields this class no longer has are dropped on load
    (:meth:`CheckpointState.from_dict`): checkpoints written before the
    folding bias and basin skeletons were removed carry
    ``folding_bias_params``, ``basin_assignments`` and ``folding_events``.
    """

    # Identifies this as a folding checkpoint when loaded generically
    checkpoint_type: str = "folding"

    # ── Folding progress at the save: Q and CA RMSD to the reference ────
    native_contacts_fraction: Any | None = None
    rmsd_to_native: Any | None = None

    # ── Q after each cycle so far (FoldingRunner.convergence_history) ───
    convergence_history: list[Any] = field(default_factory=list)


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


def saved_frame_offset(state: dict[str, Any] | CheckpointState, index: int) -> np.ndarray | None:
    """Replica ``index``'s saved frame offset, for ``set_positions(...,
    frame_offset=)``; None for a checkpoint written before offsets were
    saved, which then restores in the Context's own frame."""
    offsets = (
        state.get("replica_frame_offsets")
        if isinstance(state, dict)
        else state.replica_frame_offsets
    )
    if not offsets or index >= len(offsets) or offsets[index] is None:
        return None
    return np.asarray(offsets[index], dtype=np.float64)


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


#: The parameter set a checkpoint written before param_set was recorded
#: counts as: those builds shipped no other.
LEGACY_PARAM_SET = "mcpu08"


def param_set_record(forcefield: Any) -> dict[str, str]:
    """What a checkpoint records about the parameters ``forcefield`` reads:
    ``param_set``, ``param_signature`` (``MCPUForceField.param_signature``)
    and ``param_dir`` (the folder, as its real path, so one folder reached
    through a link is recorded as itself). All three are empty for a force
    field without a parameter set.

    Taking the signature reads the parameter files the first time a process
    asks for them (about 2 s for mcpu08), so the runners take the record once,
    when they are built; the MPI replica exchange takes it on rank 0 and
    sends it to the other ranks.
    """
    param_set = getattr(forcefield, "param_set", None)
    if not isinstance(param_set, str) or not param_set:
        return {"param_set": "", "param_signature": "", "param_dir": ""}
    signature = getattr(forcefield, "param_signature", None)
    folder = getattr(forcefield, "param_dir", None)
    return {
        "param_set": param_set,
        "param_signature": signature if isinstance(signature, str) else "",
        "param_dir": os.path.realpath(folder) if isinstance(folder, (str, os.PathLike)) else "",
    }


def describe_param_set(record: dict[str, str]) -> str | None:
    """One line naming the parameters of a :func:`param_set_record`, for a
    run's log, or None for a force field without a parameter set."""
    if not record.get("param_set"):
        return None
    return (
        f"Parameters: {record['param_set']} from {record['param_dir']} "
        f"(signature {record['param_signature'][:16]})"
    )


def checkpoint_param_set_error(
    saved: dict[str, Any] | CheckpointState,
    current: dict[str, str],
    source: Any = None,
) -> str | None:
    """Why a checkpoint cannot resume a run that reads the parameters
    ``current`` (a :func:`param_set_record`), or None.

    The run must use the parameter set the checkpoint names, and read files
    with the same signature, so the same tables and constants; where the
    folder is does not matter. A force field without a parameter set is not
    checked. A checkpoint written before parameter sets were recorded counts
    as mcpu08, the only set of those builds, and its files cannot be checked:
    resuming one under mcpu08 logs one warning that says so. Checked after
    the force field and before the atom layout, which the parameter set's
    residue templates decide.
    """

    def saved_value(key: str) -> str:
        value = saved.get(key) if isinstance(saved, dict) else getattr(saved, key, None)
        return value if isinstance(value, str) else ""

    current_set = current.get("param_set") or ""
    if not current_set:
        return None
    label = f"checkpoint {source}" if source else "the checkpoint"
    current_dir = current.get("param_dir") or "?"
    saved_set = saved_value("param_set")
    if not saved_set:
        if current_set == LEGACY_PARAM_SET:
            logger.warning(
                "%s records no parameter set, as it was written before they were recorded: "
                "it counts as %s, and its parameter files are not checked against this run's (%s)",
                label, LEGACY_PARAM_SET, current_dir,
            )
            return None
        return (
            f"{label} records no parameter set, so it counts as {LEGACY_PARAM_SET!r} (the "
            f"only set when it was written), but this run uses param_set {current_set!r}; "
            f"resume it with param_set: {LEGACY_PARAM_SET}, or start a new run without resume"
        )
    if saved_set != current_set:
        return (
            f"{label} was written with param_set {saved_set!r}, but this run uses "
            f"{current_set!r}; resume it with param_set: {saved_set}, or start a new run "
            "without resume"
        )
    saved_signature = saved_value("param_signature")
    current_signature = current.get("param_signature") or ""
    if not saved_signature or not current_signature or saved_signature == current_signature:
        return None
    saved_dir = saved_value("param_dir")
    # One folder can be reached by several paths (links, and older
    # checkpoints recorded the path as given): compare where they lead.
    if saved_dir and os.path.realpath(saved_dir) != os.path.realpath(current_dir):
        advice = (
            f"resume it with param_dir: {saved_dir} (or another folder with the same "
            "files), or start a new run without resume"
        )
    else:
        advice = (
            f"the files in {current_dir} have changed since; resume it with param_dir set "
            "to a folder that holds the files it was written with, or start a new run "
            "without resume"
        )
    return (
        f"{label} was written with param_set {saved_set!r} from parameter files with "
        f"signature {saved_signature} ({saved_dir or 'folder not recorded'}), but this "
        f"run's files have signature {current_signature} ({current_dir}): a table or "
        f"constants file differs; {advice}"
    )


def checkpoint_beta_favor_error(
    saved: float | None, current: float | None, source: Any = None
) -> str | None:
    """Why a checkpoint written with hydrogen-bond sheet factor ``saved``
    cannot resume a run whose force field uses ``current``, or None.

    ``current`` is None for a force field without the factor, which is not
    checked. A checkpoint without one (``saved`` None) was written by a build
    that used legacy MCPU's 3.0. Checked after the atom layout, since a
    checkpoint that fails both cannot be resumed with any factor.
    """
    if current is None:
        return None
    # Compared and printed as the engine's float32, so the value printed in
    # the advice resumes the run.
    saved_factor = np.float32(3.0 if saved is None else saved)
    current_factor = np.float32(current)
    if saved_factor == current_factor:
        return None
    label = f"checkpoint {source}" if source else "the checkpoint"
    return (
        f"{label} was written with hbond_beta_favor {saved_factor!s}, but this "
        f"run uses {current_factor!s}; resume it with forcefield_options: "
        f"{{hbond_beta_favor: {saved_factor!s}}}, or start a new run without resume"
    )


def checkpoint_rotamer_library_error(
    saved: str | None, current: str, sidechain_move_mode: str, source: Any = None
) -> str | None:
    """Why a checkpoint whose run used rotamer library ``saved`` cannot resume
    a run whose System holds ``current``, or None. Both are
    ``RotamerLibrary.fingerprint`` values: the library's name, the checksum of
    its data file and, for dunbrack2002_sym, the builder version.

    A resumed run must write what the uninterrupted run writes, so the
    library must be the same one. The continuous sidechain move uses no
    library and is not checked. A checkpoint without a fingerprint was
    written before it was recorded, by a build whose only library was
    dunbrack2002, so only the name is compared then. Checked after the atom
    layout, like the sheet factor.
    """
    if sidechain_move_mode == "continuous":
        return None
    current = current or ""
    saved_name = (saved or "dunbrack2002").split(" ")[0]
    if saved == current or (not saved and current.split(" ")[0] == saved_name):
        return None
    label = f"checkpoint {source}" if source else "the checkpoint"
    return (
        f"{label} was written with rotamer library {saved or saved_name!r}, but "
        f"this run uses {current!r}; resume it with rotamer_library: {saved_name} "
        "(and the same parameter files), or start a new run without resume"
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
