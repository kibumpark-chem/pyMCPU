"""MPI-parallel replica exchange via mpi4py.

Supports multiple replicas per MPI rank: each rank steps its local replicas
sequentially on one CPU, then participates in exchange attempts once all ranks
finish the MC phase.
"""

from __future__ import annotations

import sys
import csv
import logging
import os
import time
from pathlib import Path
from typing import Any, TextIO

import numpy as np

from pymcpu import mcpu_core
from pymcpu.simulation import check_state_clash
from pymcpu.config import (
    DEFAULT_CONTACT_ATOM_MODE,
    DEFAULT_CONTACT_CUTOFF,
    DEFAULT_KIC_STEP_SIZE_RAD,
    DEFAULT_MIN_SEQ_SEP,
)
from pymcpu.checkpointing import (
    CHECKPOINT_FORMAT_VERSION,
    CheckpointConfig,
    CheckpointState,
    checkpoint_cycle_filename,
    checkpoint_forcefield_error,
    checkpoint_layout_error,
    load_checkpoint,
    save_checkpoint,
    saved_frame_offset,
)
from pymcpu.sampling.replica_exchange_core import (
    EXCHANGE_CSV_FIELDS,
    ExchangeRecord,
    Replica,
    ReplicaState,
    RunSummary,
    build_grid_metadata,
    build_replica_simulation,
    build_system_and_cv,
    catch_termination_signals,
    count_exchanges,
    evaluate_exchange_acceptance,
    exchange_record_to_row,
    get_coords,
    get_frame_offset,
    replica_index,
    resolve_n_targets,
    restore_exchange_counts,
    swap_context_coordinates,
    unbiased_energy,
    write_rex_stats,
)
from pymcpu.trajectory_utils import (
    trajectory_topology_path,
    truncate_all_trajectories_on_resume,
    truncate_csv_to_cycle,
    truncate_csv_to_row,
    truncate_xtc_to_frame,
)

logger = logging.getLogger(__name__)

try:
    from mpi4py import MPI
except Exception:  # pragma: no cover - optional / env-dependent dependency
    MPI = None


def _require_mpi() -> Any:
    if MPI is None:
        raise ImportError(
            "mpi4py is required for MPI replica exchange. "
            "Install with: pip install 'pymcpu[mpi]'"
        )
    return MPI


def partition_replicas(n_replicas: int, n_ranks: int) -> tuple[list[list[int]], np.ndarray]:
    """Assign replicas to ranks in contiguous blocks.

    Examples
    --------
    20 replicas on 10 ranks -> each rank owns 2 replicas:
      rank 0: [0, 1], rank 1: [2, 3], ...

    Returns
    -------
    assignments : list[list[int]]
        Replica indices owned by each rank.
    owner_rank : ndarray of shape (n_replicas,)
        ``owner_rank[replica_idx]`` is the MPI rank that owns that replica.
    """
    if n_ranks < 1:
        raise ValueError("n_ranks must be >= 1")
    if n_replicas < n_ranks:
        raise ValueError(
            f"Need at least as many replicas ({n_replicas}) as MPI ranks ({n_ranks})."
        )

    base, extra = divmod(n_replicas, n_ranks)
    assignments: list[list[int]] = []
    owner_rank = np.empty(n_replicas, dtype=np.int32)
    replica_idx = 0
    for rank in range(n_ranks):
        count = base + (1 if rank < extra else 0)
        local = list(range(replica_idx, replica_idx + count))
        assignments.append(local)
        for idx in local:
            owner_rank[idx] = rank
        replica_idx += count
    return assignments, owner_rank


def _slot_for_index(
    replica_index: int,
    temperatures: np.ndarray,
    n_targets: np.ndarray,
) -> tuple[int, int, float, float]:
    n_q = int(n_targets.size)
    temp_index = replica_index // n_q
    q_index = replica_index % n_q
    return temp_index, q_index, float(temperatures[temp_index]), float(n_targets[q_index])


class MPIReplicaExchange:
    """MPI replica exchange with one or more replicas per rank.

    Launch with fewer MPI ranks than replicas to run multiple trajectories
    sequentially on each CPU::

        # config.yaml with 20 replicas
        mpirun -n 10 python scripts/run_mcpu_replica_exchange.py --mpi -c config.yaml

    Each rank steps its local replicas one after another, then all ranks
    participate in exchange attempts. Exchanges between two replicas on the
    same rank are handled locally; cross-rank exchanges use ``Sendrecv``.

    Bias / exchange use hard native-contact count N
    (``U = 0.5 * k * (N - N0)^2``); fraction Q is logged only.

    Keyword arguments not listed below mean the same as in
    :class:`~pymcpu.sampling.ReplicaExchange`.

    Parameters
    ----------
    comm : mpi4py.MPI.Comm
        MPI communicator (typically ``MPI.COMM_WORLD``).
    pdb_path : str
        Input structure path.
    step_size_rad, kic_step_size_rad, move_weights, sidechain_move_mode, pivot_rama_probability, pivot_rama_schedule
        Move settings for every replica, as in
        :class:`~pymcpu.sampling.ReplicaExchange`.
    full_energy_every_steps : int
        Full energy recompute cadence for every replica, in MC steps, as in
        :class:`~pymcpu.sampling.ReplicaExchange`.
    """

    def __init__(
        self,
        comm: Any,
        pdb_path: str,
        *,
        reference_pdb: str | None = None,
        temperatures: np.ndarray | list[float] | None = None,
        temp_min: float = 0.1,
        temp_step: float = 0.05,
        n_temps: int = 4,
        native_contact_targets: np.ndarray | list[float] | None = None,
        n_targets: np.ndarray | list[float] | None = None,
        q_targets: np.ndarray | list[float] | None = None,
        n_q_windows: int = 1,
        q_step: float = 0.1,
        k_bias: float = 0.0,
        contact_cutoff: float = DEFAULT_CONTACT_CUTOFF,
        q_cutoff: float | None = None,
        min_seq_sep: int = DEFAULT_MIN_SEQ_SEP,
        contact_atom_mode: str = DEFAULT_CONTACT_ATOM_MODE,
        native_contact_pairs: list[list[int]] | None = None,
        log_interval: int = 100,
        output_prefix: str = "rex",
        output_dir: str | Path | None = None,
        seed: int = 0,
        step_size_rad: float = 0.1,
        move_weights: tuple[float, float, float] | None = None,
        full_energy_every_steps: int = 1_000_000,
        sidechain_move_mode: str = "rotamer_library",
        pivot_rama_probability: float = 0.0,
        pivot_rama_schedule: dict[str, float] | None = None,
        kic_step_size_rad: float = DEFAULT_KIC_STEP_SIZE_RAD,
        fixed_residues: list[int] | None = None,
        linker_residues: list[int] | None = None,
        linker_energy_mode: str = "ignore_all",
        forcefield: str = "mcpu08",
        forcefield_options: dict[str, Any] | None = None,
        param_set: str = "mcpu08",
        param_dir: str | Path | None = None,
        checkpoint_config: CheckpointConfig | None = None,
        checkpoint_dir: str | Path | None = None,
        checkpoint_interval: int | None = None,
        keep_last_n: int | None = None,
        resume: str | bool | None = None,
        exchange_log: str = "none",
        state_log_interval: int = 0,
        log_walker_in_data_csv: bool = True,
    ):
        _require_mpi()
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        self.pdb_path = pdb_path
        self.reference_pdb = reference_pdb or pdb_path
        self.contact_cutoff = float(contact_cutoff)
        self.min_seq_sep = int(min_seq_sep)
        from pymcpu.sampling.collective_variables import normalize_contact_atom_mode

        self.contact_atom_mode = normalize_contact_atom_mode(contact_atom_mode)
        self.native_contact_pairs = (
            list(native_contact_pairs) if native_contact_pairs else None
        )
        self.k_bias = float(k_bias)
        self.log_interval = int(log_interval)
        self.fixed_residues = list(fixed_residues) if fixed_residues else []
        self.linker_residues = list(linker_residues) if linker_residues else []
        from pymcpu.config import (
            normalize_linker_energy_mode,
            normalize_move_settings,
            validate_fixed_linker_disjoint,
        )

        self.linker_energy_mode = normalize_linker_energy_mode(linker_energy_mode)
        validate_fixed_linker_disjoint(self.fixed_residues, self.linker_residues)
        self.seed = int(seed)
        self.step_size_rad = float(step_size_rad)
        self.full_energy_every_steps = int(full_energy_every_steps)
        self.move_settings = normalize_move_settings(
            move_weights=move_weights,
            sidechain_move_mode=sidechain_move_mode,
            pivot_rama_probability=pivot_rama_probability,
            pivot_rama_schedule=pivot_rama_schedule,
            kic_step_size_rad=kic_step_size_rad,
        )
        self._cycle = 0
        self._exchange_tag = 0
        # MPI tags must stay in [0, TAG_UB]. Exchanges use tag_base..tag_base+2,
        # so wrap with modulus TAG_UB-1 so tag_base+2 never exceeds TAG_UB
        # (Open MPI OFI often uses TAG_UB=2^23-1; the old `% TAG_UB` overflowed).
        tag_ub = int(comm.Get_attr(MPI.TAG_UB))
        if tag_ub < 2:
            raise RuntimeError(
                f"MPI_TAG_UB={tag_ub} is too small for replica-exchange tags "
                "(need tag_base..tag_base+2)"
            )
        self._tag_ub: int = tag_ub
        self._tag_mod: int = tag_ub - 1
        self._mc_replica_steps = 0
        ex_mode = str(exchange_log or "none").strip().lower()
        if ex_mode not in ("none", "all"):
            raise ValueError("exchange_log must be 'none' or 'all'")
        self.exchange_log_mode = ex_mode
        self.state_log_interval = max(0, int(state_log_interval))
        self.log_walker_in_data_csv = bool(log_walker_in_data_csv)

        cfg = checkpoint_config or CheckpointConfig()
        if checkpoint_dir is not None:
            cfg.checkpoint_dir = str(checkpoint_dir)
        if checkpoint_interval is not None:
            cfg.checkpoint_interval = int(checkpoint_interval)
        if keep_last_n is not None:
            cfg.keep_last_n = keep_last_n
        if resume is not None:
            cfg.resume = resume
        self.checkpoint_config = cfg

        if output_dir is not None:
            self.output_dir = Path(output_dir)
            if self.rank == 0:
                self.output_dir.mkdir(parents=True, exist_ok=True)
            comm.Barrier()
            self.output_prefix = str(self.output_dir / Path(output_prefix).name)
        else:
            self.output_dir = None
            self.output_prefix = output_prefix
        self.traj_dir = str(Path(self.output_prefix).parent)
        self._traj_writers: dict[str, Any] = {}
        self.traj_frame_counts: dict[str, int] = {}
        self._traj_prior_frames: dict[str, int] = {}
        self._reporters_attached = False
        self.sample_writer = None
        # Exchange attempts/acceptances of the whole run, checkpointed so a
        # resumed run's rex_stats.json counts every cycle.
        self._exchange_counts = restore_exchange_counts()

        if self.rank == 0:
            grid = build_grid_metadata(
                temperatures=temperatures,
                temp_min=temp_min,
                temp_step=temp_step,
                n_temps=n_temps,
                native_contact_targets=native_contact_targets,
                n_targets=n_targets,
                q_targets=q_targets,
                n_q_windows=n_q_windows,
                q_step=q_step,
            )
            payload = {
                "temperatures": grid[0],
                "n_targets_raw": grid[1],
                "q_targets_raw": grid[2],
                "n_temps": grid[3],
                "n_q_windows": grid[4],
            }
        else:
            payload = None
        payload = comm.bcast(payload, root=0)

        self.temperatures = payload["temperatures"]
        self.n_temps = int(payload["n_temps"])
        self.n_q_windows = int(payload["n_q_windows"])
        self.n_replicas = self.n_temps * self.n_q_windows
        n_targets_raw = payload["n_targets_raw"]
        q_targets_raw = payload["q_targets_raw"]

        assignments, owner_rank = partition_replicas(self.n_replicas, self.size)
        if self.rank == 0:
            payload = {
                "assignments": assignments,
                "owner_rank": owner_rank,
            }
        else:
            payload = None
        payload = comm.bcast(payload, root=0)

        self.assignments: list[list[int]] = payload["assignments"]
        self.owner_rank: np.ndarray = payload["owner_rank"]
        self.local_replica_indices: list[int] = self.assignments[self.rank]
        self.replicas_per_rank = len(self.local_replica_indices)

        self.forcefield_name = forcefield
        self.system, self.forcefield, self.topology, coords_angstroms, self.q_cv = (
            build_system_and_cv(
                pdb_path,
                reference_pdb=self.reference_pdb,
                contact_cutoff=contact_cutoff,
                min_seq_sep=min_seq_sep,
                contact_atom_mode=self.contact_atom_mode,
                q_cutoff=q_cutoff,
                native_contact_pairs=self.native_contact_pairs,
                fixed_residues=self.fixed_residues,
                linker_residues=self.linker_residues,
                linker_energy_mode=self.linker_energy_mode,
                forcefield=forcefield,
                forcefield_options=forcefield_options,
                param_set=param_set,
                param_dir=None if param_dir is None else str(param_dir),
                move_weights=self.move_settings["move_weights"],
            )
        )
        n_res = self.system.get_num_residues()
        # Topology to read this run's XTCs against. Rank 0 writes it when the
        # force field simulates fewer heavy atoms than the input. Its error,
        # if any, is broadcast so the other ranks raise instead of waiting.
        top_error = None
        try:
            self.top_path = trajectory_topology_path(
                self.forcefield, pdb_path, self.reference_pdb,
                f"{self.output_prefix}_topology.pdb", write=self.rank == 0)
        except Exception as exc:
            if self.rank != 0:
                raise
            top_error = f"rank 0 could not write the trajectory topology: {exc!r}"
        top_error = comm.bcast(top_error, root=0)
        if top_error:
            raise RuntimeError(top_error)

        n_contacts = float(self.q_cv.n_contacts)
        self.n_targets = resolve_n_targets(n_targets_raw, q_targets_raw, n_contacts)
        self.q_targets = self.n_targets / n_contacts
        if int(self.n_targets.size) != self.n_q_windows:
            raise ValueError(
                f"Resolved n_targets length {self.n_targets.size} != "
                f"n_q_windows {self.n_q_windows}"
            )

        self.replica_temperatures = np.repeat(self.temperatures, self.n_q_windows)
        self.replica_n_targets = np.tile(self.n_targets, self.n_temps)
        self.replica_q_targets = np.tile(self.q_targets, self.n_temps)

        self.replicas: dict[int, Replica] = {}
        for local_replica_index in self.local_replica_indices:
            temp_index, q_index, temperature, n_target = _slot_for_index(
                local_replica_index, self.temperatures, self.n_targets
            )
            simulation = build_replica_simulation(
                topology=self.topology,
                system=self.system,
                temperature=temperature,
                seed=self.seed,
                replica_idx=local_replica_index,
                fixed_residues=self.fixed_residues,
                n_res=n_res,
                coords_angstroms=coords_angstroms,
                k_bias=self.k_bias,
                n_target=n_target,
                step_size_rad=self.step_size_rad,
                move_settings=self.move_settings,
            )
            simulation.full_energy_every_steps = self.full_energy_every_steps

            # Reporters attached in run() so resume can truncate then append.
            self.replicas[local_replica_index] = Replica(
                index=local_replica_index,
                temp_index=temp_index,
                q_index=q_index,
                temperature=temperature,
                n_target=n_target,
                q_target=n_target / n_contacts,
                simulation=simulation,
            )

        self._walker_at_state = np.arange(self.n_replicas, dtype=np.int32)

    def _replica_traj_tag(self, replica_index: int) -> str:
        rep = self.replicas[replica_index]
        return (
            f"{self.output_prefix}_r{replica_index:03d}"
            f"_t{rep.temp_index:02d}_q{rep.q_index:02d}"
        )

    def _expected_local_filenames(self) -> set[str]:
        names: set[str] = set()
        for rid in self.local_replica_indices:
            tag = Path(self._replica_traj_tag(rid)).name
            names.add(f"{tag}.xtc")
            names.add(f"{tag}_data.csv")
        return names

    def _detach_traj_reporters(self) -> None:
        """Release local reporter file handles before truncation.

        The frame counts are kept: on resume they come from the checkpoint
        and must survive until the reporters are attached again.
        """
        for slot in self.replicas.values():
            try:
                slot.simulation.flush_reporters()
            except Exception:
                pass
            try:
                slot.simulation.reporters.clear()
            except Exception:
                pass
        for reporter in list(self._traj_writers.values()):
            try:
                reporter.close()
            except Exception:
                pass
        self._traj_writers = {}
        self._reporters_attached = False

    def _attach_traj_reporters(self, resume: bool = False) -> None:
        """
        Create or reopen trajectory reporters for local replicas.
        On resume: reporters open in append mode (files already truncated)
        and count on from the checkpoint's frame counts. Otherwise the files
        start over, and so do the counts.
        """
        self._detach_traj_reporters()
        if not resume:
            self.traj_frame_counts = {}
            self._traj_prior_frames = {}
        mapping = self.forcefield.inverse_mapping
        Path(self.traj_dir).mkdir(parents=True, exist_ok=True)

        for local_replica_index in self.local_replica_indices:
            slot = self.replicas[local_replica_index]
            tag = self._replica_traj_tag(local_replica_index)
            Path(tag).parent.mkdir(parents=True, exist_ok=True)
            xtc_path = f"{tag}.xtc"
            csv_path = f"{tag}_data.csv"
            xtc_key = Path(xtc_path).name
            csv_key = Path(csv_path).name

            xtc_reporter = mcpu_core.XtcReporter(
                xtc_path,
                self.log_interval,
                mapping,
                append=bool(resume),
            )
            energy_reporter = mcpu_core.EnergyReporter(
                csv_path,
                self.log_interval,
                append=bool(resume),
            )
            slot.simulation.add_reporter(xtc_reporter)
            slot.simulation.add_reporter(energy_reporter)
            self._traj_writers[xtc_key] = xtc_reporter
            self._traj_writers[csv_key] = energy_reporter
            prior = int(self._traj_prior_frames.get(xtc_key, 0))
            self.traj_frame_counts[xtc_key] = prior
            self.traj_frame_counts[csv_key] = int(
                self._traj_prior_frames.get(csv_key, prior)
            )

        self._reporters_attached = True
        self._sync_walker_ids_to_reporters()

    def _sync_walker_ids_to_reporters(self) -> None:
        """Stamp current walker occupancy onto each local EnergyReporter."""
        if not self.log_walker_in_data_csv or not self._reporters_attached:
            return
        for local_replica_index in self.local_replica_indices:
            tag = self._replica_traj_tag(local_replica_index)
            csv_key = Path(f"{tag}_data.csv").name
            writer = self._traj_writers.get(csv_key)
            if writer is None or not hasattr(writer, "set_walker_id"):
                continue
            wid = int(self._walker_at_state[int(local_replica_index)])
            writer.set_walker_id(wid)

    def _truncate_local_trajectories(self, checkpoint_state: dict[str, Any]) -> None:
        """Truncate this rank's trajectory files to checkpoint frame counts."""
        frame_idx = checkpoint_state.get("traj_frame_indices") or {}
        owned = self._expected_local_filenames()
        for fname, count in frame_idx.items():
            if fname not in owned:
                continue
            full_path = str(Path(self.traj_dir) / fname)
            count_i = int(count)
            self._traj_prior_frames[fname] = count_i
            if not Path(full_path).is_file() and not Path(fname).is_file():
                # Also try absolute paths stored as keys
                if Path(fname).is_file():
                    full_path = fname
                else:
                    continue
            if not Path(full_path).is_file():
                # Keys may be basenames under output_prefix parent
                cand = Path(self.output_prefix).parent / fname
                if cand.is_file():
                    full_path = str(cand)
                else:
                    continue
            if fname.endswith(".xtc"):
                if count_i <= 0:
                    Path(full_path).unlink(missing_ok=True)
                else:
                    truncate_xtc_to_frame(full_path, self.top_path, count_i - 1)
            elif fname.endswith(".csv"):
                truncate_csv_to_row(full_path, count_i)

    def _local_traj_frame_indices(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for fname, reporter in self._traj_writers.items():
            prior = int(self._traj_prior_frames.get(fname, 0))
            written = 0
            if hasattr(reporter, "n_frames_written"):
                try:
                    written = int(reporter.n_frames_written())
                except Exception:
                    written = 0
            total = prior + written
            out[fname] = total
            self.traj_frame_counts[fname] = total
        return out

    def _mpi_save_checkpoint(self, cycle: int, comm: Any | None = None) -> None:
        """
        MPI-safe checkpoint save.

        Protocol:
          1. Each rank packages its local replica state into a dict.
          2. All ranks call comm.gather() to send state to rank 0.
          3. ALL ranks hit comm.Barrier() — no rank proceeds until gather finishes.
          4. Rank 0 merges gathered state and calls save_checkpoint() atomically.
          5. ALL ranks hit comm.Barrier() — no rank proceeds until write finishes.
        """
        if comm is None:
            comm = self.comm
        rank = comm.Get_rank()

        # ── Step 1: each rank packages its local replicas ──────────
        replica_ids = list(self.local_replica_indices)
        coords: list[Any] = []
        frame_offsets: list[Any] = []
        current_steps: list[int] = []
        integrator_rng_states: list[str] = []
        move_counters: list[dict[str, int]] = []
        for rid in replica_ids:
            slot = self.replicas[rid]
            slot.simulation.recompute_and_recenter()
            coords.append(np.asarray(get_coords(slot.simulation.context), dtype=np.float64))
            frame_offsets.append(get_frame_offset(slot.simulation.context))
            current_steps.append(int(slot.simulation.current_step))
            integ = slot.simulation.integrator
            if hasattr(integ, "get_rng_state"):
                try:
                    integrator_rng_states.append(str(integ.get_rng_state()))
                except Exception:
                    integrator_rng_states.append("")
            else:
                integrator_rng_states.append("")
            move_counters.append(dict(integ.get_move_counters()))

        local_state = {
            "replica_ids": replica_ids,
            "coords": coords,
            "frame_offsets": frame_offsets,
            "current_steps": current_steps,
            "integrator_rng_states": integrator_rng_states,
            "integrator_move_counters": move_counters,
            "traj_frame_indices": self._local_traj_frame_indices(),
        }

        # Rank 0 owns the analysis HDF5/NPZ writer — flush and record count.
        if rank == 0 and getattr(self, "sample_writer", None) is not None:
            try:
                self.sample_writer.flush()
                fname = os.path.basename(str(self.sample_writer.path))
                local_state["traj_frame_indices"][fname] = int(
                    self.sample_writer.n_frames_written()
                )
            except Exception as exc:
                logger.warning(
                    "Could not record RexSampleWriter frame count: %s", exc
                )

        # ── Step 2: gather to rank 0 ───────────────────────────────
        all_states = comm.gather(local_state, root=0)

        # ── Step 3: barrier before write ──────────────────────────
        comm.Barrier()

        # ── Step 4: rank 0 merges and writes ──────────────────────
        if rank == 0:
            n_replicas = int(self.n_replicas)
            coords_ordered: list[Any] = [None] * n_replicas
            offsets_ordered: list[Any] = [None] * n_replicas
            steps_ordered: list[int] = [0] * n_replicas
            rng_ordered: list[str] = [""] * n_replicas
            counters_ordered: list[dict[str, int]] = [{} for _ in range(n_replicas)]
            frame_idx_merged: dict[str, int] = {}

            for s in all_states or []:
                for i, rid in enumerate(s["replica_ids"]):
                    coords_ordered[int(rid)] = s["coords"][i]
                    offsets_ordered[int(rid)] = s["frame_offsets"][i]
                    steps_ordered[int(rid)] = int(s["current_steps"][i])
                    rng_ordered[int(rid)] = s["integrator_rng_states"][i]
                    counters_ordered[int(rid)] = s["integrator_move_counters"][i]
                frame_idx_merged.update(s.get("traj_frame_indices") or {})

            steps_per = int(self._mc_replica_steps) if self._mc_replica_steps else 0
            state = CheckpointState(
                cycle=int(cycle),
                global_step=int(cycle) * steps_per,
                seed=int(self.seed),
                format_version=CHECKPOINT_FORMAT_VERSION,
                kind="mpi_replica_exchange",
                pdb_path=str(self.pdb_path),
                forcefield=self.forcefield_name,
                reference_pdb=str(self.reference_pdb),
                temperatures=np.asarray(self.temperatures, dtype=np.float64),
                n_targets=np.asarray(self.n_targets, dtype=np.float64),
                k_bias=float(self.k_bias),
                contact_cutoff=float(self.contact_cutoff),
                min_seq_sep=int(self.min_seq_sep),
                contact_atom_mode=str(self.contact_atom_mode),
                native_contact_pairs=(
                    [list(pair) for pair in self.native_contact_pairs]
                    if self.native_contact_pairs is not None
                    else None
                ),
                fixed_residues=list(self.fixed_residues),
                linker_residues=list(self.linker_residues),
                linker_energy_mode=str(self.linker_energy_mode),
                replica_coords=coords_ordered,
                replica_frame_offsets=offsets_ordered,
                walker_at_state=np.asarray(self._walker_at_state, dtype=np.int32).copy(),
                current_steps=steps_ordered,
                exchange_rng=None,  # MPI exchange RNG is deterministic from seed+cycle
                integrator_rng_states=rng_ordered,
                integrator_move_counters=counters_ordered,
                exchange_counts=dict(self._exchange_counts),
                n_replicas=n_replicas,
                traj_frame_indices=dict(frame_idx_merged),
            )
            out_dir = self.checkpoint_config.checkpoint_dir
            cycle_name = checkpoint_cycle_filename(int(cycle))
            save_checkpoint(
                state,
                out_dir,
                filename=cycle_name,
                cycle=int(cycle),
                keep_last_n=self.checkpoint_config.keep_last_n,
            )
            # Always refresh last.chk as the canonical resume pointer.
            save_checkpoint(
                state,
                out_dir,
                filename="last.chk",
                cycle=int(cycle),
                keep_last_n=None,
            )
            logger.info(
                "MPI checkpoint saved cycle=%s file=%s",
                cycle,
                cycle_name,
            )

        # ── Step 5: barrier after write ────────────────────────────
        comm.Barrier()

    def _mpi_load_checkpoint(self, comm: Any | None = None) -> dict[str, Any] | None:
        """
        MPI-safe checkpoint load.

        Protocol:
          1. Rank 0 checks for last.chk and loads it if present.
          2. Rank 0 broadcasts the loaded state (or None) to all ranks.
          3. All ranks apply global fields (temperatures, walker_at_state, cycle).
          4. Each rank applies only the slice of state it owns.
        """
        if comm is None:
            comm = self.comm
        rank = comm.Get_rank()
        state: dict[str, Any] | None = None

        # ── Step 1: rank 0 loads ───────────────────────────────────
        if rank == 0:
            ckpt_dir = self.checkpoint_config.checkpoint_dir
            last_chk = os.path.join(str(ckpt_dir), "last.chk") if ckpt_dir else ""
            if last_chk and os.path.exists(last_chk):
                state = load_checkpoint(last_chk)
                logger.info(
                    "[MPI RE rank 0] Loaded checkpoint: cycle=%s step=%s",
                    state.get("cycle"),
                    state.get("global_step"),
                )
            else:
                logger.info(
                    "[MPI RE rank 0] No checkpoint found — starting fresh."
                )

        # A layout mismatch is decided on rank 0 and broadcast, so every rank
        # raises at the same point. A per-rank raise during the restore below
        # would leave the other ranks waiting in the next collective.
        layout_error = None
        if rank == 0 and state is not None:
            layout_error = checkpoint_forcefield_error(
                state.get("forcefield"), self.forcefield_name, last_chk
            ) or checkpoint_layout_error(
                state.get("replica_coords"), self.system.get_num_atoms(), last_chk
            )

        # ── Step 2: broadcast to all ranks ────────────────────────
        state, layout_error = comm.bcast((state, layout_error), root=0)
        if layout_error:
            raise ValueError(layout_error)

        if state is None:
            return None

        # ── Step 3: all ranks restore global fields ───────────────
        self._walker_at_state = np.asarray(
            state["walker_at_state"], dtype=np.int32
        ).copy()
        if state.get("temperatures") is not None:
            self.temperatures = np.asarray(state["temperatures"], dtype=np.float64)
        self._cycle = int(state.get("cycle", 0))
        self._exchange_counts = restore_exchange_counts(state.get("exchange_counts"))
        self._traj_prior_frames = {
            str(k): int(v) for k, v in (state.get("traj_frame_indices") or {}).items()
        }

        # ── Step 4: each rank restores its own replicas ────────────
        coords_list = state.get("replica_coords") or []
        steps_list = state.get("current_steps") or []
        rng_list = state.get("integrator_rng_states") or []
        counters_list = state.get("integrator_move_counters") or []

        for rid in self.local_replica_indices:
            if rid >= len(coords_list) or coords_list[rid] is None:
                continue
            slot = self.replicas[rid]
            coords = np.asarray(coords_list[rid], dtype=np.float64)
            if coords.ndim != 2 or coords.shape[0] != 3:
                coords = coords.T
            slot.simulation.context.set_positions(
                coords, frame_offset=saved_frame_offset(state, rid)
            )
            slot.simulation.context.calculate_total_energy(-1)
            check_state_clash(
                slot.simulation.context, "checkpoint restore"
            )
            if rid < len(steps_list) and steps_list[rid] is not None:
                slot.simulation.current_step = int(steps_list[rid])
            if rid < len(rng_list) and rng_list[rid]:
                integ = slot.simulation.integrator
                if hasattr(integ, "set_rng_state"):
                    try:
                        integ.set_rng_state(str(rng_list[rid]))
                    except Exception:
                        pass
            if rid < len(counters_list) and counters_list[rid]:
                slot.simulation.integrator.set_move_counters(dict(counters_list[rid]))

        return state

    @property
    def cycle(self) -> int:
        return self._cycle

    def replica_index(self, temp_index: int, q_index: int) -> int:
        return replica_index(temp_index, q_index, self.n_q_windows)

    def describe(self) -> str:
        local_desc = ", ".join(
            f"#{idx} (T={self.replicas[idx].temperature:.4g}, "
            f"N*={self.replicas[idx].n_target:.4g})"
            for idx in self.local_replica_indices
        )
        if self.native_contact_pairs is not None:
            native_contacts_desc = (
                f"Native contacts: {self.q_cv.n_contacts} pairs "
                f"(explicitly specified, mode={self.contact_atom_mode}, "
                f"formed when d < {self.q_cv.q_cutoff} A)"
            )
        else:
            native_contacts_desc = (
                f"Native contacts: {self.q_cv.n_contacts} pairs "
                f"(mode={self.contact_atom_mode}, |i-j| >= {self.min_seq_sep}, "
                f"d_ref < {self.contact_cutoff} A, "
                f"formed when d < {self.q_cv.q_cutoff} A)"
            )
        lines = [
            f"MPI replica exchange on {self.size} ranks, {self.n_replicas} replicas",
            (
                f"Replicas per rank: ~{self.n_replicas / self.size:.1f} "
                f"(this rank owns {self.replicas_per_rank})"
            ),
            f"Replica grid: {self.n_temps} temperatures x {self.n_q_windows} N windows",
            f"Temperatures: {self.temperatures}",
            f"N targets: {self.n_targets}",
            f"Q targets (N / n_contacts): {self.q_targets}",
            native_contacts_desc,
            f"Rank {self.rank} replicas: {local_desc}",
        ]
        if self.k_bias > 0.0:
            lines.append(
                f"N umbrella bias: k_bias={self.k_bias} (harmonic on hard N) "
                "applied during MC moves and exchange."
            )
        return "\n".join(lines)

    def get_replica_state(self, replica_index: int) -> ReplicaState:
        rep = self.replicas[replica_index]
        coords = get_coords(rep.simulation.context)
        n_value = self.q_cv.compute_N(coords)
        n_contacts = float(self.q_cv.n_contacts)
        return ReplicaState(
            cycle=self._cycle,
            replica=replica_index,
            temp_index=rep.temp_index,
            q_index=rep.q_index,
            temperature=rep.temperature,
            n_target=rep.n_target,
            q_target=rep.q_target,
            n_value=n_value,
            q_value=n_value / n_contacts,
            energy=float(rep.simulation.context.get_state().current_energy),
            walker_id=int(self._walker_at_state[replica_index]),
        )

    def get_local_replica_states(self) -> list[ReplicaState]:
        return [self.get_replica_state(idx) for idx in self.local_replica_indices]

    def step_replicas(self, mc_replica_steps: int) -> None:
        """Advance each local replica sequentially on this rank.

        Any exception here MUST tear down the whole communicator, never just
        this rank. `simulation.step` raises StericClashError on an invariant
        violation, and run_cycle() follows this call with comm.Barrier() and
        then a comm.gather() -- so a rank that leaves via an exception strands
        every other rank in a collective that will never complete. SLURM then
        reports the job RUNNING while it makes no progress: on p18.8.7 all four
        60-replica jobs sat deadlocked for 5.7-10.5 hours, 240 cores, after a
        single rank raised. A silent hang is strictly worse than the corruption
        the guard exists to catch, so convert it into an immediate, visible
        job-wide abort.
        """
        try:
            for local_replica_index in self.local_replica_indices:
                self.replicas[local_replica_index].simulation.step(mc_replica_steps)
        except BaseException:
            import traceback
            sys.stderr.write(
                f"\n[rank {self.rank}] FATAL in step_replicas at cycle "
                f"{self._cycle}; aborting all ranks so the job fails fast "
                f"instead of deadlocking in the next collective.\n"
            )
            traceback.print_exc()
            sys.stderr.flush()
            try:
                self.comm.Abort(1)
            except Exception:
                pass
            raise

    def _exchange_rng(self, i: int, j: int) -> np.random.Generator:
        pair_seed = self.seed + self._cycle * 1_000_003 + i * 1_007 + j * 9_901
        return np.random.default_rng(pair_seed)

    def _swap_walkers(self, i: int, j: int) -> None:
        wi = int(self._walker_at_state[i])
        self._walker_at_state[i] = int(self._walker_at_state[j])
        self._walker_at_state[j] = wi

    def _broadcast_walker_update(self, i: int, j: int, accepted: bool) -> None:
        """Keep walker permutation identical on every rank after each attempt."""
        buf = np.array([i, j, int(accepted)], dtype=np.int32)
        self.comm.Bcast(buf, root=0)
        if buf[2]:
            self._swap_walkers(int(buf[0]), int(buf[1]))

    def _build_exchange_record(
        self,
        i: int,
        j: int,
        *,
        dim: str,
        accepted: bool,
        n_i: float,
        n_j: float,
        e_i_total: float,
        e_j_total: float,
    ) -> ExchangeRecord:
        n_contacts = float(self.q_cv.n_contacts)
        n_target_i = float(self.replica_n_targets[i])
        n_target_j = float(self.replica_n_targets[j])
        return ExchangeRecord(
            cycle=self._cycle,
            replica_i=i,
            replica_j=j,
            dim=dim,
            accepted=accepted,
            n_i=n_i,
            n_j=n_j,
            q_i=n_i / n_contacts,
            q_j=n_j / n_contacts,
            e_i=e_i_total,
            e_j=e_j_total,
            temperature_i=float(self.replica_temperatures[i]),
            temperature_j=float(self.replica_temperatures[j]),
            n_target_i=n_target_i,
            n_target_j=n_target_j,
            e_unbiased_i=unbiased_energy(e_i_total, n_i, n_target_i, self.k_bias),
            e_unbiased_j=unbiased_energy(e_j_total, n_j, n_target_j, self.k_bias),
            walker_id_i=int(self._walker_at_state[i]),
            walker_id_j=int(self._walker_at_state[j]),
        )

    def _attempt_local_exchange(self, i: int, j: int, *, dim: str) -> ExchangeRecord:
        rep_i = self.replicas[i]
        rep_j = self.replicas[j]
        coords_i = get_coords(rep_i.simulation.context)
        coords_j = get_coords(rep_j.simulation.context)
        n_i = self.q_cv.compute_N(coords_i)
        n_j = self.q_cv.compute_N(coords_j)
        e_i_total = float(rep_i.simulation.context.get_state().current_energy)
        e_j_total = float(rep_j.simulation.context.get_state().current_energy)
        e_i = unbiased_energy(e_i_total, n_i, rep_i.n_target, self.k_bias)
        e_j = unbiased_energy(e_j_total, n_j, rep_j.n_target, self.k_bias)

        accepted = evaluate_exchange_acceptance(
            e_i,
            n_i,
            rep_i.temperature,
            rep_i.n_target,
            e_j,
            n_j,
            rep_j.temperature,
            rep_j.n_target,
            self.k_bias,
            self._exchange_rng(i, j),
        )
        if accepted:
            swap_context_coordinates(
                rep_i.simulation.context,
                rep_j.simulation.context,
                coords_i,
                coords_j,
            )

        return self._build_exchange_record(
            i,
            j,
            dim=dim,
            accepted=accepted,
            n_i=n_i,
            n_j=n_j,
            e_i_total=e_i_total,
            e_j_total=e_j_total,
        )

    def _swap_walker_with(self, rep: Any, coords: np.ndarray, peer: int, tag: int) -> None:
        """Trade this replica's walker for the one on rank ``peer``. The frame
        offset travels with the coordinates, as one more column, so each
        walker keeps its own engine frame and re-enters bit for bit."""
        context = rep.simulation.context
        send = np.column_stack((coords, get_frame_offset(context)))
        recv = np.empty_like(send)
        self.comm.Sendrecv(
            send,
            dest=peer,
            sendtag=tag,
            recvbuf=recv,
            source=peer,
            recvtag=tag,
        )
        context.set_positions(recv[:, :-1], frame_offset=recv[:, -1])
        context.calculate_total_energy(-1)
        check_state_clash(context, "post-exchange coordinate swap")

    def _attempt_mpi_exchange(
        self,
        i: int,
        j: int,
        owner_i: int,
        owner_j: int,
        *,
        dim: str,
    ) -> ExchangeRecord | None:
        comm = self.comm
        rank = self.rank
        tag_base = self._exchange_tag
        self._exchange_tag = (self._exchange_tag + 10) % self._tag_mod

        record: ExchangeRecord | None = None

        if rank == owner_i:
            rep = self.replicas[i]
            coords = get_coords(rep.simulation.context)
            e_i_total = float(rep.simulation.context.get_state().current_energy)
            n_i = self.q_cv.compute_N(coords)

            # Step 1: owner_j sends its stats first (tag_base),
            #         owner_i receives them.
            recv_buf = np.empty(3, dtype=np.float64)
            comm.Recv(recv_buf, source=owner_j, tag=tag_base)
            e_j_total, n_j = float(recv_buf[0]), float(recv_buf[1])
            n_target_j = float(self.replica_n_targets[j])
            e_i = unbiased_energy(e_i_total, n_i, rep.n_target, self.k_bias)
            e_j = unbiased_energy(e_j_total, n_j, n_target_j, self.k_bias)

            accepted = evaluate_exchange_acceptance(
                e_i,
                n_i,
                rep.temperature,
                rep.n_target,
                e_j,
                n_j,
                float(self.replica_temperatures[j]),
                n_target_j,
                self.k_bias,
                self._exchange_rng(i, j),
            )
            # Step 2: owner_i sends decision back (tag_base + 2).
            send_buf = np.array([e_i_total, n_i, float(accepted)], dtype=np.float64)
            comm.Send(send_buf, dest=owner_j, tag=tag_base + 2)

            if accepted:
                self._swap_walker_with(rep, coords, owner_j, tag_base + 1)

            record = self._build_exchange_record(
                i,
                j,
                dim=dim,
                accepted=accepted,
                n_i=n_i,
                n_j=n_j,
                e_i_total=e_i_total,
                e_j_total=e_j_total,
            )

        elif rank == owner_j:
            rep = self.replicas[j]
            coords = get_coords(rep.simulation.context)
            e_j_total = float(rep.simulation.context.get_state().current_energy)
            n_j = self.q_cv.compute_N(coords)

            # Step 1: owner_j sends its stats (tag_base).
            send_buf = np.array([e_j_total, n_j, 0.0], dtype=np.float64)
            comm.Send(send_buf, dest=owner_i, tag=tag_base)

            # Step 2: owner_j receives decision from owner_i (tag_base + 2).
            recv_buf = np.empty(3, dtype=np.float64)
            comm.Recv(recv_buf, source=owner_i, tag=tag_base + 2)
            e_i_total, n_i = float(recv_buf[0]), float(recv_buf[1])
            accepted = bool(recv_buf[2])

            if accepted:
                self._swap_walker_with(rep, coords, owner_i, tag_base + 1)

            record = self._build_exchange_record(
                i,
                j,
                dim=dim,
                accepted=accepted,
                n_i=n_i,
                n_j=n_j,
                e_i_total=e_i_total,
                e_j_total=e_j_total,
            )

        comm.Barrier()
        return record

    def _attempt_exchange(self, i: int, j: int, *, dim: str) -> ExchangeRecord | None:
        owner_i = int(self.owner_rank[i])
        owner_j = int(self.owner_rank[j])

        if owner_i == owner_j:
            # Every rank meets one Barrier per attempt, as after a cross-rank
            # exchange. If the owner skipped it, the other ranks' Barrier met
            # the owner's next collective instead, and the run deadlocked
            # once a rank owned two neighbouring slots.
            record = (
                self._attempt_local_exchange(i, j, dim=dim)
                if self.rank == owner_i else None
            )
            self.comm.Barrier()
            return record

        return self._attempt_mpi_exchange(
            i, j, owner_i, owner_j, dim=dim
        )

    def _collect_exchange_record(
        self,
        report_rank: int,
        record: ExchangeRecord | None,
    ) -> ExchangeRecord | None:
        # Save the tag BEFORE incrementing so both the send and recv
        # use the same value regardless of wrap-around.
        tag = self._exchange_tag
        self._exchange_tag = (self._exchange_tag + 1) % self._tag_mod

        if self.rank == 0:
            if report_rank == 0:
                return record
            return self.comm.recv(source=report_rank, tag=tag)
        if self.rank == report_rank and record is not None:
            self.comm.send(record, dest=0, tag=tag)
        return None

    def _report_rank_for_exchange(self, i: int, j: int) -> int:
        owner_i = int(self.owner_rank[i])
        owner_j = int(self.owner_rank[j])
        return owner_i if owner_i == owner_j else owner_i

    def exchange_temperatures(self) -> list[ExchangeRecord]:
        records: list[ExchangeRecord] = []
        for q_idx in range(self.n_q_windows):
            for temp_idx in range(self.n_temps - 1):
                i = self.replica_index(temp_idx, q_idx)
                j = self.replica_index(temp_idx + 1, q_idx)
                local_record = self._attempt_exchange(i, j, dim="temperature")
                report_rank = self._report_rank_for_exchange(i, j)
                gathered = self._collect_exchange_record(report_rank, local_record)
                accepted = False
                if self.rank == 0 and gathered is not None:
                    records.append(gathered)
                    accepted = bool(gathered.accepted)
                self._broadcast_walker_update(i, j, accepted)
        return records

    def exchange_q_windows(self) -> list[ExchangeRecord]:
        if self.n_q_windows <= 1:
            return []
        records: list[ExchangeRecord] = []
        for temp_idx in range(self.n_temps):
            for q_idx in range(self.n_q_windows - 1):
                i = self.replica_index(temp_idx, q_idx)
                j = self.replica_index(temp_idx, q_idx + 1)
                local_record = self._attempt_exchange(i, j, dim="Q")
                report_rank = self._report_rank_for_exchange(i, j)
                gathered = self._collect_exchange_record(report_rank, local_record)
                accepted = False
                if self.rank == 0 and gathered is not None:
                    records.append(gathered)
                    accepted = bool(gathered.accepted)
                self._broadcast_walker_update(i, j, accepted)
        return records

    def run_cycle(
        self,
        mc_replica_steps: int,
    ) -> list[ReplicaState]:
        """MC block then snapshot. Caller must write MBAR samples before exchange()."""
        self.step_replicas(mc_replica_steps)
        self.comm.Barrier()
        return self.get_local_replica_states()

    def exchange_all(self) -> list[ExchangeRecord]:
        """Attempt T and Q neighbor swaps; update walker ids on all ranks."""
        if self.rank == 0:
            exchanges = self.exchange_temperatures() + self.exchange_q_windows()
        else:
            self.exchange_temperatures()
            self.exchange_q_windows()
            exchanges = []
        self._cycle += 1
        return exchanges

    def run(
        self,
        num_cycles: int,
        mc_replica_steps: int,
        *,
        write_logs: bool = True,
        verbose: bool = True,
        hdf5_path: str | Path | None = None,
        analysis_path: str | Path | None = None,
        checkpoint_dir: str | Path | None = None,
        checkpoint_interval: int | None = None,
        keep_last_n: int | None = None,
        resume: str | bool | None = None,
    ) -> RunSummary | None:
        # Merge run-time checkpoint kwargs into config.
        cfg = self.checkpoint_config
        if checkpoint_dir is not None:
            cfg.checkpoint_dir = str(checkpoint_dir)
        if checkpoint_interval is not None:
            cfg.checkpoint_interval = int(checkpoint_interval)
        if keep_last_n is not None:
            cfg.keep_last_n = keep_last_n
        if resume is not None:
            cfg.resume = resume

        self._mc_replica_steps = int(mc_replica_steps)

        # Checkpoints are written only with a checkpoint_dir and enabled=True.
        # Every rank reads the same config, so all agree on this.
        saving = bool(cfg.enabled and cfg.checkpoint_dir)

        # 1. makedirs (rank 0) + barrier
        if self.rank == 0 and saving:
            Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        self.comm.Barrier()

        # 2. Load checkpoint if resume requested
        resume_flag = bool(cfg.resolved_resume_path() if hasattr(cfg, "resolved_resume_path") else cfg.resume)
        if isinstance(cfg.resume, str) and cfg.resume:
            resume_flag = True
        elif cfg.resume is True:
            resume_flag = True

        checkpoint_state = None
        if resume_flag:
            checkpoint_state = self._mpi_load_checkpoint(self.comm)
            if verbose and self.rank == 0 and checkpoint_state is not None:
                print(
                    f"Resumed from checkpoint (cycle={self._cycle})",
                    flush=True,
                )

        # 3. Detach reporters before truncation
        self._detach_traj_reporters()

        # 4. Truncate local trajectory files to checkpointed frame counts
        if checkpoint_state is not None:
            self._truncate_local_trajectories(checkpoint_state)
            self.traj_frame_counts = {
                str(k): int(v)
                for k, v in (checkpoint_state.get("traj_frame_indices") or {}).items()
                if Path(k).name in self._expected_local_filenames()
                or k in self._expected_local_filenames()
            }

        # 5. Re-attach reporters (append on resume)
        self._attach_traj_reporters(resume=bool(checkpoint_state is not None and self._cycle > 0))

        remaining_cycles = max(0, int(num_cycles) - int(self._cycle))
        if remaining_cycles == 0 and verbose and self.rank == 0:
            print(
                f"Checkpoint already at cycle {self._cycle} >= num_cycles={num_cycles}; "
                "nothing to run."
            )

        exchange_log = Path(f"{self.output_prefix}_exchange.csv")
        state_log = Path(f"{self.output_prefix}_state.csv")
        rex_stats_path = Path(f"{self.output_prefix}_rex_stats.json")

        exchange_file: TextIO | None = None
        state_file: TextIO | None = None
        exchange_writer: csv.DictWriter | None = None
        state_writer: csv.DictWriter | None = None

        analysis_file = analysis_path if analysis_path is not None else hdf5_path
        sample_writer = None
        resume_append = checkpoint_state is not None and self._cycle > 0
        if analysis_file is not None and self.rank == 0:
            from pymcpu.analysis.pymbar_export import RexSampleWriter

            sample_writer = RexSampleWriter(
                analysis_file,
                temperatures=self.temperatures,
                n_targets=self.n_targets,
                k_bias=self.k_bias,
                n_contacts=self.q_cv.n_contacts,
            )
            self.sample_writer = sample_writer
            if resume_append:
                try:
                    loaded_n = sample_writer.load_existing()
                    fname = os.path.basename(str(sample_writer.path))
                    saved = (checkpoint_state.get("traj_frame_indices") or {}).get(fname)
                    if saved is not None and loaded_n > int(saved):
                        # Samples written after the checkpoint (the run went
                        # on past its last save): cut them like the others.
                        truncate_all_trajectories_on_resume(
                            {"traj_frame_indices": {fname: int(saved)}},
                            traj_dir=str(Path(sample_writer.path).parent),
                            top_path=self.top_path,
                        )
                        loaded_n = sample_writer.load_existing()
                    if loaded_n > 0:
                        self.traj_frame_counts[fname] = loaded_n
                except Exception as exc:
                    logger.warning(
                        "Could not reload RexSampleWriter buffer on resume: %s", exc
                    )

        write_exchange = bool(write_logs) and self.exchange_log_mode == "all"
        write_state = bool(write_logs) and self.state_log_interval > 0
        if write_exchange and self.rank == 0:
            ex_mode = "a" if resume_append and exchange_log.exists() else "w"
            if ex_mode == "a":
                truncate_csv_to_cycle(str(exchange_log), self._cycle)
            exchange_file = exchange_log.open(ex_mode, newline="")
            exchange_writer = csv.DictWriter(
                exchange_file,
                fieldnames=EXCHANGE_CSV_FIELDS,
            )
            if ex_mode == "w":
                exchange_writer.writeheader()
        if write_state and self.rank == 0:
            st_mode = "a" if resume_append and state_log.exists() else "w"
            if st_mode == "a":
                truncate_csv_to_cycle(str(state_log), self._cycle)
            state_file = state_log.open(st_mode, newline="")
            state_writer = csv.DictWriter(
                state_file,
                fieldnames=[
                    "cycle",
                    "replica",
                    "temp_index",
                    "q_index",
                    "temperature",
                    "n_target",
                    "q_target",
                    "N",
                    "Q",
                    "energy",
                    "energy_unbiased",
                    "walker_id",
                ],
            )
            if st_mode == "w":
                state_writer.writeheader()

        start_time = time.perf_counter()
        written_analysis: Path | None = None
        written_rex_stats: Path | None = None
        interval = max(1, int(cfg.checkpoint_interval))

        def _dump_rex_stats() -> Path:
            return write_rex_stats(
                rex_stats_path, **self._exchange_counts, cycles_completed=int(self._cycle)
            )

        def _flush_logs() -> None:
            # Rows reach the file before the checkpoint that covers them, so
            # a job killed later cannot lose them.
            for log_file in (exchange_file, state_file):
                if log_file is not None:
                    log_file.flush()

        try:
            with catch_termination_signals() as shutdown:
                for _ in range(remaining_cycles):
                    # Enforced order: MC → snapshot → gather → HDF5 → exchange.
                    local_states = self.run_cycle(mc_replica_steps)
                    gathered_states = self.comm.gather(local_states, root=0)

                    if self.rank == 0 and gathered_states is not None:
                        flat_states: list[ReplicaState] = []
                        for rank_states in gathered_states:
                            flat_states.extend(rank_states)
                        flat_states.sort(key=lambda s: s.replica)

                        if sample_writer is not None:
                            for replica_state in flat_states:
                                e_unbiased = unbiased_energy(
                                    replica_state.energy,
                                    replica_state.n_value,
                                    replica_state.n_target,
                                    self.k_bias,
                                )
                                sample_writer.append(
                                    state_index=self.replica_index(
                                        replica_state.temp_index, replica_state.q_index
                                    ),
                                    energy_unbiased=e_unbiased,
                                    N=replica_state.n_value,
                                    cycle=replica_state.cycle,
                                    walker_id=replica_state.walker_id,
                                )

                        if state_writer is not None and flat_states:
                            cycle_for_state = int(flat_states[0].cycle)
                            if cycle_for_state % self.state_log_interval == 0:
                                for replica_state in flat_states:
                                    e_unbiased = unbiased_energy(
                                        replica_state.energy,
                                        replica_state.n_value,
                                        replica_state.n_target,
                                        self.k_bias,
                                    )
                                    state_writer.writerow(
                                        {
                                            "cycle": replica_state.cycle,
                                            "replica": replica_state.replica,
                                            "temp_index": replica_state.temp_index,
                                            "q_index": replica_state.q_index,
                                            "temperature": replica_state.temperature,
                                            "n_target": f"{replica_state.n_target:.6f}",
                                            "q_target": f"{replica_state.q_target:.6f}",
                                            "N": f"{replica_state.n_value:.6f}",
                                            "Q": f"{replica_state.q_value:.6f}",
                                            "energy": f"{replica_state.energy:.6f}",
                                            "energy_unbiased": f"{e_unbiased:.6f}",
                                            "walker_id": replica_state.walker_id,
                                        }
                                    )

                    exchanges = self.exchange_all()
                    self._sync_walker_ids_to_reporters()
                    if self.rank == 0:
                        count_exchanges(self._exchange_counts, exchanges)
                        if exchange_writer is not None:
                            for record in exchanges:
                                exchange_writer.writerow(exchange_record_to_row(record))

                    if verbose and self.rank == 0:
                        print(f"Cycle {self._cycle}/{num_cycles} complete", flush=True)

                    if saving and self._cycle > 0 and self._cycle % interval == 0:
                        _flush_logs()
                        self._mpi_save_checkpoint(self._cycle, self.comm)
                        if self.rank == 0:
                            written_rex_stats = _dump_rex_stats()

                    stop = self.comm.bcast(shutdown.requested if self.rank == 0 else None, root=0)
                    if stop:
                        for slot in self.replicas.values():
                            slot.simulation.flush_reporters()
                        _flush_logs()
                        if saving:
                            self._mpi_save_checkpoint(self._cycle, self.comm)
                            if self.rank == 0:
                                written_rex_stats = _dump_rex_stats()
                        break
        finally:
            if exchange_file is not None:
                exchange_file.close()
            if state_file is not None:
                state_file.close()
            if sample_writer is not None:
                written_analysis = sample_writer.close()

        # 7. Final barrier
        self.comm.Barrier()

        elapsed = time.perf_counter() - start_time

        if self.rank != 0:
            return None

        written_rex_stats = _dump_rex_stats()
        counts = self._exchange_counts

        if verbose:
            print(f"Finished in {elapsed:.2f} s")
            print(
                "Temperature exchanges accepted: "
                f"{counts['n_temp_accepts']}/{counts['n_temp_attempts']}"
            )
            if self.n_q_windows > 1:
                print(f"N exchanges accepted: {counts['n_q_accepts']}/{counts['n_q_attempts']}")
            print(f"RE stats: {written_rex_stats}")
            if write_exchange:
                print(f"Exchange log: {exchange_log}")
            if write_state:
                print(f"Replica state log: {state_log}")
            if written_analysis is not None:
                print(f"Analysis samples: {written_analysis}")

        return RunSummary(
            elapsed_s=elapsed,
            **counts,
            exchange_log=exchange_log if write_exchange else None,
            state_log=state_log if write_state else None,
            analysis_path=written_analysis,
            rex_stats_path=written_rex_stats,
        )
