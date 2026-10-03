"""Replica exchange and parallel tempering for pyMCPU."""

from __future__ import annotations

import csv
import logging
import os
import time
from pathlib import Path
from typing import Any, Sequence, TextIO

import numpy as np

from pymcpu import mcpu_core
from pymcpu.simulation import check_state_clash
from pymcpu.checkpointing import (
    CHECKPOINT_FORMAT_VERSION,
    CheckpointConfig,
    checkpoint_cycle_filename,
    checkpoint_layout_error,
    find_latest_checkpoint,
    get_integrator_move_counters,
    get_integrator_rng_states,
    load_checkpoint,
    restore_numpy_rng,
    save_checkpoint,
    serialize_numpy_rng,
    set_integrator_move_counters,
    set_integrator_rng_states,
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
    evaluate_exchange_acceptance,
    exchange_record_to_row,
    get_coords,
    replica_index,
    resolve_n_targets,
    swap_context_coordinates,
    unbiased_energy,
    write_rex_stats,
)
from pymcpu.sampling.replica_exchange_core import (
    evaluate_exchange_delta,  # noqa: F401 — back-compat re-export surface (pymcpu.sampling.__init__)
    make_n_targets,  # noqa: F401 — back-compat re-export surface (pymcpu.sampling.__init__)
    make_q_targets,  # noqa: F401 — back-compat re-export surface (pymcpu.sampling.__init__)
    make_temperature_ladder,  # noqa: F401 — back-compat re-export surface (pymcpu.sampling.__init__)
)

logger = logging.getLogger(__name__)


class ReplicaExchange:
    """Temperature replica exchange with optional parallel tempering along N.

    Bias and exchange use the hard native-contact count N
    (``U = 0.5 * k * (N - N0)^2``). Fraction Q = N / n_contacts is retained
    for logging only. See :class:`~pymcpu.sampling.collective_variables.NativeContactsCV`.

    Parameters
    ----------
    pdb_path : str
        Input structure path (also used as the native reference unless
        ``reference_pdb`` is set).
    reference_pdb : str, optional
        Native reference PDB for the contact set.
    temperatures : array-like, optional
        Explicit temperature ladder. If omitted, built from ``temp_min``,
        ``temp_step``, and ``n_temps``.
    temp_min, temp_step, n_temps : float or int
        Temperature ladder parameters when ``temperatures`` is not given.
    native_contact_targets, n_targets : array-like, optional
        Explicit umbrella centers in hard-contact count N (preferred).
    q_targets : array-like, optional
        Explicit fraction-Q umbrella centers. Converted to
        ``N0 = Q * n_contacts`` after the CV is built (back-compat).
    n_q_windows, q_step : int or float
        Fraction-Q window parameters when neither N nor Q targets are given.
    k_bias : float
        Harmonic umbrella strength on N (0 = no bias in exchange criterion).
    contact_cutoff, q_cutoff, min_seq_sep
        Native-contact definition passed to :class:`NativeContactsCV`.
    log_interval : int
        Reporter interval for per-replica trajectories and energies.
    output_prefix : str
        Prefix for per-replica XTC/CSV outputs.
    seed : int
        RNG seed for exchange attempts.
    step_size_rad : float
        Monte Carlo step size for every replica, in radians.
    move_weights, sidechain_move_mode, pivot_rama_probability, pivot_rama_schedule
        Move settings for every replica, with the same meaning and defaults as
        in :class:`~pymcpu.sampling.FoldingRunner`. A schedule sets each
        replica's rama-pivot probability from its own temperature.
    """

    def __init__(
      self,
      pdb_path: str,
      *,
      reference_pdb: str | None = None,
      temperatures: np.ndarray | list[float] | None = None,
      temp_min: float = 0.4,
      temp_step: float = 0.025,
      n_temps: int = 1,
      native_contact_targets: np.ndarray | list[float] | None = None,
      n_targets: np.ndarray | list[float] | None = None,
      q_targets: np.ndarray | list[float] | None = None,
      n_q_windows: int = 1,
      q_step: float = 0.1,
      k_bias: float = 0.0,
      contact_cutoff: float = 6.0,
      q_cutoff: float | None = None,
      min_seq_sep: int = 4,
      contact_atom_mode: str = "ca",
      native_contact_pairs: Sequence[Sequence[int]] | np.ndarray | None = None,
      log_interval: int = 100,
      output_prefix: str = "rex",
      output_dir: str | Path | None = None,
      seed: int = 0,
      step_size_rad: float = 0.1,
      move_weights: tuple[float, float, float] | None = None,
      sidechain_move_mode: str = "rotamer_library",
      pivot_rama_probability: float = 0.0,
      pivot_rama_schedule: dict[str, float] | None = None,
      fixed_residues: list[int] | None = None,
      linker_residues: list[int] | None = None,
      linker_energy_mode: str = "ignore_all",
      checkpoint_dir: str | Path | None = None,
      checkpoint_interval: int = 50,
      keep_last_n: int | None = 3,
      checkpoint_config: CheckpointConfig | None = None,
      exchange_log: str = "none",
      state_log_interval: int = 0,
      log_walker_in_data_csv: bool = True,
    ):
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
      self.move_settings = normalize_move_settings(
        move_weights=move_weights,
        sidechain_move_mode=sidechain_move_mode,
        pivot_rama_probability=pivot_rama_probability,
        pivot_rama_schedule=pivot_rama_schedule,
      )
      ex_mode = str(exchange_log or "none").strip().lower()
      if ex_mode not in ("none", "all"):
        raise ValueError("exchange_log must be 'none' or 'all'")
      self.exchange_log_mode = ex_mode
      self.state_log_interval = max(0, int(state_log_interval))
      self.log_walker_in_data_csv = bool(log_walker_in_data_csv)
      if output_dir is not None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_prefix = str(self.output_dir / Path(output_prefix).name)
      else:
        self.output_dir = None
        self.output_prefix = output_prefix
      self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
      self.checkpoint_interval = max(1, int(checkpoint_interval))
      self.keep_last_n = keep_last_n
      self.checkpoint_config = checkpoint_config or CheckpointConfig(
        checkpoint_dir=str(checkpoint_dir) if checkpoint_dir is not None else "checkpoints",
        checkpoint_interval=self.checkpoint_interval,
        keep_last_n=keep_last_n if keep_last_n is not None else 3,
      )
      if self.checkpoint_dir is not None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
      self.rng = np.random.default_rng(seed)

      temperatures, n_targets_raw, q_targets_raw, _, _ = build_grid_metadata(
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
      self.temperatures = temperatures

      self._cycle = 0
      self._exchange_records: list[ExchangeRecord] = []
      self._state_records: list[ReplicaState] = []

      self.system, self.forcefield, self.filtered_traj, coords_angstroms, self.q_cv = (
        build_system_and_cv(
          pdb_path,
          reference_pdb=self.reference_pdb,
          contact_cutoff=contact_cutoff,
          min_seq_sep=min_seq_sep,
          contact_atom_mode=self.contact_atom_mode,
          q_cutoff=q_cutoff,
          fixed_residues=self.fixed_residues,
          linker_residues=self.linker_residues,
          linker_energy_mode=self.linker_energy_mode,
          native_contact_pairs=self.native_contact_pairs,
        )
      )

      n_contacts = float(self.q_cv.n_contacts)
      self.n_targets = resolve_n_targets(n_targets_raw, q_targets_raw, n_contacts)
      self.q_targets = self.n_targets / n_contacts

      self.replicas = self._build_replicas(coords_angstroms)
      # Walker id = configuration lineage currently at each thermodynamic state.
      self._walker_at_state = np.arange(self.n_replicas, dtype=np.int32)
      # basename -> frame/row counts (Python-side; synced from reporters when present)
      self.traj_frame_counts: dict[str, int] = {}
      self._traj_writers: dict[str, Any] = {}
      self._traj_prior_frames: dict[str, int] = {}
      self._reporters_attached = False
      self.sample_writer = None
      # Topology for XTC truncation (mdtraj); PDB path is fine.
      self.top_path = str(self.reference_pdb)
      self.traj_dir = str(Path(self.output_prefix).parent)

    @property
    def n_temps(self) -> int:
      return int(self.temperatures.size)

    @property
    def n_q_windows(self) -> int:
      return int(self.n_targets.size)

    @property
    def n_replicas(self) -> int:
      return len(self.replicas)

    @property
    def cycle(self) -> int:
      return self._cycle

    def replica_index(self, temp_index: int, q_index: int) -> int:
      return replica_index(temp_index, q_index, self.n_q_windows)

    def describe(self) -> str:
      if self.native_contact_pairs is not None:
        native_contacts_desc = (
          f"Native contacts: {self.q_cv.n_contacts} pairs "
          f"(mode={self.contact_atom_mode}, explicitly specified, "
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
        f"Replica grid: {self.n_temps} temperatures x {self.n_q_windows} N windows "
        f"= {self.n_replicas} replicas",
        f"Temperatures: {self.temperatures}",
        f"N targets: {self.n_targets}",
        f"Q targets (N / n_contacts): {self.q_targets}",
        native_contacts_desc,
      ]
      if self.k_bias > 0.0:
        lines.append(
          f"N umbrella bias: k_bias={self.k_bias} (harmonic on hard N) "
          "applied during MC moves and exchange."
        )
      return "\n".join(lines)

    def get_replica_state(self, replica: Replica) -> ReplicaState:
      coords = get_coords(replica.simulation.context)
      n_value = self.q_cv.compute_N(coords)
      n_contacts = float(self.q_cv.n_contacts)
      return ReplicaState(
        cycle=self._cycle,
        replica=replica.index,
        temp_index=replica.temp_index,
        q_index=replica.q_index,
        temperature=replica.temperature,
        n_target=replica.n_target,
        q_target=replica.q_target,
        n_value=n_value,
        q_value=n_value / n_contacts,
        energy=float(replica.simulation.context.get_state().current_energy),
        walker_id=int(self._walker_at_state[replica.index]),
      )

    def get_all_replica_states(self) -> list[ReplicaState]:
      return [self.get_replica_state(rep) for rep in self.replicas]

    def build_checkpoint_state(self) -> dict[str, Any]:
      """Serialize RE state needed to resume after interruption."""
      replica_coords = [get_coords(rep.simulation.context) for rep in self.replicas]
      current_steps = [int(rep.simulation.current_step) for rep in self.replicas]
      self._sync_traj_frame_counts_from_writers()

      # Flush RexSampleWriter so on-disk HDF5/NPZ matches the buffer count.
      if hasattr(self, "sample_writer") and self.sample_writer is not None:
        try:
          self.sample_writer.flush()
          fname = os.path.basename(str(self.sample_writer.path))
          self.traj_frame_counts[fname] = int(
            self.sample_writer.n_frames_written()
          )
        except Exception as exc:
          logger.warning("Could not record RexSampleWriter frame count: %s", exc)

      return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "kind": "replica_exchange",
        "cycle": int(self._cycle),
        "global_step": int(self._cycle),  # RE progress unit is cycle
        "epoch": int(self._cycle),  # alias for training-style tooling
        "seed": int(self.seed),
        "pdb_path": str(self.pdb_path),
        "reference_pdb": str(self.reference_pdb),
        "temperatures": np.asarray(self.temperatures, dtype=np.float64),
        "n_targets": np.asarray(self.n_targets, dtype=np.float64),
        "k_bias": float(self.k_bias),
        "contact_cutoff": float(self.contact_cutoff),
        "min_seq_sep": int(self.min_seq_sep),
        "contact_atom_mode": str(self.contact_atom_mode),
        "native_contact_pairs": (
          [list(pair) for pair in self.native_contact_pairs]
          if self.native_contact_pairs is not None
          else None
        ),
        "fixed_residues": list(self.fixed_residues),
        "linker_residues": list(self.linker_residues),
        "linker_energy_mode": str(self.linker_energy_mode),
        "walker_at_state": np.asarray(self._walker_at_state, dtype=np.int32),
        "replica_coords": replica_coords,
        "current_steps": current_steps,
        "exchange_rng": serialize_numpy_rng(self.rng),
        "integrator_rng_states": get_integrator_rng_states(self.replicas),
        "integrator_move_counters": get_integrator_move_counters(self.replicas),
        "n_replicas": int(self.n_replicas),
        "traj_frame_indices": {
          os.path.basename(fname): int(count)
          for fname, count in self.traj_frame_counts.items()
        },
      }

    def save_checkpoint(
      self,
      checkpoint_dir: str | Path | None = None,
      *,
      filename: str | None = None,
      is_best: bool = False,
      keep_last_n: int | None = None,
    ) -> Path:
      """Write an atomic checkpoint for the current RE state."""
      out_dir = Path(checkpoint_dir) if checkpoint_dir is not None else self.checkpoint_dir
      if out_dir is None:
        raise ValueError("checkpoint_dir is required to save a checkpoint")
      state = self.build_checkpoint_state()
      name = filename if filename is not None else checkpoint_cycle_filename(self._cycle)
      path = save_checkpoint(
        state,
        out_dir,
        filename=name,
        is_best=is_best,
        keep_last_n=self.keep_last_n if keep_last_n is None else keep_last_n,
        config=self.checkpoint_config if name == "last.chk" else None,
      )
      # Always refresh last.chk as the canonical resume pointer.
      if name != "last.chk":
        save_checkpoint(
          state,
          out_dir,
          filename="last.chk",
          is_best=False,
          config=self.checkpoint_config,
        )
      return path

    def apply_checkpoint(self, checkpoint: dict[str, Any]) -> None:
      """Restore coordinates, counters, walkers, and RNG from a checkpoint dict."""
      kind = checkpoint.get("kind", "replica_exchange")
      if kind != "replica_exchange":
        raise ValueError(f"Unsupported checkpoint kind: {kind!r}")
      layout_error = checkpoint_layout_error(
        checkpoint.get("replica_coords"), self.system.get_num_atoms()
      )
      if layout_error:
        raise ValueError(layout_error)

      n_replicas = int(checkpoint.get("n_replicas", len(checkpoint.get("replica_coords", []))))
      if n_replicas != self.n_replicas:
        raise ValueError(
          f"Checkpoint n_replicas={n_replicas} does not match current grid "
          f"({self.n_replicas})"
        )

      ck_temps = np.asarray(checkpoint["temperatures"], dtype=np.float64)
      ck_n = np.asarray(checkpoint["n_targets"], dtype=np.float64)
      if ck_temps.shape != self.temperatures.shape or not np.allclose(
        ck_temps, self.temperatures
      ):
        raise ValueError("Checkpoint temperatures do not match current ReplicaExchange")
      if ck_n.shape != self.n_targets.shape or not np.allclose(ck_n, self.n_targets):
        raise ValueError("Checkpoint n_targets do not match current ReplicaExchange")
      if abs(float(checkpoint.get("k_bias", self.k_bias)) - self.k_bias) > 1e-12:
        logger.warning(
          "Checkpoint k_bias=%s differs from current k_bias=%s; using current bias",
          checkpoint.get("k_bias"),
          self.k_bias,
        )

      coords_list = checkpoint["replica_coords"]
      steps = checkpoint.get("current_steps", [0] * self.n_replicas)
      rng_states = checkpoint.get("integrator_rng_states", [""] * self.n_replicas)

      for i, rep in enumerate(self.replicas):
        coords = np.asarray(coords_list[i], dtype=np.float32)
        rep.simulation.context.set_positions(coords)
        rep.simulation.context.set_native_contacts_bias(self.k_bias, float(rep.n_target))
        rep.simulation.context.calculate_total_energy(-1)
        check_state_clash(rep.simulation.context, f"replica {i} restored from checkpoint")
        rep.simulation.current_step = int(steps[i])

      set_integrator_rng_states(self.replicas, list(rng_states))
      set_integrator_move_counters(
        self.replicas, list(checkpoint.get("integrator_move_counters") or [])
      )

      self._cycle = int(checkpoint["cycle"])
      self._walker_at_state = np.asarray(
        checkpoint["walker_at_state"], dtype=np.int32
      ).copy()
      if self._walker_at_state.shape != (self.n_replicas,):
        raise ValueError(
          f"walker_at_state shape {self._walker_at_state.shape} != ({self.n_replicas},)"
        )

      if "exchange_rng" in checkpoint:
        restore_numpy_rng(self.rng, checkpoint["exchange_rng"])
      else:
        logger.warning("exchange_rng missing from checkpoint; exchange RNG not restored")

      logger.info(
        "Applied checkpoint at cycle %s | global_step %s",
        self._cycle,
        checkpoint.get("global_step", self._cycle),
      )

    def load_checkpoint(self, checkpoint_path: str | Path) -> dict[str, Any]:
      """Load a checkpoint file (or directory containing last.chk) and apply it."""
      path = Path(checkpoint_path)
      if path.is_dir():
        latest = find_latest_checkpoint(path)
        if latest is None:
          raise FileNotFoundError(f"No checkpoint found in directory: {path}")
        path = latest
      checkpoint = load_checkpoint(path)
      self.apply_checkpoint(checkpoint)
      return checkpoint

    def step_replicas(self, mc_replica_steps: int) -> None:
      for rep in self.replicas:
        rep.simulation.step(mc_replica_steps)

    def attempt_exchange(self, rep_a: Replica, rep_b: Replica) -> bool:
      coords_a = get_coords(rep_a.simulation.context)
      coords_b = get_coords(rep_b.simulation.context)
      n_a = self.q_cv.compute_N(coords_a)
      n_b = self.q_cv.compute_N(coords_b)
      e_a = unbiased_energy(
        float(rep_a.simulation.context.get_state().current_energy),
        n_a,
        rep_a.n_target,
        self.k_bias,
      )
      e_b = unbiased_energy(
        float(rep_b.simulation.context.get_state().current_energy),
        n_b,
        rep_b.n_target,
        self.k_bias,
      )

      accepted = evaluate_exchange_acceptance(
        e_a, n_a, rep_a.temperature, rep_a.n_target,
        e_b, n_b, rep_b.temperature, rep_b.n_target,
        self.k_bias,
        self.rng,
      )
      if accepted:
        self._swap_coordinates(rep_a, rep_b, coords_a, coords_b)
      return accepted

    def exchange_temperatures(self) -> list[ExchangeRecord]:
      records: list[ExchangeRecord] = []
      for q_idx in range(self.n_q_windows):
        for temp_idx in range(self.n_temps - 1):
          i = self.replica_index(temp_idx, q_idx)
          j = self.replica_index(temp_idx + 1, q_idx)
          record = self._exchange_pair(i, j, dim="temperature")
          records.append(record)
          self._exchange_records.append(record)
      return records

    def exchange_q_windows(self) -> list[ExchangeRecord]:
      if self.n_q_windows <= 1:
        return []
      records: list[ExchangeRecord] = []
      for temp_idx in range(self.n_temps):
        for q_idx in range(self.n_q_windows - 1):
          i = self.replica_index(temp_idx, q_idx)
          j = self.replica_index(temp_idx, q_idx + 1)
          record = self._exchange_pair(i, j, dim="Q")
          records.append(record)
          self._exchange_records.append(record)
      return records

    def run_cycle(self, mc_replica_steps: int) -> tuple[list[ReplicaState], list[ExchangeRecord]]:
      self.step_replicas(mc_replica_steps)
      states = self.get_all_replica_states()
      self._state_records.extend(states)
      exchanges = self.exchange_temperatures() + self.exchange_q_windows()
      self._cycle += 1
      return states, exchanges

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
      resume: str | Path | None = None,
      keep_last_n: int | None = None,
    ) -> RunSummary:
      if checkpoint_dir is not None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
      if checkpoint_interval is not None:
        self.checkpoint_interval = max(1, int(checkpoint_interval))
      if keep_last_n is not None:
        self.keep_last_n = keep_last_n

      if resume is not None:
        checkpoint_state = self.load_checkpoint(resume)
        if verbose:
          print(f"Resumed from checkpoint: {resume} (cycle={self._cycle})")
      else:
        checkpoint_state = None

      # Close any open trajectory handles before truncating on-disk files.
      self._detach_traj_reporters()

      if resume is not None and checkpoint_state is not None:
        from pymcpu.trajectory_utils import truncate_all_trajectories_on_resume

        truncate_all_trajectories_on_resume(
          checkpoint_state=checkpoint_state,
          traj_dir=self.traj_dir,
          top_path=self.top_path,
        )
        # Restore Python-side counters from the checkpoint after truncation.
        self.traj_frame_counts = {
          str(k): int(v)
          for k, v in (checkpoint_state.get("traj_frame_indices") or {}).items()
        }

      # Open reporters after truncation: append on resume, overwrite on fresh start.
      self._attach_traj_reporters(resume=bool(resume is not None and self._cycle > 0))

      # num_cycles is the *total* target cycle count; on resume run only remaining.
      remaining_cycles = max(0, int(num_cycles) - int(self._cycle))
      if remaining_cycles == 0 and verbose:
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
      self.sample_writer = None
      sample_writer = None
      # Append CSV logs when resuming so prior cycles are preserved.
      resume_append = resume is not None and self._cycle > 0
      if analysis_file is not None:
        from pymcpu.analysis.pymbar_export import RexSampleWriter

        sample_writer = RexSampleWriter(
          analysis_file,
          temperatures=self.temperatures,
          n_targets=self.n_targets,
          k_bias=self.k_bias,
          n_contacts=self.q_cv.n_contacts,
        )
        self.sample_writer = sample_writer
        # After truncate_all_trajectories_on_resume, rehydrate buffer so
        # flush() preserves pre-checkpoint samples.
        if resume_append:
          try:
            loaded_n = sample_writer.load_existing()
            if loaded_n > 0:
              fname = os.path.basename(str(sample_writer.path))
              self.traj_frame_counts[fname] = loaded_n
          except Exception as exc:
            logger.warning(
              "Could not reload RexSampleWriter buffer on resume: %s", exc
            )

      write_exchange = bool(write_logs) and self.exchange_log_mode == "all"
      write_state = bool(write_logs) and self.state_log_interval > 0
      if write_exchange:
        ex_mode = "a" if resume_append and exchange_log.exists() else "w"
        exchange_file = exchange_log.open(ex_mode, newline="")
        exchange_writer = csv.DictWriter(
          exchange_file,
          fieldnames=EXCHANGE_CSV_FIELDS,
        )
        if ex_mode == "w":
          exchange_writer.writeheader()
      if write_state:
        st_mode = "a" if resume_append and state_log.exists() else "w"
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
      n_temp_accepts = 0
      n_q_accepts = 0
      written_analysis: Path | None = None
      cycles_completed_this_run = 0
      written_rex_stats: Path | None = None

      def _dump_rex_stats() -> Path:
        return write_rex_stats(
          rex_stats_path,
          n_temp_accepts=n_temp_accepts,
          n_temp_attempts=cycles_completed_this_run
          * max(self.n_temps - 1, 0)
          * self.n_q_windows,
          n_q_accepts=n_q_accepts,
          n_q_attempts=cycles_completed_this_run
          * self.n_temps
          * max(self.n_q_windows - 1, 0),
          cycles_completed=int(self._cycle),
        )

      try:
        with catch_termination_signals() as shutdown:
          for _ in range(remaining_cycles):
            states, exchanges = self.run_cycle(mc_replica_steps)
            cycles_completed_this_run += 1

            if sample_writer is not None:
              for state in states:
                state_index = self.replica_index(state.temp_index, state.q_index)
                e_unbiased = unbiased_energy(
                  state.energy, state.n_value, state.n_target, self.k_bias
                )
                sample_writer.append(
                  state_index=state_index,
                  energy_unbiased=e_unbiased,
                  N=state.n_value,
                  cycle=state.cycle,
                  walker_id=state.walker_id,
                )

            if state_writer is not None and states:
              cycle_for_state = int(states[0].cycle)
              if cycle_for_state % self.state_log_interval == 0:
                for state in states:
                  e_unbiased = unbiased_energy(
                    state.energy, state.n_value, state.n_target, self.k_bias
                  )
                  state_writer.writerow(
                    {
                      "cycle": state.cycle,
                      "replica": state.replica,
                      "temp_index": state.temp_index,
                      "q_index": state.q_index,
                      "temperature": state.temperature,
                      "n_target": f"{state.n_target:.6f}",
                      "q_target": f"{state.q_target:.6f}",
                      "N": f"{state.n_value:.6f}",
                      "Q": f"{state.q_value:.6f}",
                      "energy": f"{state.energy:.6f}",
                      "energy_unbiased": f"{e_unbiased:.6f}",
                      "walker_id": state.walker_id,
                    }
                  )

            for record in exchanges:
              if record.dim == "temperature":
                n_temp_accepts += int(record.accepted)
              else:
                n_q_accepts += int(record.accepted)

              if exchange_writer is not None:
                exchange_writer.writerow(exchange_record_to_row(record))

            self._sync_walker_ids_to_reporters()

            if verbose:
              print(f"Cycle {self._cycle}/{num_cycles} complete")

            # Save after cycle N when interval divides N (and always on interrupt).
            if self.checkpoint_dir is not None and self._cycle > 0:
              if self._cycle % self.checkpoint_interval == 0:
                ckpt_path = self.save_checkpoint(self.checkpoint_dir)
                written_rex_stats = _dump_rex_stats()
                if verbose:
                  print(f"Checkpoint written: {ckpt_path}")

            if shutdown.requested:
              for rep in self.replicas:
                rep.simulation.flush_reporters()
              if self.checkpoint_dir is not None:
                ckpt_path = self.save_checkpoint(self.checkpoint_dir)
                written_rex_stats = _dump_rex_stats()
                if verbose:
                  print(f"Emergency checkpoint written: {ckpt_path}")
              if verbose:
                print(
                  "Termination signal received; "
                  "flushed reporters and stopping after this cycle."
                )
              break

          # Final checkpoint at end of successful (or empty) run when enabled.
          if self.checkpoint_dir is not None and (
            cycles_completed_this_run > 0 or resume is not None
          ):
            ckpt_path = self.save_checkpoint(self.checkpoint_dir)
            written_rex_stats = _dump_rex_stats()
            if verbose:
              print(f"Final checkpoint written: {ckpt_path}")
      finally:
        if exchange_file is not None:
          exchange_file.close()
        if state_file is not None:
          state_file.close()
        if sample_writer is not None:
          written_analysis = sample_writer.close()

      elapsed = time.perf_counter() - start_time
      n_temp_attempts = cycles_completed_this_run * max(self.n_temps - 1, 0) * self.n_q_windows
      n_q_attempts = cycles_completed_this_run * self.n_temps * max(self.n_q_windows - 1, 0)
      written_rex_stats = write_rex_stats(
        rex_stats_path,
        n_temp_accepts=n_temp_accepts,
        n_temp_attempts=n_temp_attempts,
        n_q_accepts=n_q_accepts,
        n_q_attempts=n_q_attempts,
        cycles_completed=int(self._cycle),
      )

      if verbose:
        print(f"Finished in {elapsed:.2f} s")
        print(f"Temperature exchanges accepted: {n_temp_accepts}/{n_temp_attempts}")
        if self.n_q_windows > 1:
          print(f"N exchanges accepted: {n_q_accepts}/{n_q_attempts}")
        print(f"RE stats: {written_rex_stats}")
        if write_exchange:
          print(f"Exchange log: {exchange_log}")
        if write_state:
          print(f"Replica state log: {state_log}")
        if written_analysis is not None:
          print(f"Analysis samples: {written_analysis}")

      return RunSummary(
        elapsed_s=elapsed,
        n_temp_accepts=n_temp_accepts,
        n_temp_attempts=n_temp_attempts,
        n_q_accepts=n_q_accepts,
        n_q_attempts=n_q_attempts,
        exchange_log=exchange_log if write_exchange else None,
        state_log=state_log if write_state else None,
        analysis_path=written_analysis,
        rex_stats_path=written_rex_stats,
      )

    def _build_replicas(self, coords_angstroms: np.ndarray) -> list[Replica]:
      n_contacts = float(self.q_cv.n_contacts)
      n_res = self.system.get_num_residues()
      replicas: list[Replica] = []
      replica_idx = 0
      for temp_idx, temperature in enumerate(self.temperatures):
        for q_idx, n_target in enumerate(self.n_targets):
          # Reporters are attached in run() so resume can truncate then append;
          # the tag's parent directory is created there (_attach_traj_reporters).
          simulation = build_replica_simulation(
            filtered_traj=self.filtered_traj,
            system=self.system,
            temperature=float(temperature),
            seed=self.seed,
            replica_idx=replica_idx,
            fixed_residues=self.fixed_residues,
            n_res=n_res,
            coords_angstroms=coords_angstroms,
            k_bias=self.k_bias,
            n_target=float(n_target),
            step_size_rad=self.step_size_rad,
            move_settings=self.move_settings,
          )

          replicas.append(
            Replica(
              index=replica_idx,
              temp_index=temp_idx,
              q_index=q_idx,
              temperature=float(temperature),
              n_target=float(n_target),
              q_target=float(n_target) / n_contacts,
              simulation=simulation,
            )
          )
          replica_idx += 1
      return replicas

    def _replica_traj_tag(self, replica: Replica) -> str:
      return (
        f"{self.output_prefix}_t{replica.temp_index:02d}_q{replica.q_index:02d}"
      )

    def _sync_traj_frame_counts_from_writers(self) -> None:
      """Refresh traj_frame_counts from live C++ reporters when available."""
      for path, writer in self._traj_writers.items():
        base = os.path.basename(path)
        prior = int(self._traj_prior_frames.get(path, 0))
        n_attr = getattr(writer, "n_frames_written", None)
        if callable(n_attr):
          session = int(n_attr())
        elif n_attr is not None:
          session = int(n_attr)
        else:
          session = 0
          for rep in self.replicas:
            tag = self._replica_traj_tag(rep)
            if path.startswith(tag):
              session = int(rep.simulation.current_step) // max(
                1, int(self.log_interval)
              )
              prior = 0
              break
        self.traj_frame_counts[base] = prior + session

    def _detach_traj_reporters(self) -> None:
      """Flush and drop per-replica XTC/CSV reporters (release file handles)."""
      self._sync_traj_frame_counts_from_writers()
      for rep in self.replicas:
        try:
          rep.simulation.flush_reporters()
        except Exception as exc:
          logger.warning("flush_reporters failed for replica %s: %s", rep.index, exc)
        rep.simulation.clear_reporters()
      self._traj_writers.clear()
      self._traj_prior_frames.clear()
      self._reporters_attached = False

    def _attach_traj_reporters(self, *, resume: bool = False) -> None:
      """Create XTC/CSV reporters; append when resuming after truncation."""
      if self._reporters_attached:
        self._detach_traj_reporters()

      mapping = self.forcefield.inverse_mapping
      for rep in self.replicas:
        tag = self._replica_traj_tag(rep)
        xtc_path = f"{tag}.xtc"
        csv_path = f"{tag}_data.csv"
        Path(tag).parent.mkdir(parents=True, exist_ok=True)

        xtc_append = bool(resume and Path(xtc_path).exists())
        csv_append = bool(resume and Path(csv_path).exists())

        xtc = mcpu_core.XtcReporter(
          xtc_path, self.log_interval, mapping, xtc_append
        )
        csv_rep = mcpu_core.EnergyReporter(
          csv_path, self.log_interval, csv_append
        )
        prior_xtc = int(self.traj_frame_counts.get(os.path.basename(xtc_path), 0))
        prior_csv = int(self.traj_frame_counts.get(os.path.basename(csv_path), 0))
        self.traj_frame_counts[os.path.basename(xtc_path)] = prior_xtc
        self.traj_frame_counts[os.path.basename(csv_path)] = prior_csv
        self._traj_writers[xtc_path] = xtc
        self._traj_writers[csv_path] = csv_rep
        self._traj_prior_frames[xtc_path] = prior_xtc
        self._traj_prior_frames[csv_path] = prior_csv

        rep.simulation.add_reporter(xtc)
        rep.simulation.add_reporter(csv_rep)

      self._reporters_attached = True
      self._sync_walker_ids_to_reporters()

    def _sync_walker_ids_to_reporters(self) -> None:
      """Stamp current walker occupancy onto each slot's EnergyReporter."""
      if not self.log_walker_in_data_csv or not self._reporters_attached:
        return
      for rep in self.replicas:
        csv_path = f"{self._replica_traj_tag(rep)}_data.csv"
        writer = self._traj_writers.get(csv_path)
        if writer is None or not hasattr(writer, "set_walker_id"):
          continue
        rid = int(rep.index)
        wid = int(self._walker_at_state[rid]) if rid < len(self._walker_at_state) else -1
        writer.set_walker_id(wid)

    def _swap_walkers(self, i: int, j: int) -> None:
      wi = int(self._walker_at_state[i])
      self._walker_at_state[i] = int(self._walker_at_state[j])
      self._walker_at_state[j] = wi

    def _make_exchange_record(
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
      """Build a record using walker ids *before* any swap; then update walkers."""
      rep_i, rep_j = self.replicas[i], self.replicas[j]
      n_contacts = float(self.q_cv.n_contacts)
      walker_i = int(self._walker_at_state[i])
      walker_j = int(self._walker_at_state[j])
      e_unbiased_i = unbiased_energy(e_i_total, n_i, rep_i.n_target, self.k_bias)
      e_unbiased_j = unbiased_energy(e_j_total, n_j, rep_j.n_target, self.k_bias)
      if accepted:
        self._swap_walkers(i, j)
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
        temperature_i=float(rep_i.temperature),
        temperature_j=float(rep_j.temperature),
        n_target_i=float(rep_i.n_target),
        n_target_j=float(rep_j.n_target),
        e_unbiased_i=e_unbiased_i,
        e_unbiased_j=e_unbiased_j,
        walker_id_i=walker_i,
        walker_id_j=walker_j,
      )

    def _exchange_pair(self, i: int, j: int, *, dim: str) -> ExchangeRecord:
      rep_i, rep_j = self.replicas[i], self.replicas[j]
      coords_i = get_coords(rep_i.simulation.context)
      coords_j = get_coords(rep_j.simulation.context)
      n_i = self.q_cv.compute_N(coords_i)
      n_j = self.q_cv.compute_N(coords_j)
      e_i = float(rep_i.simulation.context.get_state().current_energy)
      e_j = float(rep_j.simulation.context.get_state().current_energy)
      accepted = self.attempt_exchange(rep_i, rep_j)
      return self._make_exchange_record(
        i,
        j,
        dim=dim,
        accepted=accepted,
        n_i=n_i,
        n_j=n_j,
        e_i_total=e_i,
        e_j_total=e_j,
      )

    @staticmethod
    def _swap_coordinates(
      rep_a: Replica,
      rep_b: Replica,
      coords_a: np.ndarray,
      coords_b: np.ndarray,
    ) -> None:
      swap_context_coordinates(
        rep_a.simulation.context,
        rep_b.simulation.context,
        coords_a,
        coords_b,
      )
