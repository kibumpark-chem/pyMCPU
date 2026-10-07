"""Single-temperature MC folding runner with checkpoint / resume support."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Sequence

import mdtraj as md
import numpy as np

from pymcpu import mcpu_core
from pymcpu.checkpointing import (
    CHECKPOINT_FORMAT_VERSION,
    CheckpointConfig,
    FoldingCheckpointState,
    checkpoint_cycle_filename,
    checkpoint_layout_error,
    get_integrator_move_counters,
    get_integrator_rng_states,
    load_checkpoint as _load_checkpoint,
    save_checkpoint as _save_checkpoint,
    set_integrator_move_counters,
    set_integrator_rng_states,
)
from pymcpu.forcefields.mcpu import MCPUForceField
from pymcpu.sampling.collective_variables import (
    NativeContactsCV,
    build_ca_index,
    build_contact_atom_index,
    kabsch_rmsd,
    reference_ca_from_pdb,
    reference_contact_from_pdb,
)
from pymcpu.sampling.folding_bias import BasinTracker, FoldingBias
from pymcpu.sampling.replica_exchange import get_coords
from pymcpu.simulation import Simulation, check_state_clash
from pymcpu.trajectory_utils import truncate_all_trajectories_on_resume

logger = logging.getLogger(__name__)


class FoldingRunner:
    """Single-temperature MC folding, run in cycles, with checkpoints.

    Each cycle is ``steps_per_cycle`` MC steps (by default ``report_interval``).
    After each cycle, Q, the fraction of native contacts formed, is counted
    against ``reference_pdb`` (by default the starting structure). The run
    stops early once Q has stayed at or above ``q_threshold`` (default 0.75)
    for ``convergence_window`` (default 10) cycles in a row.
    ``full_energy_every_steps`` sets the simulation's full energy recompute
    cadence (:attr:`pymcpu.Simulation.full_energy_every_steps`); a checkpoint
    save also recomputes.
    """

    def __init__(
        self,
        pdb: str | Path,
        *,
        temperature: float = 0.6,
        report_interval: int = 100,
        steps_per_cycle: int | None = None,
        output_dir: str | Path = "./out_folding",
        seed: int = 42,
        param_dir: str | Path | None = None,
        param_set: str = "mcpu08",
        compute_dssp: bool = False,
        dssp_coil_state: str = "C",
        step_size_rad: float = 0.1,
        sidechain_move_mode: str = "rotamer_library",
        pivot_rama_probability: float = 0.0,
        move_weights: tuple[float, float, float] | None = None,
        full_energy_every_steps: int = 1_000_000,
        pivot_rama_schedule: dict[str, float] | None = None,
        prefix: str = "folding",
        fixed_residues: list[int] | None = None,
        linker_residues: list[int] | None = None,
        linker_energy_mode: str = "ignore_all",
        reference_pdb: str | Path | None = None,
        contact_cutoff_ang: float = 8.0,
        min_seq_sep: int = 4,
        contact_atom_mode: str = "ca",
        native_contact_pairs: Sequence[Sequence[int]] | None = None,
        q_threshold: float = 0.75,
        convergence_window: int = 10,
        enable_bias: bool = False,
        bias_k: float = 0.0,
        bias_r0: float = 0.0,
        bias_mode: str = "none",
        enable_basins: bool = False,
        n_basins: int = 1,
        checkpoint_config: CheckpointConfig | None = None,
        checkpoint_dir: str | Path | None = None,
        checkpoint_interval: int | None = None,
        keep_last_n: int | None = None,
        resume: str | bool | None = None,
        verbose: bool = True,
    ) -> None:
        self.pdb_path = Path(pdb)
        self.reference_pdb = str(reference_pdb) if reference_pdb is not None else str(self.pdb_path)
        self.contact_cutoff_ang = float(contact_cutoff_ang)
        self.min_seq_sep = int(min_seq_sep)
        from pymcpu.sampling.collective_variables import normalize_contact_atom_mode

        self.contact_atom_mode = normalize_contact_atom_mode(contact_atom_mode)
        self.native_contact_pairs = (
            list(native_contact_pairs) if native_contact_pairs else None
        )
        self.q_threshold = float(q_threshold)
        self.convergence_window = int(convergence_window)
        self._native_contacts: list[tuple[int, int]] | None = None
        self._q_cv: NativeContactsCV | None = None
        self._ca_internal_idx: np.ndarray | None = None
        self.temperature = float(temperature)
        self.report_interval = int(report_interval)
        self.steps_per_cycle = int(
            steps_per_cycle if steps_per_cycle is not None else report_interval
        )
        self.seed = int(seed)
        self.param_dir = param_dir
        self.param_set = param_set
        self.compute_dssp = bool(compute_dssp)
        self.dssp_coil_state = dssp_coil_state
        self.step_size_rad = float(step_size_rad)
        self.prefix = str(prefix)
        self.fixed_residues = list(fixed_residues) if fixed_residues else []
        self.linker_residues = list(linker_residues) if linker_residues else []
        from pymcpu.config import (
            normalize_linker_energy_mode,
            normalize_move_weights,
            normalize_pivot_rama_probability,
            normalize_pivot_rama_schedule,
            normalize_sidechain_move_mode,
            validate_fixed_linker_disjoint,
        )

        self.linker_energy_mode = normalize_linker_energy_mode(linker_energy_mode)
        self.sidechain_move_mode = normalize_sidechain_move_mode(sidechain_move_mode)
        self.pivot_rama_probability = normalize_pivot_rama_probability(pivot_rama_probability)
        self.move_weights = normalize_move_weights(move_weights)
        self.full_energy_every_steps = int(full_energy_every_steps)
        self.pivot_rama_schedule = normalize_pivot_rama_schedule(pivot_rama_schedule)
        validate_fixed_linker_disjoint(self.fixed_residues, self.linker_residues)
        self.verbose = bool(verbose)

        self.folding_bias = (
            FoldingBias(k=bias_k, r0=bias_r0, mode=bias_mode)
            if enable_bias
            else None
        )
        self.basin_tracker = (
            BasinTracker(n_basins=n_basins, n_replicas=1)
            if enable_basins
            else None
        )

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

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.traj_dir = str(self.output_dir)
        self.top_path = str(self.pdb_path)
        self.output_prefix = str(self.output_dir / self.prefix)

        self._cycle = 0
        self._total_steps_target = 0
        self.traj_reporters: dict[str, Any] = {}
        self.traj_frame_counts: dict[str, int] = {}
        self._traj_prior_frames: dict[str, int] = {}
        self._reporters_attached = False
        self.sample_writer = None

        # Optional analytics hooks (empty until folding analysis is added)
        self.convergence_history: list[Any] = []
        self.folding_events: list[Any] = []

        traj = md.load(str(self.pdb_path))
        indices = traj.topology.select("not element H")
        self.filtered = traj.atom_slice(indices)

        ff_kwargs: dict[str, Any] = {
            "param_set": param_set,
            "compute_dssp": self.compute_dssp,
            "dssp_coil_state": self.dssp_coil_state,
        }
        if param_dir is not None:
            ff_kwargs["param_dir"] = str(param_dir)

        self.forcefield = MCPUForceField(self.filtered, **ff_kwargs)
        self.system = self.forcefield.create_system(self.filtered.topology)
        self.mapping = self.forcefield.inverse_mapping

        from pymcpu.config import apply_linker_energy_mask

        apply_linker_energy_mask(
            self.system,
            self.linker_residues,
            self.linker_energy_mode,
            fixed_residues=self.fixed_residues,
        )

        from pymcpu.config import configure_integrator

        integrator = mcpu_core.Integrator(self.temperature, self.step_size_rad)
        configure_integrator(
            integrator,
            move_weights=self.move_weights,
            sidechain_move_mode=self.sidechain_move_mode,
            pivot_rama_probability=self.pivot_rama_probability,
            pivot_rama_schedule=self.pivot_rama_schedule,
        )
        if hasattr(integrator, "set_seed"):
            integrator.set_seed(self.seed)
        if self.fixed_residues:
            integrator.set_fixed_residues(
                self.fixed_residues, self.system.get_num_residues()
            )

        self.simulation = Simulation(
            self.filtered.topology, self.system, integrator
        )
        self.simulation.full_energy_every_steps = self.full_energy_every_steps
        coords_angstroms = self.forcefield.coords[0] * 10.0
        self.simulation.context.set_positions(coords_angstroms.T.astype(np.float32))
        self.simulation.context.calculate_total_energy(-1)
        check_state_clash(self.simulation.context, "start structure")

        # Single-"replica" list for RNG helpers shared with RE checkpointing.
        self.replicas = [self.simulation]

    @property
    def cycle(self) -> int:
        return self._cycle

    def _coords_3xn(self) -> np.ndarray:
        """Return current engine coordinates as (3, n_atoms) Angstrom."""
        # Prefer Simulation.get_coords if present (tests monkeypatch this).
        if hasattr(self.simulation, "get_coords") and callable(
            getattr(self.simulation, "get_coords")
        ):
            try:
                raw = np.asarray(self.simulation.get_coords(), dtype=np.float64)
            except Exception:
                raw = np.asarray(
                    get_coords(self.simulation.context), dtype=np.float64
                )
        else:
            raw = np.asarray(get_coords(self.simulation.context), dtype=np.float64)

        if raw.ndim != 2:
            raise ValueError(f"coords must be 2-D, got shape {raw.shape}")
        if raw.shape[0] == 3:
            return raw
        if raw.shape[1] == 3:
            return raw.T
        raise ValueError(
            f"coords must be (3, n) or (n, 3) Angstrom, got {raw.shape}"
        )

    def _ensure_q_cv(self) -> NativeContactsCV | None:
        """Lazy-build NativeContactsCV on engine CA indices (Angstrom)."""
        if self._q_cv is not None:
            return self._q_cv
        if not self.reference_pdb:
            return None
        if not hasattr(self, "forcefield") or self.forcefield is None:
            return None
        try:
            ca_idx = build_contact_atom_index(
                self.forcefield, mode=self.contact_atom_mode
            )
            ref_ca = reference_contact_from_pdb(
                str(self.reference_pdb), mode=self.contact_atom_mode
            )
            n_res = int(ref_ca.shape[0])
            energy_mask = None
            if self.linker_residues:
                energy_mask = np.zeros(n_res, dtype=bool)
                for r in self.linker_residues:
                    if 0 <= r < n_res:
                        energy_mask[r] = True
            self._q_cv = NativeContactsCV(
                ca_internal_idx=ca_idx,
                ref_ca_xyz=ref_ca,
                contact_cutoff=self.contact_cutoff_ang,
                min_seq_sep=self.min_seq_sep,
                mode="hard",
                q_cutoff=self.contact_cutoff_ang,
                energy_ignored_residue_mask=energy_mask,
                contact_atom_mode=self.contact_atom_mode,
                native_contact_pairs=self.native_contact_pairs,
            )
            self._ca_internal_idx = ca_idx
            return self._q_cv
        except Exception as exc:
            logger.warning("[Folding] Failed to build NativeContactsCV: %s", exc)
            return None

    def _get_engine_ca_indices(self) -> list[int]:
        """
        Resolve engine-internal CA atom indices.

        Priority:
          1. ``build_ca_index(self.forcefield)``
          2. cached ``self._ca_internal_idx``
          3. mdtraj topology CA indices from the reference/input PDB
        Always returns a ``list`` (possibly empty); never ``None``.
        """
        # Priority 1: forcefield CA index builder
        if getattr(self, "forcefield", None) is not None:
            try:
                ca_idx = build_ca_index(self.forcefield)
                out = [int(i) for i in np.asarray(ca_idx).ravel().tolist()]
                if out:
                    self._ca_internal_idx = np.asarray(out, dtype=np.int64)
                    return out
            except Exception as exc:
                logger.warning("[Folding] build_ca_index failed: %s", exc)

        # Priority 2: previously cached engine indices
        if self._ca_internal_idx is not None:
            out = [int(i) for i in np.asarray(self._ca_internal_idx).ravel().tolist()]
            if out:
                return out

        # Priority 3: mdtraj topology CA indices
        ref_pdb = getattr(self, "reference_pdb", None) or getattr(
            self, "pdb_path", None
        )
        if ref_pdb is not None:
            try:
                ref = md.load(str(ref_pdb))
                out = [int(a.index) for a in ref.topology.atoms if a.name == "CA"]
                if out:
                    self._ca_internal_idx = np.asarray(out, dtype=np.int64)
                    return out
            except Exception as exc:
                logger.warning("[Folding] topology CA fallback failed: %s", exc)

        return []

    def _check_ca_index_alignment(self, engine_ca: list[int]) -> None:
        """Log a warning if engine CA count disagrees with reference CA count."""
        ref_pdb = getattr(self, "reference_pdb", None) or getattr(
            self, "pdb_path", None
        )
        if not ref_pdb or not engine_ca:
            return
        try:
            ref = md.load(str(ref_pdb))
            n_ref = sum(1 for a in ref.topology.atoms if a.name == "CA")
            if n_ref and n_ref != len(engine_ca):
                logger.warning(
                    "[Folding] CA index count mismatch: engine=%s reference=%s",
                    len(engine_ca),
                    n_ref,
                )
        except Exception:
            pass

    def _compute_native_contacts(self) -> list[tuple[int, int]]:
        """
        Build native CA–CA contact pairs from the reference structure.

        Uses engine CA indices (via ``_get_engine_ca_indices``) and residue
        lookup through ``self.mapping`` (inverse_mapping). Caches in
        ``self._native_contacts``. Returns
        ``[]`` (never ``None``) when unavailable.
        """
        if self._native_contacts is not None:
            return self._native_contacts

        ref_pdb = getattr(self, "reference_pdb", None) or getattr(
            self, "pdb_path", None
        )
        if ref_pdb is None:
            logger.warning(
                "[Folding] No reference PDB found on self — "
                "native contacts unavailable."
            )
            self._native_contacts = []
            return self._native_contacts

        engine_ca = self._get_engine_ca_indices()
        if not engine_ca:
            logger.warning("[Folding] No engine CA indices available.")
            self._native_contacts = []
            return self._native_contacts

        # Prefer NativeContactsCV when forcefield is available (engine path).
        cv = self._ensure_q_cv()
        if cv is not None:
            atom_i, atom_j = cv.atom_pair_indices()
            contacts = [
                (int(i), int(j))
                for i, j in zip(atom_i.tolist(), atom_j.tolist())
            ]
            if self.native_contact_pairs:
                logger.info(
                    "[Folding] Using %s explicitly specified native contacts "
                    "in %s (engine CA path; native_contact_pairs explicit)",
                    len(contacts),
                    ref_pdb,
                )
            else:
                logger.info(
                    "[Folding] Found %s native contacts in %s "
                    "(engine CA path; cutoff=%.1f Å, min_seq_sep=%s)",
                    len(contacts),
                    ref_pdb,
                    self.contact_cutoff_ang,
                    self.min_seq_sep,
                )
            self._native_contacts = contacts
            self._check_ca_index_alignment(engine_ca)
            return self._native_contacts

        # Fallback (no forcefield / CV): build contacts from the reference PDB
        # using topology.select + engine_ca / inverse_mapping. Hardcoded CA name
        # scans belong only in CA-index helpers / RMSD reference loading.
        #
        # GOTCHA: this fallback always selects "name CA" regardless of
        # self.contact_atom_mode. If _ensure_q_cv() fails for any reason on a
        # cb-mode run, native contacts silently switch from CB to CA here with
        # no warning to the caller.
        try:
            if self.native_contact_pairs:
                # Explicit pairs override cutoff-based discovery entirely: map
                # each (i, j) residue-index pair through engine_ca (same
                # helper NativeContactsCV would use via ca_internal_idx),
                # with the same range/self-pair validation as the CV class.
                n_res = len(engine_ca)
                contacts = []
                for i, j in self.native_contact_pairs:
                    i, j = int(i), int(j)
                    if not (0 <= i < n_res) or not (0 <= j < n_res):
                        raise ValueError(
                            f"native_contact_pairs index out of range: "
                            f"({i}, {j}) for n_res={n_res}"
                        )
                    if i == j:
                        raise ValueError(
                            f"native_contact_pairs contains a self-pair: "
                            f"({i}, {j})"
                        )
                    contacts.append((int(engine_ca[i]), int(engine_ca[j])))
            else:
                ref = md.load(str(ref_pdb))
                ca_top_idx = [int(i) for i in ref.topology.select("name CA")]
                ca_atoms = [
                    (idx, int(ref.topology.atom(idx).residue.index))
                    for idx in ca_top_idx
                ]
                mapping = list(getattr(self, "mapping", []) or [])
                # -1 marks an explicit amide H, which has no topology atom.
                top_to_engine = {
                    int(top_i): eng_i for eng_i, top_i in enumerate(mapping) if top_i >= 0
                }

                ref_xyz_nm = ref.xyz[0]
                cutoff_nm = self.contact_cutoff_ang / 10.0
                contacts: list[tuple[int, int]] = []

                if top_to_engine:
                    engine_pairs: list[tuple[int, int, int]] = []
                    for top_idx, res_idx in ca_atoms:
                        if top_idx in top_to_engine:
                            engine_pairs.append(
                                (top_to_engine[top_idx], top_idx, res_idx)
                            )
                    for ii, (eng_i, top_i, res_i) in enumerate(engine_pairs):
                        for jj, (eng_j, top_j, res_j) in enumerate(engine_pairs):
                            if jj <= ii:
                                continue
                            if abs(res_i - res_j) < self.min_seq_sep:
                                continue
                            dist_nm = float(
                                np.linalg.norm(ref_xyz_nm[top_i] - ref_xyz_nm[top_j])
                            )
                            if dist_nm < cutoff_nm:
                                contacts.append((eng_i, eng_j))
                else:
                    # No mapping: use topology / engine_ca indices (unit-test path).
                    use_ca = ca_atoms
                    if engine_ca and len(engine_ca) == len(ca_atoms):
                        use_ca = [
                            (int(engine_ca[k]), res_idx)
                            for k, (_, res_idx) in enumerate(ca_atoms)
                        ]
                    for ii, (idx_i, res_i) in enumerate(use_ca):
                        for jj, (idx_j, res_j) in enumerate(use_ca):
                            if jj <= ii:
                                continue
                            if abs(res_i - res_j) < self.min_seq_sep:
                                continue
                            # Distances always from reference topology CA order.
                            top_i = ca_atoms[ii][0]
                            top_j = ca_atoms[jj][0]
                            dist_nm = float(
                                np.linalg.norm(ref_xyz_nm[top_i] - ref_xyz_nm[top_j])
                            )
                            if dist_nm < cutoff_nm:
                                contacts.append((idx_i, idx_j))

            if self.native_contact_pairs:
                logger.info(
                    "[Folding] Using %s explicitly specified native contacts "
                    "in %s (fallback path; native_contact_pairs explicit)",
                    len(contacts),
                    ref_pdb,
                )
            else:
                logger.info(
                    "[Folding] Found %s native contacts in %s "
                    "(fallback path; cutoff=%.1f Å, min_seq_sep=%s)",
                    len(contacts),
                    ref_pdb,
                    self.contact_cutoff_ang,
                    self.min_seq_sep,
                )
            self._native_contacts = contacts
            self._check_ca_index_alignment(engine_ca)
            return self._native_contacts
        except Exception as exc:
            logger.error("[Folding] Failed to compute native contacts: %s", exc)
            self._native_contacts = []
            return self._native_contacts

    def _fraction_native_contacts(self, coords: np.ndarray) -> float:
        """
        Compute Q = (formed native contacts) / (total native contacts).

        ``coords`` must be engine coordinates in **Angstrom**, shape ``(3, n)``.
        Uses column slicing ``coords[:, i]``.
        """
        if not self._native_contacts:
            return 0.0

        arr = np.asarray(coords, dtype=np.float64)
        if arr.shape[0] != 3:
            # Accept (n, 3) by converting once; primary layout is (3, n).
            if arr.ndim == 2 and arr.shape[1] == 3:
                arr = arr.T
            else:
                raise ValueError(
                    f"coords must be (3, n) Angstrom, got shape {arr.shape}"
                )

        cutoff = float(self.contact_cutoff_ang)
        formed = 0
        for i, j in self._native_contacts:
            dist = float(np.linalg.norm(arr[:, i] - arr[:, j]))
            if dist < cutoff:
                formed += 1
        return formed / len(self._native_contacts)

    def _compute_Q_values(self) -> np.ndarray:
        """Native contact fraction Q for the current simulation. Shape (1,)."""
        if self._native_contacts is None:
            self._compute_native_contacts()

        if not self._native_contacts:
            return np.array([0.0])

        try:
            cv = self._q_cv
            if cv is not None:
                q = float(cv.compute_Q(self._coords_3xn()))
            else:
                q = self._fraction_native_contacts(self._coords_3xn())
            return np.array([q])
        except Exception as exc:
            logger.warning("[Folding] _compute_Q_values failed: %s", exc)
            return np.array([0.0])

    def _compute_rmsd_values(self) -> np.ndarray:
        """CA RMSD to native (Angstrom) after Kabsch superposition. Shape (1,)."""
        ref_pdb = getattr(self, "reference_pdb", None) or getattr(
            self, "pdb_path", None
        )
        if ref_pdb is None:
            return np.array([0.0])

        try:
            engine_ca = self._get_engine_ca_indices()
            if not engine_ca:
                return np.array([0.0])
            ca_idx = np.asarray(engine_ca, dtype=np.int64)

            # Reference CA in Angstrom.
            if getattr(self, "forcefield", None) is not None:
                ref_ca_ang = np.asarray(
                    reference_ca_from_pdb(str(ref_pdb)), dtype=np.float64
                )
            else:
                ref = md.load(str(ref_pdb))
                ref_ca_atoms = [
                    a.index for a in ref.topology.atoms if a.name == "CA"
                ]
                ref_ca_ang = ref.xyz[0, ref_ca_atoms, :] * 10.0

            coords_3xn = self._coords_3xn()  # Angstrom (3, n)
            mobile_ca = coords_3xn[:, ca_idx].T  # (n_ca, 3)

            if mobile_ca.shape != ref_ca_ang.shape:
                # Align lengths if reference used topology CA count.
                n = min(mobile_ca.shape[0], ref_ca_ang.shape[0])
                mobile_ca = mobile_ca[:n]
                ref_ca_ang = ref_ca_ang[:n]

            return np.array([kabsch_rmsd(mobile_ca, ref_ca_ang)])
        except Exception as exc:
            logger.warning("[Folding] _compute_rmsd_values failed: %s", exc)
            return np.array([0.0])

    def _check_folding_convergence(self) -> bool:
        """True when the last ``convergence_window`` Q values are all >= threshold."""
        if len(self.convergence_history) < self.convergence_window:
            return False
        recent = self.convergence_history[-self.convergence_window :]
        return all(q >= self.q_threshold for q in recent)

    def _update_convergence_history(self) -> None:
        """Append current max Q to ``convergence_history`` (call once per cycle)."""
        q_vals = self._compute_Q_values()
        self.convergence_history.append(float(np.max(q_vals)))

    def _detach_traj_reporters(self) -> None:
        """Release reporter file handles before truncation."""
        try:
            self.simulation.flush_reporters()
        except Exception:
            pass
        self.simulation.clear_reporters()
        for reporter in list(self.traj_reporters.values()):
            try:
                if hasattr(reporter, "close"):
                    reporter.close()
            except Exception:
                pass
        self.traj_reporters = {}
        self._traj_prior_frames = {}
        self._reporters_attached = False

    def _attach_traj_reporters(self, resume: bool = False) -> None:
        """Create XTC/CSV reporters; append when resuming after truncation."""
        if self._reporters_attached:
            self._detach_traj_reporters()

        tag = self.output_prefix
        Path(tag).parent.mkdir(parents=True, exist_ok=True)
        xtc_path = f"{tag}.xtc"
        csv_path = f"{tag}_data.csv"
        xtc_key = Path(xtc_path).name
        csv_key = Path(csv_path).name

        xtc_append = bool(resume and Path(xtc_path).exists())
        csv_append = bool(resume and Path(csv_path).exists())

        xtc = mcpu_core.XtcReporter(
            xtc_path, self.report_interval, self.mapping, xtc_append
        )
        csv_rep = mcpu_core.EnergyReporter(
            csv_path, self.report_interval, csv_append
        )
        self.simulation.add_reporter(xtc)
        self.simulation.add_reporter(csv_rep)

        prior_xtc = int(self.traj_frame_counts.get(xtc_key, 0))
        prior_csv = int(self.traj_frame_counts.get(csv_key, 0))
        self.traj_frame_counts[xtc_key] = prior_xtc
        self.traj_frame_counts[csv_key] = prior_csv
        self._traj_prior_frames[xtc_key] = prior_xtc
        self._traj_prior_frames[csv_key] = prior_csv
        self.traj_reporters[xtc_key] = xtc
        self.traj_reporters[csv_key] = csv_rep
        self._reporters_attached = True

    def _local_traj_frame_indices(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for fname, reporter in self.traj_reporters.items():
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

        # RexSampleWriter is unused in folding today; keep the hook for parity.
        if hasattr(self, "sample_writer") and self.sample_writer is not None:
            try:
                self.sample_writer.flush()
                fname = os.path.basename(str(self.sample_writer.path))
                out[fname] = int(self.sample_writer.n_frames_written())
                self.traj_frame_counts[fname] = out[fname]
            except Exception:
                pass
        return out

    def _run_folding_cycle(self, cycle: int) -> None:
        """Advance one cycle (up to ``steps_per_cycle`` MC steps)."""
        if self._total_steps_target > 0:
            remaining = self._total_steps_target - int(self.simulation.current_step)
            n_steps = min(self.steps_per_cycle, max(0, remaining))
        else:
            n_steps = self.steps_per_cycle
        if n_steps > 0:
            self.simulation.step(int(n_steps))
        self._cycle = int(cycle) + 1

    def save_checkpoint(self, cycle: int) -> None:
        """Save complete folding simulation state atomically."""
        self.simulation.recompute_energy()
        q_values = self._compute_Q_values()
        rmsd_values = self._compute_rmsd_values()

        bias_params = None
        if self.folding_bias is not None:
            try:
                bias_params = self.folding_bias.get_params()
            except Exception as e:
                logger.warning("[Folding] get_params() failed: %s", e)

        basin_assign = None
        if self.basin_tracker is not None:
            try:
                basin_assign = self.basin_tracker.assignments.copy()
            except Exception as e:
                logger.warning(
                    "[Folding] basin_tracker.assignments failed: %s", e
                )

        coords = np.asarray(get_coords(self.simulation.context), dtype=np.float64)
        state = FoldingCheckpointState(
            cycle=int(cycle),
            global_step=int(self.simulation.current_step),
            seed=int(self.seed),
            format_version=CHECKPOINT_FORMAT_VERSION,
            kind="folding",
            checkpoint_type="folding",
            pdb_path=str(self.pdb_path),
            reference_pdb=str(self.reference_pdb),
            temperatures=np.asarray([self.temperature], dtype=np.float64),
            n_targets=None,
            k_bias=0.0,
            fixed_residues=list(self.fixed_residues),
            linker_residues=list(self.linker_residues),
            linker_energy_mode=str(self.linker_energy_mode),
            replica_coords=[coords],
            walker_at_state=np.array([0], dtype=np.int32),
            current_steps=[int(self.simulation.current_step)],
            exchange_rng=None,
            integrator_rng_states=get_integrator_rng_states(self.replicas),
            integrator_move_counters=get_integrator_move_counters(self.replicas),
            n_replicas=1,
            traj_frame_indices=self._local_traj_frame_indices(),
            native_contacts_fraction=q_values,
            rmsd_to_native=rmsd_values,
            folding_bias_params=bias_params,
            basin_assignments=basin_assign,
            convergence_history=list(self.convergence_history),
            folding_events=list(self.folding_events),
        )

        out_dir = self.checkpoint_config.checkpoint_dir
        if not out_dir:
            raise ValueError("checkpoint_dir is required to save a folding checkpoint")

        cycle_name = checkpoint_cycle_filename(int(cycle))
        _save_checkpoint(
            state,
            out_dir,
            filename=cycle_name,
            cycle=int(cycle),
            keep_last_n=self.checkpoint_config.keep_last_n,
        )
        _save_checkpoint(
            state,
            out_dir,
            filename="last.chk",
            cycle=int(cycle),
            keep_last_n=None,
        )
        logger.info("[Folding] checkpoint saved cycle=%s file=%s", cycle, cycle_name)

    def load_checkpoint(self, checkpoint_path: str | Path) -> FoldingCheckpointState:
        """Load a folding checkpoint and restore simulation state."""
        raw = _load_checkpoint(checkpoint_path)
        state = FoldingCheckpointState.from_dict(raw)

        chk_type = getattr(state, "checkpoint_type", None) or raw.get(
            "checkpoint_type", raw.get("kind", "base")
        )
        if chk_type not in ("folding",):
            logger.warning(
                "[Folding] checkpoint_type=%r — expected 'folding'. "
                "Applying as base state. Folding-specific fields may be None.",
                chk_type,
            )

        # Restore coordinates / step / RNG
        layout_error = checkpoint_layout_error(
            state.replica_coords, self.system.get_num_atoms(), checkpoint_path
        )
        if layout_error:
            raise ValueError(layout_error)
        if state.replica_coords:
            coords = np.asarray(state.replica_coords[0], dtype=np.float64)
            if coords.ndim == 2 and coords.shape[0] == 3:
                self.simulation.context.set_positions(coords)
            else:
                self.simulation.context.set_positions(coords.T)
            self.simulation.context.calculate_total_energy(-1)
            check_state_clash(self.simulation.context, f"checkpoint restore from {checkpoint_path}")

        if state.current_steps:
            self.simulation.current_step = int(state.current_steps[0])

        if state.temperatures is not None:
            temps = np.asarray(state.temperatures, dtype=np.float64).ravel()
            if temps.size:
                self.temperature = float(temps[0])

        set_integrator_rng_states(self.replicas, state.integrator_rng_states or [])
        set_integrator_move_counters(self.replicas, state.integrator_move_counters or [])

        self._cycle = int(state.cycle)
        self.traj_frame_counts = {
            Path(str(k)).name: int(v)
            for k, v in (state.traj_frame_indices or {}).items()
        }

        # Folding-specific restores
        if state.folding_bias_params is not None:
            if self.folding_bias is not None:
                try:
                    self.folding_bias.set_params(state.folding_bias_params)
                except Exception as e:
                    logger.warning(
                        "[Folding] Could not restore bias params: %s", e
                    )

        if state.basin_assignments is not None:
            if self.basin_tracker is not None:
                try:
                    self.basin_tracker.assignments = (
                        state.basin_assignments.copy()
                    )
                except Exception as e:
                    logger.warning(
                        "[Folding] Could not restore basin state: %s", e
                    )

        if state.convergence_history:
            self.convergence_history = list(state.convergence_history)
        if state.folding_events:
            self.folding_events = list(state.folding_events)

        logger.info(
            "[Folding] Restored from cycle=%s step=%s type=%s",
            state.cycle,
            state.global_step,
            chk_type,
        )
        return state

    def run(
        self,
        n_cycles: int | None = None,
        *,
        steps: int | None = None,
        checkpoint_dir: str | None = None,
        checkpoint_interval: int | None = None,
        keep_last_n: int | None = None,
        resume: bool | str | None = None,
    ) -> Path:
        """
        Run folding MC in report-sized cycles with optional checkpoint/resume.

        Provide either ``n_cycles`` or ``steps`` (converted to cycles using
        ``steps_per_cycle``, defaulting to ``report_interval``). The run can
        stop sooner, once Q converges; see the class description.
        """
        cfg = self.checkpoint_config
        if checkpoint_dir is not None:
            cfg.checkpoint_dir = checkpoint_dir
        if checkpoint_interval is not None:
            cfg.checkpoint_interval = int(checkpoint_interval)
        if keep_last_n is not None:
            cfg.keep_last_n = keep_last_n
        if resume is not None:
            cfg.resume = resume

        if steps is not None:
            self._total_steps_target = int(steps)
            if n_cycles is None:
                n_cycles = max(
                    1,
                    (self._total_steps_target + self.steps_per_cycle - 1)
                    // self.steps_per_cycle,
                )
        elif n_cycles is not None:
            self._total_steps_target = int(n_cycles) * self.steps_per_cycle
        else:
            raise ValueError("run() requires n_cycles or steps")

        n_cycles = int(n_cycles)

        if cfg.checkpoint_dir:
            os.makedirs(cfg.checkpoint_dir, exist_ok=True)

        start_cycle = 0
        checkpoint_state: FoldingCheckpointState | None = None

        # ── Resume: load → detach → truncate ──────────────────────
        if cfg.resolved_resume_path() and cfg.checkpoint_dir:
            last_chk = os.path.join(str(cfg.checkpoint_dir), "last.chk")
            if os.path.exists(last_chk):
                self._detach_traj_reporters()
                checkpoint_state = self.load_checkpoint(last_chk)
                start_cycle = int(checkpoint_state.cycle) + 1
                truncate_all_trajectories_on_resume(
                    checkpoint_state=checkpoint_state,
                    traj_dir=str(self.traj_dir),
                    top_path=str(self.top_path),
                )
                if self.verbose:
                    print(f"[Folding] Resuming from cycle {start_cycle}")
            else:
                if self.verbose:
                    print("[Folding] No checkpoint found — starting fresh.")

        # ── Attach reporters (append if resuming) ─────────────────
        self._attach_traj_reporters(resume=(checkpoint_state is not None))

        if self.verbose:
            print(f"PDB: {self.pdb_path}")
            print(
                f"Temperature: {self.temperature}  "
                f"steps_target: {self._total_steps_target}  "
                f"cycles: {n_cycles}  seed: {self.seed}"
            )
            if self.fixed_residues:
                print(f"Fixed residues (engine indices): {self.fixed_residues}")
            if self.linker_residues:
                print(
                    f"Linker residues (engine indices): {self.linker_residues} "
                    f"(mode={self.linker_energy_mode})"
                )
            print(f"Writing outputs under {self.output_dir}")

        interval = max(1, int(cfg.checkpoint_interval))

        # ── Main cycle loop ───────────────────────────────────────
        for cycle in range(start_cycle, n_cycles):
            self._run_folding_cycle(cycle)
            self._update_convergence_history()

            if cfg.checkpoint_dir and cfg.enabled:
                is_interval = (cycle % interval == 0)
                is_last = (cycle == n_cycles - 1)
                if is_interval or is_last:
                    self.save_checkpoint(cycle)

            # Convergence check — always emergency-save when True
            converged = bool(self._check_folding_convergence())

            if converged:
                print(f"[Folding] Converged at cycle {cycle}")
                if cfg.checkpoint_dir and cfg.enabled:
                    self.save_checkpoint(cycle)  # emergency save on convergence
                break

        try:
            self.simulation.flush_reporters()
        except Exception:
            pass

        return self.output_dir
