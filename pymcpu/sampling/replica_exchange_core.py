"""Shared, engine-agnostic building blocks for replica exchange / parallel
tempering: pure math (temperature/N/Q ladders, the Metropolis exchange
criterion), plain data containers (:class:`Replica`, :class:`ReplicaState`,
:class:`ExchangeRecord`, :class:`RunSummary`), and the construction steps
that are verbatim-identical between the serial (:class:`~pymcpu.sampling.replica_exchange.ReplicaExchange`)
and MPI-parallel (:class:`~pymcpu.sampling.mpi_replica_exchange.MPIReplicaExchange`)
engines.

Anything that depends on which replicas a process owns, or on MPI
communication/rank gating, stays in ``pymcpu.sampling.mpi_replica_exchange``
-- this module holds only what both engines can safely share unchanged.
"""

from __future__ import annotations

import json
import signal
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import mdtraj as md
import numpy as np

from pymcpu import mcpu_core
from pymcpu.forcefields.mcpu import MCPUForceField
from pymcpu.sampling.collective_variables import (
    NativeContactsCV,
    attach_native_contacts_bias_potential,
    build_contact_atom_index,
    reference_contact_from_pdb,
)
from pymcpu.simulation import Simulation, check_state_clash


@dataclass
class _ShutdownFlag:
    """Mutable flag set by SIGTERM/SIGINT; safe to check between RE cycles."""

    requested: bool = False

    def request(self, signum: int, frame: object | None) -> None:
        self.requested = True


@contextmanager
def catch_termination_signals() -> Iterator[_ShutdownFlag]:
    """Install SIGTERM/SIGINT handlers that only set a flag (no I/O in-handler).

    Flush reporters and exit cleanly after the current RE cycle completes.
    """
    flag = _ShutdownFlag()
    prev_term = signal.signal(signal.SIGTERM, flag.request)
    prev_int = signal.signal(signal.SIGINT, flag.request)
    try:
        yield flag
    finally:
        signal.signal(signal.SIGTERM, prev_term)
        signal.signal(signal.SIGINT, prev_int)


@dataclass
class Replica:
    """One replica's identity and live simulation state, owned entirely by
    one process: the serial engine owns every ``Replica``, the MPI engine
    owns only the subset assigned to its rank."""

    index: int
    temp_index: int
    q_index: int
    temperature: float
    n_target: float
    q_target: float  # fraction = n_target / n_contacts (logging)
    simulation: Simulation


@dataclass
class ReplicaState:
    cycle: int
    replica: int
    temp_index: int
    q_index: int
    temperature: float
    n_target: float
    q_target: float  # fraction
    n_value: float
    q_value: float  # fraction
    energy: float
    walker_id: int = 0


@dataclass
class ExchangeRecord:
    cycle: int
    replica_i: int
    replica_j: int
    dim: str
    accepted: bool
    n_i: float
    n_j: float
    q_i: float  # fraction
    q_j: float  # fraction
    e_i: float  # biased total
    e_j: float  # biased total
    temperature_i: float
    temperature_j: float
    n_target_i: float
    n_target_j: float
    e_unbiased_i: float
    e_unbiased_j: float
    walker_id_i: int
    walker_id_j: int


EXCHANGE_CSV_FIELDS = [
    "cycle",
    "replica_i",
    "replica_j",
    "dim",
    "accepted",
    "temperature_i",
    "temperature_j",
    "n_target_i",
    "n_target_j",
    "N_i",
    "N_j",
    "Q_i",
    "Q_j",
    "E_i",
    "E_j",
    "E_unbiased_i",
    "E_unbiased_j",
    "walker_id_i",
    "walker_id_j",
]


def exchange_record_to_row(record: ExchangeRecord) -> dict[str, object]:
    return {
        "cycle": record.cycle,
        "replica_i": record.replica_i,
        "replica_j": record.replica_j,
        "dim": record.dim,
        "accepted": int(record.accepted),
        "temperature_i": f"{record.temperature_i:.6f}",
        "temperature_j": f"{record.temperature_j:.6f}",
        "n_target_i": f"{record.n_target_i:.6f}",
        "n_target_j": f"{record.n_target_j:.6f}",
        "N_i": f"{record.n_i:.6f}",
        "N_j": f"{record.n_j:.6f}",
        "Q_i": f"{record.q_i:.6f}",
        "Q_j": f"{record.q_j:.6f}",
        "E_i": f"{record.e_i:.6f}",
        "E_j": f"{record.e_j:.6f}",
        "E_unbiased_i": f"{record.e_unbiased_i:.6f}",
        "E_unbiased_j": f"{record.e_unbiased_j:.6f}",
        "walker_id_i": record.walker_id_i,
        "walker_id_j": record.walker_id_j,
    }


@dataclass
class RunSummary:
    elapsed_s: float
    n_temp_accepts: int
    n_temp_attempts: int
    n_q_accepts: int
    n_q_attempts: int
    exchange_log: Path | None = None
    state_log: Path | None = None
    analysis_path: Path | None = None
    rex_stats_path: Path | None = None


def write_rex_stats(
    path: str | Path,
    *,
    n_temp_accepts: int,
    n_temp_attempts: int,
    n_q_accepts: int,
    n_q_attempts: int,
    cycles_completed: int,
) -> Path:
    """Write compact RE acceptance summary (O(1) size)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    def _rate(accepted: int, attempts: int) -> float | None:
        if attempts <= 0:
            return None
        return float(accepted) / float(attempts)

    payload = {
        "temperature": {
            "attempts": int(n_temp_attempts),
            "accepted": int(n_temp_accepts),
            "rate": _rate(n_temp_accepts, n_temp_attempts),
        },
        "Q": {
            "attempts": int(n_q_attempts),
            "accepted": int(n_q_accepts),
            "rate": _rate(n_q_accepts, n_q_attempts),
        },
        "cycles_completed": int(cycles_completed),
    }
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return out


def make_temperature_ladder(temp_min: float, temp_step: float, n_temps: int) -> np.ndarray:
    return temp_min + temp_step * np.arange(n_temps, dtype=np.float64)


def make_n_targets(n_windows: int, n_step: float) -> np.ndarray:
    """Build umbrella centers along native-contact count N."""
    if n_windows <= 1:
        return np.array([0.0], dtype=np.float64)
    return n_step * np.arange(n_windows, dtype=np.float64)


def make_q_targets(n_q_windows: int, q_step: float) -> np.ndarray:
    """Build umbrella centers along fraction Q (back-compat; convert to N for bias)."""
    if n_q_windows <= 1:
        return np.array([0.0], dtype=np.float64)
    return q_step * np.arange(n_q_windows, dtype=np.float64)


def biased_energy(energy: float, n_value: float, n_target: float, k_bias: float) -> float:
    if k_bias == 0.0:
        return energy
    return energy + 0.5 * k_bias * (n_value - n_target) ** 2


def unbiased_energy(energy: float, n_value: float, n_target: float, k_bias: float) -> float:
    """Strip the harmonic N bias from a total energy that includes it."""
    if k_bias == 0.0:
        return energy
    return energy - 0.5 * k_bias * (n_value - n_target) ** 2


def evaluate_exchange_delta(
    e_a: float,
    n_a: float,
    temp_a: float,
    n_target_a: float,
    e_b: float,
    n_b: float,
    temp_b: float,
    n_target_b: float,
    k_bias: float,
) -> float:
    """Return the Metropolis log-acceptance factor for swapping configs between two replicas."""
    beta_a = 1.0 / temp_a
    beta_b = 1.0 / temp_b
    u_a_before = biased_energy(e_a, n_a, n_target_a, k_bias)
    u_b_before = biased_energy(e_b, n_b, n_target_b, k_bias)
    u_a_after = biased_energy(e_b, n_b, n_target_a, k_bias)
    u_b_after = biased_energy(e_a, n_a, n_target_b, k_bias)
    return beta_a * (u_a_after - u_a_before) + beta_b * (u_b_after - u_b_before)


def evaluate_exchange_acceptance(
    e_a: float,
    n_a: float,
    temp_a: float,
    n_target_a: float,
    e_b: float,
    n_b: float,
    temp_b: float,
    n_target_b: float,
    k_bias: float,
    rng: np.random.Generator,
) -> bool:
    delta = evaluate_exchange_delta(
        e_a, n_a, temp_a, n_target_a,
        e_b, n_b, temp_b, n_target_b,
        k_bias,
    )
    return delta <= 0.0 or rng.random() < np.exp(-delta)


def get_coords(context: mcpu_core.Context) -> np.ndarray:
    """The context's coordinates in build (topology) order, the order
    ``set_positions`` takes them in. ``get_state().coords`` is storage
    order, which differs after an ``init_only`` atom reorder."""
    return np.asarray(context.coords, dtype=np.float32)


def swap_context_coordinates(
    context_a: mcpu_core.Context,
    context_b: mcpu_core.Context,
    coords_a: np.ndarray,
    coords_b: np.ndarray,
) -> None:
    context_a.set_positions(coords_b)
    context_b.set_positions(coords_a)
    context_a.calculate_total_energy(-1)
    context_b.calculate_total_energy(-1)
    check_state_clash(context_a, "replica swap")
    check_state_clash(context_b, "replica swap")


def replica_index(temp_index: int, q_index: int, n_q_windows: int) -> int:
    return temp_index * n_q_windows + q_index


def build_grid_metadata(
    *,
    temperatures: np.ndarray | list[float] | None,
    temp_min: float,
    temp_step: float,
    n_temps: int,
    native_contact_targets: np.ndarray | list[float] | None,
    n_targets: np.ndarray | list[float] | None,
    q_targets: np.ndarray | list[float] | None,
    n_q_windows: int,
    q_step: float,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, int, int]:
    """Return temperatures and raw N/Q window specs (N conversion needs n_contacts)."""
    if temperatures is None:
        temperatures = make_temperature_ladder(temp_min, temp_step, n_temps)
    temperatures = np.asarray(temperatures, dtype=np.float64)

    n_targets_in = n_targets if n_targets is not None else native_contact_targets
    if n_targets_in is not None:
        n_targets_arr = np.asarray(n_targets_in, dtype=np.float64)
        q_targets_arr = None
        n_windows = int(n_targets_arr.size)
    elif q_targets is not None:
        n_targets_arr = None
        q_targets_arr = np.asarray(q_targets, dtype=np.float64)
        n_windows = int(q_targets_arr.size)
    else:
        n_targets_arr = None
        q_targets_arr = make_q_targets(n_q_windows, q_step)
        n_windows = int(q_targets_arr.size)

    n_temps = int(temperatures.size)
    return temperatures, n_targets_arr, q_targets_arr, n_temps, n_windows


def resolve_n_targets(
    n_targets: np.ndarray | None,
    q_targets: np.ndarray | None,
    n_contacts: float,
) -> np.ndarray:
    if n_targets is not None:
        return np.asarray(n_targets, dtype=np.float64)
    if q_targets is None:
        raise ValueError("Either n_targets or q_targets must be provided.")
    return np.asarray(q_targets, dtype=np.float64) * float(n_contacts)


def replica_seed(seed: int, replica_idx: int) -> int:
    """Per-replica deterministic integrator seed derived from the run seed."""
    return int(seed) + 1_000_003 * (replica_idx + 1)


def build_system_and_cv(
    pdb_path: str,
    *,
    reference_pdb: str,
    contact_cutoff: float,
    min_seq_sep: int,
    contact_atom_mode: str,
    q_cutoff: float | None,
    fixed_residues: list[int],
    linker_residues: list[int],
    linker_energy_mode: str,
    native_contact_pairs: Sequence[Sequence[int]] | np.ndarray | None = None,
) -> tuple[mcpu_core.System, MCPUForceField, md.Trajectory, np.ndarray, NativeContactsCV]:
    """Load the structure, build the C++ System, and attach the native-contacts
    bias force -- identical setup shared by the serial and MPI engines.

    Returns ``(system, forcefield, filtered_traj, coords_angstroms, q_cv)``.
    """
    from pymcpu.config import apply_linker_energy_mask

    traj = md.load(pdb_path)
    indices_to_keep = traj.topology.select("not element H")
    filtered_traj = traj.atom_slice(indices_to_keep)

    forcefield = MCPUForceField(filtered_traj)
    system = forcefield.create_system(filtered_traj.topology)
    coords_angstroms = forcefield.coords[0] * 10.0
    n_res = system.get_num_residues()

    apply_linker_energy_mask(
        system, linker_residues, linker_energy_mode, fixed_residues=fixed_residues
    )

    fixed_mask: np.ndarray | None = None
    if fixed_residues:
        fixed_mask = np.zeros(n_res, dtype=bool)
        for r in fixed_residues:
            if 0 <= r < n_res:
                fixed_mask[r] = True

    energy_mask: np.ndarray | None = None
    if linker_residues:
        energy_mask = np.zeros(n_res, dtype=bool)
        for r in linker_residues:
            if 0 <= r < n_res:
                energy_mask[r] = True

    ca_internal_idx = build_contact_atom_index(forcefield, mode=contact_atom_mode)
    ref_ca_xyz = reference_contact_from_pdb(reference_pdb, mode=contact_atom_mode)
    q_cv = NativeContactsCV(
        ca_internal_idx=ca_internal_idx,
        ref_ca_xyz=ref_ca_xyz,
        contact_cutoff=contact_cutoff,
        min_seq_sep=min_seq_sep,
        mode="hard",
        q_cutoff=q_cutoff,
        fixed_residue_mask=fixed_mask,
        energy_ignored_residue_mask=energy_mask,
        contact_atom_mode=contact_atom_mode,
        native_contact_pairs=native_contact_pairs,
    )
    attach_native_contacts_bias_potential(system, q_cv)

    return system, forcefield, filtered_traj, coords_angstroms, q_cv


def build_replica_simulation(
    *,
    filtered_traj: md.Trajectory,
    system: mcpu_core.System,
    temperature: float,
    seed: int,
    replica_idx: int,
    fixed_residues: list[int],
    n_res: int,
    coords_angstroms: np.ndarray,
    k_bias: float,
    n_target: float,
    step_size_rad: float,
    move_settings: Mapping[str, Any],
) -> Simulation:
    """Build one replica's Integrator + Simulation, positioned and biased,
    ready to step -- identical construction shared by the serial and MPI
    engines (only *which* replicas get built here differs between them).

    ``move_settings`` comes from :func:`pymcpu.config.normalize_move_settings`.
    """
    from pymcpu.config import configure_integrator

    integrator = mcpu_core.Integrator(float(temperature), float(step_size_rad))
    configure_integrator(integrator, **move_settings)
    if hasattr(integrator, "set_seed"):
        integrator.set_seed(replica_seed(seed, replica_idx))
    if fixed_residues:
        integrator.set_fixed_residues(fixed_residues, n_res)
    simulation = Simulation(filtered_traj.topology, system, integrator)
    simulation.context.set_positions(coords_angstroms.T.astype(np.float32))
    simulation.context.set_native_contacts_bias(k_bias, float(n_target))
    simulation.context.calculate_total_energy(-1)
    return simulation
