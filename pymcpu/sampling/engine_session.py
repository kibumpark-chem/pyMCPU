"""One pyMCPU engine, built once and reused across many trajectory segments.

This is the object an external sampling framework holds per worker process.
Building the engine takes from under a second for a small protein to several
seconds for a large one, so rebuilding it for every segment would dominate a
large run. :class:`EngineSession` defers all of it until first use and then
holds it for the life of the process.

It deliberately does **not** decide when a new session is needed. That
lazy/per-PID policy belongs to the caller, because only the caller knows
its own process model -- whether it forks workers, when, and how many. A
caller that forks typically keys a session on ``(thread, PID)``. Creating
one builds nothing, so the object can be made in the main process before the
fork, as long as each worker builds its own engine.

**It never seeds the integrator.** The caller seeds each segment itself,
from :func:`pymcpu.sampling.derive_seed`; see that function for why
inheriting a parent's RNG stream across a clone silently ruins the
statistics. This is also why :class:`~pymcpu.config.EngineSpec` has no
``seed`` field to read.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import mdtraj as md
import numpy as np

from pymcpu import mcpu_core
from pymcpu.config import (
    EngineSpec,
    apply_linker_energy_mask,
    check_move_weights,
    configure_integrator,
)
from pymcpu.forcefields import load_forcefield
from pymcpu.forcefields.base import BaseForceField
from pymcpu.sampling.cv_factory import build_cv
from pymcpu.sampling.replica_exchange_core import get_frame_offset
from pymcpu.simulation import Simulation, check_state_clash

__all__ = ["EngineSession", "build_forcefield", "compute_fingerprint"]


def compute_fingerprint(
    pdb_path: str,
    param_set: str,
    n_atoms: int,
    *,
    compute_dssp: bool = False,
    dssp_coil_state: str = "C",
) -> str:
    """Short hash identifying what an engine was built from.

    Lets a restart state loaded against a different system fail loudly
    instead of producing silent nonsense -- the coordinates would have the
    wrong atom count, or the right count in a different order.
    """
    payload = (
        f"{Path(pdb_path).resolve()}|{param_set}|{int(n_atoms)}|"
        f"{bool(compute_dssp)}|{dssp_coil_state}"
    ).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def build_forcefield(spec: EngineSpec) -> tuple[BaseForceField, md.Topology]:
    """Load ``spec.pdb`` and build the force field it names.

    Shared by :class:`EngineSession` (which also needs a full
    ``Context``/``Integrator``) and a consumer that needs only the force field
    -- to read off a CV's dimensionality and labels, say -- and must not pay
    for a ``Context``. Analysis tools typically fall in that category.

    Returns the force field and the topology the engine actually simulates,
    which is *not* necessarily the input's: ``KORPForceField`` drops
    sidechains, so trajectories written from it must be read back against
    the returned topology.
    """
    forcefield = _load_spec_forcefield(spec, spec.pdb)
    return forcefield, forcefield.output_topology


def _load_spec_forcefield(
    spec: EngineSpec, structure: Any, *, compute_dssp: bool | None = None
) -> BaseForceField:
    return load_forcefield(
        structure, spec.forcefield, spec.forcefield_options,
        param_set=spec.param_set, param_dir=spec.param_dir,
        compute_dssp=spec.compute_dssp if compute_dssp is None else compute_dssp,
        dssp_coil_state=spec.dssp_coil_state,
    )


class EngineSession:
    """Lazily built, cached pyMCPU force field, system, context and
    integrator for one :class:`~pymcpu.config.EngineSpec`."""

    def __init__(self, spec: EngineSpec):
        self.spec = spec
        self._forcefield: BaseForceField | None = None
        self._topology: md.Topology | None = None
        self._sim: Simulation | None = None
        self._cv: Any = None
        self._fingerprint: str | None = None
        self._auxref_cache: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Lazy construction
    # ------------------------------------------------------------------
    def _ensure_forcefield(self) -> BaseForceField:
        if self._forcefield is None:
            self._forcefield, self._topology = build_forcefield(self.spec)
        return self._forcefield

    def _ensure_cv(self) -> Any:
        if self._cv is None:
            self._cv = build_cv(self.spec.cv, self._ensure_forcefield())
        return self._cv

    def _ensure_sim(self) -> Simulation:
        if self._sim is None:
            ff = self._ensure_forcefield()
            check_move_weights(ff, self.spec.move_weights, self.spec.forcefield)
            system = ff.create_system(self._topology)
            apply_linker_energy_mask(
                system,
                list(self.spec.linker_residues),
                self.spec.linker_energy_mode,
                fixed_residues=list(self.spec.fixed_residues),
            )
            integrator = mcpu_core.Integrator(self.spec.temperature, self.spec.step_size_rad)
            configure_integrator(
                integrator,
                move_weights=self.spec.move_weights,
                sidechain_move_mode=self.spec.sidechain_move_mode,
                pivot_rama_probability=self.spec.pivot_rama_probability,
                pivot_rama_schedule=self.spec.pivot_rama_schedule,
            )
            if self.spec.fixed_residues:
                integrator.set_fixed_residues(list(self.spec.fixed_residues), system.get_num_residues())

            self._sim = Simulation(self._topology, system, integrator)
            coords0 = (ff.coords[0] * 10.0).T.astype(np.float32)
            self._sim.context.set_positions(coords0)
            self._sim.context.calculate_total_energy(-1)
            check_state_clash(self._sim.context, "start structure")
            self._fingerprint = compute_fingerprint(
                self.spec.pdb, self.spec.param_set, system.get_num_atoms(),
                compute_dssp=self.spec.compute_dssp, dssp_coil_state=self.spec.dssp_coil_state,
            )
        return self._sim

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def fingerprint(self) -> str:
        self._ensure_sim()
        assert self._fingerprint is not None
        return self._fingerprint

    def compute_cv(self, coords_3xn: np.ndarray) -> np.ndarray:
        return np.asarray(self._ensure_cv()(coords_3xn), dtype=np.float64)

    def coords(self) -> np.ndarray:
        """``(3, n_atoms)`` float64 Angstrom, in the input structure's frame.
        Stored as float64, with :meth:`frame_offset`, they restore the run bit
        for bit through :meth:`set_coords` (except for an engine coordinate
        within a few 1e-6 A of zero); as float32 they would not for a
        structure the engine runs shifted (see ``Context.frame_offset``)."""
        return np.asarray(self._ensure_sim().context.coords, dtype=np.float64)

    def frame_offset(self) -> np.ndarray:
        """The engine frame's offset, float64, shape (3,), in Angstrom. It
        moves with the chain (``Context.recenter``), so store it with
        :meth:`coords` and pass both to :meth:`set_coords` to continue a run
        bit for bit."""
        return get_frame_offset(self._ensure_sim().context)

    def set_coords(self, coords_3xn: np.ndarray, *, frame_offset: Any = None) -> None:
        """Place ``(3, n_atoms)`` coordinates in Angstrom. ``frame_offset``,
        from :meth:`frame_offset` when the coordinates were saved, re-enters
        them bit for bit; without it they enter in the session's current
        frame and are rounded once to float32 there."""
        sim = self._ensure_sim()
        sim.context.set_positions(
            np.asarray(coords_3xn, dtype=np.float64),
            frame_offset=None if frame_offset is None else np.asarray(frame_offset, dtype=np.float64),
        )
        sim.context.calculate_total_energy(-1)
        check_state_clash(sim.context, "EngineSession.set_coords")

    def set_seed(self, seed: int) -> None:
        self._ensure_sim().integrator.set_seed(int(seed))

    def get_rng_state(self) -> str:
        """Exact MC random state, for replaying a single trajectory.
        Restoring it across a clone (several children sharing one parent
        state) would make every child bitwise identical; see
        :func:`pymcpu.sampling.derive_seed`."""
        return str(self._ensure_sim().integrator.get_rng_state())

    def restore_rng_state(self, rng_state: str) -> None:
        self._ensure_sim().integrator.set_rng_state(rng_state)

    def step(self, n_steps: int) -> None:
        self._ensure_sim().step(int(n_steps))

    @property
    def current_step(self) -> int:
        return int(self._ensure_sim().current_step)

    @current_step.setter
    def current_step(self, value: int) -> None:
        self._ensure_sim().current_step = int(value)

    def coords_from_auxref(self, auxref: str) -> np.ndarray:
        """Read starting coordinates from a file (``auxref``), in engine
        order, as ``(3, n_atoms)`` float64 Angstrom. Supports:

        * ``.npz`` -- anything carrying a ``coords`` array, which includes
          a restart state written by an external sampler.
        * ``.chk`` — a :func:`pymcpu.checkpointing.load_checkpoint` payload
          (the endpoint of an existing ``FoldingRunner``/``ReplicaExchange``
          production run; for replica exchange, its first replica) -- lets an
          external sampler start from prior pyMCPU output with no new API.
          The checkpoint's frame offset is not returned, so a state saved far
          from the origin is rounded once to float32 when it is placed.
        * ``.pdb`` -- an arbitrary starting structure, mapped into engine atom
          order by a throwaway force field of the session's own kind
          (``spec.forcefield``) built from it. This assumes the PDB is the
          same protein/topology as ``spec.pdb``; a different atom count is
          caught when the coordinates are loaded.

        Results are cached per resolved path (a starting state is typically read
        many times, once per new or recycled trajectory).
        """
        if auxref in self._auxref_cache:
            return self._auxref_cache[auxref].copy()

        path = Path(auxref)
        suffix = path.suffix.lower()
        if suffix == ".npz":
            # Any .npz carrying a `coords` array, NOT a specific framework's
            # restart format. Restart files an external sampler writes
            # satisfy this by construction, so pyMCPU needs to know nothing
            # about their format.
            with np.load(path) as payload:
                if "coords" not in payload:
                    raise ValueError(
                        f"{auxref!r} has no 'coords' array (found "
                        f"{sorted(payload.files)})"
                    )
                coords = np.asarray(payload["coords"], dtype=np.float64)
            if coords.ndim != 2 or 3 not in coords.shape:
                raise ValueError(
                    f"{auxref!r} 'coords' has shape {coords.shape}; expected "
                    f"(3, n_atoms) or (n_atoms, 3)"
                )
            if coords.shape[0] != 3:
                coords = coords.T
        elif suffix == ".chk":
            from pymcpu.checkpointing import load_checkpoint

            data = load_checkpoint(path)
            coords = np.asarray(data["replica_coords"][0], dtype=np.float64)
            if coords.ndim == 2 and coords.shape[0] != 3 and coords.shape[1] == 3:
                coords = coords.T
        elif suffix == ".pdb":
            # The session's own force field, so the coordinates come out in
            # its layout (KORP keeps only the backbone). DSSP is skipped: this
            # throwaway force field is used only for its coordinates.
            local_ff = _load_spec_forcefield(self.spec, path, compute_dssp=False)
            coords = (local_ff.coords[0] * 10.0).T.astype(np.float32)
        else:
            raise ValueError(
                f"unsupported basis-state auxref extension {suffix!r} for {auxref!r}; "
                "expected .npz, .chk, or .pdb"
            )

        coords = np.asarray(coords, dtype=np.float64)
        self._auxref_cache[auxref] = coords
        return coords.copy()
