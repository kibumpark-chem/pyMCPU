"""One pyMCPU engine, built once and reused across many trajectory segments.

This is the object an external sampling framework holds per worker process.
Building an :class:`~pymcpu.forcefields.mcpu.MCPUForceField` reads the
parameter tables from disk (~0.5 s, independent of protein size), so
rebuilding one per segment turns a large run into a parameter-parsing
benchmark rather than a sampling one. :class:`EngineSession` defers all of
it until first use and then holds it for the life of the process.

It deliberately does **not** decide when a new session is needed. That
lazy/per-PID policy belongs to the caller, because only the caller knows
its own process model -- whether it forks workers, when, and how many. The
WESTPA add-on's propagator is a worked example: it keys a session on
``(thread, PID)`` and keeps construction cheap so the object survives being
created in a master process before workers fork.

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
from pymcpu.config import EngineSpec, apply_linker_energy_mask, configure_integrator
from pymcpu.forcefields import build_forcefield as _build_registered_forcefield
from pymcpu.forcefields import get_forcefield
from pymcpu.forcefields.base import BaseForceField
from pymcpu.forcefields.mcpu import MCPUForceField
from pymcpu.sampling.cv_factory import build_cv
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


def _forcefield_options(spec: EngineSpec, *, include_dssp: bool = True) -> dict[str, Any]:
    """Constructor options for the force field ``spec`` names.

    `param_set` and friends are top-level EngineSpec fields because they
    predate forcefield_options and existing configs set them there. They are
    MCPU's, though, so they are only forwarded to MCPU -- KORP has no
    parameter set and would reject them. An explicit entry in
    forcefield_options still wins.
    """
    options: dict[str, Any] = dict(spec.forcefield_options)
    if issubclass(get_forcefield(spec.forcefield), MCPUForceField):
        options.setdefault("param_set", spec.param_set)
        if include_dssp:
            options.setdefault("compute_dssp", spec.compute_dssp)
            options.setdefault("dssp_coil_state", spec.dssp_coil_state)
        if spec.param_dir is not None:
            options.setdefault("param_dir", spec.param_dir)
    return options


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
    forcefield = _build_registered_forcefield(
        spec.forcefield, md.load(str(spec.pdb)), _forcefield_options(spec))
    return forcefield, forcefield.output_topology


class EngineSession:
    """Lazily-built, cached pyMCPU forcefield/system/context/integrator for
    one WE sim root's configuration."""

    def __init__(self, spec: EngineSpec):
        self.spec = spec
        self._forcefield: MCPUForceField | None = None
        self._topology: md.Topology | None = None
        self._sim: Simulation | None = None
        self._cv: Any = None
        self._fingerprint: str | None = None
        self._auxref_cache: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Lazy construction
    # ------------------------------------------------------------------
    def _ensure_forcefield(self) -> MCPUForceField:
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
            self._fingerprint = compute_fingerprint(
                self.spec.pdb, self.spec.param_set, system.get_num_atoms(),
                compute_dssp=self.spec.compute_dssp, dssp_coil_state=self.spec.dssp_coil_state,
            )
        return self._sim

    # ------------------------------------------------------------------
    # Public API used by MCPUPropagator
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
        Stored as float64 they restore the run bit for bit through
        :meth:`set_coords` (except for an engine coordinate within a few 1e-6 A
        of zero); as float32 they would not for a structure the engine runs
        shifted (see ``Context.frame_offset``)."""
        return np.asarray(self._ensure_sim().context.coords, dtype=np.float64)

    def set_coords(self, coords_3xn: np.ndarray) -> None:
        sim = self._ensure_sim()
        sim.context.set_positions(np.asarray(coords_3xn, dtype=np.float64))
        sim.context.calculate_total_energy(-1)
        check_state_clash(sim.context, "EngineSession.set_coords")

    def set_seed(self, seed: int) -> None:
        self._ensure_sim().integrator.set_seed(int(seed))

    def get_rng_state(self) -> str:
        """Exact MC RNG stream state — only ever saved/restored when
        ``store_rng_state: true`` (debug single-segment replay). Restoring
        this across a *split* (multiple children sharing one parent state)
        would make every child bitwise-identical; see
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
        """Resolve a basis/initial-state ``auxref`` to engine-order
        coordinates ``(3, n_atoms)`` float64 Angstrom. Supports:

        * ``.npz`` -- anything carrying a ``coords`` array, which includes
          a restart state written by an external sampler.
        * ``.chk`` — a :func:`pymcpu.checkpointing.load_checkpoint` payload
          (the endpoint of an existing ``FoldingRunner``/``ReplicaExchange``
          production run) -- lets an external sampler start from prior
          pyMCPU output with no new API.
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
            # restart format. The add-on's own segment-state files satisfy
            # this by construction, which is what keeps the interop working
            # without core knowing anything about them.
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
            local_ff = _build_registered_forcefield(
                self.spec.forcefield, md.load(str(path)),
                _forcefield_options(self.spec, include_dssp=False))
            coords = (local_ff.coords[0] * 10.0).T.astype(np.float32)
        else:
            raise ValueError(
                f"unsupported basis-state auxref extension {suffix!r} for {auxref!r}; "
                "expected .npz, .chk, or .pdb"
            )

        coords = np.asarray(coords, dtype=np.float64)
        self._auxref_cache[auxref] = coords
        return coords.copy()
