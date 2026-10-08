"""High-level Simulation conductor (OpenMM-style)."""

from __future__ import annotations

import logging
import os
from typing import Sequence

import mdtraj as md

from pymcpu import mcpu_core

logger = logging.getLogger(__name__)


def _clash_is_fatal() -> bool:
    """Whether an accepted-state steric clash aborts the run (default) or is
    counted and warned about (MCPU_CLASH_FATAL=0)."""
    return os.environ.get("MCPU_CLASH_FATAL", "1") != "0"


_FULL_ENERGY_EVERY_RENAMED = (
    "Simulation.full_energy_every was renamed to full_energy_every_steps, "
    "which counts MC steps rather than step() calls (default 1_000_000)"
)


class StericClashError(RuntimeError):
    """An accepted state contains a hard-core overlap.

    Moves are tested against each pair's hard-core cutoff, so no move that
    re-measures a pair puts it under that cutoff. A rigid pivot carries most
    pairs inside its segment without re-measuring them, and float rounding
    moves such a pair a few 1e-6 A per carry, adding up over carries as a
    random walk. A whole state is therefore judged against cutoffs
    ``mcpu_core.STATE_CLASH_BUFFER_A`` (0.001 A) looser, and a rigid pivot
    that would carry a pair under that looser cutoff is rejected. A clash here
    therefore means coordinates that did not come from a move
    (``set_positions``, a restore, a start structure other than the one the
    force field was built from) or a pair a delta path missed.

    The integrator raises it too, before the first move of a run, when the
    recompute it does after the coordinates or the energy definition changed
    finds an overlap, e.g. a cleared ``ignore_all`` mask whose residues
    overlap the rest of the chain. ``MCPU_CLASH_FATAL`` does not apply there.

    Fatal by default, because the previous silent behaviour let
    ``weight * 99999`` flow into the REMD Metropolis criterion as if it were an
    energy. Set ``MCPU_CLASH_FATAL=0`` to count and warn instead:
    Context::calculate_total_energy does not cache the sentinel, so continuing
    costs a single exchange attempt on a stale energy.
    Simulation.steric_clash_events counts occurrences either way.
    """


def check_state_clash(context: mcpu_core.Context, where: str) -> bool:
    """Check coordinates that did not come from a move for a hard-core overlap.

    For a state just set with ``set_positions`` and recomputed with
    ``calculate_total_energy(-1)`` (it reads that recompute's verdict): a
    checkpoint restore, a replica swap, coordinates a caller supplies. Raises
    :class:`StericClashError`; with ``MCPU_CLASH_FATAL=0`` it logs a warning
    and returns True instead. The recompute keeps the previous energy when it
    finds a clash, so ``current_energy`` stays stale until the overlap is
    gone. Returns False for a clean state.
    """
    if not context.has_steric_clash():
        return False
    detail = (
        f"steric clash at {where}: a pair is more than 0.001 A "
        "(STATE_CLASH_BUFFER_A) under its hard-core cutoff. No move can do "
        "that, so the coordinates came in that way, or a delta path missed "
        "the pair where they were produced."
    )
    if _clash_is_fatal():
        raise StericClashError(detail)
    logger.warning("%s (MCPU_CLASH_FATAL=0: continuing on the previous energy)", detail)
    return True


class Simulation:
    """Bind topology + system + integrator and drive MC steps with reporters.

    Parameters
    ----------
    topology :
        MDTraj topology for the simulated molecule.
    system :
        C++ ``System`` holding forcefield parameters and layout.
    integrator :
        C++ ``Integrator`` used to advance the state.

    Attributes
    ----------
    context :
        C++ ``Context`` owning the current configuration.
    reporters :
        Python-side list of attached reporters. Synced to the C++ integrator
        on every :meth:`step` (and on add/remove helper calls).
    """

    def __init__(
        self,
        topology: md.Topology,
        system: mcpu_core.System,
        integrator: mcpu_core.Integrator,
    ):
        self.topology = topology
        self.system = system
        self.context = mcpu_core.Context(self.system)
        self.integrator = integrator
        self.reporters: list = []
        self._current_step = 0
        # Full O(N^2) energy recompute cadence, in MC steps (checked after each
        # step() call). The running total is kept from double sums of the
        # accepted moves' deltas and stays within ~1e-10 of a full recompute
        # over 1e7 steps, so the recompute is a check (a clash or a pair a delta
        # path missed), not a correction. One costs 2-6 ms at 270-415 residues.
        # The Integrator also recomputes on entry whenever the coordinates or
        # the energy definition changed (Context.energy_resyncs).
        self.full_energy_every_steps = 1_000_000
        # Warn if the incremental energy has drifted from a full recompute by
        # more than this (absolute). Only checked on recomputes. Rounding
        # stays far below it at any run length, so a warning means a pair the
        # incremental path missed or a stale cache, not precision.
        self.energy_drift_warn_atol = 1e-3
        self._steps_since_full_energy = 0
        self.steric_clash_events = 0

    @property
    def full_energy_every(self):
        raise AttributeError(_FULL_ENERGY_EVERY_RENAMED)

    @full_energy_every.setter
    def full_energy_every(self, value):
        raise AttributeError(_FULL_ENERGY_EVERY_RENAMED)

    def describe(self) -> str:
        """Return a human-readable summary of the simulation configuration."""
        lines = [
            f"Simulation: {self.system.get_num_atoms()} atoms, "
            f"{self.system.get_num_residues()} residues",
            f"Reporters: {len(self.reporters)}",
        ]
        fixed = self.integrator.get_fixed_residues()
        if fixed:
            lines.append(f"Fixed residues: {len(fixed)}")
        return "\n".join(lines)

    def set_fixed_residues(self, residues: Sequence[int]) -> None:
        """Mark residues as fixed (0-based engine indices)."""
        n_res = int(self.system.get_num_residues())
        self.integrator.set_fixed_residues(list(residues), n_res)

    def get_fixed_residues(self) -> list[int]:
        """Return currently fixed residue indices."""
        return list(self.integrator.get_fixed_residues())

    def clear_fixed_residues(self) -> None:
        """Clear all fixed-residue constraints."""
        self.integrator.clear_fixed_residues()

    # ------------------------------------------------------------------
    # Reporters
    # ------------------------------------------------------------------
    def _sync_reporters(self) -> None:
        """Make the C++ integrator reporter list match ``self.reporters``."""
        self.integrator.clear_reporters()
        for reporter in self.reporters:
            self.integrator.add_reporter(reporter)

    def add_reporter(self, reporter: mcpu_core.Reporter) -> None:
        """Attach a reporter. Raises ``TypeError`` for anything else."""
        if not isinstance(reporter, mcpu_core.Reporter):
            raise TypeError(
                f"reporter must be a pymcpu Reporter, got {type(reporter)!r}"
            )
        self.reporters.append(reporter)
        self.integrator.add_reporter(reporter)

    def remove_reporter(self, reporter: mcpu_core.Reporter) -> None:
        """Remove a reporter from the Python list and resync the C++ integrator."""
        self.reporters.remove(reporter)
        self._sync_reporters()

    def clear_reporters(self) -> None:
        """Drop all reporters from Python and C++."""
        self.reporters.clear()
        self.integrator.clear_reporters()

    def add_xtc_reporter(
        self,
        path: str,
        interval: int,
        inverse_mapping: Sequence[int] | None = None,
    ) -> mcpu_core.XtcReporter:
        """Write a trajectory frame every ``interval`` steps, and return the reporter.

        Pass the force field's ``inverse_mapping`` to write the frames in the
        topology's atom order.
        """
        mapping = list(inverse_mapping) if inverse_mapping is not None else []
        reporter = mcpu_core.XtcReporter(str(path), int(interval), mapping)
        self.add_reporter(reporter)
        return reporter

    def add_energy_reporter(
        self, path: str, interval: int
    ) -> mcpu_core.EnergyReporter:
        """Write a CSV row of energies and move counts every ``interval`` steps.

        Returns the reporter.
        """
        reporter = mcpu_core.EnergyReporter(str(path), int(interval))
        self.add_reporter(reporter)
        return reporter

    def add_simulation_reporter(
        self, interval: int
    ) -> mcpu_core.SimulationReporter:
        """Print the move counts and the energy every ``interval`` steps.

        Returns the reporter.
        """
        reporter = mcpu_core.SimulationReporter(int(interval))
        self.add_reporter(reporter)
        return reporter

    # ------------------------------------------------------------------
    # Propagation
    # ------------------------------------------------------------------
    def step(self, n_steps: int) -> None:
        """Run ``n_steps`` Monte Carlo steps.

        Step numbers keep counting across calls, so reporters see the
        global step.
        """
        self._sync_reporters()
        offset = int(self._current_step)
        self.integrator.run(self.context, int(n_steps), int(offset))
        self._current_step = offset + int(n_steps)
        self._steps_since_full_energy += int(n_steps)
        if self._steps_since_full_energy >= int(self.full_energy_every_steps):
            self.recompute_energy()

    def recompute_energy(self) -> float:
        """Recompute the total energy in full and check the state.

        Replaces ``current_energy`` with the full recompute and returns it.
        :meth:`step` calls this every :attr:`full_energy_every_steps` steps, and
        the folding and replica-exchange drivers before every checkpoint save.
        A hard-core overlap raises :class:`StericClashError` (with
        ``MCPU_CLASH_FATAL=0`` it is counted and warned about instead), and a
        running total more than :attr:`energy_drift_warn_atol` away from the
        recompute logs a warning.
        """
        self._steps_since_full_energy = 0
        incremental = float(self.context.get_state().current_energy)
        raw = self.context.calculate_total_energy(-1)
        # A steric clash in the ACCEPTED state is never tolerated (see
        # StericClashError): no move can make one, so it means coordinates
        # from outside the moves or a pair a delta path missed. Fail loudly:
        # the previous behaviour silently fed weight*99999 into the REMD
        # Metropolis criterion, which on p19.14.3 corrupted 41% of the top
        # rung's cycles while looking like nothing more than a large energy.
        if self.context.has_steric_clash():
            # Context::calculate_total_energy does not cache the clash
            # sentinel, so continuing costs one exchange attempt using a
            # slightly stale energy. MCPU_CLASH_FATAL=0 counts and warns
            # instead; the default stays fatal so reproduction runs and CI stop
            # at the first occurrence.
            self.steric_clash_events += 1
            _detail = (
                "steric clash in the ACCEPTED state after "
                f"{self._current_step} steps: the full recompute finds a pair "
                "more than 0.001 A (STATE_CLASH_BUFFER_A) under its hard-core "
                "cutoff.\n"
                f"  incremental energy : {incremental:.6f}\n"
                f"  full recompute     : {raw:.6f}  (~weight * 99999 sentinel)\n"
                f"  steric_rejected so far: {self.integrator.get_steric_rejected()}\n"
                "No move can do that: moves are tested against the cutoff itself, "
                "and a rigid pivot whose rounding would carry a pair more than "
                "0.001 A under it is rejected. Either the coordinates came in that way "
                "(set_positions, a restore, a start structure other than the force "
                "field's) or a delta path missed the pair."
            )
            if _clash_is_fatal():
                raise StericClashError(_detail)
            logger.warning(
                "%s\n  (MCPU_CLASH_FATAL=0: continuing; event #%d this run)",
                _detail, self.steric_clash_events,
            )
        exact = float(self.context.get_state().current_energy)
        drift = abs(exact - incremental)
        if drift > float(self.energy_drift_warn_atol):
            logger.warning(
                "energy drift: incremental %.6f vs recomputed %.6f "
                "(|d|=%.3e > %.3e) after %d steps -- the incremental "
                "delta-E path and the full recompute disagree",
                incremental, exact, drift,
                float(self.energy_drift_warn_atol), self._current_step,
            )
        return exact

    # ------------------------------------------------------------------
    # Last-move accessors (crash / failure snapshot helpers)
    # ------------------------------------------------------------------
    @property
    def last_move_kind(self) -> str:
        """Kind of the last proposed MC move (Pivot/KIC/Sidechain/Other)."""
        try:
            return str(self.integrator.last_move_kind())
        except Exception:
            return "unknown"

    @property
    def last_move_is_rigid(self) -> bool:
        """Whether the last proposal was marked rigid."""
        try:
            return bool(self.integrator.last_move_is_rigid())
        except Exception:
            return False

    @property
    def last_moved_indices(self) -> list[int]:
        """Atom indices moved in the last proposal."""
        try:
            return list(self.integrator.last_moved_indices())
        except Exception:
            return []

    @property
    def last_delta_energy(self) -> float:
        """ΔE of the last accepted move (0 if rejected/invalid)."""
        try:
            return float(self.integrator.last_delta_energy())
        except Exception:
            return 0.0

    @property
    def current_step(self) -> int:
        """Best-effort step counter (reporters / REMD may track separately)."""
        return int(self._current_step)

    @current_step.setter
    def current_step(self, value: int) -> None:
        self._current_step = int(value)

    def flush_reporters(self) -> None:
        """Flush any attached reporters that expose a ``flush()`` method."""
        for reporter in self.reporters:
            flush = getattr(reporter, "flush", None)
            if callable(flush):
                flush()
