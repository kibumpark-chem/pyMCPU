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


class StericClashError(RuntimeError):
    """An accepted state contains a hard-core overlap.

    Moves are tested against each pair's hard-core cutoff, so no accepted move
    puts a pair under it. A rigid pivot carries the pairs inside its segment
    without re-checking them, and float rounding can leave such a pair a few
    1e-6 A under its cutoff, so a whole state is judged against cutoffs
    ``mcpu_core.STATE_CLASH_BUFFER_A`` (0.001 A) looser. A clash here therefore
    means coordinates that did not come from a move (``set_positions``, a
    restore, a start structure other than the one the force field was built
    from) or a pair a delta path missed.

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
        # Full O(N^2) energy recompute cadence, in step() calls. 1 = every call
        # (the historical behaviour). Raise it to trade an exact `current_energy`
        # for the incrementally-maintained one, which tests/physics/
        # test_energy_consistency.py pins to within 1e-3 of a full recompute.
        # At actin one recompute is ~36.8 ms; at a 10k-step exchange interval the
        # pair of them was ~5.9% of wall (~2 h per 1e9-step replica).
        self.full_energy_every = 1
        # Warn if the incremental energy has drifted from a full recompute by
        # more than this (absolute). Only checked on recompute cycles.
        self.energy_drift_warn_atol = 1e-3
        self._steps_since_full_energy = 0
        self.steric_clash_events = 0
        # Whether `current_energy` has been seeded for this Simulation. See the
        # note in step(): the three paths that normally keep it exact on entry
        # all require a PREVIOUS cycle, so none of them covers the first call.
        self._energy_seeded = False

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
        """Attach a reporter (no kwargs — C++ accepts only the reporter object)."""
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
        mapping = list(inverse_mapping) if inverse_mapping is not None else []
        reporter = mcpu_core.XtcReporter(str(path), int(interval), mapping)
        self.add_reporter(reporter)
        return reporter

    def add_energy_reporter(
        self, path: str, interval: int
    ) -> mcpu_core.EnergyReporter:
        reporter = mcpu_core.EnergyReporter(str(path), int(interval))
        self.add_reporter(reporter)
        return reporter

    def add_simulation_reporter(
        self, interval: int
    ) -> mcpu_core.SimulationReporter:
        reporter = mcpu_core.SimulationReporter(int(interval))
        self.add_reporter(reporter)
        return reporter

    # ------------------------------------------------------------------
    # Propagation
    # ------------------------------------------------------------------
    def step(self, n_steps: int) -> None:
        """Advance the simulation by ``n_steps`` Monte Carlo steps.

        Passes ``step_offset=current_step`` into the integrator so reporters
        see a monotonically increasing global step across REMD cycles.
        """
        self._sync_reporters()
        # A pre-run full O(N^2) recompute used to sit here unconditionally. It
        # was dropped as redundant, because `current_energy` is already exact on
        # entry via three independent paths: the post-run recompute below
        # (previous cycle), swap_context_coordinates() after an accepted REMD
        # exchange, and the checkpoint-restore path. That halved the per-cycle
        # full-energy cost.
        #
        # But all three require a PREVIOUS cycle, so none covers the FIRST call
        # on a fresh Simulation. `Context::set_positions` does not seed the
        # running total, so `current_energy` was still 0.0 there and the
        # incremental accumulator stayed off by exactly the starting energy for
        # the rest of the run -- measured at 14.706589 on the 1UAO quickstart,
        # constant in step count (10k steps drift only 2.9e-5 once seeded).
        # Accept bits are unaffected (Metropolis uses dE, not the total), so
        # this only ever corrupted REPORTED energies -- but that included the
        # energy-drift warning the documented quickstart printed, and the value
        # attempt_exchange() reads for REMD acceptance on a first cycle.
        # Every scripts/parity_*.py already called calculate_total_energy(-1)
        # by hand after set_positions for this reason.
        #
        # Seeding ONCE per Simulation restores correctness without giving back
        # the per-cycle win.
        if not self._energy_seeded:
            self.context.calculate_total_energy(-1)
            self._energy_seeded = True
        offset = int(self._current_step)
        self.integrator.run(self.context, int(n_steps), int(offset))

        # The post-run recompute IS load-bearing: attempt_exchange() reads
        # current_energy for the REMD acceptance criterion. Keeping it every
        # cycle preserves the historical exact value; raising
        # `full_energy_every` falls back to the incremental value between
        # recomputes and turns this into the periodic drift check that a
        # running-E_total scheme is supposed to have.
        self._steps_since_full_energy += 1
        if self._steps_since_full_energy >= max(1, int(self.full_energy_every)):
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
                # Since Context::calculate_total_energy does not cache the clash
                # sentinel, continuing costs one exchange attempt using a slightly
                # stale energy. MCPU_CLASH_FATAL=0 counts and warns instead; the
                # default stays fatal so reproduction runs and CI stop at the first
                # occurrence.
                self.steric_clash_events += 1
                _detail = (
                    "steric clash in the ACCEPTED state after "
                    f"{offset + int(n_steps)} steps: the full recompute finds a pair "
                    "more than 0.001 A (STATE_CLASH_BUFFER_A) under its hard-core "
                    "cutoff.\n"
                    f"  incremental energy : {incremental:.6f}\n"
                    f"  full recompute     : {raw:.6f}  (~weight * 99999 sentinel)\n"
                    f"  steric_rejected so far: {self.integrator.get_steric_rejected()}\n"
                    "No move can do that: moves are tested against the cutoff itself, "
                    "and rounding moves a pair a rigid pivot carries by a few 1e-6 A at "
                    "most near the origin. Either the coordinates came in that way "
                    "(set_positions, a restore, a start structure other than the force "
                    "field's), they lie thousands of A from the origin, where that "
                    "rounding is far larger (centre the structure), or a delta path "
                    "missed the pair."
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
                    float(self.energy_drift_warn_atol), offset + int(n_steps),
                )
        self._current_step = offset + int(n_steps)

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
