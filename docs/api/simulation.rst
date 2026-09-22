Simulation
==========

:class:`~pymcpu.Simulation` is the primary entry point: it binds a
topology, a system and an integrator, owns the ``Context`` that holds
the current configuration, keeps the reporter list in sync with the C++
integrator, and advances the chain with
:meth:`~pymcpu.Simulation.step`. Driving the engine through it is
preferred over using ``Context`` and ``Integrator`` directly, because it
takes care of the energy bookkeeping described below.

Composing a run
---------------

.. code-block:: python

   import mdtraj as md
   import numpy as np
   import pymcpu as mc

   traj = md.load("examples/data/1uao.pdb")
   heavy = traj.atom_slice(traj.topology.select("not element H"))

   ff = mc.MCPUForceField(heavy, param_set="mcpu_v1")
   system = ff.create_system(heavy.topology)

   integrator = mc.Integrator(temperature=0.6, step_size_rad=0.1)
   integrator.set_seed(42)

   sim = mc.Simulation(heavy.topology, system, integrator)

   # nm -> Angstrom, (n, 3) -> (3, n), float32.
   sim.context.set_positions((ff.coords[0] * 10.0).T.astype(np.float32))

   sim.add_xtc_reporter("traj.xtc", interval=100,
                        inverse_mapping=ff.inverse_mapping)
   sim.add_energy_reporter("energy.csv", interval=100)

   sim.step(1000)
   print(sim.describe())
   print(float(sim.context.get_state().current_energy))

Temperature here is a dimensionless reduced parameter (roughly 0.3 cold
to 0.6 hot), and energies are unitless sums of knowledge-based table
entries.

.. note::
   ``Simulation`` constructs its own ``Context`` from the system and
   exposes it as :attr:`~pymcpu.Simulation.context`; you do not pass one
   in, and ``Context`` itself takes only a system. Positions are set on
   that context, not on the simulation.

Class reference
---------------

.. autoclass:: pymcpu.Simulation
   :members:
   :member-order: bysource
   :special-members: __init__

Reporter helpers
----------------

:meth:`~pymcpu.Simulation.add_reporter` accepts any
``mcpu_core.Reporter`` and raises ``TypeError`` for anything else. The
three factory methods construct a reporter, attach it, and return it:

* ``add_xtc_reporter(path, interval, inverse_mapping=None)`` --
  trajectory output. Pass
  :attr:`~pymcpu.MCPUForceField.inverse_mapping` so frames are written
  in the original topology order; without it the engine's internal atom
  order is written.
* ``add_energy_reporter(path, interval)`` -- per-frame energy log.
* ``add_simulation_reporter(interval)`` -- progress/status reporter with
  no output file.

The Python-side list is :attr:`~pymcpu.Simulation.reporters`; it is
pushed to the C++ integrator on every :meth:`~pymcpu.Simulation.step` as
well as on the add/remove helpers, so mutating the list directly still
takes effect at the next step.
:meth:`~pymcpu.Simulation.flush_reporters` flushes those reporters that
expose a ``flush()`` method.

Energy bookkeeping
------------------

The engine maintains a running total energy incrementally from accepted
move deltas, and periodically checks it against a full O(N^2)
recompute. Three attributes control that, none of which appears in the
class docstring:

.. py:attribute:: pymcpu.Simulation.full_energy_every
   :type: int
   :value: 1

   How often, measured in :meth:`~pymcpu.Simulation.step` calls, to do
   the full recompute. ``1`` (the default) recomputes after every call,
   which keeps ``context.get_state().current_energy`` exact -- the value
   replica exchange reads for its acceptance test. Raising it trades
   that exactness for speed and turns the recompute into a periodic
   drift check.

.. py:attribute:: pymcpu.Simulation.energy_drift_warn_atol
   :type: float
   :value: 0.001

   Absolute tolerance for the incremental-versus-recomputed comparison.
   A larger disagreement logs a warning. Only checked on recompute
   cycles, so raising :attr:`full_energy_every` also makes this check
   less frequent.

.. py:attribute:: pymcpu.Simulation.steric_clash_events
   :type: int
   :value: 0

   Count of hard-core overlaps found in an accepted state (see below).

Seeding the running total
~~~~~~~~~~~~~~~~~~~~~~~~~

``Context.set_positions`` does **not** seed the running total energy: it
leaves ``current_energy`` at ``0.0``, and an incremental accumulator
started from zero stays off by exactly the starting energy for the rest
of the run. Accept/reject decisions are unaffected (the Metropolis test
uses the move delta, not the total), but every reported energy is wrong,
including replica-exchange acceptance on a first cycle.

:meth:`~pymcpu.Simulation.step` therefore seeds the total once per
``Simulation`` object, on its first call. Code that drives a raw
``Context`` instead must do it by hand::

   context.set_positions(positions)
   context.calculate_total_energy(-1)   # seed the running total

The one-time seeding is why the recipe above does not call
``calculate_total_energy`` itself; calling it anyway is harmless.

Steric clashes in accepted states
---------------------------------

The integrator rejects any proposal whose incremental delta reports a
steric clash, so an accepted state cannot contain one. If the periodic
full recompute nevertheless finds a hard-core overlap, that is a
detection gap in the delta path, and
:meth:`~pymcpu.Simulation.step` raises:

.. autoexception:: pymcpu.simulation.StericClashError

:attr:`~pymcpu.Simulation.steric_clash_events` is incremented either
way. Setting the environment variable ``MCPU_CLASH_FATAL=0`` downgrades
the exception to a counted warning, which is the right choice for a long
production run; the default is fatal so that reproduction runs and CI
stop at the first occurrence.

Inspecting the last move
------------------------

:attr:`~pymcpu.Simulation.last_move_kind`,
:attr:`~pymcpu.Simulation.last_move_is_rigid`,
:attr:`~pymcpu.Simulation.last_moved_indices` and
:attr:`~pymcpu.Simulation.last_delta_energy` forward to the integrator
and are meant for failure snapshots and debugging: each returns a
neutral value (``"unknown"``, ``False``, ``[]``, ``0.0``) rather than
raising if the underlying call fails.
:attr:`~pymcpu.Simulation.current_step` is a best-effort global step
counter, and is writable so that replica-exchange and
checkpoint-restore paths can realign it.

.. seealso::

   :doc:`forcefield`
       Builds the ``system`` argument and the coordinates to load.
   :doc:`integrator`
       The move set, step sizes and acceptance statistics.
   :doc:`context`
       Positions, energies and the state object.
   :doc:`reporters`
       Reporter types the helpers above construct.
