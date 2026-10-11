Reporters
=========

Reporters observe a running simulation at a fixed step interval.
:py:class:`EnergyReporter` and :py:class:`XtcReporter` write files;
:py:class:`SimulationReporter` prints progress, for interactive use. Files
are the default because under replica exchange many processes run at once,
and their printed output would interleave.

Attach a reporter either through the ``Simulation`` helpers, which construct
and register it in one call, or by constructing it yourself and calling
:meth:`~pymcpu.Simulation.add_reporter`.

.. code-block:: python

   sim.add_energy_reporter("energies.csv", interval=100)
   sim.add_xtc_reporter("traj.xtc", interval=1000,
                        inverse_mapping=forcefield.inverse_mapping)

   # equivalent, if you want to keep a handle on the object
   reporter = mc.EnergyReporter("energies.csv", 100)
   sim.add_reporter(reporter)

EnergyReporter
--------------

.. py:class:: EnergyReporter(energy_filename, report_interval, append=False)

   Writes a CSV row every ``report_interval`` steps holding each energy term
   and the cumulative move accept/attempt counts.

   :param str energy_filename: Output CSV path.
   :param int report_interval: Steps between rows.
   :param bool append: Append to an existing file instead of truncating it.
      Used when resuming from a checkpoint, after the file has been truncated
      back to the checkpointed row count.

   The columns follow the simulation rather than a fixed list:

   * ``step`` and ``total`` (the weighted total energy);
   * one column per energy term, named as in
     :py:meth:`System.energy_terms() <pymcpu.mcpu_core.System.energy_terms>`
     -- ``mu``, ``backbone_torsion``, ``sidechain_torsion``,
     ``hydrogen_bond`` and ``aromatic`` for MCPU (plus
     ``native_contacts_bias`` under replica exchange);
   * ``<kind>_accepted`` and ``<kind>_attempted`` for each move kind the
     move weights and sidechain mode can propose, as in
     :py:meth:`Integrator.move_counts() <pymcpu.mcpu_core.Integrator.move_counts>`
     -- by default ``pivot``, ``rama_pivot``, ``kic`` and ``rotamer``;
   * ``walker_id``, always last.

   The header is written when the first ``run()`` starts, because that is
   when the terms and move settings are known. With ``append=True`` an
   existing header must match exactly; if it does not -- a different force
   field or different move settings -- ``run()`` raises before making any
   move rather than writing misaligned columns. A missing or empty file gets
   a fresh header.

   ``total`` includes the native-contacts bias when one is enabled. That makes
   it the right quantity for monitoring a biased run and the **wrong** one for
   MBAR reweighting — use the replica-exchange HDF5 samples and
   :doc:`analysis` for that.

   .. py:attribute:: filename
      :type: str

      The path being written.

   .. py:attribute:: n_frames_written
      :type: int

      Rows written so far. The resume path uses this to align the file with a
      checkpoint.

   .. py:attribute:: walker_id
      :type: int

      Value written in the ``walker_id`` column; ``-1`` when unset.

   .. py:method:: set_walker_id(walker_id)

      Tag subsequent rows with a walker/replica identifier, so output from
      several replicas can be told apart after the fact.

.. note::

   There is no accessor for reading energies back out of an
   ``EnergyReporter`` — it is a writer, not a buffer. Read the CSV it
   produced, for example with ``pandas.read_csv(reporter.filename)``. To query
   current energies directly, use ``Context.energy_breakdown()``.

XtcReporter
-----------

.. py:class:: XtcReporter(xtc_filename, report_interval, inverse_mapping=[], append=False)

   Writes coordinates to a GROMACS XTC trajectory.

   :param str xtc_filename: Output ``.xtc`` path.
   :param int report_interval: Steps between frames.
   :param inverse_mapping: Permutation mapping the engine's internal atom order
      back to the input topology order, so the trajectory is readable against
      the original PDB. Pass ``MCPUForceField.inverse_mapping``. An empty
      sequence writes internal order. With the default virtual amide
      hydrogens that has the topology's atom count but not its atom order,
      so MDTraj loads it against the topology without complaint and the
      structure comes out scrambled.
   :param bool append: Append to an existing trajectory, for resume.

   .. py:attribute:: filename
      :type: str

   .. py:attribute:: n_frames_written
      :type: int

   .. py:method:: flush()

      Force buffered frames to disk. The resume path calls this before
      detaching reporters, so the on-disk frame count matches
      ``n_frames_written``.

XTC is compressed, and a resumed run cuts it, like the energy CSV, back to the
checkpoint before appending; see :doc:`../checkpointing`.

SimulationReporter
------------------

.. py:class:: SimulationReporter(report_interval)

   Prints the move counts and the energy to standard output every
   ``report_interval`` steps.

   :param int report_interval: Steps between lines.

   A convenience for interactive use only. It takes no filename and has no
   verbosity control — there is deliberately nothing to configure. For anything
   you intend to keep or parse, use :py:class:`EnergyReporter`, whose output is
   structured and goes to a file.

Reporter
--------

.. py:class:: Reporter

   Base class of the three reporters above. It is exposed so that
   :meth:`~pymcpu.Simulation.add_reporter` has a type to accept and so
   ``isinstance`` checks work. It has no public members of its own and is not
   intended to be subclassed from Python.
