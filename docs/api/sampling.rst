Sampling
========

``pymcpu.sampling`` holds the Monte Carlo drivers and the
native-contacts collective variable. The subpackage is not pulled in by
``import pymcpu``; import it explicitly::

    import pymcpu.sampling

Three drivers are provided: :class:`~pymcpu.sampling.FoldingRunner` for
a single-temperature run, :class:`~pymcpu.sampling.ReplicaExchange` for
in-process replica exchange, and
:class:`~pymcpu.sampling.MPIReplicaExchange` for the same algorithm
spread over MPI ranks. All three build the
:class:`~pymcpu.MCPUForceField`, the :doc:`System <system>` and one
:doc:`Integrator <integrator>` per replica internally, so a run is
configured entirely through constructor keywords.

.. note::
   Temperature is a dimensionless reduced parameter, not a physical
   temperature: roughly 0.3 (cold, folded) to 0.6 (hot, unfolded).
   Energies are unitless sums of knowledge-based potential table
   entries.

Replica exchange
----------------

The replica grid is the outer product of the temperature ladder and the
umbrella windows in native-contact count N, so
``n_replicas == n_temps * n_q_windows``. Exchange is attempted between
temperature neighbours and, when there is more than one window, between
window neighbours. Both the bias and the exchange criterion use the
hard count N; the fraction ``Q = N / n_contacts`` is logged only.

.. autoclass:: pymcpu.sampling.ReplicaExchange
   :members:

``run(num_cycles, mc_replica_steps, ...)`` is the entry point: it steps
every replica for ``mc_replica_steps`` MC steps, attempts exchanges,
and repeats ``num_cycles`` times, returning a ``RunSummary``. Passing
``analysis_path=`` (or ``hdf5_path=``) makes it also write the MBAR
sample file described in :doc:`analysis`. ``run_cycle``,
``step_replicas``, ``exchange_temperatures`` and ``exchange_q_windows``
expose the same loop one piece at a time for custom schedules.

MPI replica exchange
--------------------

.. note::
   Requires ``mpi4py`` built against a working MPI library:
   ``pip install "pymcpu[mpi]"``. When the import fails, both
   ``pymcpu.sampling.MPIReplicaExchange`` and
   ``pymcpu.sampling.partition_replicas`` are set to ``None`` rather
   than raising, so guard on them before use.

.. note::
   **HPC-specific:** on a cluster, build ``mpi4py`` against the site
   MPI installation (make the system MPI available first, then install
   ``mpi4py`` from source) instead of taking a prebuilt wheel, which
   may not match the launcher.

Rank count need not equal replica count. Launch with
``mpirun -n R python your_script.py`` for any ``R`` from 1 up to the
replica count; more ranks than replicas is an error. Each rank steps
its assigned replicas one after another, then all ranks join the
exchange sweep, with swaps between two replicas on the same rank
handled locally. See :doc:`../running_remd` for the launch and
configuration workflow.

.. autoclass:: pymcpu.sampling.MPIReplicaExchange
   :members:

Single-temperature folding
--------------------------

.. autoclass:: pymcpu.sampling.FoldingRunner
   :members:

Collective variables
--------------------

.. autoclass:: pymcpu.sampling.NativeContactsCV
   :members:

The two required arguments, ``ca_internal_idx`` and ``ref_ca_xyz``, are
engine-internal contact-atom indices and reference contact-atom
coordinates in Angstroms. The drivers derive both from their
``pdb_path`` / ``reference_pdb`` arguments, using this module's
``build_contact_atom_index`` and ``reference_contact_from_pdb``
helpers; construct the CV directly only when you need a contact set the
drivers cannot express.

.. autofunction:: pymcpu.sampling.attach_native_contacts_bias_potential

.. autoclass:: pymcpu.sampling.CARMSDCV
   :members:

CA-RMSD to a reference structure, superposed with Kabsch before
measuring, so the value is invariant to rigid-body motion of the mobile
structure. Takes the same engine-internal CA indices as
``NativeContactsCV`` and a reference in Angstroms; build them with
``build_ca_index`` and ``reference_ca_from_pdb``.

Both CV classes satisfy the same structural contract -- ``ndim``,
``labels`` and ``__call__(coords_3xn) -> np.ndarray`` -- which is what
lets an external sampling framework consume either one without knowing
which it has.

Ladders and umbrella windows
----------------------------

.. autofunction:: pymcpu.sampling.make_temperature_ladder

Returns the arithmetic ladder ``temp_min + temp_step * arange(n_temps)``
as a ``float64`` array. This is what the drivers use when
``temperatures`` is not given explicitly.

.. autofunction:: pymcpu.sampling.make_n_targets

.. autofunction:: pymcpu.sampling.make_q_targets

Both window builders return ``[0.0]`` for a single window, i.e. no
umbrella sampling. ``make_n_targets`` is the preferred form, since the
bias and exchange criterion are defined on the count N; Q targets are
converted to ``N0 = Q * n_contacts`` once the contact set is known.

Internal helpers
----------------

``pymcpu.sampling.__all__`` also exports the plumbing the drivers are
built from. These are not part of the supported public surface and may
change without notice:

* Exchange records and state snapshots: ``Replica``, ``ReplicaState``,
  ``ExchangeRecord``, ``RunSummary``.
* Exchange primitives: ``evaluate_exchange_delta``,
  ``evaluate_exchange_acceptance``, ``get_coords``,
  ``swap_context_coordinates``.
* Contact-set construction: ``build_ca_index``,
  ``build_contact_atom_index``, ``reference_ca_from_pdb``,
  ``reference_contact_from_pdb``.
* MPI rank assignment: ``partition_replicas``.
* ``FoldingBias`` and ``BasinTracker`` are skeletons: the bias applies
  no energy and the tracker assigns every replica to basin 0.
