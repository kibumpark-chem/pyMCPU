Analysis Utilities
==================

``pymcpu.analysis`` is the export and reweighting bridge between a
replica-exchange run and an MBAR (``pymbar``) or umbrella-sampling
analysis. :class:`~pymcpu.analysis.RexSampleWriter` accumulates one
sample record per replica per cycle and flushes them to HDF5 (or to
NPZ plus a JSON metadata sidecar); :func:`~pymcpu.analysis.reduced_potentials`
reads such a file back and rebuilds the reduced-potential matrix
``u_kn`` that MBAR consumes.

The subpackage is not pulled in by ``import pymcpu``. Import it
explicitly::

    import pymcpu.analysis

:class:`~pymcpu.sampling.ReplicaExchange` and
:class:`~pymcpu.sampling.MPIReplicaExchange` construct the writer for
you when ``run()`` is given ``analysis_path=`` (or ``hdf5_path=``); in a
config, ``hdf5: FILE`` does the same. So most users only need the reader
side of this module.

.. note::
   Energies are unitless sums of knowledge-based potential table
   entries, and temperature is a dimensionless reduced parameter, so
   ``kB`` defaults to ``1.0`` and a reduced potential is just
   ``(E + bias) / T``. The ``kB`` argument is a rescaling hook for
   other unit conventions, not a physical Boltzmann constant.

.. note::
   ``h5py`` comes with pyMCPU. The writer picks the format from the file
   name: ``.h5``, ``.hdf5`` or no extension for HDF5, ``.npz`` for NPZ plus
   ``<stem>_meta.json``. It falls back to NPZ if ``h5py`` cannot be loaded,
   and the reader accepts both.

Sample records
--------------

Every record has five fields. MBAR needs the first three; the other two
are bookkeeping.

``state_index``
    Thermodynamic state the sample was drawn from, indexed
    ``k = temp_index * n_windows + n_index``.
``energy_unbiased``
    Total energy with the harmonic N bias subtracted back out.
``N``
    Hard native-contact count at the time of the sample.
``cycle``
    Exchange cycle the sample came from.
``walker_id``
    Configuration lineage occupying that state; for replica-mixing
    diagnostics only.

Writer
------

``RexSampleWriter`` buffers records in memory and writes the whole
buffer out on
:meth:`~pymcpu.analysis.RexSampleWriter.flush` or ``close()``. The
constructor takes the output ``path`` plus the run metadata the reader
needs in order to rebuild ``u_kn``:

``temperatures``
    The reduced-temperature ladder, shape ``(n_temps,)``.
``n_targets``
    Umbrella centers in native-contact count N, shape ``(n_windows,)``.
``k_bias``
    Harmonic strength of the N bias, ``U = 0.5 * k * (N - N0)^2``.
``n_contacts``
    Size of the native-contact set, so that ``Q = N / n_contacts`` can
    be recovered downstream.
``kB``
    Unit-convention rescaling; defaults to ``1.0``.
``bias_cv``
    Label for the biased collective variable stored in the file
    metadata; defaults to ``"hard_N"``.

``RexSampleWriter`` is also a context manager, and closes itself on exit.

.. autoclass:: pymcpu.analysis.RexSampleWriter
   :members:

.. autofunction:: pymcpu.analysis.open_writer

``open_writer`` forwards its arguments to
:class:`~pymcpu.analysis.RexSampleWriter` unchanged and returns the
instance; it exists so that call sites read as an open rather than a
construction.

Reduced potentials
------------------

.. autofunction:: pymcpu.analysis.reduced_potentials

.. autofunction:: pymcpu.analysis.reduced_potentials_from_arrays

.. seealso::
   Q and N themselves are computed by
   :class:`pymcpu.sampling.NativeContactsCV` (``compute_N`` /
   ``compute_Q``), not by this module. pyMCPU ships no RMSD or
   radius-of-gyration helper: run ``mdtraj.rmsd`` and
   ``mdtraj.compute_rg`` on the XTC trajectories a run writes.
