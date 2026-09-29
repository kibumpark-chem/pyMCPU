Installation
============

.. code-block:: bash

   pip install pymcpu

or

.. code-block:: bash

   conda install -c conda-forge pymcpu

Requirements
------------

* Linux x86-64
* Python 3.9 or newer
* A CPU with AVX2 and FMA (Intel Haswell or AMD Zen, or newer). This is the
  baseline the published package is built for; see `CPU baseline`_ to change it.

MPI (optional)
--------------

You only need MPI to spread replica exchange over several processes with
:class:`~pymcpu.sampling.MPIReplicaExchange`. Single runs and in-process
replica exchange (:class:`~pymcpu.sampling.ReplicaExchange`) work without
it. ``import pymcpu`` works whether or not MPI is installed.

MPI support is made of two pieces, and **they must match**:

* an **MPI library** (Open MPI, MPICH, Intel MPI, ...), which provides the
  ``mpirun`` command, and
* **mpi4py**, the Python interface to it, built for that same library.

Almost every MPI problem comes from these two not matching. Choose the route
that fits your machine.

On your own workstation
~~~~~~~~~~~~~~~~~~~~~~~

With conda, install both together so they match automatically:

.. code-block:: bash

   conda install -c conda-forge mpi4py openmpi

Without conda, install an MPI library with your system package manager, then
build mpi4py against it:

.. code-block:: bash

   sudo apt install openmpi-bin libopenmpi-dev   # Debian/Ubuntu; other systems differ
   pip install --no-binary mpi4py mpi4py

On a cluster
~~~~~~~~~~~~

Use the cluster's own MPI, not one from conda. The cluster's MPI is set up
for its network and job scheduler; a generic one may not be. Load it, then
build mpi4py against it:

.. code-block:: bash

   module load openmpi                      # the module name varies by site
   pip install --no-binary mpi4py mpi4py

``--no-binary mpi4py`` makes pip compile mpi4py against the MPI you just
loaded, instead of downloading a pre-built copy made for some other MPI.
Load the same module in your job scripts.

Check that it works
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   mpirun -n 2 python -c "from mpi4py import MPI; c = MPI.COMM_WORLD; print(c.Get_rank(), 'of', c.Get_size())"

You should see ``0 of 2`` and ``1 of 2``, in either order.

* **You see** ``0 of 1`` **twice.** ``mpirun`` and mpi4py come from different
  MPI libraries, so each process started on its own. A replica-exchange job
  would run as two unrelated copies instead of one run. Reinstall mpi4py with
  the right MPI loaded, as above.
* **The import fails with** ``cannot load MPI library``. mpi4py is installed
  but no MPI library is available. Install or load one as above.

Before launching a job
~~~~~~~~~~~~~~~~~~~~~~

Unpack the parameters once, before ``mpirun``. Otherwise every process tries
to unpack them at the same moment into your shared home directory:

.. code-block:: bash

   export MCPU_PARAMS_DIR="$(mcpu materialize-params --set mcpu08)"
   mpirun -n 32 python my_remd_run.py

See :doc:`running_remd` for how processes are assigned to replicas.

KORP energy map
---------------

The MCPU force field needs no download. :class:`~pymcpu.KORPForceField`
needs one extra file, the KORP 6D energy map ``korp6Dv1.bin``. It is 316 MiB,
too large to ship with the package, so you download it once:

1. Download ``Korp6Dv1.txz`` from https://chaconlab.org/modeling/korp. The
   site asks you to accept its license first.
2. Unpack it and tell pyMCPU where the map is:

   .. code-block:: bash

      tar xJf Korp6Dv1.txz
      export KORP_MAP_PATH=$PWD/Korp6Dv1/korp6Dv1.bin

Put the ``export`` line in your shell profile or job script so it is always
set. Two alternatives work as well: pass the path directly with
``KORPForceField(traj, map_path=...)``, or place the file at
``~/.cache/pymcpu/korp/Korp6Dv1/korp6Dv1.bin`` where it is found
automatically. If you set ``MCPU_CACHE_DIR``, that location moves with it,
to ``$MCPU_CACHE_DIR/korp/Korp6Dv1/korp6Dv1.bin``.

If you publish results that use KORP, please cite López-Blanco & Chacón,
*Bioinformatics* 35(17):3013–3019 (2019).

Building from source
--------------------

You need a C++20 compiler (GCC 8.5 or newer), CMake 3.15 or newer, and
pybind11 2.12 or newer. Eigen 3.4 is downloaded automatically if it is not
found. ``environment.yml`` provides all of these.

.. code-block:: bash

   git clone https://github.com/kibumpark-chem/pyMCPU.git
   cd pyMCPU
   conda env create -f environment.yml && conda activate mcpu
   pip install --no-build-isolation -e .
   python scripts/install_check.py

``--no-build-isolation`` is required. It builds against the ``pybind11`` and
``numpy`` already in your environment, which keeps the C++ runtime consistent
with the interpreter that loads the extension. The first compile takes a few
minutes. Root access is not needed.

CPU baseline
~~~~~~~~~~~~

The default baseline is ``v3`` (AVX2 + FMA), the same as the published
package. Set ``MCPU_ARCH`` to change it:

.. code-block:: bash

   MCPU_ARCH=v4     pip install --no-build-isolation -e .   # AVX-512
   MCPU_ARCH=native pip install --no-build-isolation -e .   # this machine only

Accepted values are ``v2``, ``v3``, ``v4``, ``native``, ``none`` (no
``-march`` flag; use your own ``CXXFLAGS``), or any ``-march`` value.

.. warning::

   On a cluster with mixed hardware, do not build with ``native`` on a login
   node and then run on compute nodes. That is the usual cause of
   ``Illegal instruction`` errors. Changing the baseline can also change
   floating-point results in the last bit, which can flip a single Metropolis
   decision. Runs you intend to compare should use the same build.

C++ runtime compatibility
~~~~~~~~~~~~~~~~~~~~~~~~~

The ``libstdc++`` your compiler targets must be no newer than the one loaded
at run time. If it is newer, the build succeeds but ``import pymcpu`` fails
with ``version 'CXXABI_x.y.z' not found``.

The conda environment above avoids this: ``gxx_linux-64`` (the compiler) and
``libstdcxx-ng`` (the runtime) are versioned together. A conda Python loads
its own environment's ``lib/libstdc++.so.6`` before anything on
``LD_LIBRARY_PATH``, so setting that variable will not fix a mismatch.

Building on a cluster
~~~~~~~~~~~~~~~~~~~~~

With ``environment.yml`` you do not need compiler or CMake modules. Conda
provides both, and they match the C++ runtime automatically.

If you build with the cluster's own compiler instead, load it before
building. Module names vary by site:

.. code-block:: bash

   module load gcc cmake

Then check that this compiler is not newer than the ``libstdc++`` your Python
loads, as described above.

MCPU parameter lookup
---------------------

The MCPU force field uses the parameter set ``mcpu08``. It is the default,
so ``MCPUForceField(traj)`` needs no ``param_set`` argument. pyMCPU looks for
the parameters in this order and uses the first match:

1. ``MCPU_PARAMS_DIR``: a directory containing ``constants/`` and
   ``mcpu_params/``
2. The source tree, when installed from a checkout with ``pip install -e``
3. The cache (``MCPU_CACHE_DIR``, default ``~/.cache/pymcpu``), if it is
   already filled
4. ``MCPU_PARAMS_BUNDLE``: a local ``.tar.gz``, unpacked into the cache
5. The copy shipped inside the package, unpacked into the cache on first use

A directory you set explicitly therefore always wins over the shipped copy.
On a cluster, a node-local cache is usually faster than a shared home
directory:

.. code-block:: bash

   export MCPU_CACHE_DIR="$TMPDIR"

Verifying the installation
--------------------------

.. code-block:: bash

   mcpu version

or, from Python:

.. code-block:: bash

   python -c "import pymcpu; print(pymcpu.__version__)"

Either one prints the version number. Getting that far means the compiled
engine loaded. For a full end-to-end check, including the parameters, run the
:doc:`quickstart` example.

From a source checkout, ``python scripts/install_check.py`` also checks the
reporters and prints the build flags.

Next: :doc:`quickstart`, which starts with the two conventions you need
before your first run.
