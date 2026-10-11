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

Building from source
--------------------

You need a C++20 compiler, CMake 3.15 or newer, and pybind11 2.12 or newer.
GCC 15 is the default compiler and the one the published wheels are built
with. GCC 8.5 is the oldest that builds and passes the tests, but its builds
run about 3-10% slower (about 10% on pivot moves), and
CMake warns when it finds a GCC older than 15. Eigen 3.4 is downloaded
automatically if it is not found. ``environment.yml`` provides all of these.

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

Branch padding on Intel CPUs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Intel CPUs from Skylake to Cascade Lake run code more slowly when a jump
sits on a 32-byte boundary (the "JCC erratum"). When the toolchain allows it,
the build pads jumps away from those boundaries. On those CPUs this makes
runs 3-8% faster, and 8-14% faster together with a newer compiler. Results
do not change. On other CPUs the only effect is about 2% more machine code.

Padding needs GCC with binutils 2.34 or newer, and with LTO (the Release
default) GCC 11 or newer. Older toolchains build without it. To check a
build, run:

.. code-block:: bash

   python -c "import pymcpu.mcpu_core as m; b = m.build_info()['build']; print(b['jcc_pad'], b['jcc_pad_reason'])"

Set ``MCPU_JCC_PAD`` to choose:

.. code-block:: bash

   MCPU_JCC_PAD=ON  pip install --no-build-isolation -e .   # fail if it cannot pad
   MCPU_JCC_PAD=OFF pip install --no-build-isolation -e .   # never pad

The default, ``AUTO``, pads when it can and otherwise builds without padding.

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

To build with the cluster's own compiler instead, on RHEL 8 or Rocky 8
(FASRC included) enable the GCC 15 toolset in the same shell, before
``pip install`` or ``cmake``:

.. code-block:: bash

   source /opt/rh/gcc-toolset-15/enable
   g++ --version | head -1                  # should print 15.x
   pip install --no-build-isolation -e .

The toolset compiles the parts of the C++ runtime that are newer than the
system's into the extension itself, so the result imports under any Python,
conda's included, and its assembler pads branches (see above). Two things
override the toolset: a ``CXX`` environment variable (conda's compiler
activation sets one; ``unset CXX``), and the compiler cached in an existing
CMake build directory (configure a fresh one).

If a node has no ``/opt/rh/gcc-toolset-15``, use the conda environment
instead. A plain module GCC 14 or newer (``module load gcc``) builds an
extension that fails at import under a stock Miniforge or Mambaforge Python
(``CXXABI_1.3.15 not found``, see above). If you must use one, link the C++
runtime statically. On FASRC, with ``gcc/15.2.0-fasrc01``:

.. code-block:: bash

   module load gcc/15.2.0-fasrc01
   unset CXX
   STATIC="-static-libstdc++ -static-libgcc"
   pip install --no-build-isolation -e . \
       -Ccmake.define.CMAKE_CXX_COMPILER=$(which g++) \
       -Ccmake.define.CMAKE_SHARED_LINKER_FLAGS="$STATIC" \
       -Ccmake.define.CMAKE_MODULE_LINKER_FLAGS="$STATIC"

Such a build uses the system assembler (binutils 2.30 on RHEL 8), which
cannot pad branches, so it may run somewhat slower than the toolset build.

MCPU parameter lookup
---------------------

The MCPU force field uses the parameter set ``mcpu08``. It is the default,
so ``MCPUForceField(traj)`` needs no ``param_set`` argument. pyMCPU looks for
the parameters in this order and uses the first match:

1. ``MCPU_PARAMS_DIR``: a directory containing ``constants/`` and
   ``mcpu_params/``
2. The source tree, when installed from a checkout with ``pip install -e``
3. The copy shipped inside the package, unpacked into the cache
   (``MCPU_CACHE_DIR``, default ``~/.cache/pymcpu``) on first use

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
