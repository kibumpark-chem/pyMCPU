Installation
============

.. code-block:: bash

   pip install pymcpu

or

.. code-block:: bash

   conda install -c conda-forge pymcpu

That is the whole installation. The fitted potentials ship inside the package,
so there is no download step, no environment variable to set, and no network
access required at first use.

Requirements
------------

* Linux x86-64
* Python 3.9 or newer

Building from source additionally needs a C++20 compiler, CMake 3.15 or newer,
and pybind11 2.12 or newer. Eigen 3.4 is fetched automatically if it is not
already present.

Optional extras
---------------

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Extra
     - Provides
   * - ``analysis``
     - HDF5 replica-exchange output (``h5py``)
   * - ``mpi``
     - MPI replica exchange (``mpi4py``)
   * - ``training``
     - Refitting the potentials from a structure corpus (``scipy``)
   * - ``docs``
     - Building this documentation
   * - ``dev``
     - Test suite and linter

.. note::
   **WESTPA support is a separate package**, not an extra:
   ``pip install pymcpu-westpa``. It requires Python 3.10 or newer even
   though pyMCPU itself supports 3.9, because ``westpa`` 2022.15 declares
   ``requires-python >=3.10``.

   If you install WESTPA from conda-forge instead, check the version. The
   newest conda-forge build is ``2022.04`` (that is 2022.4), which is *older*
   than the 2022.15 this project asks for; PyPI has the newer one.

   One failure worth recognizing: WESTPA imports its MPI work manager
   defensively, but only catches ``ImportError``. A modern ``mpi4py``
   installed with no MPI runtime behind it raises ``RuntimeError`` instead,
   so ``import westpa`` fails with ``cannot load MPI library`` even though
   WESTPA is fine. Install an MPI library, or uninstall ``mpi4py``.

.. code-block:: bash

   pip install "pymcpu[analysis,mpi]"

Building from source
--------------------

.. code-block:: bash

   git clone https://github.com/kibumpark-chem/pyMCPU.git
   cd pyMCPU
   conda env create -f environment.yml && conda activate mcpu
   pip install --no-build-isolation -e .
   python scripts/install_check.py

``--no-build-isolation`` is required. It makes the build use the ``pybind11``
and ``numpy`` already in your environment rather than downloading an isolated
toolchain, which is faster and — more importantly — keeps the C++ runtime
consistent with the interpreter that will load the extension.

Expect a few minutes for the first compile. Root is not needed; a user conda
environment or venv is sufficient.

Choosing a CPU baseline
~~~~~~~~~~~~~~~~~~~~~~~

The default baseline is AVX-512 (Skylake-SP or newer). Override it for other
hardware:

.. code-block:: bash

   MCPU_ARCH=x86-64-v3 pip install --no-build-isolation -e .   # AVX2 + FMA
   MCPU_ARCH=native    pip install --no-build-isolation -e .   # this machine

.. warning::

   On a heterogeneous cluster, do not build with ``native`` on a login node and
   run on compute nodes — that is the classic route to ``Illegal instruction``
   at run time. Changing the baseline can also change floating-point results in
   the last bit, which is enough to flip one Metropolis decision, so runs you
   intend to compare should share a build.

The C++ runtime rule
~~~~~~~~~~~~~~~~~~~~

The ``libstdc++`` your compiler targets must be no newer than the one present
at run time. Violating this produces an extension that builds cleanly and then
fails at ``import pymcpu`` with ``version 'CXXABI_x.y.z' not found``.

The reliable way to satisfy it is to let conda supply both halves —
``gxx_linux-64`` for the compiler and ``libstdcxx-ng`` for the runtime, both
already in ``environment.yml``. They are versioned together, so they cannot
drift apart.

Note that a conda interpreter carries an ``RPATH`` of ``$ORIGIN/../lib``, so
the environment's own ``lib/libstdc++.so.6`` wins over anything on
``LD_LIBRARY_PATH``. A mismatch cannot be repaired by setting that variable.

C++20 is required. GCC 8.5 is the oldest version verified to build the tree and
pass the full test suite.

HPC notes
---------

On a module-based system, load a compiler and CMake before building — module
names vary by site:

.. code-block:: bash

   module load gcc cmake
   conda activate mcpu

Then load the *same* compiler at run time, and make sure it is not newer than
the ``libstdc++`` your interpreter loads. See the rule above.

For multi-rank MPI jobs, decode the bundled parameters once before launching,
so that N ranks do not race to do it simultaneously against a shared ``$HOME``:

.. code-block:: bash

   export MCPU_PARAMS_DIR="$(mcpu materialize-params --set mcpu_v1)"
   mpirun -n 32 python my_remd_run.py

The number of ranks may be anything from 1 up to the number of replicas;
requesting more ranks than replicas is an error. See :doc:`api/sampling`.

MPI support
~~~~~~~~~~~

``mpi4py`` must be built against the site's own MPI, so install it after
loading the MPI module rather than taking a generic wheel:

.. code-block:: bash

   module load openmpi          # site-specific name
   pip install --no-binary mpi4py mpi4py
   python -c "from mpi4py import MPI; print('MPI OK')"

Parameter resolution
--------------------

Only one parameter set is published: ``mcpu_v1``. It is the default, so
``MCPUForceField(traj)`` and ``MCPUForceField(traj, param_set="mcpu_v1")`` are
equivalent.

``pymcpu.params.ensure_params()`` resolves it in this order, first hit wins:

.. list-table::
   :header-rows: 1
   :widths: 8 92

   * - #
     - Source
   * - 1
     - ``MCPU_PARAMS_DIR`` — a pre-staged root holding ``constants/`` and
       ``mcpu_params/``
   * - 2
     - An editable checkout's own tree,
       ``src/pymcpu/parameters/pretrained/mcpu08``
   * - 3
     - An already-populated cache (``MCPU_CACHE_DIR``, default
       ``~/.cache/pymcpu``)
   * - 4
     - ``MCPU_PARAMS_BUNDLE`` — a local ``.tar.gz``
   * - 5
     - The compact archive shipped inside the package, decoded into the cache
   * - 6
     - A registry URL via ``pooch``, for a future published parameter release

Step 5 is what makes ``pip install pymcpu`` work offline. It sits below steps
1–4 on purpose: a pre-staged root, or a table you refitted locally, continues
to take precedence over the shipped one.

Set ``MCPU_NO_DOWNLOAD=1`` to make step 6 fail loudly rather than reach the
network.

On a cluster, pointing ``MCPU_CACHE_DIR`` at node-local scratch is usually
faster than a shared home directory:

.. code-block:: bash

   export MCPU_CACHE_DIR="$TMPDIR"

Verifying the installation
--------------------------

.. code-block:: bash

   python scripts/install_check.py

or, minimally:

.. code-block:: bash

   python -c "import pymcpu; print(pymcpu.__version__)"

Conventions worth knowing before your first run
-----------------------------------------------

**Temperature is dimensionless.** It is a reduced parameter in the Metropolis
criterion, not a physical unit. Useful values run from about ``0.3`` (cold,
folded) to ``0.6`` (hot, unfolded). ``Integrator`` defaults to ``300.0``, which
is *not* a reduced temperature — always pass one explicitly.

**Energies are unitless.** They are sums of knowledge-based table entries
scaled by a dimensionless per-group weight. There is no Boltzmann constant and
no Kelvin anywhere in the engine.

**The engine is heavy-atom.** Strip hydrogens before building a force field;
the hydrogen-bond term constructs the virtual amide hydrogens it needs.

**Atom counts differ between the file and the engine.** The bundled 1UAO
structure has 77 heavy atoms, while ``System.get_num_atoms()`` reports 80. The
three extra slots are per-glycine bookkeeping entries, and chignolin has three
glycines. Backbone torsion terms additionally require at least three residues.
