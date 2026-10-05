Command-line interface
======================

Installing pyMCPU puts an ``mcpu`` command on your ``PATH``. It prints the
version, prepares the force-field parameters, and runs or checks a simulation
config.

.. code-block:: bash

   mcpu --help
   mcpu <subcommand> --help

``mcpu version``
----------------

Print the installed pyMCPU version.

.. code-block:: console

   $ mcpu version
   0.1.0

``mcpu download-params``
------------------------

.. program:: mcpu download-params

Find the parameter set the way the force field does (the order is under
"MCPU parameter lookup" in :doc:`installation`) and print its directory.
Despite the name it needs no network: if nothing else provides the set, it
unpacks the copy shipped with pyMCPU. If no source has the set, it says what
to do and exits with status 1.

.. code-block:: bash

   mcpu download-params --set mcpu08
   mcpu download-params --set mcpu08 --dir /scratch/$USER/mcpu_params

.. option:: --set SET

   Parameter set name. Default ``mcpu08``.

.. option:: --dir DIR

   Copy the parameters into ``DIR`` and print ``DIR`` instead, for example to
   put them on node-local scratch; then ``export MCPU_PARAMS_DIR=DIR``. Run it
   with ``MCPU_PARAMS_DIR`` unset: if that already points at ``DIR``, the
   command copies ``DIR`` onto itself, which fails and can delete part of it.

``mcpu materialize-params``
---------------------------

.. program:: mcpu materialize-params

Unpack the parameters shipped with pyMCPU into the cache and print the
directory. pyMCPU does this itself on first use when nothing earlier in the
lookup order provides the set (see "MCPU parameter lookup" in
:doc:`installation`). Before an MPI job that uses the shipped parameters, run
it once yourself, so the processes do not all unpack at once into a shared
home directory:

.. code-block:: bash

   export MCPU_PARAMS_DIR="$(mcpu materialize-params --set mcpu08)"
   mpirun -n 32 python my_remd_run.py

The command always unpacks the shipped copy, even when an earlier step of the
lookup, such as ``MCPU_PARAMS_BUNDLE``, would supply other parameters; in that
case do not export its output. The directory name includes a hash of the
shipped tables, so a release with different tables unpacks into a new
directory, and running the command again is quick.

.. option:: --set SET

   Parameter set name. Default ``mcpu08``.

.. option:: --timeout SECONDS

   How long to wait while another process is unpacking the same set. A lock
   older than this is taken to be abandoned and is taken over. Default
   ``900``.

.. option:: --no-verify

   Skip the SHA-256 check of each unpacked table. Faster, but a damaged table
   goes unnoticed. Not recommended.

``mcpu run``
------------

.. program:: mcpu run

Run a simulation from a JSON or YAML config.

.. code-block:: bash

   mcpu run config.yaml
   mcpu run config.yaml --resume --checkpoint-dir checkpoints/

``mcpu run`` uses a single process. Under ``mpirun`` every process would run
its own full copy. For MPI replica exchange, start a short script like this
one under ``mpirun``, or use ``scripts/run_mcpu_replica_exchange.py --mpi``
from a source checkout (see :doc:`running_remd`):

.. code-block:: python

   from mpi4py import MPI

   from pymcpu.config import load_config_auto
   from pymcpu.runners import run_from_config

   run_from_config(load_config_auto("config.yaml"), comm=MPI.COMM_WORLD)

.. option:: config

   Path to the config file (``.json``, ``.yaml`` or ``.yml``).

.. option:: --quiet

   Print less progress output.

The checkpoint options below override the matching settings in the config
(its ``checkpointing`` block in YAML, ``checkpoint`` in JSON), so you can
resume a run without editing the config. An option you leave out keeps the
config's setting; the defaults given are those of a config that does not set
it either. See :doc:`checkpointing`.

.. option:: --checkpoint-interval N

   Save a checkpoint every ``N`` cycles. Default ``50``.

.. option:: --checkpoint-dir DIR

   Directory for checkpoint files. Default ``checkpoints/``.

.. option:: --keep-last-n N

   Numbered checkpoints to keep. Default ``3``.

.. option:: --resume

   Resume from the latest checkpoint in the checkpoint directory.

``mcpu validate``
-----------------

Load a config and check its settings without running anything, then print the
mode, the PDB path, and the checkpoint directory and interval it resolved. It
takes the same checkpoint options as ``mcpu run``, so you can check a config in
the form you will run it.

.. code-block:: bash

   mcpu validate config.yaml

It exits with status 1 and the reason if the config has an unknown key or a
value the loader rejects, or if the ``pdb`` file, or the ``reference_pdb`` file
of a replica exchange config, is missing or is not a file. It does not read
the structure or the force-field parameters, so a problem inside the PDB file
shows up only when the run starts.
