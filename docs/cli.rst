Command-line interface
======================

Installing pyMCPU puts a single ``mcpu`` executable on your ``PATH``. It covers
parameter resolution, running and validating configs, and WESTPA scaffolding.

.. code-block:: bash

   mcpu --help
   mcpu <subcommand> --help

``mcpu version``
----------------

Print the installed package version. Useful in bug reports and job logs.

.. code-block:: bash

   $ mcpu version
   0.1.0

``mcpu download-params``
------------------------

Resolve the pretrained potentials and print the resulting directory. Runs the
full resolution order documented in :doc:`installation` and exits non-zero with
an explanation if no source can satisfy the request.

.. code-block:: bash

   mcpu download-params --set mcpu08
   mcpu download-params --set mcpu08 --dir /scratch/$USER/mcpu_params

.. option:: --set SET

   Parameter set name from the registry. Default ``mcpu08``.

.. option:: --dir DIR

   Optional staging directory to copy the resolved parameters into. Use this to
   pre-place parameters on node-local scratch.

``mcpu materialize-params``
---------------------------

Decode the compact parameter archive that ships inside the package into a full
parameters root, and print its path.

You rarely need this interactively — ``ensure_params()`` does it on demand. It
exists for **multi-rank MPI jobs**, where N ranks cold-starting against a
shared ``$HOME`` would otherwise all try to decode at once. Running it once
before ``mpirun`` turns that race into a single serial call and pins every rank
to one identical root:

.. code-block:: bash

   export MCPU_PARAMS_DIR="$(mcpu materialize-params --set mcpu08)"
   mpirun -n 32 python my_remd_run.py

The cache directory is content-addressed (``<set>-<sha256[:12]>``), so
upgrading the package produces a different directory rather than requiring
invalidation, and repeated calls are cheap no-ops.

.. option:: --set SET

   Parameter set name from the registry. Default ``mcpu08``.

.. option:: --timeout TIMEOUT

   Seconds to wait for another process that holds the materialization lock.
   Default ``900``.

.. option:: --no-verify

   Skip the per-table SHA-256 check. Faster, but you lose the guarantee that
   the decoded tables match the shipped ones. Not recommended.

``mcpu run``
------------

Run a simulation from a JSON or YAML config.

.. code-block:: bash

   mcpu run config.yaml
   mcpu run config.yaml --resume --checkpoint-dir checkpoints/

.. option:: config

   Path to the config file (``.json`` / ``.yaml`` / ``.yml``).

.. option:: --quiet

   Reduce stdout chatter.

Checkpointing options — these override the config's ``checkpointing`` block, so
you can restart a run without editing it. See :doc:`checkpointing`.

.. option:: --checkpoint-interval N

   Save a checkpoint every ``N`` cycles. Default ``50``.

.. option:: --checkpoint-dir CHECKPOINT_DIR

   Directory for checkpoint files. Default ``checkpoints/``.

.. option:: --keep-last-n KEEP_LAST_N

   Number of versioned checkpoints to retain. Default ``3``.

.. option:: --resume

   Resume from the newest checkpoint in ``--checkpoint-dir``.

.. option:: --cloud-sync

   Upload ``last.chk`` after every save.

.. option:: --cloud-bucket URI

   Destination URI, for example ``s3://my-bucket/run-01/``.

.. option:: --cloud-sync-cmd CLOUD_SYNC_CMD

   Upload command. Default ``aws s3 cp``.

``mcpu validate``
-----------------

Parse and validate a config without running anything, and report what it
resolved to. Accepts the same checkpointing options as ``run`` so that a
resume-capable config can be checked in the exact form it will be run.

.. code-block:: bash

   mcpu validate config.yaml

Use it in a submission script before requesting a large allocation — a typo in
a config is much cheaper to find here than after the job starts.

WESTPA subcommands
------------------

WESTPA scaffolding moved out of ``mcpu`` when the integration became the
separate ``pymcpu-westpa`` distribution. Install it and you get a
``mcpu-westpa`` command with ``init`` and ``check`` subcommands:

.. code-block:: bash

   pip install pymcpu-westpa
   mcpu-westpa init --pdb protein.pdb --out we_run/ --mode equilibrium
   mcpu-westpa check we_run/west.cfg --bstates we_run/bstates/bstates.txt

They are not on ``mcpu`` itself because argparse cannot register a
subcommand lazily: ``mcpu --help`` would advertise them to every user,
almost none of whom have WESTPA installed.

