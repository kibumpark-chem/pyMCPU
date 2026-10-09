Quickstart
==========

Two conventions to know before you start:

* **Temperature is dimensionless** — a reduced parameter, not Kelvin. Where a
  protein unfolds depends on the protein: chignolin, the example structure
  below, melts at about ``0.65`` to ``0.7`` (see
  :doc:`tutorials/02_replica_exchange`).
* **Energies are unitless** — sums of knowledge-based table entries scaled by a
  dimensionless per-group weight.

A complete run
--------------

This is the whole pipeline. It works from any working directory on a plain
``pip install pymcpu``, with no parameter download and no environment
variables. It writes ``energies.csv`` and ``traj.xtc`` to the current
directory and takes under a minute.

.. code-block:: python

   import numpy as np
   import mdtraj as md

   import pymcpu as mc
   from pymcpu.forcefields.mcpu import MCPUForceField
   from pymcpu.runners import default_example_pdb

   # 1. Load a structure. The engine is heavy-atom, so strip hydrogens --
   #    the H-bond term builds the virtual amide hydrogens it needs.
   traj = md.load(str(default_example_pdb()))
   heavy = traj.atom_slice(traj.topology.select("not element H"))

   # 2. Build the force field, then the system. The force field assigns atom
   #    types, reorders atoms into the engine's contiguous layout, and loads
   #    the fitted potentials.
   forcefield = MCPUForceField(heavy, param_set="mcpu08")
   system = forcefield.create_system(heavy.topology)

   # 3. Set up sampling. The temperature is required, and is a reduced value
   #    (see the conventions above). Set the seed: the same seed, input and
   #    pyMCPU build reproduce a run exactly (other compilers can differ; see
   #    Known issues).
   integrator = mc.Integrator(temperature=0.6, step_size_rad=0.1)
   integrator.set_seed(42)

   sim = mc.Simulation(heavy.topology, system, integrator)

   # 4. Give it coordinates. The cleanest way is to keep feeding in the
   #    mdtraj object: the force field read your structure from `heavy`, and
   #    forcefield.coords holds it ready for the engine, so never build this
   #    array yourself. The engine wants one frame as (3, n_atoms) in
   #    Angstroms, float32 -- hence the x10, the transpose and the cast.
   sim.context.set_positions((forcefield.coords[0] * 10.0).T.astype(np.float32))

   # 5. Attach reporters. They write to these files as the run goes.
   #    inverse_mapping writes the frames back in your topology's atom order,
   #    so md.load("traj.xtc", top=heavy.topology) reads them.
   sim.add_energy_reporter("energies.csv", interval=100)
   sim.add_xtc_reporter("traj.xtc", interval=1000,
                        inverse_mapping=forcefield.inverse_mapping)

   # 6. Run.
   sim.step(10_000)

   print(sim.context.energy_breakdown(weighted=True))

The composition pattern
-----------------------

1. :doc:`api/forcefield` — ``MCPUForceField(trajectory, param_set=...)`` turns
   an mdtraj structure into typed, engine-ordered atoms and loads the fitted
   potentials.
2. ``forcefield.create_system(topology)`` returns a :doc:`api/system` with all
   five energy terms already registered.
3. :doc:`api/integrator` — ``Integrator(temperature, step_size_rad)`` holds the
   Monte Carlo move settings.
4. :doc:`api/simulation` — ``Simulation(topology, system, integrator)`` owns the
   :doc:`api/context` and the reporters; you call ``step()`` on it.
5. :doc:`api/reporters` write output during the run; :doc:`api/analysis`
   prepares replica-exchange output for MBAR reweighting.

Reading the results
-------------------

``energy_breakdown()`` returns the total energy both raw and weighted
(``raw_total`` and ``weighted_total``), plus the energy of each term in
``by_name``, keyed by the term's name (``by_group`` holds the same values keyed
by group number). The per-term values are weighted by default; pass
``weighted=False`` for the raw table sums.

.. code-block:: python

   breakdown = sim.context.energy_breakdown(weighted=True)

   for name, value in breakdown["by_name"].items():
       print(f"{name:20} {value:12.4f}")

Acceptance rates are the first thing to check on any Monte Carlo run. For
``pivot`` and ``kic``, a very low rate means ``step_size_rad`` is too large for
the temperature, and a very high rate means the moves are too small to
explore. The ``rotamer`` move draws whole rotamers from a library and does not
use ``step_size_rad``:

.. code-block:: python

   for kind, (accepted, attempted) in integrator.move_counts().items():
       rate = accepted / attempted if attempted else float("nan")
       print(f"{kind:12} {accepted:6d} / {attempted:6d}  {rate:6.1%}")

``move_counts()`` lists each kind of move the run can propose. With the
default settings these are the continuous ``pivot``, the knowledge-based
``rama_pivot`` (off until you set a rama probability, so it shows ``0 / 0``),
``kic`` and the ``rotamer`` sidechain move.

The energy reporter only writes to disk. To analyse energies, read the CSV
back. It has a row for the starting structure (step 0, all move counts zero)
and then one every ``interval`` steps -- 101 rows for this run. The columns
are ``step``, ``total``, one column per energy term, ``<kind>_accepted`` and
``<kind>_attempted`` for each move kind, and ``walker_id``:

.. code-block:: python

   import pandas as pd

   energies = pd.read_csv("energies.csv")
   print(energies[["step", "total", "mu", "hydrogen_bond"]].tail())

Using your own structure
------------------------

``default_example_pdb()`` returns the bundled 1UAO chignolin structure. For
your own protein, load its file instead and keep the heavy atoms, dropping
water:

.. code-block:: python

   traj = md.load("my_protein.pdb")
   heavy = traj.atom_slice(traj.topology.select("not water and not element H"))

The MCPU force field has parameters for the 20 standard amino acids.
Protonation-state variants such as ``HID``, ``CYX`` or ``ASH`` are renamed to
the standard residue automatically. Anything else -- ions, ligands, ``MSE``,
phosphorylated residues -- makes ``MCPUForceField`` raise a ``ValueError``
that names the residues; remove them by name, for example by adding
``and not resname NA CL LIG`` to the selection.

The structure must be **one continuous chain** of at least three residues,
with **every heavy atom present**. ``MCPUForceField`` raises a ``ValueError``
otherwise:

* For a file with several chains, or a chain with missing residues, which it
  finds by a jump in the residue numbering or by a C atom more than 2 Å from
  the next residue's N. Keep one chain (for example, add ``and chainid 0`` to
  the selection) and model any missing residues first. To simulate the pieces
  joined as if they were bonded anyway, pass ``allow_chain_breaks=True``
  (in a YAML config, ``forcefield_options: {allow_chain_breaks: true}``).
* For a residue with missing heavy atoms, common in crystal structures; the
  message names each residue and atom. Rebuild missing atoms first, for
  example with PDBFixer.

Avoid ``select("protein")`` here: mdtraj does not count some
protonation-variant names (``ASH``, for example) as protein, so that selection
silently drops those residues and breaks the chain.

Beyond that, no preparation is needed: the potentials are general
knowledge-based tables, not fitted to a particular structure.

Where to go next
----------------

* :doc:`tutorials/01_single_trajectory` — a guided first simulation, with
  plots of the energy, the structure and the effect of temperature
* :doc:`tutorials/02_replica_exchange` — a small replica exchange, its swap
  rates and a melting curve
* :doc:`running_remd` — replica exchange, serial and MPI
* :doc:`checkpointing` — surviving a scheduler timeout
* :doc:`cli` — the ``mcpu`` command line
* :doc:`physics_notes/background` — what each energy term computes
