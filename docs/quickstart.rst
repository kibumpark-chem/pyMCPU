Quickstart
==========

Two conventions to know before you start:

* **Temperature is dimensionless** — a reduced parameter, roughly ``0.3``
  (cold, folded) to ``0.6`` (hot, unfolded). It is not in Kelvin.
* **Energies are unitless** — sums of knowledge-based table entries scaled by a
  dimensionless per-group weight.

A complete run
--------------

This is the whole pipeline. It works from any working directory on a plain
``pip install pymcpu``, with no parameter download and no environment
variables.

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

   # 3. Set up sampling. Always pass a temperature: the default is 300.0,
   #    which is not a reduced temperature. Set the seed -- the trajectory is
   #    fully determined by it.
   integrator = mc.Integrator(temperature=0.6, step_size_rad=0.1)
   integrator.set_seed(42)

   sim = mc.Simulation(heavy.topology, system, integrator)

   # 4. Give it coordinates. Note the transpose and the nm -> Angstrom factor:
   #    mdtraj stores (n_frames, n_atoms, 3) in nm, the engine wants (3, n_atoms)
   #    in Angstroms, float32.
   sim.context.set_positions((forcefield.coords[0] * 10.0).T.astype(np.float32))

   # 5. Attach reporters. These write files; the library does not print.
   sim.add_energy_reporter("energies.csv", interval=100)
   sim.add_xtc_reporter("traj.xtc", interval=1000)

   # 6. Run.
   sim.step(10_000)

   print(sim.context.energy_breakdown(weighted=True))

The composition pattern
-----------------------

1. :doc:`api/forcefield` — ``MCPUForceField(trajectory, param_set=...)`` turns
   an mdtraj topology into typed, ordered atoms plus loaded potentials.
2. ``forcefield.create_system(topology)`` produces a :doc:`api/system`, with all
   five energy terms already registered. You do not add them individually.
3. :doc:`api/integrator` — ``Integrator(temperature, step_size_rad)`` holds the
   Monte Carlo move parameters.
4. :doc:`api/simulation` — ``Simulation(topology, system, integrator)`` owns the
   :doc:`api/context` and the reporter list, and is what you call ``step()`` on.
5. :doc:`api/reporters` write output; :doc:`api/analysis` reweights it.

Unlike OpenMM, you do not compose the energy terms yourself. The five
knowledge-based potentials are a fitted set that is only meaningful together,
so ``create_system`` registers all of them.

Reading the results
-------------------

``energy_breakdown()`` gives both raw table sums and weighted values, keyed by
energy group:

.. code-block:: python

   breakdown = sim.context.energy_breakdown(weighted=True)

   names = {1: "mu", 2: "backbone_torsion", 3: "sidechain_torsion",
            4: "hydrogen_bond", 5: "aromatic"}
   for group, value in sorted(breakdown["by_group"].items()):
       print(f"{names.get(group, group):20} {value:12.4f}")

Acceptance rates are the first diagnostic to check on any Monte Carlo run — very
low means the step size is too large for the temperature, very high means the
moves are too timid to explore:

.. code-block:: python

   for label, accepted, attempted in [
       ("pivot", integrator.get_bb_accepted(), integrator.get_bb_attempted()),
       ("sidechain", integrator.get_sc_accepted(), integrator.get_sc_attempted()),
       ("KIC", integrator.get_kic_accepted(), integrator.get_kic_attempted()),
   ]:
       rate = accepted / attempted if attempted else float("nan")
       print(f"{label:10} {accepted:6d} / {attempted:6d}  {rate:6.1%}")

The ``EnergyReporter`` is a writer, not a buffer — there is no accessor to read
energies back out of it. Read the CSV:

.. code-block:: python

   import pandas as pd

   energies = pd.read_csv("energies.csv")

Using your own structure
------------------------

``default_example_pdb()`` is a convenience for the bundled 1UAO chignolin
model. Any PDB works — pass it to ``md.load`` instead:

.. code-block:: python

   traj = md.load("my_protein.pdb")

No per-protein preparation step is required: the potentials are general
knowledge-based tables, not per-structure caches.

Where to go next
----------------

* :doc:`tutorials/01_single_trajectory` — the same material as a runnable
  notebook, with commentary
* :doc:`running_remd` — replica exchange, serial and MPI
* :doc:`checkpointing` — surviving a scheduler timeout
* :doc:`cli` — the ``mcpu`` command line
* :doc:`physics_notes/background` — what each energy term computes
