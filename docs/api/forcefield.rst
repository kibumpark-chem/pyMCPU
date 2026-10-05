Force fields
============

A force field turns an MDTraj structure into a ``mcpu_core.System`` the
engine can simulate: it brings the potentials (the energy terms), their
parameters, and the atom order they need. There are two, and you use one or
the other:

- ``mcpu08``, built by :class:`~pymcpu.MCPUForceField`: all-atom, with the
  five MCPU knowledge-based potentials.
- ``korp``, built by :class:`~pymcpu.KORPForceField`: backbone-only, with the
  KORP 6D orientational potential and a steric filter.

A config names one with ``forcefield:`` (default ``mcpu08``), but so far
only an :doc:`EngineSession <../integrating_pymcpu>` reads it: ``mcpu run``
and the :doc:`sampling` drivers always build mcpu08. Both classes follow
:class:`~pymcpu.forcefields.base.BaseForceField`.

MCPUForceField (mcpu08)
=======================

:class:`~pymcpu.MCPUForceField` builds the mcpu08 force field from a
structure. The constructor loads the mcpu08 parameter set, renames
protonation variants to the standard residues, checks the structure, and
puts the atoms in the order the engine works in.
:meth:`~pymcpu.MCPUForceField.create_system` then builds a system from it, as
many times as you need.

.. code-block:: python

   import mdtraj as md
   import numpy as np
   import pymcpu as mc
   from pymcpu.runners import default_example_pdb

   traj = md.load(str(default_example_pdb()))
   heavy = traj.atom_slice(traj.topology.select("not element H"))

   ff = mc.MCPUForceField(heavy, param_set="mcpu08")
   system = ff.create_system(heavy.topology)

   # Engine coordinates: Å, shape (3, n_atoms), float32.
   positions = (ff.coords[0] * 10.0).T.astype(np.float32)

The parameters are read in the constructor, so a missing or incomplete
parameter directory fails there. Without ``param_dir``, the set named by
``param_set`` is found as described under "MCPU parameter lookup" in
:doc:`../installation`.

The structure must be one continuous chain of at least three residues, with
every heavy atom present; "Using your own structure" in :doc:`../quickstart`
says what happens otherwise.

Class reference
---------------

.. autoclass:: pymcpu.MCPUForceField
   :members:
   :member-order: bysource
   :special-members: __init__

Besides :meth:`~pymcpu.MCPUForceField.create_system`, the class has the
:attr:`~pymcpu.MCPUForceField.inverse_mapping` and
:attr:`~pymcpu.MCPUForceField.output_topology` properties and
:meth:`~pymcpu.MCPUForceField.prepare_trajectory`, which drops hydrogens. The
attributes worth reading are listed under `Attributes`_.

Atom order, ``coords`` and ``inverse_mapping``
----------------------------------------------

The engine keeps all atoms in one order: the backbone ``N``, ``CA`` and ``C``
of every residue, then all backbone oxygens (``O``, ``OXT``, ``OCT``), then
all side-chain atoms. ``MCPUForceField`` builds that order and records where
each atom came from: :attr:`~pymcpu.MCPUForceField.coords` holds the
coordinates in engine order, and
:attr:`~pymcpu.MCPUForceField.inverse_mapping` gives the topology index of
each engine atom.

With the default virtual amide hydrogens, the engine has one slot per heavy
atom, so its atom count equals the topology's: 77 for the bundled chignolin.
Glycine, which has no side chain, has no side-chain atoms in that order. Only
``virtual_amide_h=False`` adds slots, one explicit amide hydrogen per
non-proline residue after the first; the topology has no such atoms, so
:attr:`~pymcpu.MCPUForceField.inverse_mapping` holds ``-1`` for them, and the
XTC reporter skips them.

Pass :attr:`~pymcpu.MCPUForceField.inverse_mapping` to a reporter so that
frames come out in topology order. To go the other way, index a frame in
topology order with the same mapping: ``xyz[ff.inverse_mapping]`` is in
engine order, as a collective variable needs it (with the default virtual
hydrogens).

Units and layout differ between MDTraj and the engine, and converting is up
to you:

============================  ====================================
``ff.coords``                 nanometres, ``(n_frames, n_atoms, 3)``
``Context.set_positions``     Å, ``(3, n_atoms)``
============================  ====================================

so the first frame goes in as
``(ff.coords[0] * 10.0).T.astype(np.float32)``. A structure far from the
origin runs in a shifted frame; see :ref:`context-frame`.

The mcpu08 terms and weights
----------------------------

:meth:`~pymcpu.MCPUForceField.create_system` adds the five mcpu08 potentials
to a new system, one per energy group, and installs the rotamer library. Each
group's energy is multiplied by its weight:

=====  =====================  =============================  ======
Group  Term                   Potential                      Weight
=====  =====================  =============================  ======
1      ``mu``                 ``MuPotential``                0.4
2      ``backbone_torsion``   ``TripletPotential``           1.35
3      ``sidechain_torsion``  ``SidechainTripletPotential``  2.5
4      ``hydrogen_bond``      ``HBondPotential``             2.7
5      ``aromatic``           ``AromaticPotential``          5.0
=====  =====================  =============================  ======

The hydrogen-bond weight is set as 1.35, and the term is doubled as in legacy
MCPU, so it counts 2.7. The weights are dimensionless, like the energies.
These are the mcpu08 values; another MCPU parameter set can change the
weights or a potential.

The Ramachandran mixture library is loaded too when the parameter set has one
(``constants/rama_mixture.json``). Only the ``rama_pivot`` backbone move uses
it, and that move is off by default, so a set without the library still
loads.

Attributes
----------

Set during construction and safe to read:

.. py:attribute:: pymcpu.MCPUForceField.coords
   :type: numpy.ndarray

   Reordered coordinates in the engine's atom order, in nanometres,
   shape ``(n_frames, n_atoms, 3)``, float32.

.. py:attribute:: pymcpu.MCPUForceField.n_atoms
   :type: int

   Number of atoms in the engine layout: ``topology.n_atoms``, plus
   :attr:`~pymcpu.MCPUForceField.total_h_atoms` explicit amide hydrogens
   when ``virtual_amide_h=False``.

.. py:attribute:: pymcpu.MCPUForceField.n_res
   :type: int

   Number of residues.

.. py:attribute:: pymcpu.MCPUForceField.total_bb_atoms
   :type: int

   Size of the backbone ``N``/``CA``/``C`` segment, which begins at
   index 0.

.. py:attribute:: pymcpu.MCPUForceField.total_o_atoms
   :type: int

   Size of the backbone-oxygen segment, which follows the backbone
   segment.

.. py:attribute:: pymcpu.MCPUForceField.total_sc_atoms
   :type: int

   Size of the sidechain segment, which follows the oxygen segment.

.. py:attribute:: pymcpu.MCPUForceField.total_h_atoms
   :type: int

   Number of explicit amide hydrogens appended to the layout. ``0``
   with the default ``virtual_amide_h=True``.

.. py:attribute:: pymcpu.MCPUForceField.atom_to_res
   :type: list[int]

   Residue index for each engine atom index.

.. py:attribute:: pymcpu.MCPUForceField.ordered_indices
   :type: list[int]

   Original MDTraj atom index for each heavy-atom engine slot, in
   engine order. Unlike :attr:`~pymcpu.MCPUForceField.inverse_mapping`
   it has no entries for explicit amide hydrogens.

.. py:attribute:: pymcpu.MCPUForceField.ordered_atom_list
   :type: list[MCPUAtom]

   Per-atom identity records in engine order (see
   :class:`~pymcpu.forcefields.mcpu.MCPUAtom`).

.. py:attribute:: pymcpu.MCPUForceField.amino_index
   :type: list[int]

   Engine amino-acid type code per residue.

.. py:attribute:: pymcpu.MCPUForceField.is_proline
   :type: list[int]

   ``1`` for proline residues, ``0`` otherwise.

.. py:attribute:: pymcpu.MCPUForceField.secondary_structure
   :type: str

   Per-residue legacy secstr string (``H``/``E``/``C``) when
   ``compute_dssp=True``, otherwise ``""``, which the engine reads as
   all-coil.

.. py:attribute:: pymcpu.MCPUForceField.param_dir
   :type: pathlib.Path

   Resolved parameter root, containing ``constants/`` and
   ``mcpu_params/``.

The constructor arguments ``param_set``, ``virtual_amide_h``,
``compute_dssp``, ``dssp_coil_state`` and ``allow_provisional_rama`` are kept
as attributes of the same name. Everything else on the instance, such as the
loaded parameter tables and the per-residue bookkeeping, is internal and may
change.

Supporting types
----------------

.. autoclass:: pymcpu.forcefields.mcpu.MCPUAtom
   :members:

KORPForceField (korp)
=====================

:class:`~pymcpu.KORPForceField` is a backbone-only force field built on
KORP's 6D orientational potential. KORP reads only N, CA and C, so this force
field drops the side chains: the engine holds N, CA, C and O, with O kept
only so that the moves work as they do for MCPU.

.. warning::

   **The trajectory you get back has no side chains.** Load a trajectory
   written from this force field against
   :attr:`~pymcpu.KORPForceField.output_topology`, not against your input's
   topology. Save that topology next to the trajectory::

       import mdtraj as md
       md.Trajectory(ff.coords[:1], ff.output_topology).save_pdb("top.pdb")

.. warning::

   **Switch the side-chain moves off.** These residues have no side-chain
   torsions, so call ``integrator.set_move_weights(pivot, kic, 0.0)``;
   :py:meth:`Integrator.run` raises an error otherwise.

The energy map is not shipped with pyMCPU: at 316 MiB it is over PyPI's file
size limit. Download it once and point ``KORP_MAP_PATH`` at it, as described
in :ref:`korp-map`.

.. code-block:: python

   import mdtraj as md
   import numpy as np
   import pymcpu as mc
   from pymcpu.forcefields.korp import KORPForceField

   traj = md.load("protein.pdb")
   ff = KORPForceField(traj)                       # or map_path=...
   system = ff.create_system(traj.topology)

   integrator = mc.Integrator(temperature=0.6, step_size_rad=0.05)
   integrator.set_move_weights(0.5, 0.5, 0.0)      # backbone moves only
   sim = mc.Simulation(ff.output_topology, system, integrator)
   sim.context.set_positions((ff.coords[0] * 10.0).T.astype(np.float32))
   ff.apply_energy_weights(sim.context)
   sim.step(10_000)

.. autoclass:: pymcpu.forcefields.korp.KORPForceField
   :members: create_system, apply_energy_weights, inverse_mapping

.. py:attribute:: KORPForceField.output_topology

   The backbone-only MDTraj topology the engine actually simulates.
   Load any trajectory this force field produces against *this*, not
   against the input.

.. seealso::

   :doc:`/physics_notes/korp_6d`
       The potential itself: frame, coordinates, binning, and the
       paper-versus-code discrepancy in the frame definition.

.. autoclass:: pymcpu.forcefields.base.BaseForceField
   :members:

.. seealso::

   :doc:`simulation`
       Binds the system produced here to an integrator and runs it.
   :doc:`system`, :doc:`context`
       The compiled objects ``create_system`` produces and populates.
   :doc:`reporters`
       Where :attr:`~pymcpu.MCPUForceField.inverse_mapping` is used.
