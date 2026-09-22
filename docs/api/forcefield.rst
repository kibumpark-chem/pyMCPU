MCPUForceField
==============

:class:`~pymcpu.MCPUForceField` is the bridge between an MDTraj
topology and the C++ engine: it is the only object that turns a
topology into a simulatable ``mcpu_core.System``, so every simulation
starts here. It is constructed once from a trajectory -- which loads
the parameter set, canonicalises residue names, validates the topology,
and reorders atoms into the engine's global layout -- and then builds
one or more systems with
:meth:`~pymcpu.MCPUForceField.create_system`.

.. code-block:: python

   import mdtraj as md
   import numpy as np
   import pymcpu as mc

   traj = md.load("examples/data/1uao.pdb")
   heavy = traj.atom_slice(traj.topology.select("not element H"))

   ff = mc.MCPUForceField(heavy, param_set="mcpu_v1")
   system = ff.create_system(heavy.topology)

   # Engine coordinates: Angstrom, shape (3, n_atoms), float32.
   positions = (ff.coords[0] * 10.0).T.astype(np.float32)

.. note::
   The parameter set is read in the constructor, so a missing or
   incomplete parameter directory fails there rather than in
   :meth:`~pymcpu.MCPUForceField.create_system`. When ``param_dir`` is
   omitted the set named by ``param_set`` is resolved through
   ``pymcpu.params.ensure_params``.

Class reference
---------------

.. autoclass:: pymcpu.MCPUForceField
   :members:
   :member-order: bysource
   :special-members: __init__

The public surface is deliberately small: one method
(:meth:`~pymcpu.MCPUForceField.create_system`) and one property
(:attr:`~pymcpu.MCPUForceField.inverse_mapping`). Everything else a
caller needs is an attribute set during construction; the useful ones
are listed below.

Atom ordering, ``coords`` and ``inverse_mapping``
-------------------------------------------------

The engine requires a single global atom order: all backbone
``N``/``CA``/``C`` for every residue, then all backbone oxygens
(``O``, ``OXT``, ``OCT``), then all sidechain atoms in template order.
``MCPUForceField`` builds that order at construction time, records the
originating MDTraj index for each atom, and exposes the result as
:attr:`~pymcpu.MCPUForceField.coords` (reordered coordinates) and
:attr:`~pymcpu.MCPUForceField.inverse_mapping` (internal index ->
topology index).

Two consequences matter in practice:

* The engine atom count can exceed the topology's. Glycine has no
  sidechain, so its ``CA`` is listed a second time as that residue's
  sidechain entry. The shipped ``1uao.pdb`` has 10 residues, 3 of them
  glycine, so its 77 heavy atoms become 80 engine atoms.
* :attr:`~pymcpu.MCPUForceField.inverse_mapping` holds ``-1`` for those
  duplicate entries, which tells ``XtcReporter`` to skip the coordinate
  instead of writing it twice. Pass it to the reporter so frames come
  out in topology order with the original atom count.

Units and layout differ between MDTraj and the engine, and the
conversion is the caller's job:

============================  ====================================
``ff.coords``                 nanometres, ``(n_frames, n_atoms, 3)``
``Context.set_positions``     Angstrom, ``(3, n_atoms)``, float32
============================  ====================================

so the first frame is passed as
``(ff.coords[0] * 10.0).T.astype(np.float32)``.

What ``create_system`` installs
-------------------------------

:meth:`~pymcpu.MCPUForceField.create_system` transfers the per-residue
bookkeeping (block indices, torsion counts, chi atom indices, downstream
cache, amino-acid indices, proline flags, secondary structure) onto a
fresh ``mcpu_core.System``, installs the rotamer library, and adds
exactly five potentials, one per energy group:

=====  =============================  ====================
Group  Potential                      Legacy outer weight
=====  =============================  ====================
1      ``MuPotential``                0.4
2      ``TripletPotential``           1.35
3      ``SidechainTripletPotential``  2.5
4      ``HBondPotential``             1.35 (effective 2.7)
5      ``AromaticPotential``          5.0
=====  =============================  ====================

The hydrogen-bond row is the subtle one: 1.35 is the configured weight,
but the group's effective multiplier is 2.7, because the legacy
``RDTHREE_CON`` factor of 2.0 is folded in. Energies are unitless sums
of knowledge-based table entries, so these weights are dimensionless.

The Ramachandran mixture library is loaded as well when the parameter
set ships ``constants/rama_mixture.json``; it is optional, because the
knowledge-based backbone pivot it serves is off by default
(``pivot_rama_probability = 0.0``). Without it that move is unavailable
and construction still succeeds.

Attributes
----------

Set during construction and safe to read:

.. py:attribute:: pymcpu.MCPUForceField.coords
   :type: numpy.ndarray

   Reordered coordinates in the engine's atom order, in nanometres,
   shape ``(n_frames, n_atoms, 3)``, float32.

.. py:attribute:: pymcpu.MCPUForceField.n_atoms
   :type: int

   Number of atoms in the engine layout (see the glycine note above:
   this can exceed ``topology.n_atoms``).

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

   Original MDTraj atom index for each engine atom index, before the
   ``to_write`` filtering that
   :attr:`~pymcpu.MCPUForceField.inverse_mapping` applies.

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
``compute_dssp``, ``dssp_coil_state`` and ``allow_provisional_rama`` are
also retained as attributes of the same name. Everything else on the
instance -- the loaded parameter tables (``mu_energies``, ``bb_triplet``,
``sc_triplet``, ``hbond``, ``hbond_seq_dep``, ``aromatic``,
``rotamer_lib_raw``, ``rama_mixture_raw``), the residue template
``ff_template``, and the block bookkeeping ``blocks``, ``downstream``,
``chi_atom_indices``, ``chi_moved_atom_ranges`` -- is internal and may
change without notice.

Supporting types
----------------

.. autoclass:: pymcpu.forcefields.mcpu.MCPUAtom
   :members:

.. autoclass:: pymcpu.forcefields.base.BaseForceField
   :members:

.. seealso::

   :doc:`simulation`
       Binds the system produced here to an integrator and runs it.
   :doc:`system`, :doc:`context`
       The compiled objects ``create_system`` produces and populates.
   :doc:`reporters`
       Where :attr:`~pymcpu.MCPUForceField.inverse_mapping` is used.
