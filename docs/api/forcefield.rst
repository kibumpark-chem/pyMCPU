Force fields
============

A force field turns an MDTraj topology into a simulatable
``mcpu_core.System``. Two are available, and they are alternatives
rather than layers: :class:`~pymcpu.MCPUForceField` (all-atom, the five
MCPU knowledge-based terms) and :class:`~pymcpu.KORPForceField`
(backbone-only, the KORP 6D orientational potential plus a steric
filter). Both implement the one-method
:class:`~pymcpu.forcefields.base.BaseForceField` contract.

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

   ff = mc.MCPUForceField(heavy, param_set="mcpu08")
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

Each heavy atom has one engine slot, so with the default virtual amide
hydrogens the engine atom count equals the topology's: 77 for the shipped
``1uao.pdb``. Glycine, which has no sidechain, simply has an empty
sidechain block. Only ``virtual_amide_h=False`` adds slots, one explicit
amide hydrogen per non-proline residue after the first; the topology has
no such atoms, so :attr:`~pymcpu.MCPUForceField.inverse_mapping` holds
``-1`` for them, which tells ``XtcReporter`` to skip them. Pass
``inverse_mapping`` to the reporter so frames come out in topology order.

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

KORPForceField
==============

:class:`~pymcpu.KORPForceField` is a **backbone-only** force field built
on KORP's 6D orientational potential. KORP reads only N, CA and C -- the
pair coordinate is CA-CA -- so rather than carry sidechains along unused
this force field drops them: the engine sees N, CA, C and O only, with O
kept purely so the existing move machinery and segment bookkeeping work
unchanged.

.. warning::

   **The trajectory you get back is not the trajectory you put in.**
   Sidechain atoms are gone, so an XTC written from this force field
   must be loaded against
   :attr:`~pymcpu.KORPForceField.output_topology`, not against your
   input PDB's topology. Save that topology next to the trajectory::

       import mdtraj as md
       md.Trajectory(ff.coords[:1], ff.output_topology).save_pdb("top.pdb")

.. warning::

   **Sidechain moves must be switched off.** These residues have no chi
   angles, so every sidechain proposal would return without proposing
   anything. Call ``integrator.set_move_weights(pivot, kic, 0.0)``;
   :py:meth:`Integrator.run` raises rather than silently discarding that
   share of the run.

The energy map is **not distributed with pyMCPU**: at 316 MiB it is well
over PyPI's per-file limit. Obtain it once and point ``KORP_MAP_PATH`` at
it; the error raised when it is missing says exactly how. See the
project README for the download and checksum.

.. code-block:: python

   import mdtraj as md, numpy as np, pymcpu as mc
   from pymcpu import mcpu_core
   from pymcpu.forcefields.korp import KORPForceField

   traj = md.load("protein.pdb")
   ff = KORPForceField(traj)                 # or map_path=...
   system = ff.create_system(traj.topology)

   integrator = mcpu_core.Integrator(temperature=0.6, step_size_rad=0.05)
   integrator.set_move_weights(0.5, 0.5, 0.0)     # backbone moves only
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
