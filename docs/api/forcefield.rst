Force fields
============

A force field turns an MDTraj structure into a ``mcpu_core.System`` the
engine can simulate: it brings the potentials (the energy terms), their
parameters, and the atom order they need. pyMCPU has one, ``mcpu08`` (alias
``mcpu``), built by :class:`~pymcpu.MCPUForceField`: all-atom, with the five
MCPU knowledge-based potentials.

A config names it with ``forcefield:`` (default ``mcpu08``) and passes
constructor arguments in ``forcefield_options:``. ``mcpu run``, the
:doc:`sampling` drivers and an :doc:`EngineSession <../integrating_pymcpu>`
all build it through :func:`pymcpu.forcefields.load_forcefield`; an unknown
name fails when the config is loaded. A force field follows
:class:`~pymcpu.forcefields.base.BaseForceField`.

.. autofunction:: pymcpu.forcefields.load_forcefield

MCPUForceField (mcpu08)
=======================

:class:`~pymcpu.MCPUForceField` builds the mcpu08 force field from a
structure. The constructor loads its parameter set (mcpu08 unless
``param_set`` names another, see `The mcpu26 parameter set`_), renames
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
parameter directory fails there. So does a residue template
(``constants/standard_amino_acids.json``) in which a residue lacks, for any
chi angle it has (its ``ntorsions``, which is how the engine counts them),
the four atoms that define the angle (``chi_atoms``) or the atoms it moves
(``chi_moved_atoms``), one row per chi in both. The side-chain moves turn
only the atoms in the ``chi_moved_atoms`` rows, and the side-chain torsion
energy reads each chi from its ``chi_atoms`` row, so a chi angle without its
rows would never change, and the energies would not show it. Without ``param_dir``, the set named by
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
:meth:`~pymcpu.MCPUForceField.prepare_trajectory`, which drops hydrogens.
:attr:`~pymcpu.MCPUForceField.param_signature` identifies the parameter files
it read; checkpoints record it with the parameter set, so a run cannot resume
with other tables (see :doc:`../checkpointing`). Taking it reads every
parameter file in full, so a force field takes it only when asked, and a run
only when it first saves or resumes a checkpoint. The attributes worth
reading are listed under `Attributes`_.

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
to a new system, one per energy group, and installs the default rotamer
library (see `Rotamer libraries`_). Each group's energy is multiplied by its
weight:

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
They are the same for every MCPU parameter set: a set that needs another
weight multiplies it into its table (see `A parameter set's own defaults`_).

Within the hydrogen-bond term, a bond between residues more than four apart
is multiplied by the sheet factor ``hbond_beta_favor``. A force field that
is not given one takes the parameter set's own, or 2.0 when the set gives
none, as for mcpu08. Legacy MCPU compiled 3.0 (``beta_favor`` in its
``define.h``); ``MCPUForceField(..., hbond_beta_favor=3.0)``, or
``forcefield_options: {hbond_beta_favor: 3.0}`` in a config, scores hydrogen
bonds as it did, whatever the set says. Bonds four apart, the helical ones,
are not scaled.

The Ramachandran mixture library is loaded too when the parameter set has one
(``constants/rama_mixture.json``). Only the ``rama_pivot`` backbone move uses
it, and that move is off by default, so a set without the library still
loads.

The hard core
-------------

Two atoms that can clash may come no closer than a hard core, a factor times
the sum of their radii; a move that would put them closer is rejected. Three
options set it. The defaults are mcpu08's, and checkpoints record all three,
so a run resumes only with the values it started with.

- ``clash_alpha`` (0.75): the factor. It must be above 0 and below 1.35; the
  contact cutoff stays at 1.35 radius sums.
- ``clash_bond_exclusion`` (0): 1, 2 or 3 leaves out of the clash check every
  pair at most that many covalent bonds apart, by the residue templates and
  the start structure's disulfide bridges (two CYS SG atoms under 2.5 Å
  apart). mcpu08's rules already leave out bonded pairs, but not all pairs
  two or three bonds apart, whose distance a bond angle or one torsion sets.
  A bridge the start structure lacks is not counted; at ``clash_alpha`` 0.85
  its CB-SG pairs, about 3.0 Å apart in a bridge, are inside the hard core.
  MCPU has no S-S bond energy, so no setting holds a bridge together during
  a run.
- ``clash_start_exemption`` (True): a pair already inside its hard core in the
  start structure (a start overlap) is exempt for the whole run. False
  penalises it instead.

To relax a start structure, build it with ``clash_start_exemption: false``.
Each start overlap then adds 1000 to the raw ``mu`` energy (400 at its
weight) while it stays inside its hard core, and every other pair keeps the
hard core. In a cold run of local moves, for example ``move_weights: [0,
0.5, 0.5]`` (loop closure and side chains) at temperature 0.1, a move that
takes an overlap out lowers the energy by the penalty, and one that puts it
back raises it by as much. Such a run takes out the overlaps a local move can
reach, but not always all of them: an atom caged by its neighbours can stay
inside for millions of steps. So check how many are left:
:func:`pymcpu.simulation.start_overlaps` (``MuPotential.count_start_overlaps``)
gives the count, and a folding run prints it at its start and end. The run's
energies include 400 for each overlap still inside.
Start production runs from the relaxed structure, with the same
``clash_alpha`` and ``clash_bond_exclusion``. With the default
``clash_start_exemption``, an overlap still inside is exempt for that whole
run.

A parameter set's own defaults
------------------------------

A parameter folder can describe its set in ``constants/paramset.json``
(format ``pymcpu-paramset/1``; mcpu08 has none):

- its ``name``, which must be the set the folder is loaded as
  (``param_set``), and optionally a ``description``. A force field refuses a
  folder whose manifest names another set, as the folder's tables and
  defaults are that set's;
- ``hbond_beta_favor``, the set's sheet factor, which
  :class:`~pymcpu.MCPUForceField` uses when it is given none. A factor given
  explicitly, as an argument or in ``forcefield_options``, wins;
- ``folded_weights``: for any of the five terms, the factor
  w_new / w_legacy that the set multiplied into that term's table, 0 or
  above (0 for a term the fit switched off). It is a record only. The engine
  applies nothing, and the group weights above stay as they are;
- ``provenance``, of any content, except that an ``md5`` entry lists files
  of the folder with the md5 each had when the set was written.

The force field reads the file before the tables. It refuses one with an
unknown key, a key given twice, a value of the wrong kind, or a listed file
that is missing or no longer has its md5, and the error names the file.
Checking the md5s reads each listed file the first time. The md5 and the
sha256 (for :attr:`~pymcpu.MCPUForceField.param_signature`) come from that
one read and are kept under the cache root (``MCPU_CACHE_DIR``, in
``params/digests.json``) while the file is unchanged, so later processes do
not read it again. ``verify_provenance=False`` builds the force field
without comparing them, and :meth:`~pymcpu.MCPUForceField.check_provenance`
compares them later; MPI replica exchange does so on rank 0 alone. The
checked contents are
:attr:`~pymcpu.MCPUForceField.paramset_manifest`. The file counts in
:attr:`~pymcpu.MCPUForceField.param_signature`, so a checkpoint resumes only
with the same file, and checkpoints record the sheet factor a run used,
wherever it came from (see :doc:`../checkpointing`). The file ships with its
set: ``scripts/encode_params.py`` copies it next to the table archive, and a
wheel install unpacks it with the tables, so a set reads the same from a
wheel as from a checkout or ``MCPU_PARAMS_DIR``.

.. autofunction:: pymcpu.params.load_paramset_manifest

.. autofunction:: pymcpu.params.check_paramset_provenance

The mcpu26 parameter set
------------------------

pyMCPU ships a second parameter set, ``mcpu26`` (legacy form): mcpu08's
energy function, with its tables re-estimated from PISCES 2026 and its weights
fitted on the folding landscapes of natural domains from Tsuboyama et al.
(2023), then folded into the tables. Name it with ``param_set``:

.. code-block:: python

   ff = mc.MCPUForceField(heavy, param_set="mcpu26")

``param_set: mcpu26`` does the same in a config, and
``mcpu materialize-params --set mcpu26`` unpacks it before an MPI job. Its
``constants/paramset.json`` gives its sheet factor, 2.3994, which a force
field uses when it is given none, and records the weights folded into its
tables (``mu`` 1.058, ``bb_torsion`` 0.820, ``sc_torsion`` 0.184, ``hbond``
1.158, ``aromatic`` 0.863) and the md5 of each table. A folder of the set
without that file is incomplete. Its constants are mcpu08's. Its tables are
in ``mcpu26_params/``, not ``mcpu_params/``, so a folder of one set given for
the other, as ``MCPU_PARAMS_DIR`` or ``param_dir``, fails.

Its weights were fitted with ``clash_alpha`` 0.85 and ``clash_bond_exclusion``
3 (see `The hard core`_), with start overlaps exempt in runs and penalised
when a start structure was relaxed. Pass the same options to run it as it was
fitted: ``forcefield_options: {clash_alpha: 0.85, clash_bond_exclusion: 3}``.

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

.. py:attribute:: pymcpu.MCPUForceField.hbond_beta_favor
   :type: float

   The hydrogen-bond sheet factor the systems it builds use: the
   ``hbond_beta_favor`` argument, or, when that is None, the parameter
   set's own (see `A parameter set's own defaults`_), else 2.0.

.. py:attribute:: pymcpu.MCPUForceField.clash_alpha
   :type: float

   The hard-core factor (see `The hard core`_).

.. py:attribute:: pymcpu.MCPUForceField.clash_bond_exclusion
   :type: int

   Pairs at most this many covalent bonds apart cannot clash; 0 keeps
   mcpu08's rules.

.. py:attribute:: pymcpu.MCPUForceField.clash_start_exemption
   :type: bool

   Whether start overlaps are exempt (True) or penalised (False).

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

   Resolved parameter root, laid out as ``param_set``: ``constants/`` and
   the set's table folder (``mcpu_params/`` for mcpu08, ``mcpu26_params/``
   for mcpu26).

.. py:attribute:: pymcpu.MCPUForceField.paramset_manifest
   :type: dict | None

   The checked contents of the folder's ``constants/paramset.json``
   (:func:`~pymcpu.params.load_paramset_manifest`), or ``None`` when it has
   none, as for mcpu08.

The constructor arguments ``param_set``, ``virtual_amide_h``,
``compute_dssp``, ``dssp_coil_state`` and ``allow_provisional_rama`` are kept
as attributes of the same name. Everything else on the instance, such as the
loaded parameter tables and the per-residue bookkeeping, is internal and may
change.

Rotamer libraries
-----------------

The rotamer sidechain move draws from a rotamer library held by the system.
:meth:`~pymcpu.MCPUForceField.create_system` installs
``dunbrack2002_sym``, built from the parameter set's 2002 library
(``constants/bbind02.May.lib``); a run selects the other proposal,
``dunbrack2002`` (legacy MCPU's draw), with the ``rotamer_library`` setting.
In Python, build one with this function and pass it to
``System.set_rotamer_library``. See :doc:`../physics_notes/rotamer_proposal`.

.. autofunction:: pymcpu.forcefields.builders.rotamer_builder.load_rotamer_library

Supporting types
----------------

.. autoclass:: pymcpu.forcefields.mcpu.MCPUAtom
   :members:

BaseForceField
==============

.. autoclass:: pymcpu.forcefields.base.BaseForceField
   :members:

.. seealso::

   :doc:`simulation`
       Binds the system produced here to an integrator and runs it.
   :doc:`system`, :doc:`context`
       The compiled objects ``create_system`` produces and populates.
   :doc:`reporters`
       Where :attr:`~pymcpu.MCPUForceField.inverse_mapping` is used.
