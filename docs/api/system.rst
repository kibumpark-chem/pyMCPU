System
======

``System`` is the immutable-topology half of a pyMCPU simulation: it
owns the atom/residue layout, the per-residue chemistry and torsion
bookkeeping, the move libraries, and the list of registered
:doc:`potentials <forces>`. Coordinates and energies live in
:doc:`Context <context>`, never here.

You normally do not construct a ``System`` yourself.
``MCPUForceField.create_system(topology)`` builds one from an MDTraj
topology: it sizes the object, fills every table below from the
parameter files, and registers the five physics potentials.

.. note::
   ``System`` is a compiled ``pymcpu.mcpu_core`` class. The
   documentation build mocks that module, so the directives on this
   page are hand-authored; the signatures were read off the built
   extension. The class and its methods carry **no docstrings**, so
   every description here is written by hand rather than extracted.

.. note::
   Many setters were bound without ``py::arg`` names, so pybind11
   exposes no keywords for them and they must be called positionally.
   Those are marked with a trailing ``, /`` in the signatures below.

.. py:currentmodule:: pymcpu

.. py:class:: System(n_atoms, n_residues, /)

   :param n_atoms: number of atom slots in the engine layout.
   :param n_residues: number of residues.

   ``repr()`` renders as ``<System: 80 atoms, 10 residues>``.

   .. rubric:: Size and layout

   .. py:method:: get_num_atoms() -> int

      Number of atom slots in the engine layout. This is not
      necessarily the atom count of the input topology: the builder can
      allocate extra slots (for example for virtual amide hydrogens),
      and ``MCPUForceField.inverse_mapping`` uses ``-1`` for engine
      slots with no counterpart in the input topology.

   .. py:method:: get_num_residues() -> int

      Number of residues.

   .. py:method:: get_total_h_atoms() -> int

      Number of explicit hydrogen slots. This is 0 when virtual amide
      hydrogens are in use, since then no explicit H is stored.

   .. py:method:: residue_contiguous_layout() -> bool

      Whether every residue's atoms occupy one contiguous index range.
      The default builder layout is category-segmented -- all backbone
      atoms, then carbonyl oxygens, then sidechain atoms, then
      hydrogens -- so this reports ``False``. It becomes ``True`` only
      after an atom-locality reorder has regrouped the blocks per
      residue. Several neighbour-list and Mu fast paths branch on it.

   .. py:attribute:: atom_to_residue

      Read/write ``list[int]`` of length ``get_num_atoms()``: the
      residue index owning each atom slot.

   .. py:method:: set_atom_counts(total_bb, total_o, total_sc, total_h, /) -> None

      Record the per-category atom counts (backbone, carbonyl oxygen,
      sidechain, hydrogen) that the block bookkeeping is built against.

   .. rubric:: Potential registry

   .. py:method:: add_potential(potential, /) -> int

      Register a :py:class:`~pymcpu.mcpu_core.Potential` and return its
      index in the registry. Passing ``None`` raises ``ValueError``.

   .. py:method:: get_potentials() -> list

      The registered potentials, in registration order.

      .. note::
         pybind11 renders the element type of this method and of
         ``add_potential`` using the raw C++ name ``mcpu::Potential``
         rather than ``pymcpu.mcpu_core.Potential``. The objects
         themselves are ordinary ``Potential`` instances.

   .. py:method:: energy_terms() -> dict[int, str]

      The energy terms as ``{group: name}``, sorted by group -- for MCPU
      ``{1: 'mu', 2: 'backbone_torsion', 3: 'sidechain_torsion',
      4: 'hydrogen_bond', 5: 'aromatic'}``. A group whose potentials have no
      name is reported as ``'group_<n>'``. These names label
      ``energy_breakdown()['by_name']`` and the energy CSV columns.

      ``add_potential`` raises ``ValueError`` if a potential would give one
      group two names, or use one name for two groups.

   .. rubric:: Residue chemistry

   .. py:method:: amino_index(res_id) -> int

      Amino-acid type index of residue ``res_id``.

   .. py:method:: set_amino_index(indices) -> None

      Set all amino-acid type indices from a ``Sequence[int]``.

   .. py:method:: is_proline(res_id) -> bool

      Whether residue ``res_id`` is a proline. Proline needs special
      handling in the pivot and sidechain proposals.

   .. py:method:: set_is_proline(flags) -> None

      Set the proline flags from a ``Sequence[int]``.

   .. py:method:: secondary_structure(res_id) -> str

      One-character secondary-structure code for residue ``res_id``.

   .. py:method:: set_secondary_structure(ss) -> None

      Set every residue's code from one string of length
      ``get_num_residues()``.

   .. rubric:: Torsion bookkeeping

   .. py:method:: get_torsions_per_residue() -> list[int]

      Number of sidechain chi torsions per residue (0 for glycine and
      alanine).

   .. py:method:: set_torsions_per_residue(counts, /) -> None

      Set the per-residue chi counts from a ``Sequence[int]``.

   .. py:method:: get_chi_atom_indices() -> list

      Shape ``[n_residues][4][4]``: the four atom indices defining each
      of the (up to four) chi torsions of each residue.

   .. py:method:: set_chi_atom_indices(indices, /) -> None

      Set the chi atom indices; same nesting as the getter.

   .. py:method:: get_chi_moved_atom_ranges() -> list

      Shape ``[n_residues][4][2]``: the ``[begin, end)`` atom range
      moved when each chi torsion is rotated.

   .. py:method:: set_chi_moved_atom_ranges(ranges, /) -> None

      Set the moved-atom ranges; same nesting as the getter.

   .. rubric:: Block bookkeeping and move libraries

   .. py:method:: get_block_indices() -> list

      One ``pymcpu.mcpu_core.BlockIndices`` record per residue, giving
      the backbone/carbonyl/sidechain/hydrogen sub-ranges the geometry
      kernels index through.

   .. py:method:: set_block_indices(blocks, /) -> None

      Install the per-residue block records.

   .. py:method:: set_downstream_cache(cache, /) -> None

      Install the precomputed downstream-atom cache used to decide
      which atoms a torsion rotation moves.

   .. py:method:: get_rotamer_library() -> RotamerLibrary

      The rotamer library backing the ``rotamer_library`` sidechain
      move mode.

   .. py:method:: set_rotamer_library(library, /) -> None

      Install the rotamer library.

   .. py:method:: get_rama_mixture_library() -> RamaMixtureLibrary

      The Ramachandran mixture library backing the knowledge-based
      ``(phi, psi)`` pivot proposal.

   .. py:method:: set_rama_mixture_library(library, /) -> None

      Install the Ramachandran mixture library.

   .. rubric:: Loop-closure (KIC) targets

   .. py:method:: set_kic_reference(start_coords) -> None

      Store the bond lengths, bond angles and peptide omegas that the
      KIC loop-closure move closes every window to. They are measured
      once, in double precision, from ``start_coords``: a
      ``(3, n_atoms)`` float32 array in Angstrom, in build order -- the
      same array the context is positioned with.
      ``MCPUForceField.create_system`` and ``KORPForceField.create_system``
      call this for you. A ``System`` built by hand must call it before a
      run that uses KIC moves; KIC raises ``RuntimeError`` otherwise,
      rather than measure the targets from whatever the chain looks like
      at the time. Only internal coordinates are stored, so
      ``Context.set_positions`` (a replica swap, a checkpoint restore)
      can never change them. Raises ``ValueError`` for the wrong shape,
      and ``RuntimeError`` if called before ``set_block_indices`` or
      after a context has reordered the atoms.

   .. py:method:: has_kic_reference() -> bool

      Whether ``set_kic_reference`` has been called.

   .. py:method:: get_kic_reference() -> dict

      The stored targets, in Angstrom and radians: ``len_na``,
      ``len_ac`` and ``ang_nac`` (N-CA, CA-C, N-CA-C) per residue, and
      ``len_cn``, ``ang_acn``, ``ang_cna`` and ``omega`` per peptide bond
      ``k -> k+1``.

   .. rubric:: Virtual amide hydrogens

   .. py:method:: virtual_amide_h() -> bool

      Whether backbone amide hydrogens are reconstructed on demand
      from the N, CA and preceding C positions instead of being stored
      as explicit atoms.

   .. py:method:: set_virtual_amide_h(on) -> None

      Enable or disable virtual amide hydrogens. Set by
      ``MCPUForceField(..., virtual_amide_h=...)``; changing it after
      the system is built invalidates the block bookkeeping.

   .. rubric:: Energy masking

   Masking removes residues from the *energy*, unlike
   :py:meth:`Integrator.set_fixed_residues`, which removes them from
   the *moves*.

   .. py:method:: set_energy_ignored_residues(residues, mode='ignore_all') -> None

      Mask the given residues (0-based engine indices).

      ``mode='ignore_all'`` drops every energy contribution involving
      those residues. ``mode='clash_only'`` keeps the hard-core clash
      test but drops the contact and directional terms. Any other value
      raises ``ValueError``.

   .. py:method:: clear_energy_ignored_residues() -> None

      Remove the mask and reset the mode to ``'ignore_all'``.

   .. py:method:: is_residue_energy_ignored(res) -> bool

      Whether residue ``res`` is currently masked.

   .. py:method:: energy_mask_mode() -> str

      The active mode, ``'ignore_all'`` or ``'clash_only'``. This
      reports the stored mode even when no residues are masked.

.. seealso::
   :doc:`context` for coordinates and energy evaluation,
   :doc:`forces` for the potentials a system holds, and
   :doc:`../architecture/topology_bridge` for how an MDTraj topology is
   translated into the tables above.
