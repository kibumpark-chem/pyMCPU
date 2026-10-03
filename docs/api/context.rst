Context
=======

``Context`` is the mutable half of a pyMCPU simulation. A ``Context``
is created from one :doc:`System <system>`, holds the coordinates and
the running energy, and evaluates the registered
:doc:`potentials <forces>`. The topology never changes; the context
does.

.. note::
   ``Context`` is a compiled ``pymcpu.mcpu_core`` class. The
   documentation build mocks that module, so the directives on this
   page are hand-authored; the signatures were read off the built
   extension.

.. note::
   ``Context`` exposes 67 public members. This page documents the
   user-facing subset. The remainder -- neighbour-list cell sizing and
   Verlet skin knobs (``mu_cell_size_angstrom``, ``mu_skin``,
   ``mu_verlet_enabled``, ``verlet_moved_threshold``,
   ``verlet_partial_threshold``, ``use_cell_pair``,
   ``cell_pair_min_moved``, ``clash_first_min_moved``,
   ``skip_rigid_mm`` and their setters), rebuild counters (``neighbor_aabb_rebuilds``,
   ``neighbor_dense_cap_fallbacks``, ``neighbor_proxy_stats``,
   ``print_neighbor_audit``) and the atom-permutation internals -- are
   **performance and diagnostic knobs and are not part of the stable
   API**. They exist for benchmarking and bug reproduction, they change
   between releases, and correct results must never depend on them.

.. py:currentmodule:: pymcpu

.. py:class:: Context(system)

   :param system: the :py:class:`System` to bind to. This is the
      **only** constructor argument; the integrator is passed to
      :py:meth:`Integrator.run` instead, not to the context. The
      context keeps the system alive for its own lifetime.

   .. code-block:: python

      context = pymcpu.Context(system)
      context.set_positions(coords_3xn)      # float32, shape (3, n)
      context.calculate_total_energy(-1)     # seeds the running total

Core state
----------

.. py:method:: Context.get_system() -> pymcpu.System

   The system this context was built from.

.. py:method:: Context.get_state() -> pymcpu.mcpu_core.State

   The live state object (coordinates, torsions, running energy).

.. py:method:: Context.set_positions(coords, /) -> None

   Replace the coordinates. ``coords`` is a ``float32`` array of shape
   ``(3, n)`` -- **atom-major columns**, i.e. the transpose of MDTraj's
   ``(n, 3)`` frame -- in Ångström, with ``n`` equal to
   ``System.get_num_atoms()``; any other ``n`` raises ``ValueError``. The
   argument is positional-only.

   .. important::
      ``set_positions()`` does not seed the running total energy.
      ``Simulation.step()`` seeds it once per object, but code driving
      a raw ``Context`` must call ``calculate_total_energy(-1)``
      itself afterwards, or the first reported total (and the
      ``total`` column of an energy report) will read 0.

.. py:attribute:: Context.coords

   Read/write ``float32`` ``(3, n)`` array. Writing an array with a
   different ``n`` raises ``ValueError``, as for :py:meth:`Context.set_positions`.

   Coordinates in external (build) order by default;
   ``set_output_internal_order(True)`` for storage order.

.. py:method:: Context.coords_for_python() -> numpy.ndarray

   Method form of reading :py:attr:`Context.coords`; returns a ``float32``
   ``(3, n)`` array.

.. py:method:: Context.set_coords_from_python(coords, /) -> None

   Method form of writing :py:attr:`Context.coords`.

Energy
------

.. py:method:: Context.calculate_total_energy(target_group=-1) -> float

   Weighted total energy: ``sum_g weight[g] * E_raw[g]``. With
   ``target_group >= 0``, returns ``weight[g] * E_raw[g]`` for that
   group alone. ``target_group=-1`` means all groups, and also seeds
   the running total carried in the state.

   The value is a **unitless** sum of knowledge-based table entries.

.. py:method:: Context.calculate_total_energy_raw(target_group=-1) -> float

   As above but without the outer weights.

.. py:method:: Context.calculate_delta_energy(proposed_state, patch) -> float

   Incremental energy of one proposed move. Called by the integrator;
   direct use requires a ``ProposalPatch`` from the proposal
   machinery.

.. py:method:: Context.energy_breakdown(weighted=True) -> dict

   Return per-term energies. ``weighted=True`` uses legacy outer
   weights (incl. HBond ``RDTHREE_CON``); ``weighted=False`` returns
   raw per-potential energies.

   The returned dict has keys ``raw_total``, ``weighted_total``,
   ``by_group`` (a ``dict[int, float]`` keyed by energy group),
   ``by_name`` (the same values keyed by term name, in group order --
   see :py:meth:`System.energy_terms() <pymcpu.mcpu_core.System.energy_terms>`),
   ``weighted`` and ``use_legacy_weights``. Note that ``raw_total``
   and ``weighted_total`` are both always present; ``weighted`` says
   which one ``by_group`` and ``by_name`` were computed with.

.. py:method:: Context.get_energy_weights() -> dict

   The **effective** outer weight of every tracked energy group
   (integer keys 1 through 15), plus ``'use_legacy_weights'`` and
   ``'hbond_rdthree'``. With the legacy defaults this reads
   ``{1: 0.4, 2: 1.35, 3: 2.5, 4: 2.7, 5: 5.0, 6..15: 1.0,
   'hbond_rdthree': 2.0}``.

.. py:method:: Context.set_energy_weight(group_id, w) -> None

   Set outer weight for an energy group. For group 4 (HBond), the
   effective multiplier is ``outer * hbond_rdthree`` when
   ``use_legacy_weights`` is True.

.. py:method:: Context.set_use_legacy_weights(on) -> None

   Turn the legacy weight set on or off. With ``on=False`` every outer
   weight is 1.0 and the group-4 ``hbond_rdthree`` factor is dropped.

.. py:method:: Context.use_legacy_weights() -> bool

   Whether the legacy weights are active. Default ``True``.

.. seealso::
   The full group/weight table is on the :doc:`potentials <forces>`
   page.

Clash and constraint queries
----------------------------

.. py:method:: Context.has_steric_clash() -> bool

   Whether the most recent total-energy evaluation was rejected for a
   hard-core overlap.

.. py:method:: Context.has_hard_constraint_violation() -> bool

   Whether the most recent total-energy evaluation was hard-rejected
   for any reason. Steric clash is currently the only such reason, so
   this and :py:meth:`Context.has_steric_clash` agree today.

.. important::
   Both queries report the result of the **last** total-energy
   evaluation; neither re-evaluates. Call
   :py:meth:`Context.calculate_total_energy` first.

Native-contact bias
-------------------

.. py:method:: Context.set_native_contacts_bias(k_bias, n_target) -> None

   Harmonic umbrella on hard native-contact count N:
   ``U = 0.5 * k_bias * (N - n_target)^2``.

   Requires a :py:class:`~pymcpu.NativeContactsBiasPotential`
   registered on the system (normally via
   ``pymcpu.sampling.attach_native_contacts_bias_potential``). Setting
   the target on the context rather than on the potential is what lets
   one registered potential serve a whole umbrella ladder.

.. py:method:: Context.native_contacts_bias_k() -> float

   The current force constant. Defaults to 0.0, i.e. no bias.

.. py:method:: Context.native_contacts_bias_n_target() -> float

   The current contact-count target ``N0``. Defaults to 0.0.

.. py:method:: Context.set_q_bias(k_bias, n_target) -> None

   Legacy name for :py:meth:`Context.set_native_contacts_bias`; prefer the
   longer name in new code.

.. py:attribute:: Context.mu_potential

   Read-only. First :py:class:`~pymcpu.MuPotential` registered on the
   system, or ``None``.

Fixed residues
--------------

Immobilizing part of a structure is an **integrator** feature, not a
context or potential one: see
:py:meth:`Integrator.set_fixed_residues` (or
``Simulation.set_fixed_residues()``, which supplies the residue count
for you). To drop residues from the energy instead of from the moves,
use :py:meth:`System.set_energy_ignored_residues`.

Atom ordering
-------------

.. py:method:: Context.set_output_internal_order(on) -> None

   When ``True``, :py:attr:`Context.coords` reads and writes in engine storage
   order instead of the external build order.

.. py:method:: Context.output_internal_order() -> bool

   Whether internal-order output is active. Default ``False``.

.. py:method:: Context.set_atom_reorder_mode(mode) -> None

   Atom locality reorder: ``"off"`` (default) or ``"init_only"``.
   ``init_only`` renumbers the System's atoms in place and remaps its
   energy terms. A Context created on that System afterwards (REMD
   replicas share one) adopts the same order and still takes build-order
   coordinates. A Context created *before* the reorder holds coordinates
   in the old order and raises ``RuntimeError`` when used; create it
   again.

.. py:method:: Context.get_atom_reorder_mode() -> str

   The active reorder mode.

.. py:method:: Context.atom_permutation_info() -> dict

   Diagnostic snapshot of the permutation, with keys ``enabled``,
   ``mode``, ``permutation_checksum``, ``n_atoms``, ``n_res``,
   ``int_to_ext`` and ``ext_to_int``.

Backend introspection
---------------------

These report which neighbour-search backend was selected. They are
useful in bug reports and for confirming that no fallback path is in
play; nothing about the physics depends on them.

.. py:method:: Context.mu_backend_name() -> str

   Name of the Mu neighbour backend in use.

.. py:method:: Context.hbond_backend_name() -> str

   Name of the hydrogen-bond neighbour backend in use.

.. py:method:: Context.hbond_index_ok() -> bool

   Whether the hydrogen-bond donor/acceptor index built cleanly.

.. py:method:: Context.hbond_uses_fallback() -> bool

   Whether the hydrogen-bond term fell back off its indexed path.

.. py:method:: Context.coord_sync_stats() -> dict

   Coordinate-materialization counters, with keys
   ``num_coords_eigen_materializations`` and
   ``num_coords_eigen_writes_back``.

.. py:method:: Context.reset_coord_sync_stats() -> None

   Zero the counters above.

Deprecated
----------

.. py:method:: Context.print_neighbor_proxy_stats(tag='neighbor-proxy') -> None

   .. deprecated:: 0.1.0
      Print neighbor-list proxy statistics (developer tuning only).
      Will be removed in a future version.

.. py:method:: Context.set_proxy_print_every(n) -> None

   .. deprecated:: 0.1.0
      Was used to set auto-print interval for neighbor-list stats. Now
      a no-op. Will be removed in a future version.

State
-----

.. py:currentmodule:: pymcpu.mcpu_core

.. py:class:: State

   The coordinate/torsion/energy snapshot returned by
   :py:meth:`pymcpu.Context.get_state`. Not re-exported at the top
   level; reference it as ``pymcpu.mcpu_core.State``.

   .. py:attribute:: coords

      ``float32`` ``(3, n)`` coordinate array, in storage order
      (:py:attr:`pymcpu.Context.coords` uses build order by default; they
      differ after an ``init_only`` atom reorder). Writing it discards this
      state's Mu contact list. To move a ``Context``, use
      :py:meth:`pymcpu.Context.set_positions` or
      :py:attr:`pymcpu.Context.coords`, which also refresh its neighbour
      grids; writing ``ctx.get_state().coords`` does not.

   .. py:attribute:: current_energy

      The running total energy. This is the value an energy report
      writes as ``total``, and it is 0 until something seeds it -- see
      the warning under :py:meth:`pymcpu.Context.set_positions`.

   .. py:attribute:: backbone_torsions

      Per-residue backbone torsion records.

   .. py:attribute:: sidechain_torsions

      Per-residue sidechain torsion records.

Checkpointing
-------------

``Context`` holds no checkpoint state. There is no
``Context.save_checkpoint``, no ``Context.from_checkpoint`` and no
``Context.step_count``. The RNG state a checkpoint round-trips lives on
the integrator (:py:meth:`pymcpu.Integrator.get_rng_state` /
:py:meth:`pymcpu.Integrator.set_rng_state`), the step counter is
``Simulation.current_step``, and the checkpoint files themselves are
written by the pure-Python drivers -- see :doc:`sampling`.
