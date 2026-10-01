Integrator
==========

``Integrator`` drives the Metropolis Monte Carlo loop: it proposes
moves, asks a :doc:`Context <context>` for the energy change, accepts
or rejects, and feeds the attached :doc:`reporters <reporters>`.

.. note::
   ``temperature`` is a dimensionless reduced parameter, not a physical
   temperature unit. Typical values: 0.3 (cold/folded) to 0.6
   (hot/unfolded). Energies are unitless knowledge-based table sums, so
   there is no kT and no Boltzmann constant in the acceptance test --
   see :doc:`../physics_notes/mc_acceptance`.

.. note::
   ``Integrator`` is a compiled ``pymcpu.mcpu_core`` class (the C++
   type is ``MCIntegrator``; ``Integrator`` is the only name exposed to
   Python). The documentation build mocks that module, so the
   directives on this page are hand-authored; the signatures and
   defaults below were read off the built extension.

.. py:currentmodule:: pymcpu

.. py:class:: Integrator(temperature, step_size_rad=0.1, sidechain_step_size_rad=-1.0)

   :param temperature: reduced temperature of this replica. Required:
      there is no default. Fixed for the object's lifetime -- there is no
      ``set_temperature``, and replica exchange swaps coordinates between
      fixed-temperature replicas rather than changing a replica's
      temperature.
   :param step_size_rad: backbone torsion step amplitude in radians.
   :param sidechain_step_size_rad: chi step amplitude in radians for
      the continuous sidechain mode. A negative value means "same as
      backbone", which :py:meth:`Integrator.sidechain_step_size_rad`
      then resolves to the actual number in use.

   All three arguments accept keywords.

   .. code-block:: python

      integrator = pymcpu.Integrator(temperature=0.6, step_size_rad=0.1)
      integrator.set_seed(1234)
      integrator.run(context, 10_000)

Propagation
-----------

.. py:method:: Integrator.run(context, num_steps, step_offset=0) -> None

   Run ``num_steps`` Monte Carlo steps against ``context``.
   ``step_offset`` is added to the step number handed to reporters, so
   successive calls can produce a continuous trajectory.

.. py:method:: Integrator.set_seed(seed) -> None

   Seed the mt19937 generator.

.. py:method:: Integrator.get_rng_state() -> str

   Serialize mt19937 RNG state for checkpoint/resume.

.. py:method:: Integrator.set_rng_state(state) -> None

   Restore mt19937 RNG state previously returned by
   :py:meth:`Integrator.get_rng_state`.

Move set
--------

The move mix has three slots -- Pivot, Sidechain and KIC (concerted
loop closure). :py:meth:`Integrator.set_move_weights` sets how often
each slot is chosen; the remaining knobs select the algorithm used
*inside* a slot and none of them adds a fourth move kind.

.. py:method:: Integrator.set_move_weights(pivot, kic, sidechain) -> None

   Relative probabilities of the three slots, normalized internally --
   ``(1, 1, 0)`` and ``(0.5, 0.5, 0)`` mean the same thing. All three
   must be non-negative with a positive, finite sum; anything else
   raises ``ValueError``.

   The default is ``(0.25, 0.25, 0.50)``, the mix the engine has always
   used. Passing it explicitly is byte-identical to never calling this
   method: exactly one RNG draw is consumed per step whatever the
   weights are, so the stream does not shift.

   .. warning::
      Pass ``sidechain=0.0`` for a force field whose residues have no
      chi angles -- a backbone-only one, for instance. Otherwise every
      sidechain proposal returns without proposing anything, and that
      share of the run is spent producing nothing. Rather than let that
      happen quietly, :py:meth:`Integrator.run` raises when the
      sidechain weight is positive and no residue in the system has a
      chi angle.

   Note that KIC needs a residue index in ``[1, n_residues - 4]``, so on
   a short chain a large KIC weight buys less than it looks like: the
   fraction of KIC draws that can produce a move is roughly
   ``(N - 4) / (N - 2)``, which is 0 for four residues or fewer.

.. py:method:: Integrator.move_weights() -> tuple

   ``(pivot, kic, sidechain)``, normalized to sum 1.

.. py:method:: Integrator.set_sidechain_move_mode(mode) -> None

   Selects the Sidechain-slot proposal algorithm: ``'continuous'`` or
   ``'rotamer_library'``. Any other value raises ``ValueError``.

   The built extension defaults to ``'rotamer_library'``, which draws
   from the Dunbrack-style rotamer table on the system.
   ``'continuous'`` perturbs each chi by a Gaussian of width
   :py:meth:`Integrator.sidechain_step_size_rad`.

.. py:method:: Integrator.sidechain_move_mode() -> str

   The active sidechain mode.

.. py:method:: Integrator.set_sidechain_step_size_rad(sigma_rad) -> None

   Continuous-sidechain chi amplitude in radians; negative restores
   "same as backbone". Legacy has two independent amplitudes
   (``MC_STEP_SIZE`` 2 deg backbone, ``SIDECHAIN_NOISE`` 10 deg chi),
   so a like-for-like comparison needs this set separately. Ignored by
   the ``rotamer_library`` sidechain mode, which takes per-chi widths
   from the library rows.

.. py:method:: Integrator.sidechain_step_size_rad() -> float

   The effective chi amplitude, with the "same as backbone" sentinel
   already resolved (so never negative).

.. py:method:: Integrator.backbone_step_size_rad() -> float

   The backbone torsion amplitude given to the constructor.

.. py:method:: Integrator.set_pivot_rama_probability(p) -> None

   Fraction of Pivot-slot attempts using the knowledge-based
   ``(phi, psi)`` rama-mixture proposal instead of the continuous
   single-dihedral pivot. ``p`` outside ``[0, 1]`` raises
   ``ValueError``.

   The default is **0.0**: the move is opt-in, because it rotates the
   whole C-terminal segment and is rejected on displacement grounds
   most of the time. ``p=0.0`` also reproduces legacy behaviour
   exactly, RNG draw count included.

.. py:method:: Integrator.pivot_rama_probability() -> float

   The active probability.

.. py:method:: Integrator.set_pivot_rama_schedule(t_low, t_high, p_min, p_max) -> None

   Set :py:meth:`Integrator.pivot_rama_probability` from a
   piecewise-linear schedule in this integrator's own (fixed) reduced
   temperature: ``p_min`` at or below ``t_low``, ``p_max`` at or above
   ``t_high``, linear in between. Evaluated once, immediately.
   ``t_low >= t_high``, or a probability outside ``[0, 1]``, raises
   ``ValueError``.

Fixed residues
--------------

.. py:method:: Integrator.set_fixed_residues(residue_indices, n_residues) -> None

   Mark residues as fixed (0-based engine indices). Fixed residues will
   not be moved by any MC proposal.

   ``n_residues`` sizes the mask and must be the system's residue
   count. ``Simulation.set_fixed_residues(residues)`` fills it in from
   ``System.get_num_residues()``, and both ``ReplicaExchange`` and
   ``FoldingRunner`` accept a ``fixed_residues`` constructor argument.

.. py:method:: Integrator.get_fixed_residues() -> list[int]

   The fixed residue indices, or an empty list.

.. py:method:: Integrator.get_fixed_residue_mask() -> list[int]

   Per-residue 0/1 mask of length ``n_residues``.

.. py:method:: Integrator.has_fixed_residues() -> bool

   Whether any residue is fixed.

.. py:method:: Integrator.clear_fixed_residues() -> None

   Release every fixed residue.

Reporters
---------

.. py:method:: Integrator.add_reporter(reporter, /) -> None

   Attach a :py:class:`~pymcpu.Reporter`. Positional-only.

   Prefer the ``Simulation`` helpers (``add_xtc_reporter``,
   ``add_energy_reporter``, ``add_simulation_reporter``), which keep a
   Python-side list alive alongside the C++ one -- see
   :doc:`reporters`.

.. py:method:: Integrator.clear_reporters() -> None

   Detach every reporter.

.. py:method:: Integrator.num_reporters() -> int

   Number of attached reporters.

Acceptance counters
-------------------

All of the following return ``int``. They count from construction and
accumulate across :py:meth:`Integrator.run` calls; nothing resets them
(:py:meth:`Integrator.reset_step_stats` clears only the timing
statistics). Take differences if you want a per-window rate. The samplers
save them in every checkpoint and restore them on resume, so the running
totals in a resumed energy CSV continue rather than restarting at 0.

.. list-table::
   :header-rows: 1
   :widths: 46 54

   * - Method
     - Counts
   * - ``get_bb_accepted()`` / ``get_bb_attempted()``
     - Pivot-slot (backbone) moves
   * - ``get_sc_accepted()`` / ``get_sc_attempted()``
     - Sidechain-slot moves
   * - ``get_kic_accepted()`` / ``get_kic_attempted()``
     - KIC concerted loop-closure moves
   * - ``get_rotamer_accepted()`` / ``get_rotamer_attempted()``
     - Sidechain moves drawn from the rotamer library
   * - ``get_rama_pivot_accepted()`` / ``get_rama_pivot_attempted()``
     - Pivot moves drawn from the rama mixture
   * - ``get_steric_rejected()``
     - Proposals rejected for hard-core overlap
   * - ``get_fixed_rejected()``
     - Proposals rejected for touching a fixed residue
   * - ``get_kic_geometry_invalid()``
     - KIC closures the solver dropped because an N-CA-C angle missed
       its target by more than 1e-6 rad (counts closures, not moves)
   * - ``get_kic_jacobian_invalid()``
     - KIC proposals with a non-finite Jacobian
   * - ``get_kic_presolve_zero()``
     - KIC proposals with no closure solution
   * - ``get_kic_reverse_missing()``
     - KIC proposals refused because the current window is not among
       its own closure solutions, so the move could not be reversed
   * - ``get_kic_proline_skipped()``
     - KIC draws skipped because the move would change a proline's phi
   * - ``num_pivot_resample_pro_phi()``
     - Pivot phi draws resampled because of proline
   * - ``num_sc_resample_pro()``
     - Sidechain draws resampled because of proline

.. py:method:: Integrator.move_counts(include_unused=False) -> dict

   Accept/attempt counts per move kind, as
   ``{kind: (accepted, attempted)}``, with each move counted once. The kinds,
   in order, are ``pivot`` (continuous pivot), ``rama_pivot``, ``kic``,
   ``sidechain`` (continuous sidechain) and ``rotamer``. The slot counters
   above are sums of these: ``get_bb_*`` is ``pivot`` + ``rama_pivot`` and
   ``get_sc_*`` is ``sidechain`` + ``rotamer``.

   Kinds the current move weights and sidechain mode cannot propose are left
   out -- with the defaults that is ``sidechain`` -- unless
   ``include_unused=True``. A kind that has already been proposed is always
   included. These are the move columns of the energy CSV.

   .. code-block:: python

      for kind, (accepted, attempted) in integrator.move_counts().items():
          rate = accepted / attempted if attempted else float("nan")
          print(f"{kind:12} {accepted:6d} / {attempted:6d}  {rate:6.1%}")

.. py:method:: Integrator.move_stats() -> dict

   Aggregate of the counters above, with keys ``num_propose_pivot``,
   ``num_accept_pivot``, ``num_propose_sc``, ``num_accept_sc``,
   ``num_propose_kic``, ``num_accept_kic``, ``num_propose_rotamer``,
   ``num_accept_rotamer``, ``num_propose_rama_pivot``,
   ``num_accept_rama_pivot``, ``steric_rejected``,
   ``kic_geometry_invalid``, ``kic_jacobian_invalid``,
   ``kic_presolve_zero``, ``kic_reverse_missing`` and
   ``kic_proline_skipped``.

.. py:method:: Integrator.get_move_counters() -> dict[str, int]

   Every counter in the table above as ``{name: count}`` (``bb_attempted``,
   ``bb_accepted``, ..., ``num_sc_resample_pro``), for checkpointing.

.. py:method:: Integrator.set_move_counters(counters) -> None

   Restore counters saved by :py:meth:`get_move_counters`. Every counter is
   reset to 0 first, then the given ones are applied; an unknown name raises
   ``ValueError`` and changes nothing.

.. py:method:: Integrator.reset_step_stats() -> None

   Zero the per-step timing statistics reported by ``step_stats()``.
   The acceptance counters above are not affected. ``run()`` calls
   this itself at entry.

Last-move introspection
-----------------------

These describe the most recent proposal and are what the
``Simulation`` properties of the same names read.

.. py:method:: Integrator.last_move_kind() -> str

   Last proposed move kind string (``Pivot``/``KIC``/``Sidechain``/
   ``Other``). The same four names are exposed as the enum
   ``pymcpu.mcpu_core.MoveKind`` with members ``Pivot``, ``KIC``,
   ``Sidechain`` and ``Other``.

.. py:method:: Integrator.last_move_is_rigid() -> bool

   Whether the last proposal was a rigid body move.

.. py:method:: Integrator.last_moved_indices() -> list[int]

   Atom indices moved in the last proposal.

.. py:method:: Integrator.last_delta_energy() -> float

   After :py:meth:`Integrator.run`, the last step's energy change if it
   was accepted, else 0. After a ``debug_force_*`` call (see Test-only
   hooks below), the forced proposal's energy change.

.. py:method:: Integrator.last_accept_bits() -> list[int]

   One accept/reject bit per step of the last
   :py:meth:`Integrator.run` (1 = accept, 0 = reject or invalid), so
   the list is ``num_steps`` long. Intended for determinism
   regression tests.

.. py:method:: Integrator.last_log_jacobian_weight() -> float

   Metropolis-Hastings correction term of the most recent forced
   proposal from a ``debug_force_*`` call (0 for a symmetric move). See
   :doc:`../physics_notes/kic_jacobian` for why a torsion-space
   proposal needs one.

Performance and diagnostics
---------------------------

These are tuning and instrumentation knobs. They do not change the
sampled distribution, they are **not part of the stable API**, and
correct results must not depend on them:
``set_use_pooled_proposal(on)`` / ``use_pooled_proposal()``,
``set_use_sparse_proposal(on)`` / ``use_sparse_proposal`` (read/write
property: "If True (default): skip per-step ``copy_dynamic_from``;
O(n_moved) restore on reject"), ``reject_restore_enabled()``,
``proposal_is_dynamic_only()``, ``proposal_lifecycle_info()``,
``step_stats()``, ``set_step_stats_verbose(on)`` and
``step_stats_verbose()``.

Test-only hooks
---------------

The following force a specific move instead of drawing one, and exist
for the test suite and for physics bug reproduction. They bypass the
move mix, so a run that uses them is not a valid sample:

``debug_force_pivot(context, residue, is_phi) -> bool``,
``debug_force_rama_pivot(context, residue) -> bool``,
``debug_force_rama_pivot_to(context, residue, phi, psi) -> bool``,
``debug_force_rotamer(context, residue) -> bool``,
``debug_force_sc(context, residue) -> bool`` and
``verify_physics_consistency(context, num_steps, atol=0.001)``.

A ``debug_force_*`` hook never commits its move. When it returns
``True``, ``last_move_kind()``, ``last_move_is_rigid()``,
``last_moved_indices()``, ``last_delta_energy()`` and
``last_log_jacobian_weight()`` describe the move it proposed.

.. seealso::
   :doc:`../physics_notes/mc_acceptance` for the acceptance rule,
   :doc:`../physics_notes/kic_jacobian` for the concerted-move
   Jacobian, and :doc:`sampling` for the replica-exchange drivers that
   own one ``Integrator`` per replica.
