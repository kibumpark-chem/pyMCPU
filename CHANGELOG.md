# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `Integrator.set_move_weights(pivot, kic, sidechain)` and
  `Integrator.move_weights()`, plus a `move_weights` key on `IntegratorConfig`
  and `EngineSpec`. The Pivot/KIC/Sidechain mix was previously three
  function-local floats in `MCIntegrator::run` with no way to reach them.
  The default `(0.25, 0.25, 0.50)` is unchanged and bit-identical: exactly one
  RNG draw is consumed per step whatever the weights are, so the stream does
  not shift.
- **`KORPForceField`**: a backbone-only force field built on the KORP 6D
  orientational potential (Lopez-Blanco & Chacon, *Bioinformatics* 2019). KORP
  reads only N, CA and C and its pair coordinate is CA-CA, so this force field
  drops sidechains rather than carrying them unused — the engine sees N, CA, C
  and O only. Two new C++ terms back it, in the new `forces/korp/` lineage:
  `OrientationalPairPotential` (energy group 7) and
  `CalphaExcludedVolumePotential` (group 8), the latter supplying the
  excluded volume KORP lacks as a pure filter that contributes zero to every
  accepted state.

  The compiled term reproduces the reference `korpe` binary to 4e-8 relative
  on four structures, and its incremental energy agrees with a full recompute
  over real integrator moves.

  Two things to know before using it: the trajectory it produces contains
  backbone only and must be loaded against `KORPForceField.output_topology`,
  and sidechain moves must be switched off with
  `set_move_weights(pivot, kic, 0.0)`.
- `pymcpu.forcefields.korp_map`: a reader for the KORP 6D energy map, with a
  reference (non-hot-path) scorer. **The map is not distributed** — at 316 MiB
  it is well over PyPI's per-file limit — so it is supplied via
  `KORP_MAP_PATH`. Verified against the reference `korpe` binary on four
  structures to within 5e-9 relative.

- **A force-field registry.** `pymcpu.forcefields.get_forcefield` /
  `build_forcefield` / `available_forcefields`, plus `forcefield` and
  `forcefield_options` keys on `SimulationConfig` and `EngineSpec`, so a config
  file can select KORP instead of every call site constructing
  `MCPUForceField` directly. The default is `"mcpu08"`, so a config that does
  not mention a force field means exactly what it did before. Each force field
  now declares its own preprocessing through
  `BaseForceField.prepare_trajectory` (MCPU drops hydrogens; KORP slices its
  own backbone), and both expose `output_topology`.
- **Energy terms have names.** Each force field names the terms it creates
  (`mu`, `backbone_torsion`, `sidechain_torsion`, `hydrogen_bond`, `aromatic`;
  `korp_6d`, `calpha_excluded_volume`; `native_contacts_bias`) with the new
  `Potential.set_name()`. `System.energy_terms()` returns `{group: name}`,
  `energy_breakdown()` gains a `by_name` key, and `step_stats()` labels its
  per-term timings from the System. Before this the names lived in separate
  hardcoded tables, one of which (`step_stats`) silently left out KORP.
  `System.add_potential` rejects a name that would give one group two names or
  one name two groups.
- **`Integrator.move_counts()`** reports accept/attempt counts per move kind
  -- `pivot`, `rama_pivot`, `kic`, `sidechain`, `rotamer` -- with each move
  counted once. The slot getters count the knowledge-based sub-kinds inside
  their slot, and the continuous moves had no counter of their own. The
  counts are derived from the existing counters, so trajectories are
  unchanged.

### Fixed

- `Context.set_mu_cell_size_scale` below 1, or `set_mu_cell_size_angstrom`
  below the Mu cutoff, crashed the interpreter in `set_positions`: the
  neighbour grid only supports a one-cell stencil, and a smaller cell
  overflowed its buffers (the cell-pair path also silently dropped cells).
  Such a cell is now raised to the cutoff. Larger cells are unchanged.
- The KORP CA-CA steric guard re-checks the pairs a rigid pivot carries. It
  skipped them because a rigid move keeps their distances, but the pivot is
  applied in float32, so a pair sitting exactly on the floor could be rounded
  under it and accepted with a clash the full energy then reports.
- **KORP keeps chain IDs and residue numbers.** Its backbone slice goes through
  mdtraj's `Topology.subset`, which drops every chain ID and renumbers a
  residue numbered 0. Multi-chain inputs were therefore numbering-checked,
  scored and steric-guarded as one chain: a homo-oligomer numbered from 1 in
  each chain was refused, chains with distinct numbers were scored as one, and
  the CA-CA guard excused cross-chain contacts as bonded neighbours. Both are
  now put back, also on `output_topology`. Single-chain inputs are unchanged.
- **The native-contacts bias follows the `init_only` atom reorder.** The
  reorder renumbers atoms and asks each energy term to remap the atom ids it
  holds; the bias kept its pairs in the old numbering and so measured
  unrelated atoms (a native bias of 289560 instead of 8 on actin). It now
  remaps them, and a term added after the reorder is remapped when it is
  added. The reorder rewrites the System, which REMD replicas share: a
  Context created on it afterwards now adopts the same atom order, and one
  created before it raises instead of scoring garbage (40778 instead of -551
  on actin). Runs without the reorder (every shipped runner) are unchanged.
- **Masked runs no longer accept a clash carried by a large rigid move.** A
  rigid pivot keeps the distances inside the moved segment, so Mu re-checks
  those pairs for a hard-core clash instead of scoring them. In the delta
  paths every run with a residue energy mask takes, that check looked for
  each atom's moved partners around its new position in a grid of old
  positions, and lost them once the segment travelled about a cell (6 A or
  more), so the move could be accepted with a clash in it. The check now
  runs around each atom's old position, where every such partner is. The
  `MCPU_PIVOT_MU_BREAKDOWN` diagnostic path, which had no such check, gets
  it too. Runs without a mask take a different path and are bit-identical.
- Writing `Context.coords` after the `init_only` atom reorder now discards
  Mu's live contact list, as `set_positions` does, so a run after such a
  reset matches a fresh start. Under `set_output_internal_order(True)` it
  also takes the array in storage order, the order the getter returns; it
  used to treat it as build order and scramble the atoms.
- **Rigid moves no longer stick on overlaps that an `ignore_all` mask
  allows.** With `ignore_all`, a pair involving a masked residue (a linker,
  for example) neither clashes nor makes a contact, and the full energy and
  the ordinary delta paths respected that. The guard that re-checks the
  moved-moved pairs of a rigid move did not, so a rigid move carrying such an
  overlap was rejected as a steric clash, and masked residues that overlapped
  could get stuck. `clash_only` is unchanged: moves are still rejected on a
  clash. Runs without a mask are bit-identical. Clearing a `clash_only` mask
  also brings back clash reporting in the full energy, which used to keep
  dropping every clash.
- Mu checks that every atom of a type has one radius. The engine keeps one
  hard-core distance, contact distance and energy per pair of atom types,
  filled in atom order, so a parameter set that gave one type two radii
  would have made Mu depend on atom order. Reading such an atom-type file
  now raises `ValueError`, and so does an asymmetric or non-finite
  `mu_potentials.bin`, or handing `MuPotential` such matrices. mcpu08 is
  unaffected.
- **Collective variables work with `KORPForceField`.** Every CV finds its
  atoms through `build_contact_atom_index`, which read MCPU's per-atom
  list, so an `EngineSession` with KORP and any CV stopped with an
  `AttributeError`. It now reads the per-residue blocks every force field
  provides. `contact_atom_mode="cb"` raises a `ValueError` for a force field
  with no sidechain atoms, such as KORP, instead of quietly using the CA
  against a CB reference. A reference structure now contributes only
  residues with a backbone N, CA and C, so a calcium ion (atom name CA), a
  ligand or a water in it no longer breaks the CV.
- Every `Integrator.debug_force_*` test hook records the move it proposes:
  `last_move_kind()`, `last_moved_indices()`, `last_delta_energy()` and
  `last_log_jacobian_weight()` then describe it. `debug_force_sc`,
  `debug_force_rotamer` and `debug_force_rama_pivot` used to leave the
  previous move's values in place. A forced move also no longer leaves its
  queued changes behind: with the native-contacts bias attached, the next
  accepted step of `run()` used to commit the forced move's pair flips too.
  Like `run()`, the hooks now refuse a fixed residue.
- **Coordinates of the wrong size are rejected.** `Context.set_positions`,
  `Context.coords` and `State.coords` used to resize the engine state to
  whatever array they were given, so a checkpoint or restart file written
  with a different atom layout loaded with every atom after the first
  difference shifted. Only replica exchange noticed, by accident, through a
  steric clash. They now raise `ValueError` naming both counts, and the
  folding, serial and MPI replica-exchange resume paths check the stored
  coordinates before restoring anything; under MPI every rank raises
  together.
- `EngineSession.coords_from_auxref` built an `MCPUForceField` for a `.pdb`
  starting state even in a KORP session, so the coordinates had MCPU's
  layout. It now builds the session's own force field.
- The two copies of each glycine CA (see Changed) drifted apart, by about
  1e-5 Å over 10^5-2x10^5 steps, because moves updated them separately.
- **Replica exchange ignored every move setting.** Each replica's integrator
  was built from its temperature alone, so `move_weights`,
  `sidechain_move_mode`, `pivot_rama_probability`/`pivot_rama_schedule` and
  `step_size_rad` had no effect under REMD, from a config or the Python API.
  `ReplicaExchange`, `MPIReplicaExchange` and the REMD runners now take them.
  The defaults equal what an unconfigured integrator got, so default runs are
  bit-identical.
- **YAML configs turned on the rama pivot.** The YAML loader filled a missing
  `pivot_rama_probability` with 0.05, where the engine, `IntegratorConfig`,
  JSON configs and every Python signature default to 0.0 (the move is
  opt-in). An engine built from such a config with
  `EngineSpec.from_simulation_config` ran rama pivots at 5% without the config
  asking for them; with the move settings now reaching folding and REMD, so
  would those. A missing value now means 0.0.
- **`mcpu run` of a folding config raised `TypeError`** before starting:
  `run_from_config` passed the rama-pivot settings to `run_folding`, which did
  not accept them. The existing tests mocked `run_folding`; a new one runs a
  real simulation.
- `scripts/install_check.py` reported 2 of 7 checks failed on a working
  install (a `SimulationReporter` call with the wrong arguments, and a check
  of an unexported class). The `set_sidechain_move_mode` and
  `set_pivot_rama_probability` docstrings gave `continuous` and `0.05` as
  defaults; the defaults are `rotamer_library` and `0.0`.
- **A move slot with zero weight could still be chosen.** `set_move_weights`
  normalizes in float32, so weights such as `(0.4, 0.2, 0.0)` -- the
  backbone-only setup KORP needs -- leave the pivot and KIC weights summing to
  just under 1, and about one roll in 10^7 fell through to the sidechain slot.
  That wasted the step; with per-kind CSV columns it also made the next
  `run()` raise. A zero-weight slot is now unreachable. The same single roll
  is used, so the RNG stream and every run with nonzero weights are unchanged.
- **Move counters survive a checkpoint resume.** Checkpoints saved only the
  RNG state, so a resumed run restarted every accept/attempt counter at 0
  while its energy CSV kept appending, and the cumulative move columns dropped
  back to zero mid-file. Folding, serial REMD and MPI REMD checkpoints now
  save the counters (`Integrator.get_move_counters()` /
  `set_move_counters()`) and restore them. A checkpoint that has no saved
  counters still loads, and its counters start from 0 as before.
- Building a force field no longer prints `Maximum contact distance
  (squared): ...`, and the first H-bond energy change no longer prints a
  `[neighbor-audit]` line to stderr on every rank. Both were developer
  diagnostics; the audit is still available as
  `Context.print_neighbor_audit()`.

- **KIC loop-closure moves no longer bend the backbone, and four related move
  bugs are fixed.** Every move is meant to keep bond lengths and bond angles
  fixed. KIC did not. Its solver found the root of the closure polynomial
  accurately, then recovered the other two torsions as ratios (`calc_t2`,
  `calc_t1`) whose top and bottom both fall to rounding level when a torsion is
  near 180°, and returned "closed" windows with N-CA-C wrong by up to ~40°.
  Nothing checked them — `kic_geometry_invalid` was declared and exported but
  never incremented — and each move re-measured its target lengths and angles
  from the current coordinates, so every bad closure became the next move's
  target. N-CA-C random-walked without bound: in GA/GB folding runs 9–16 % of
  residue-frames were more than 5° off, up to 69°, in every force field
  including mcpu08. Legacy MCPU checks each closure against the start structure
  and does not drift. The fix, all unconditional:

  * a pole-free back-substitution replaces the ratios (each closure equation is
    used in the `(1, cos, sin)` basis, where nothing has a pole);
  * the solver drops any closure whose three N-CA-C angles miss their targets
    by more than 1e-6 rad, in the pre-move and post-move solves alike, so the
    solution-count ratio stays balanced; drops are counted in
    `kic_geometry_invalid`;
  * the targets are measured once, in double, from the start structure
    (`System.set_kic_reference`, called by `MCPUForceField.create_system` and
    `KORPForceField.create_system`), so replica swaps and checkpoint restores
    cannot change them;
  * the driver-moved anchor atoms are rounded to float before the post-move
    solve, so the next move re-solves exactly this move's reverse problem;
  * a move is refused unless the current window is one of its own pre-move
    solutions (within 1e-3 Å), since it could not be reversed otherwise; new
    counter `kic_reverse_missing`.

  The four related bugs:

  * **The KIC Jacobian depended on the lab frame.** It used the lab x/y
    components of the CA→C bond, so it equalled the true Jacobian divided by
    `|u_z|`, and a phi-driver move's weight changed (by up to 0.39 in log) when
    the molecule was rotated. It is now the orientation-free twist determinant.
  * **KIC changed proline phi**, by up to 76°. A window that would is now
    skipped — residues r..r+2, plus r+3 for the phi driver, the set legacy
    `loop.h` refuses — and counted in `kic_proline_skipped`. Under
    `KORPForceField` the pivot turned proline phi as well, because
    `create_system` never flagged prolines, so every `System.is_proline` was
    False (10k moves on chignolin: 47° by KIC, 5.6° by the pivot). It now
    flags them.
  * **N-terminal psi pivots swung the carbonyl O(r)** with the moving side,
    though it is bonded to C(r) on the axis. O=C-N was more than 5° off in
    16–60 % of residue-frames of the N-terminal half of every production run, up
    to 118°. O(r) now stays. With explicit amide H (`virtual_amide_h=False`) the
    same kind of error left H(r) behind on N-terminal phi pivots and H(r+3)
    behind on KIC's phi driver; both are fixed. So are the atom-reorder
    layout's pivot branches, which had the same errors.
  * **KIC left stale cached backbone torsions** on residue r-1 (phi driver) and
    r-2, r+3 (psi driver), whose pCA/bCA read atoms KIC moves. The next move
    touching them was billed the difference: more than 0.1 on 4–9 % of KIC
    moves, up to 3.6. Every residue whose cache reads a moved atom is now
    refreshed.

  No energy code changed: every term is bit-identical on a fixed structure. In
  a 5M-move mcpu08 chain N-CA-C drift fell from 45.8° max (9.8° rms) to 0.006°
  (0.0008° rms); KIC refuses 0.1–0.4 % of proposals on the reverse check; a
  rough timing showed no slowdown. **This changes every trajectory that uses
  pivot or KIC moves**, mcpu08 included, and two things follow. A checkpoint
  whose backbone has already drifted cannot be resumed usefully — KIC refuses
  nearly every window — so start new runs from the start structure. And a
  `System` built by hand rather than by `create_system` must call
  `System.set_kic_reference(start_coords)` before KIC can run; KIC raises
  `RuntimeError` otherwise, rather than measure targets from whatever the chain
  looks like. `tests/physics/moves/test_kic_closure_fixes.py` covers each bug;
  against the unfixed engine every one of its tests fails. Two frozen
  trajectory baselines that run KIC were re-captured, since the first accepted
  KIC move now lands a few float32 steps away and the runs then separate:
  `test_coords_soa.py` (accepted moves 258 → 264) and
  `test_rama_pivot_move.py`'s p = 0 check (92 → 79). The old values still
  come out of the unfixed engine.

- **KORP's rigid-pivot moved-moved elision is now off by default** — it gave
  Metropolis a wrong delta-E. `OrientationalPairPotential` skipped every pair of
  residues carried by the same rigid pivot, on the grounds that their six pair
  coordinates are unchanged. That holds in real arithmetic only: the pivot is
  applied in float32 and the table is nearest-bin, so a co-moving pair within
  rounding of a bin edge could change bin with nothing entering delta-E.
  Measured on CLN025 at T = 8: accepted moves scored delta-E 0.052 / 1.221 /
  0.000 against a true 3.801 / 2.186 / 5.764, and the running total drifted
  17.4 from a full recompute within 1e5 steps (2.4e-4 with the elision off).
  Events are rare (~1e-4 per accepted move) but individually large, and
  Metropolis preferentially accepts the ones with a hidden positive cost, so the
  error carried a sign. `set_rigid_skip_enabled(True)` still exists, for
  measuring what the elision would buy; leaving it off costs < 6 % of wall time.
  **This changes KORP trajectories**; mcpu08 is untouched.

  The shipped test for it ran 400 cold steps and checked that accept bits
  matched, which they do over that window; its docstring claimed they match in
  general, which they do not, and is corrected.
  `tests/physics/forces/test_korp_exact_delta.py` adds a 40k-step, T = 8,
  pivot-only guard (the pre-fix build drifts 1.73; the bound is 1e-2).

- **The KORP energy table can no longer be freed under a live potential.**
  `OrientationalPairMap` references the table in place by raw pointer. Its
  lifetime was tied, by `py::keep_alive`, to the map's *Python wrapper* — but
  potentials own the map through a `shared_ptr` and outlive that wrapper, and
  `KORPForceField` keeps only its latest map. So a table with no other owner
  (for example one swapped in for a single system) was freed once the next
  system was built, and the older potential then read freed memory: the new
  test's pre-fix run returns **23,780,400 for a structure that scores
  -3693.59**, silently, rather than crashing. The map's `shared_ptr` deleter now
  owns a reference to the array, so the table lives exactly as long as any
  potential using it. Ordinary use — every system built from one memmap — was
  unaffected, which is why it went unnoticed.

- **`KORPForceField.output_topology` no longer carries an atom the engine does
  not hold.** `_BACKBONE_SELECTION` admits `OXT`/`OCT` so a residue with no
  plain `O` can still supply one, and `_collect_residues` takes the first of
  `O`, `OXT`, `OCT`. A C-terminus with *both* `O` and `OXT` therefore left the
  unused one in `output_topology` with no engine slot, so `inverse_mapping` was
  sparse — the one thing its own docstring says would break the XTC reporter,
  which sizes output from `max(mapping) + 1` and zero-fills the rest.

  Two symptoms, by where the unused atom sorted. Mid-file (a PDB listing `OXT`
  before `N`/`CA`, as CLN025 does): a trajectory with one extra atom pinned at
  the origin that **loaded cleanly and was silently wrong**. Last: a trajectory
  with fewer atoms than the topology, which `mdtraj` refused outright.

  `output_topology` is now built after `_build_layout` and restricted to the
  atoms the engine actually holds, which makes `inverse_mapping` a dense
  permutation of `range(n_atoms)` by construction. The `O`/`OXT`/`OCT` fallback
  is unaffected — a residue with `OXT` and no `O` still scores.

  **No energy changes**: `create_system` discards the topology argument and
  builds from the layout, so nothing here reaches the physics. Verified
  bit-identical on twelve structures spanning 10–80 residues. Structures
  without a terminal extra oxygen are untouched.

  Neither shipped fixture has a terminal `OXT`, so nothing caught this;
  `tests/physics/forcefield/test_korp_terminal_oxygen.py` builds its own and
  covers both orderings plus the fallback. Against the unfixed code five of
  its eight cases fail.

### Changed

- **Glycine's CA has one engine slot.** `MCPUForceField` stored each glycine
  CA twice, in the backbone segment and again as the residue's sidechain,
  and the Mu potential muted the backbone copy so the atom was scored once.
  The engine now has one slot per heavy atom, plus the explicit amide
  hydrogens when `virtual_amide_h=False`: 77 atoms for `1uao.pdb` instead of
  80, and 2943 for actin instead of 2971. `forcefield.coords`, `n_atoms`,
  `System.get_num_atoms()` and its `repr`, and checkpoint coordinates change
  to match, and `inverse_mapping` now holds `-1` only for explicit
  hydrogens. Written trajectories already had one CA per glycine and are
  unchanged. Energies agree to float rounding (Mu within 1e-4, the other
  terms bit for bit), and on the parity cases and three 50,000-step
  chignolin runs a given seed reproduces the old trajectory bit for bit; a
  much longer run can eventually diverge, because the old copy drifted.
  **Checkpoints and `EngineSession` restart states of a protein with glycine
  from earlier versions cannot be resumed**: they stop with an error naming
  both atom counts. The checkpoint `format_version` is now 2, and
  `EngineSession.fingerprint` changes for proteins with glycine.
- **`Integrator` requires a temperature.** Its Python constructor defaulted to
  `temperature=300.0`, a physical-units value about 500x the top of the
  useful reduced range (0.3-0.6), so `Integrator()` silently ran a
  near-random walk. Leaving the temperature out now raises `TypeError`.
- **The energy CSV's columns follow the simulation, in snake_case.** The
  header is `step,total`, then one column per energy term
  (`System.energy_terms()`), then `<kind>_accepted,<kind>_attempted` for each
  move kind in use (`Integrator.move_counts()`), then `walker_id`. It used to
  be a fixed MCPU list (`Step,Total,Mu,...,KicAttempted,WalkerId`), so a KORP
  run wrote six always-zero columns with its own energies only in `Total`,
  and rama-pivot and rotamer counts were never written. The header is now
  written when the first `run()` starts. In append mode an existing header
  must match exactly or `run()` raises before any move, where a changed force
  field or move setting used to misalign the columns silently; a missing or
  empty file gets a header, where the MPI resume path could produce a file
  with none. **A run started before this change cannot resume into its old
  energy file**: that file has the old fixed header, so `run()` stops with an
  error saying so. Resume it into a new file (a different output prefix or
  directory), or move the old file aside.
- `SimulationReporter` prints the move kinds in use (from `move_counts()`)
  instead of three fixed labels, and flushes each report.
- **`Integrator.run` now raises** when the sidechain move weight is positive
  but no residue in the system has a chi angle. Previously that combination —
  which is what any backbone-only force field produces — silently discarded
  that share of the step budget, since every sidechain proposal returns
  without proposing anything. Call `set_move_weights(pivot, kic, 0.0)`.
  Systems with sidechains are unaffected.
- **`Integrator.run` now raises** on systems with fewer than three residues.
  The pivot residue distribution is `uniform_int_distribution(1, n_res - 2)`,
  which is undefined below that and returned unspecified values rather than
  failing.
- KORP's polar angles are compared as cosines rather than as angles, which is
  an exact reordering of the same comparison (`acos` is monotonic) and removes
  two inverse-trig calls from the per-pair hot path. Worth ~20% of the step;
  the reference energies are unchanged.
- `PhysicsVerifier::verify_potential_delta` decides whether a clash sentinel is
  expected by asking the potential (`canHardReject()`) instead of testing
  `energy_group == 1`. The old test was correct only while `MuPotential` was
  the only hard-rejecting term.

### Removed

- `Context.set_mm_clash_margin` / `mm_clash_margin`,
  `Context.set_mm_double_boundary` / `mm_double_boundary`, the matching
  `MuPotential` properties and the `MCPU_MM_CLASH_MARGIN` and
  `MCPU_MM_DOUBLE_BOUNDARY` environment variables. They were experiments for
  the rigid-move clash problem that the shared hard-core threshold has since
  solved, and only the cell-pair path read them. The double-boundary check had
  become the default check without its prefilter, so it changed only speed.
  The margin rejected rigid moves that have no clash, on that one path only,
  so a masked run's trajectory depended on which path a move took. Runs that
  did not set them are bit-identical.
- `MCPUAtom.to_write` and `MCPUAtom.is_sidechain`. They existed to tell
  glycine's second CA slot (see Changed) apart from real atoms. `to_write`
  was then false only for explicit amide hydrogens, exactly when
  `original_index` is -1, and nothing read `is_sidechain`.
  `MCPUForceField.inverse_mapping` is now each atom's `original_index`.
- `mcpu_core.EnergyComponents`. It held the fixed MCPU column set the energy
  reporter used to write; nothing exported or used it.

## [0.1.0] — 2026-09-16

First public release. Everything below describes the state of the engine as
published, not a delta against a previous public version — there was none.

Because this is the first release, the API surface is stated here explicitly so
that later versions have something concrete to be compatible with.

### Physics

The engine evaluates five knowledge-based statistical potentials, fitted to the
PDB and shipped inside the package:

| Energy group | Term | Default outer weight |
|---|---|---|
| 1 | Contact/solvation ("Mu") | 0.4 |
| 2 | Backbone virtual-torsion | 1.35 |
| 3 | Sidechain χ torsion | 2.5 |
| 4 | Directional hydrogen bond | 1.35 (effective **2.7** = 1.35 × `RDTHREE_CON`) |
| 5 | Aromatic ring stacking | 5.0 |
| 6 | Native-contact umbrella bias | 1.0 (opt-in) |

Energies are unitless sums of table entries scaled by a dimensionless
per-group weight; temperature is a dimensionless reduced parameter. There is no
Boltzmann constant or Kelvin anywhere in the engine.

Sampling: Metropolis Monte Carlo with backbone pivot, continuous sidechain,
rotamer-library and kinematic-closure (KIC) loop moves; temperature and
umbrella replica exchange, serial and MPI; WESTPA weighted-ensemble support.

#### Hydrogen-bond legacy parity

The hydrogen-bond potential reproduces legacy MCPU's `hbonds.h` exactly. Four
independent gaps were closed during development and are noted here as
provenance for the shipped numbers:

1. `ang_CACA` — the helix/sheet orientation angle for long-range pairs — used
   transposed cross-chain operands and compared a radian value against a
   degree-scale threshold. Now uses the correct intra-chain-axis vectors
   compared against π/2.
2. The secondary-structure `'H'` (helix) gate applied to *every* residue
   separation; it now applies only to `|i-j| > 4` pairs, matching legacy.
3. Added the four Ramachandran hard-reject gates (donor/acceptor φ/ψ) that
   legacy applies before scoring a candidate pair.
4. Added sequence-dependent multiplicative scaling
   (`seq_hb[helix_sheet][aa_i][aa_j]`) and the `beta_favor` (3×) long-range
   bonus.

On `examples/actin/input_pdb/acta.pdb` the weighted H-bond group energy is
**-172.9607**, exact parity with the legacy log's printed value
(`-128.12 × 1.35`). The derivation and ablation ladder are in
`docs/hbond_legacy_parity.md`.

```{note}
If you have energies or trajectories from a pre-release checkout, they are not
comparable to 0.1.0 output — they reflect the earlier, incorrect H-bond
physics. This affects saved checkpoints, WESTPA `.h5` files and trajectories.
```

### Added

- CI that builds and tests (`ci.yml`). The previous — and only — workflow ran
  `ruff check .` and nothing else: it never installed the package, built the
  extension, imported `pymcpu`, or ran a test. Includes a
  `clean-checkout-imports` job that statically resolves every relative import
  against `git ls-files`, which reproduces and would have prevented the
  nine-commit broken-import window above.
- `wheels.yml`: cibuildwheel on `manylinux_2_28` for cp39–cp313, a job that
  rejects any `linux_x86_64`-tagged wheel (which PyPI refuses) and also fails
  if there are no wheels at all, and tag-triggered publishing via PyPI trusted
  publishing.
- `scripts/ci_check_wheel.py`, a wheel acceptance check that refuses to run
  against a source or editable install.
- `CONTRIBUTING.md` and `SECURITY.md`.

- **Parameters ship inside the package.** The fitted potentials are stored as a
  compact lossless archive (678 MiB of raw tables → ~2 MiB, bit-exact on every
  table) and decoded on first use into a content-addressed cache. A plain
  `pip install pymcpu` therefore works offline with no download step and no
  environment variables.
- **`mcpu materialize-params`** decodes that archive once and prints the path,
  so an MPI launcher can serialize the decode instead of having N ranks race
  against a shared `$HOME`:
  `export MCPU_PARAMS_DIR="$(mcpu materialize-params --set mcpu08)"`.
- **Manual native-contact-pair mode** for the native-contacts bias and CV.
  `NativeContactsCV` — and everything built on it
  (`attach_native_contacts_bias_potential`, `ReplicaExchangeConfig`,
  `ReplicaExchange`, `MPIReplicaExchange`, `FoldingRunner`, and the WESTPA
  `native_contacts_q` / `native_contacts_n` pcoord specs) — accepts an explicit
  `native_contact_pairs` list of 0-based residue-index pairs instead of
  deriving the contact set from `contact_cutoff` / `min_seq_sep` against a
  reference structure.

  Explicit pairs **replace** the derived set rather than being unioned with it.
  They are mapped through the existing `contact_atom_mode` mechanism, so one
  atom-type choice (CA or CB) applies to the whole CV. `contact_cutoff` keeps
  its independent role of setting the "is this contact currently formed"
  threshold; `q_cutoff` defaults to it.

  Two deliberate asymmetries: `fixed_residue_mask` does **not** silently drop
  explicit fixed–fixed pairs (auto-derivation drops them as an efficiency
  choice, but a user who typed a pair made a deliberate choice), while
  `energy_ignored_residue_mask` hard-rejects any explicit pair touching a
  masked linker/ghost residue, because those coordinates are physically
  meaningless and a "contact" there would look like real physics.
- **Optional DSSP-derived secondary structure** for the H-bond SS-dependent
  gates: `MCPUForceField(..., compute_dssp=True)`, plus matching options on
  `FoldingRunner` and the WESTPA config block. Default is `False`, where every
  residue reads as `'C'` — byte-identical to not having the feature. Legacy's
  4th secstr state `'L'` has no DSSP equivalent and is never inferred.
- **Checkpointing and resume** for every runner, with atomic writes
  (temp file → `fsync` → `os.replace`) so a crash mid-write cannot corrupt a
  snapshot. Both RNG streams are captured, making a resumed run a continuation
  rather than a fresh sample. Resume truncates XTC, CSV, HDF5 and NPZ outputs
  back to the checkpointed frame count.
- **Documentation**: a single Sphinx tree covering installation, quickstart,
  the CLI, REMD, WESTPA, checkpointing, physics background and the API
  reference.

### Fixed

- **Link-time optimisation was silently disabled in every wheel, CI and
  conda-forge build.** `check_ipo_supported()` was called without `LANGUAGES`,
  so it checked every enabled language; FetchContent'd Eigen calls
  `enable_language(Fortran)` from its `blas/`, `lapack/` and `test/`
  subdirectories, CMake has no Fortran IPO, and the check returned false. A
  local `cmake` build against a system Eigen kept LTO, so the configuration
  being measured was never the configuration being shipped. Scoped the check to
  `LANGUAGES CXX`. `build_info()["build"]["lto"]` is now asserted in CI and by
  the wheel acceptance check.
- **Training cache pipeline failed on every structure** with
  `maximum recursion depth exceeded`. mdtraj enables pyparsing packrat caching
  and its selection DSL recurses for hundreds of frames; the nested sidechain
  selection exceeded Python's default 1000-frame limit. The failure was
  caller-depth dependent — the same call succeeded from a shallow script and
  failed from a pytest test body ~33 frames deeper — so headroom is now
  reserved relative to the current stack depth rather than as an absolute
  limit.
- `MCPUForceField` raised `KeyError: 'rama mixture'` for any parameter set
  whose registry entry declares no `optional` block — the first thing an
  outside developer adding their own set would hit.
- `CITATION.cff` now exists. `README.md` had claimed it was included in the
  repository while it was not.

- **`Simulation` now seeds its running total energy on the first `step()`.**
  `Context.set_positions()` does not compute an energy, and the three paths
  that normally keep `current_energy` exact on entry (the previous cycle's
  recompute, an accepted REMD exchange, checkpoint restore) all require a
  *previous* cycle — so the first call left the accumulator off by exactly the
  starting energy for the rest of the run. On the quickstart that was a
  constant 14.706589 offset, independent of step count; the genuine float32
  accumulation over 10 000 steps is 2.9e-5. Accept bits were never affected
  (Metropolis consumes ΔE, not the total), so only reported energies were
  wrong — including the energy-drift warning the documented quickstart printed
  and the step-0 row of the `EnergyReporter` CSV.
- **`pymcpu.runners.default_example_pdb()`** resolved relative to the package's
  parent directory, which is the repo root only for an editable install. From a
  wheel it pointed at a nonexistent `site-packages/examples/...`, so the
  documented quickstart failed with `OSError: No such file`. The 1UAO structure
  now ships as `pymcpu/data/1uao.pdb` and is resolved with
  `importlib.resources`.
- **Neighbour-list proxy stats no longer auto-print from `Integrator::run()`.**
  They were emitted at the end of every `run()` call whenever
  `proxy_print_every` was 0 (the historical default), which under REMD meant
  many calls × many MPI ranks of stderr spam. Unsolicited `[neighbor-audit]`
  lines from the position-setting and H-bond hot paths were also removed.

### Changed

- **The WESTPA integration is a separate distribution, `pymcpu-westpa`.** It
  used to ship inside the `pymcpu` wheel as `pymcpu.we`, where two of its
  modules imported `westpa` at module scope and so were dead weight in every
  install that did not have it. It now lives in its own
  repository (https://github.com/kibumpark-chem/pymcpu-westpa), with its own version and its own PyPI project:
  `pip install pymcpu-westpa`. The `[westpa]` extra is gone, and
  `mcpu westpa-init` / `mcpu westpa-check` are now `mcpu-westpa init` /
  `mcpu-westpa check`, shipped by that package — argparse cannot register a
  subcommand lazily, so the core CLI was advertising two subcommands almost
  no user could run.

  The half of that code which was never WESTPA-specific moved the other way,
  **into** core, where it is now public API for any framework driving pyMCPU:
  `pymcpu.sampling.EngineSession` (a cached engine, was `MCPUEngine`),
  `pymcpu.sampling.build_cv` (a declarative CV factory, was `build_pcoord`),
  `pymcpu.sampling.derive_seed`, `pymcpu.sampling.compute_fingerprint`, and
  `pymcpu.config.EngineSpec`. `pymcpu.sampling.CARMSDCV` and an `__all__` for
  `pymcpu.config` are exported for the same reason: the integration depended
  on six names that were public only by accident.

- `ParityTolerance.EXACT` is an absolute `1e-4`, which at float32 magnitudes
  spans 3,355 ULP around `0.4f` — a `0.4 → 0.40001` transcription error passed
  it. Added a genuinely bitwise `ParityTolerance.BITWISE` tier and pointed the
  six published energy-weight constants at it. That test also carried
  `pytest.mark.slow`, so the only direct check of the published weights never
  ran in `pytest -q`; it does now.
- `tests/physics/test_coords_soa.py` `BASELINE_E_HBOND` re-captured
  (eighth capture). The previous value is not reproducible from any commit:
  it was recorded by `e57ac48`, which imports `RotamerLibraryBuilder` from a
  file that stayed untracked for nine commits, so `import pymcpu` fails there
  and at the eight commits after it.

- **Energy terms are named `*Potential`, not `*Force`.** The engine computes no
  force vector — there is no gradient method on the base class or any subclass,
  and the values are consumed directly as energies. The base class is
  `Potential`; `System.add_potential()` / `System.get_potentials()` replace the
  former `addForce` / `getForces`; and an energy term's group is its
  `energy_group`, set with `set_energy_group()`.
- **The Python API is uniformly snake_case.** 41 exposed camelCase names were
  renamed (`setPositions` → `set_positions`, `getState` → `get_state`,
  `getSystem` → `get_system`, `getNumAtoms` → `get_num_atoms`, and so on), and
  `EnergyWeights.kLegacyMu` and friends became `LEGACY_MU`. The C++ method
  names keep camelCase; only the Python-facing bindings changed.
- `pooch` is no longer required at import time; the download path imports it
  lazily, so an offline install never needs it.

### Removed

- The `QBiasPotential` alias for `NativeContactsBiasPotential`, and the
  `attach_q_bias_force` alias for `attach_native_contacts_bias_potential`.
  Each name now has exactly one spelling.
- `set_proxy_print_every()`. It had been a no-op emitting a
  `DeprecationWarning`; call `Context.print_neighbor_proxy_stats()` explicitly
  when you want the statistics.

### Known limitations

- Linux x86-64 only. macOS and arm64 wheels are not built.
- The published wheel targets **x86-64-v3** (AVX2 + FMA + BMI2; Haswell and
  later, Zen and later). Older CPUs need a source build with `MCPU_ARCH=v2`
  or `MCPU_ARCH=none`; `MCPU_ARCH=native` gets the most from a newer one.
  No AVX-512 is used, deliberately — an AVX-512 baseline would fault on
  every AMD Zen 1–3 part and every 12th-generation-or-later Intel consumer
  part, with no diagnostic.
- **DCD trajectories are not truncated on resume** — the only output format
  that is not. `truncate_all_trajectories_on_resume` warns and skips them, so
  delete the DCD manually before resuming, or use XTC.
- Exact MC RNG restore requires resuming with a compatible `mcpu_core` build:
  the state is serialized through `std::mt19937`'s stream operators, so it is a
  C++ standard-library text format rather than a pyMCPU-defined one.
  Coordinates, counters and exchange state restore regardless.

[Unreleased]: https://github.com/kibumpark-chem/pyMCPU/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/kibumpark-chem/pyMCPU/releases/tag/v0.1.0
