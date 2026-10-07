# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `MuPotential.clist_fallbacks`: moves the contact list could not follow
  (a grid overflow, or a rigid move past the drift budget). Each accepted
  one costs an O(N^2) list rebuild, which is what slows hot replicas.

- `scripts/tolerance_check.py`: accepts or rejects a build that is correct
  but not bit-identical to a reference, for speedups that change float
  rounding. It compares per-group static energies (1e-5 relative), checks
  the running energy against a full recompute after long runs
  (max(1e-3, 1e-5·|E|); the relative term admits reference builds that
  accumulated in float32), compares sampling statistics over several seeds within
  statistical error, and checks KORP against the reference `korpe`
  energies. `CONTRIBUTING.md` says when to use it instead of
  `arch_parity_dump.py`.
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

- **A fixed first residue stays in place.** A pivot that rotates the
  N-terminal end moves residues 0 to r-1, but the fixed-residue check
  looked at residues 1 to r-1 only, so a fixed residue 0 moved whenever
  residue 1 was free (up to 1.3 A in 2,000 pivot steps on T4 lysozyme).
  Such a pivot now rotates the C-terminal end, and is redrawn if that end
  holds a fixed residue too. Runs that do not fix residue 0 are unchanged.

- **The running energy is exact after a run, whatever changed before it.**
  `current_energy` is updated from each accepted move's delta, so it kept a
  constant offset until the next full recompute whenever something other than
  a move changed the energy: setting or clearing an energy mask (-123.1 on T4L
  for `ignore_all`, -57.6 for `clash_only`), `Context.set_energy_weight`
  (-83.4), disabling a potential (-74.2), `set_use_legacy_weights`, the
  native-contacts bias, and `set_positions` (which is why `Simulation` seeded
  the total on its first `step()`). The deltas were right, so accept
  decisions never changed, but reported energies and the value replica
  exchange reads were off. The context now keeps a fingerprint of what
  defines the energy (mask epoch, legacy-weights switch, group weights, bias,
  each potential's group or disabled state) and whether the coordinates were
  replaced, and `Integrator.run` recomputes in full on entry only when one of
  them changed: O(#potentials) otherwise, 2-6 ms per actual change at
  270-415 residues. `Context.energy_resyncs` counts those recomputes. A clash
  that recompute finds raises `StericClashError` before the first move:
  clearing an `ignore_all` mask whose residues overlap the rest of the chain
  now does that, where it used to carry the clash into the next recompute.
  `Simulation`'s one-time seeding and the "set_positions does not seed"
  caveat are gone; calling `calculate_total_energy(-1)` after
  `set_positions` is optional. That call counts as the run's recompute, so
  it does not raise on an overlap: the caller checks `has_steric_clash()`.
  The folding, replica-exchange and `EngineSession` drivers now do that for
  their start structure (`check_state_clash`, which honours
  `MCPU_CLASH_FATAL=0`). Before, a clashing start raised after the first
  `step()`, and with this change alone it would have run on an unseeded
  energy until the periodic check. Trajectories and accept bits are
  bit-identical (`arch_parity_dump.py`).

- **The x86-64-v2 build compiles again.** `CoordsSoA.h` included
  `pair_r2.h` only inside its AVX2+FMA block, so a `MCPU_ARCH=v2` build failed
  with "'pair_r2' was not declared" (the CI job for that tier had failed
  since the pair-distance change). The include is now unconditional. The CI
  job also named `tests/physics/test_mu_cell_size.py`, which the overflow-proof
  grid removed; it now runs `test_mu_grid_membership.py` and
  `test_grid_overflow.py` in its place. The default v3 build is unchanged.

- **The init_only atom reorder works on a chain that ends in OXT.**
  `Context.set_atom_reorder_mode("init_only")` raised `RuntimeError:
  compute_init_only_atom_permutation: atom count mismatch after tree emit`
  for any chain with a C-terminal OXT (T4L, LDHA, chignolin and most PDB
  files; actin has none). The builder puts OXT in the O segment after the
  last residue's O, no `BlockIndices` field names it, and the reorder emitted
  only the atoms the blocks name, so it came up one atom short. It now emits
  every atom no block names right after its residue's O (by
  `atom_to_residue`), inside that residue's span, so a pivot carries OXT with
  the last residue. Static energies match reorder off up to summation order
  (T4L total -394.75668 off, -394.75681 init_only; largest term difference
  9e-7 relative; actin, which already worked: 1.5e-6). Pivot, KIC and
  side-chain moves carry OXT with its residue. The default ("off") is
  untouched and bit-identical. `tests/integration/test_reorder_terminal_oxt.py`.

- **A rigid pivot re-decides the H-bonds it carries.** Under a pivot, an
  H-bond between two residues that moved as one body was carried at its old
  energy, but the rotation rounds every moved coordinate, so such a pair
  could cross a bin edge or the 2.5 A cutoff without the running energy
  noticing. 1000 A from the origin, an actin pivot-only run drifted 0.212
  (one H-bond) from the full energy at step 7092
  (`test_the_drift_budget_rebuilds_the_list_in_time`, which is
  slow-marked, so CI never ran it). The H-bond ledger now lists every pair
  whose H and O are within 2.55 A, with energy 0 if it does not score; a
  pivot re-scores the listed pairs it carries; and the ledger is measured
  again from the coordinates before the summed carry rounding of the
  unlisted ones could reach the 0.05 A band, like the Mu contact list. The
  H-bond grid cells are 2.55 A so one stencil walk finds every listed pair.
  The run now ends 6e-4 from the full energy. Every arch_parity_dump case
  stays bit-identical.

- **Every pair-distance cutoff test rounds the squared distance the same
  way.** GCC (default `-ffp-contract=fast`) contracts
  `dx * dx + dy * dy + dz * dz` into fused multiply-adds in a different
  order at different inlined call sites, so the incremental delta and the
  full energy could decide the same pair on different sides of a cutoff.
  All scalar cutoff tests (Mu, pair search, H-bond, CA excluded volume,
  KORP near filter, aromatic, Q bias) now go through one helper,
  `pair_r2`, which fuses in the order of the AVX2 pair search, or not at
  all under `MCPU_FP_CONTRACT=off` or without FMA.

- **`examples/configs/template.yaml` runs on its own structure.** It held
  residues 88-199 fixed, but the 1uao structure it points to has 10
  residues, so any run of it stopped with an `IndexError`. That line is now
  a commented example. Its comments no longer mark keys that have defaults
  as required, and no longer point to `inputs/template.yaml`, which does
  not exist.

- **Checkpoint flags override only the settings they are given.**
  `examples/gromacs_style/run.py` replaced the YAML's checkpoint directory,
  interval and keep count with the flags' defaults. A flag left out now
  keeps the config's setting. `mcpu run`, `mcpu validate`,
  `scripts/run_mcpu_replica_exchange.py --config` and the example now share
  `pymcpu.utils.cli.apply_checkpoint_args` to apply the flags.

- **`mcpu validate` checks that the structure files exist.** For a YAML
  config it reported OK when `pdb`, or a replica exchange config's
  `reference_pdb`, did not exist, so the mistake surfaced only when the run
  started; the JSON loader already checked. For either format it now fails,
  naming the file, when the file is missing or is not a file. Any other
  error while loading the config is now reported in one line rather than as
  a traceback, for example a single number where a list is expected.

- **KIC no longer stretches the bonds of the atoms it carries.** A KIC move
  carries each window residue's O, sidechain and amide H rigidly with its
  backbone frame, and `transfer_dependent_atoms` built that frame in
  float32. The error was a bias, not noise: every accepted move pushed the
  carried bonds the same way. Over 3M default-mix chignolin steps, in the
  residues KIC moves, sidechain bonds grew by 7.6e-4 Å on average (up to
  1.6e-3 Å), CA-CB shrank by 2.2e-4 Å and C=O drifted by 7e-4 Å rms, while
  the backbone bonds KIC re-closes stayed within 4e-5 Å. The frame is now
  built in double and each coordinate rounded to float once, as
  `CoordsSoA::rotate_atoms` already does: the same bonds stay within 2e-4 Å
  (rms 5e-5 to 9e-5 Å), and over 200k KIC-only steps within 2e-5 Å, against
  1e-3 Å before. Trajectories change from the first accepted KIC move;
  energies and acceptance rates were statistically unchanged over 7 seeds.
  The double maths costs about 3% per step on chignolin's default move mix
  and 4% with KIC moves only.

- **Rigid pivots score the carried pairs that rounding takes across their Mu
  contact cutoff.** A rigid pivot does not re-measure the pairs it carries,
  but its rounding moves each carried distance by up to sqrt(3) float steps
  of the largest coordinate (6.7e-6 Å below 64 Å), so a pair sitting on its
  contact cutoff could cross it unseen and leave the running energy one
  contact energy off until the next recompute: on actin, energy-drift
  warnings up to a few times per million steps. The contact list now also
  holds, with energy 0, every contact pair less than 0.05 Å outside its
  cutoff, and a rigid pivot re-decides each listed pair it carries. The
  others cannot cross: the bound is summed over accepted pivots, and the list
  is rebuilt from the coordinates before the sum reaches 0.05 Å (Simulation's
  recompute after each `step()` resets it anyway). A pivot out of the
  neighbour grid re-decides them too: the listed ones, or every carried pair
  when there is no list. The running energy now equals the full energy after
  every move. The Mu neighbour cutoff grows by the band (5.0765 to 5.1265 Å on
  actin), and `MuPotential.contact_list_rebuilds` counts rebuilds. The full Mu
  energy now skips pairs beyond that cutoff, which makes the actin recompute
  about 40% faster again. Moves inside the neighbour grid take the same time;
  a pivot out of it with no list to go by (under a mask, say) takes up to 1.3x
  longer. Trajectories match the previous build until the first such crossing.

- **Replica exchange and folding read coordinates in the order they write
  them.** `pymcpu.sampling.get_coords`, which exchanges, checkpoints and the
  folding CVs use, read `State.coords` (storage order) while
  `set_positions` takes build order. The two differ after an `init_only`
  atom reorder, so a swap or a checkpoint round trip scrambled the atoms and
  the CVs read the wrong ones. It now reads `Context.coords`. No shipped
  runner enables the reorder, so their runs are unchanged.
- **Restores and replica swaps check the state they set for clashes, and
  honour `MCPU_CLASH_FATAL=0`.** A hard-core overlap that no move can make
  is reported by `Simulation.step` (fatal by default, a warning under
  `MCPU_CLASH_FATAL=0`). The serial and MPI replica-exchange checkpoint
  restores and the MPI cross-rank swap checked too, but raised even under
  `MCPU_CLASH_FATAL=0`; folding resume, `EngineSession.set_coords` and
  same-rank replica swaps did not check, and since the recompute keeps the
  previous energy when it finds a clash, they carried on with a stale
  energy until the next step reported it. All of them now go through
  `pymcpu.simulation.check_state_clash`.
- **Mu no longer exempts a native pair that does not clash.** A pair already
  under its hard-core distance in the structure a force field is built from
  is exempt from the clash test for the whole run. The test used the exact
  hard-core radius, which lies 0.0015 Å or more above the cutoff moves are
  tested against, so a pair in between -- no clash under either cutoff --
  lost its protection for good and could later overlap to any depth. A
  structure written by one run and used as the input of the next can hold
  such pairs. Only pairs under the move cutoff are exempt now; 1UAO and actin
  have none in between, so their runs are unchanged.
- `Context.set_mu_cell_size_scale` below 1, or `set_mu_cell_size_angstrom`
  below the Mu cutoff, crashed the interpreter in `set_positions`: the
  neighbour grid only supports a one-cell stencil, and a smaller cell
  overflowed its buffers.
  Such a cell is now raised to the cutoff. Larger cells are unchanged.
- **KORP keeps chain IDs and residue numbers.** Its backbone slice goes through
  mdtraj's `Topology.subset`, which drops every chain ID and renumbers a
  residue numbered 0. Multi-chain inputs were therefore numbering-checked,
  scored and steric-guarded as one chain: a homo-oligomer numbered from 1 in
  each chain was refused, chains with distinct numbers were scored as one, and
  the CA-CA guard excused cross-chain contacts as bonded neighbours. Both are
  now put back, also on `output_topology`, and the inter-chain energy of the
  bundle's two-chain structures matches korpe. Single-chain inputs are
  unchanged. The moves still treat all chains as one bonded backbone, so
  KORPForceField warns on multi-chain input: use it for scoring, or sample
  one chain.
- **The native-contacts bias follows the `init_only` atom reorder.** The
  reorder renumbers atoms and asks each energy term to remap the atom ids it
  holds; the bias kept its pairs in the old numbering and so measured
  unrelated atoms (a native bias of 289560 instead of 8 on actin). It now
  remaps them, and a term added after the reorder is remapped when it is
  added. The reorder rewrites the System, which REMD replicas share: a
  Context created on it afterwards now adopts the same atom order, and one
  created before it raises instead of scoring garbage (40778 instead of -551
  on actin). Runs without the reorder (every shipped runner) are unchanged.
- Writing `Context.coords` after the `init_only` atom reorder now discards
  Mu's live contact list, as `set_positions` does, so a run after such a
  reset matches a fresh start. Under `set_output_internal_order(True)` it
  also takes the array in storage order, the order the getter returns; it
  used to treat it as build order and scramble the atoms.
- Clearing a `clash_only` residue mask brings back clash reporting in the
  full energy, which used to keep dropping every clash.
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

- **A full energy recompute also refreshes Mu's contact list.** A move's Mu
  energy change reads the old contacts off a live list of the accepted
  state's contacts. A rigid pivot does not re-decide the pairs it carries,
  and its rounding can carry one across its contact cutoff by about 1e-6 Å,
  so neither the running energy nor the list sees the change.
  `calculate_total_energy(-1)`, which `Simulation` runs after every `step()`
  by default, corrected the energy but kept the stale entry, and the next
  move that separated the pair was scored one contact energy wrong: on
  actin, energy-drift warnings came in pairs of opposite sign, up to a few
  per million steps. That recompute now rewrites the list from the same pass
  (through the new `Potential::resyncEnergy`); read-only evaluations such as
  `energy_breakdown` leave the list alone. (The crossing itself is now scored
  too; see "Rigid pivots score the carried pairs..." above.) The pass also
  stopped writing a per-state N^2 pair-flag cache that only the legacy build
  read (see Removed), which makes the actin recompute about a quarter faster
  (45 to 34 ms). Writing
  `State.coords` now discards that state's contact list too.
  `docs/physics_notes/mc_acceptance.md` described a periodic contact-list
  rebuild (`contact_rebuild_interval`) the engine never had; it now
  describes the list as it is. `MuPotential::calculateEnergyBrute` and a
  three-argument `calculateEnergy` overload, which nothing called, are gone.

- **Pivots no longer shrink the protein.** The pivot, rama-pivot and
  continuous sidechain moves turn groups of atoms rigidly, and did so in
  float32, rounding each coordinate twice: relative to the pivot atom, then
  in lab coordinates. That left every distance a rotation should keep
  slightly shorter on average, and the losses added up. Over 5M pivot-only
  chignolin steps, CA-C bonds shrank by 3e-3 Å and the distances inside the
  rigid pieces by 3e-3 Å on average, up to 2.6e-2 Å; the default move mix
  lost about 6e-5 Å per million steps. The rotation is now done in double and
  each coordinate rounded to float once, which leaves unbiased noise (+6e-6 Å
  on bonds over the same 5M steps), at no measurable cost. **This changes
  every trajectory that uses these moves**, under both force fields:
  `test_rama_pivot_move.py`'s p = 0 check was re-captured (79 → 81), and
  `test_coords_soa.py`'s frozen baseline, which a GCC 8.5 build missed
  (271 accepted moves for 264), now comes out of it exactly.
  `tests/physics/moves/test_rotation_keeps_distances.py` fails on the float32
  build.

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

- `Context.neighbor_aabb_rebuilds()` is now `neighbor_grid_rebuilds()`,
  and the `neighbor_proxy_stats()` key `num_aabb_rebuild_accept` is now
  `num_grid_rebuilds`: since grids wrap, the counter counts full grid
  rebuilds (set_positions and overflow recovery), not bounding-box
  rebuilds after a move.

- **With fixed residues, no step is spent on a fixed residue.** A KIC step
  drew its window once and gave up when the window touched a fixed residue;
  it now draws again, as pivot and sidechain steps already did, so the
  window is uniform over the windows that move no fixed residue and the
  proposal stays symmetric. This changes what a step is when residues are
  fixed: in the production sce setup (272 residues, 88-192 fixed) 8.1% of
  steps were KIC draws thrown away on the fixed block. Now every KIC step
  draws a window off the block, KIC windows per KIC step rise from 0.50 to
  0.83, and valid moves per step from 0.733 to 0.771 (native) and 0.687 to
  0.693 (hot; most hot KIC closures fail). Pivot and sidechain steps are
  unchanged. A native replica (T=0.4) costs 15% more per step (24.9k to
  28.8k cycles) from the extra KIC work. On a hot replica (T=1.0), which
  paces the exchange, the change is not resolvable: +0.9% +/- 4.1% over 8
  seeds (45.9k to 46.3k cycles), because the cost per step there depends
  more on the conformation than on the move mix. A Ramachandran pivot with
  the last residue fixed has no allowed residue; it now ends at once
  instead of drawing 64 times. `get_fixed_rejected()` now counts only steps
  that end with no move because of fixed residues; it used to count every
  redraw, which read as 55% of sce steps. Without fixed residues nothing is
  redrawn and trajectories are bit-identical. With them, trajectories
  change; fixed atoms stayed exactly in place in every run (24 sce runs of
  1M steps, 24 T4 lysozyme runs of 200k, 32 chignolin runs of 10M). At
  equilibrium (chignolin, residues 4-5 fixed) energy, native contacts and
  Rg agree with the old code within statistical error (|z| < 2).
  Relaxation runs compared per mobile proposal agree for T4 lysozyme (40-79
  fixed) and native sce (|z| <= 1.1); hot sce loses native contacts a
  little more slowly (z = 2.5), since the added proposals are KIC, which
  rarely succeeds in hot states.

- **The neighbour grids wrap: an atom outside the grid's box is filed in
  the cell its coordinates give mod the cell count, instead of leaving the
  grid.** Hot and unfolded replicas spent most of their time on atoms
  outside the box: such a trial could not use the Mu contact list and took
  the all-pairs delta, and an accepted one rebuilt every grid (O(box
  volume)) and dropped the list. Now a cell may also list atoms whole grid
  periods away; every walk measures true distances, so they are dropped
  and the pairs found are the same. No move leaves a grid, so the
  out-of-box rebuild and the per-move box test (`trial_in_bounds`) are
  gone; a grid is rebuilt, centred on the atoms, only by `set_positions`
  or after a cell overflows. Grids have at least 2R + 1 cells per axis
  (R = stencil radius) and at most 16 cells per atom (at least 32,768): a
  folded protein keeps its exact box, an unfolded one a trimmed grid, so
  memory no longer grows with the unfolded volume. NaN and huge
  coordinates get a cell instead of an undefined int conversion. The
  rigid-carry rounding bound now follows atoms past the box.
  States that stay in the box are bit-identical (parity vs d38a246; the
  production `sce` checkpoint at T=0.4 and 0.65, actin and PGK1 at
  T=0.4-1.0: same energies, accept counts and final coordinates). Once
  atoms pass the old box, the contact list replaces the all-pairs delta:
  exact, with sums in another order (running vs full energy within 4e-14
  relative over 1e6 steps on five hot states, no overflow). User-space
  cycles per step against the previous commit, n=3, 50k steps: production
  `sce` checkpoint at T=0.95 -24.7%, T=1.0 -23.7%, T<=0.91 within
  -5.4%..+1.3%; melted `sce` -49.9%, melted actin -59.0%, melted PGK1
  -55.9%..-59.1%. Peak RSS of a melted actin or PGK1 replica: about 2.0 GB
  -> 1.2 GB (the native level).

- **`Simulation.full_energy_every` is now `full_energy_every_steps`, counted
  in MC steps, default 1,000,000.** The old setting counted `step()` calls
  and defaulted to 1, so every 10k-step exchange paid a full O(N^2)
  recompute (2.4-5.8 ms at 270-415 residues; 0.7-1% of a hot replica's wall,
  3.7-6.2% on cold rungs) to correct a running total that, with double sums,
  stays within 1e-10 of a full recompute over 1e7 steps (1e-7 at 1e8). Accept
  bits are the same either way. Replica exchanges and the energy columns of
  folding and replica-exchange logs now read that running total instead of a
  fresh recompute, so their last digits can differ from before, and in
  principle so can an exchange decided at the margin. The recompute is now a periodic check for a
  clash or a missed pair, also done before every folding and
  replica-exchange checkpoint save, and available as
  `Simulation.recompute_energy()`. The YAML config sets it with
  `full_energy_every_steps`, and `FoldingRunner`, `ReplicaExchange` and
  `MPIReplicaExchange` take it as an argument; it must be a positive whole
  number (`ValueError` otherwise, where a YAML `null` used to give a bare
  `TypeError` and 0 recomputed at every call). Assigning
  `full_energy_every` raises `AttributeError` naming the new attribute, so a
  script that set it to a large value to skip the recompute fails instead of
  silently recomputing.

- **The native-contacts bias checks only the atoms that end a native
  pair.** Its energy change scanned every atom up to the highest pair atom,
  testing four moved flags each, and allocated a new vector on every move;
  it now walks the short list of pair atoms built with the pair table and
  reuses a buffer in `Context`'s bias workspace. Bit-identical (parity vs
  d38a246; same energies and accept counts). User-space cycles per step,
  n=3, 50k steps, on top of the change below: production `sce` checkpoint
  -15.8% at T=0.4 (slot 0) and -14.8% at T=1.0 (slot 57), melted `sce`
  -7.6% at T=1.0. Runs without the bias are unaffected.

- **The Mu grid no longer keeps a list of occupied neighbour cells.** Every
  cell carried the occupied cells of its 27-cell stencil, updated whenever a
  cell turned empty or occupied, but no energy term read it: the Mu walks
  step through the stencil and skip empty cells themselves. The list, its
  per-cell valid-stencil table and the debug check that compared the two are
  gone, with the unused neighbour-cell helpers that read them. Hot replicas
  gain most, since their atoms keep moving between cells. Bit-identical
  (parity vs d38a246; same energies and accept counts). User-space cycles
  per step, n=3, 50k steps, production `sce` checkpoint: -1.2% at T=0.4
  (slot 0), -9.7% at T=1.0 (slot 57); melted `sce` -20.6% and melted actin
  -24.5% at T=1.0. Actin and PGK1 near the native state, default and
  pivot-only, are no slower.

- **GCC 15 is the default compiler, and the wheels are built with it.** Its
  extension runs about 3-15% faster than GCC 8.5's in user-space cycles per
  step: about 10% on pivot moves (actin, same trajectory), 14% with KORP,
  and 3-5% averaged over seeds on the default move mix and KIC. Instructions
  per step fall 5-11%, so the gain is code generation, not layout. On RHEL 8
  the compiler is gcc-toolset-15 (`source /opt/rh/gcc-toolset-15/enable`).
  Its binutils 2.44 pads branches, and it links the libstdc++ parts newer
  than RHEL 8's into the extension, so the symbol floor stays GLIBC_2.27,
  GLIBCXX_3.4.21 and CXXABI_1.3.11, the same as GCC 8.5, and the extension
  imports under a conda Python with no extra flags. The wheel job installs
  the toolset in manylinux_2_28 and puts it first on PATH, and
  `ci_check_wheel.py` fails a wheel built with an older GCC. A new CI job
  builds and tests with the same toolset and image; the ubuntu GCC 13 job
  stays as the older-compiler check. GCC 8.5 is still the supported
  minimum, and CMake warns when it finds a GCC older than 15. The install
  docs replace `module load gcc` (a module GCC 14 or newer fails to import
  under a stock Miniforge or Mambaforge Python) with the toolset. Not
  bit-identical to GCC 8.5: chignolin trajectories part at step 1188 (seed
  1337) and 939 (seed 42), likely from different multiply-add fusion, as
  between other GCC versions. `scripts/tolerance_check.py` passes in full mode
  with KORP against ec954cf.

- **Energy sums, deltas and the running total are kept in double
  precision, so the running energy no longer drifts from a full
  recompute.** Every energy accumulator and total is double: the per-term
  deltas and full sums, `EnergyChangeResult`/`TotalEnergyResult`, the
  `Potential` interface, `System`'s sums and breakdown,
  `State.current_energy`, the Metropolis input (`beta`, `beta*dE`) and the
  contact-list and H-bond drift budgets. `Integrator(temperature=...)` is
  double too, so `beta` is the replica-exchange driver's `1/T` exactly.
  Per-pair values, caches and coordinates stay float32; KORP's full sum now
  adds the same float pair terms its cache and delta use. In float32 the
  running total random-walked with the number of accepted moves (actin
  7.9e-4 after 50k steps, KORP actin about 0.1 after 3e6, about 1.7
  projected at 1e9); it now stays within 1e-10 of a recompute with no
  growth (actin 3.2e-11 after 1e7 steps, KORP actin exactly 0 after 3e6).
  Accept bits and coordinates are unchanged on the
  parity cases; reported energies change in the last float digits (actin
  Mu full sum 1.6e-6 relative, the float32 summation error that is gone).
  Python API unchanged. Instructions/step within 0.1%, cycles within noise.

- **The neighbour grid keeps each cell once.** With the linked lists gone
  (see Removed), an insert or removal touches only the cell's packed block,
  and the occupied stencil is no longer timed with two clock reads per
  update. Bit-identical (parity vs ec954cf; actin, PGK1 and T4L default,
  actin pivot-only and KIC-only, PGK1 pivot-only, and the masked runs all
  give the same accept bits and final energy as before). Cycles per step,
  n=3, 50k steps: actin -1.3% default, -2.4% pivot-only, PGK1 -1.6% and
  -2.6%, T4L -0.8%, with instructions within 0.3%; masked runs (with the
  grid change above) actin ignore_all:0:40 -8.7%, ignore_all:150:30 -5.1%,
  clash_only:0:40 -1.1%, PGK1 ignore_all:0:40 -5.9%. About 650 lines of C++
  go.

- **A neighbour-grid cell that fills up switches its grid off instead of
  falling back to linked lists.** Cells hold 48 atoms. A cell asked to hold
  more used to switch the whole grid to per-cell linked lists for the rest
  of the grid's life, about 35x slower (actin with 10 A cells: 398 s against
  11 s per million steps). The atom is now left out, the grid goes inactive
  (Mu takes the exact all-pairs delta, H-bonds the brute-force search), and
  every accepted move retries the rebuild until the grid fits.
  `neighbor_proxy_stats()` gains `mu_grid_active`, `mu_grid_overflows` and
  `hbond_grid_overflows`; the warning prints once under `MCPU_VERBOSE`.

- **The Mu neighbour grid leaves out the atoms of residues an `ignore_all`
  energy mask switches off.** Every pair with such an atom scores 0 and
  cannot clash, so only zero terms go and the order of the other atoms in
  each cell is kept: masked runs are bit-identical (actin ignore_all at
  residues 0-39 and 150-179, clash_only at 0-39, PGK1 ignore_all at 0-39,
  same accept bits and final energy), and faster, about 7% fewer
  instructions per step on actin ignore_all:0:40. It also restores the
  hard-core bound on how many atoms share a cell: masked atoms overlap
  freely, and actin with the whole chain masked filled a 48-atom cell. A
  mask set or cleared between runs updates the membership in place: at
  `MCIntegrator.run`, on the next accepted move, or lazily when
  `MuPotential::calculateEnergyChange` sees the mask has changed, so Mu
  never runs on a stale grid (until the sync, `neighbor_proxy_stats()`
  shows the grid as inactive).
  `neighbor_proxy_stats()` gains `mu_grid_n_atoms`.

- **Rigid pivots carry an H-bond again until rounding could change its
  score, which gives back the 6-8% pivot cost of re-scoring them.** Each
  listed H-bond pair now records its slack: the smallest change in the
  ledger's drift bound that could flip any test its score went through
  (the 2.5 A cutoff, the CA-CA and Ramachandran gates, the orientation
  sign and the six angle-bin edges). A pivot carries a pair of two rigid
  sites at its ledger energy while the drift since it was scored stays
  within that slack, and re-scores it otherwise, so the running energy
  stays exact. Pivot-only cycles/step against main ec954cf (n=3, 20k
  steps, GCC 8.5): actin -0.4%, PGK1 +0.5%, LDH-A -0.7%, where re-scoring
  every carried pair cost -7.2%, -7.8% and -7.8%. Default moves +1.4%,
  +1.5% and +0.2%. The H-bond ledger check finds 0 mismatches in 1e6
  pivot-only steps on actin and PGK1 and in the far-from-origin runs, and
  every arch_parity_dump case stays bit-identical.

- **Runs under a residue energy mask no longer pay for moves that leave the
  neighbour grid: masked actin uses 27-46% fewer cycles per step.** Such a
  move cannot use the contact list. Unmasked, a check on the grid rejects
  nearly all of them for an overlap, but it was switched off under a mask, so
  every one ran the all-pairs delta, about 1.3 ms on actin. The check now runs
  under a mask too. It drops the pairs `ignore_all` switches off and keeps the
  clashes `clash_only` keeps, exactly as the delta does, and reads the mask in
  force at each move, so an overlap it finds is one the delta finds. The
  all-pairs delta also leaves out moved atoms of masked residues where all
  their terms are zero (the old positions in either mode, and the new ones
  under `ignore_all`), so a masked tail that leaves the grid on its own costs
  almost nothing. On masked actin (residues 0-39 `ignore_all`, 150k steps) the
  delta's distance tests fell from 186k to 3.9k and its share of the profile
  from 26% to under 1%. Interleaved A/B against the previous main (n=3, user
  cycles/step, 150k steps): actin `ignore_all` 0-39 -27%, `ignore_all` 150-179
  -31%, `clash_only` 0-39 -46%, PGK1 `ignore_all` 0-39 -15%; unmasked actin
  and PGK1, default mix and pivot-only, unchanged (instructions/step equal).
  Bit-identical: the parity dump, and the final energies and accept bits of
  every A/B run.
- **The KORP map is read into memory on 2 MiB pages by default.** The
  energy table is far larger than what 4 KiB pages keep in the TLB, so
  with the old memory-mapped default about 11% of every KORP step's
  cycles were page walks. `load_korp_map` now reads the table into a
  private anonymous mapping, aligned to 2 MiB and advised `MADV_HUGEPAGE`
  (it gets huge pages when THP is `always` or `madvise`, and is ordinary
  memory elsewhere; Pythons built without the constant use the kernel's
  value) and returns it as a read-only array. Interleaved A/B against the
  previous main (n=3, 20k steps, pivot+KIC, user cycles/step): T4L
  -4.7%, CA2 -6.4%, actin -8.0%, PGK1 -7.7%, page walks 11.2% of cycles
  -> 0.1%, instructions unchanged. The cost is ~316 MiB of private memory
  per process (actin: RSS 422 -> 526 MB) and a longer start: ~0.15 s when
  the file is in the page cache, 10-17 s when it is read cold from NFS
  (the memory-mapped mode paid that time as 4 KiB page faults during the
  first steps instead). The old behaviour, one copy shared through the page
  cache by every process on a node, is `KORPForceField(..., map_mmap=True)`
  (`forcefield_options: {map_mmap: true}` in a config) or `load_korp_map(path,
  mmap=True)`; use it when many ranks share a node short of memory. Loads of
  the same unchanged file in the same mode now return one shared `KorpMap` per
  process, so several force fields or systems hold one table. `sha256=True`
  works in both modes and, in the default one, digests the bytes already read
  instead of reading the file again. Bit-identical: the parity dump, the KORP
  checks of `scripts/tolerance_check.py` (zero difference against the previous
  main) and the accept counts and final energies of every A/B run match.
- **The KIC root solver does less bookkeeping per solve.** The Sturm
  sequence is packed for vector evaluation without looking up each
  member's order per entry (that loop was about a fifth of a solve with
  roots), each remainder step reads its leading coefficient once so the
  update vectorises, and the Newton finish runs on up to four isolated
  roots at once in AVX lanes, each lane making the scalar loop's
  decisions with the same operations. On 3,240 T4L closure polynomials
  a standalone solve went from 13.0k to 9.1k TSC ticks. Interleaved A/B
  against the previous main (n=3, user cycles/step) on T4L, CA2, LDH-A,
  actin and PGK1: KIC-only -2% to -4%, default mix -1% to -2%.
  Bit-identical in every check (the parity dump, the closure coordinates
  of 21,230 KIC solutions in the T4L and actin dumps, the A/B energies),
  though not by construction: the lanes repeat the scalar loop's FMAs, so
  this relies on the compiler contracting the scalar Horner step the same
  way. Passes `scripts/tolerance_check.py`. The non-AVX2 build runs the
  scalar Newton loop as before. Polynomials with tightly clustered roots,
  whose floating-point Sturm counts are not monotone, still return every
  value isolation produces (hundreds in synthetic tests), as before.
- **KORP moves cost about a fifth less.** On builds with AVX2 and FMA
  (the default `-DMCPU_ARCH=v3`), the pair binning counts the distance
  shell and the two polar rings with vector compares and finds the three
  angle bins (psi_a, psi_b, chi) together, one per vector lane, with no
  data-dependent branch. The bins are the same as before by construction,
  in every build mode: the counts are the same comparisons, and the edge
  tests that decide each angle bin are rounded the same way in both paths
  (one fused multiply-subtract by default, the form GCC already compiled
  the scalar test to; two rounded products under `MCPU_FP_CONTRACT=off`).

  Interleaved A/B against main, pivot + KIC, user-space cycles per step
  (n = 3): actin 557k -> 436k (-22%), PGK1 612k -> 482k (-21%),
  beta-galactosidase (AF-P00722) 1.81M -> 1.46M (-19%). Branch mispredicts
  per step fell about tenfold (actin 4726 -> 486). Every run ended on the
  same energy and accept sequence as main, as did a 200k-step pivot-only
  actin run, whose running and recomputed energies also match main's.
- **The Mu contact walk is about 5-7% faster per step.** The walk over a
  moved atom's new neighbours now keeps the distance its 8-slot prefilter
  computed: each kept slot's partner id and r2 are compressed into two
  lists, and the energy is scored from them, so a candidate is no longer
  looked up again and its coordinates reloaded to recompute r2. The
  prefilter now applies the exact cutoff. With FMA and the default
  contraction, r2 is formed in the same FMA order GCC 8.5 emits for the
  scalar form at -march=haswell; with `MCPU_FP_CONTRACT` set to anything
  but `fast`, or without FMA, it is the unfused sum, as the scalar form is
  in those builds. The pending contact lists also push with an inlined
  capacity test and store, where std::vector::push_back had been left out
  of line. Interleaved A/B against the previous main (n=3, user
  cycles/step, two runs) on T4L, CA2, LDH-A, actin and PGK1: default mix
  -4.7% to -6.8%, pivot-only -5.2% to -7.3%, actin with a residue energy
  mask -3.9%. Bit-identical in these builds (`scripts/arch_parity_dump.py`
  and every A/B run); a compiler that contracts the scalar distance
  differently could move a pair lying on the cutoff by an ulp, within
  tolerance.
- **Pivot moves cost 3-4% less, the default mix about 2% less.** Three
  scans now use AVX2 compares and stop at the same first match as before:
  the live-cell pass that H-bond and Mu probes run over their 27 stencil
  cells, the search for a dropped partner when an accepted move updates a
  pair ledger, and the search for an atom's slot when it moves within its
  grid cell. Interleaved A/B against the previous main (n=3, user
  cycles/step): pivot-only actin -4.2%, PGK1 -3.3%; default mix -0.4% to
  -2.0% on T4L, CA2, LDH-A, actin and PGK1; actin with 30 residues masked
  -1.4%; KIC-only unchanged. Bit-identical.
- **Rotamer-library sidechain moves compute log(sigma) once, at load.**
  The proposal density stores each component's log(sigma) when the table
  is loaded instead of calling `logf` per component and chi angle (one
  `logf` per call remains, for the log-sum-exp), and keeps its
  per-component terms on the stack instead of a new heap vector per call.
  Writing `RotamerLibrary.row(...).sigma` from Python refreshes the cached
  value. `wrap_angle_to_pi` rounds with `std::trunc` plus a fix-up that
  equals `std::round` bit for bit, which drops the `roundf` call on SSE4.1
  builds (`MCPU_ARCH` v2 and up). The per-energy-group timers in
  `System::evaluateDeltaEnergy`, on by default, read the TSC instead of
  calling `steady_clock::now()` twice per potential per move; the
  `step_stats` timings and `MCPU_ENERGY_TIMING=0` are unchanged.
  Interleaved A/B against the previous main (n=3, user cycles/step) on
  T4L, CA2, LDH-A, actin and PGK1: default mix -1.0% to -2.9% (actin
  instructions/step -4.7%), pivot-only and KIC-only -1.4% to +0.4%.
  Bit-identical.

- **KIC proposals cost about a fifth less.** The degree-16 closure
  polynomial is built from products whose degrees are fixed when
  compiling, so each unrolls to straight-line code (bit-identical
  coefficients). Each root the Sturm counts isolate is then finished by
  Newton's method kept inside its sign-change bracket, about 9 Horner
  passes per root where regula falsi plus sign halvings took about 50; the
  roots are also more accurate (largest relative error against 50-digit
  roots 4.7e-10 -> 5.7e-12), so fewer closures fail the 1e-6 rad N-CA-C
  check. Interleaved A/B against the previous main (n=3, user
  cycles/step) on T4L, CA2, LDH-A, actin and PGK1: KIC-only -21% to -23%,
  default mix -8% to -11%. Not bit-identical; passes
  `scripts/tolerance_check.py`, and KIC accept, presolve-zero and
  reverse-missing counts over 8 seeds x 200k KIC-only steps match the
  previous main within statistical error.

- **The H-bond energy change walks the shared pair-search layer and skips
  pairs inside a rigid pivot body, which makes pivot moves 4-8% faster.**
  The walks from a moved donor's H into the O grid and from a moved
  acceptor's O into the H grid now use the layer's 8-slot distance
  prefilter and skip cells whose sites all belong to affected residues.
  Under a rigid move, a pair whose donor and acceptor geometry (backbone of
  residues r-1 to r+1) moved as one body keeps its energy: it is left out
  of both sides of the delta and its ledger entry is carried over.
  `Context.set_skip_rigid_mm(False)` scores those pairs again, as a
  reference. Interleaved A/B against the previous main (n=3, user
  cycles/step) on T4L, CA2, LDH-A, actin and PGK1: pivot-only -4.5% to
  -8.2%, default mix about -1% to -4%, masked actin -3%, KIC-only +0.1% to
  +1.3% (within run-to-run spread; KIC moves have no rigid sites). Short
  runs (3k-60k steps) and `arch_parity_dump` are bit-identical to before.
  Over long pivot runs, rotation rounding now and then moves a skipped pair
  across a bin edge and the trajectory departs from the previous one: 2 of
  8 runs of 300k pivot-only steps did, and the H-bond ledger check reports
  those pairs as mismatches. With `set_skip_rigid_mm(False)` both runs
  matched the previous main exactly. |running - full| stayed within
  max(1e-3, 1e-5 |E|) in all 8 runs and `tolerance_check` passes.

- **KORP moves are about 1.3x faster again, bit-identical.** The pair
  angles psi_a, psi_b and chi are no longer computed: each is binned by
  testing its (y, x) vector against precomputed bin-edge directions, after
  a rough float angle has picked the bin (the libm-free atan2 and the
  divisions by the bin widths were about 35% of KORP cycles). The shell
  and ring are counted against padded boundary arrays instead of found by
  loops with data-dependent exits, and the candidate pass over a changed
  residue's partners runs eight at a time with AVX2. Interleaved A/B,
  pivot + KIC, user-space cycles per step (n = 3): actin 720k -> 547k
  (-24%), PGK1 832k -> 628k (-24%), beta-galactosidase (AF-P00722)
  2.51M -> 1.75M (-30%); pivot-only actin 1.31M -> 972k. Every run ended
  with the same energy, coordinates and accept count as before,
  `arch_parity_dump` is unchanged, and the checks of
  `scripts/tolerance_check.py --mode full` (including `korpe`) pass; a bin
  can differ only for an angle within a few ulp of a bin edge.

- **Pivot bookkeeping runs over index ranges, and the rotation is
  vectorised; bit-identical.** A pivot moves one contiguous block of each
  atom kind, and `ProposalPatch::mark_moved_range` now records those
  ranges, so marking, clearing the masks, copying coordinates on accept
  and reject, the bounds test and the displacement check run per range
  (memset, memcpy, branch-free loops) instead of per atom. Moves that mark
  single atoms, and patches whose `moved_indices` Python sets, keep the
  per-atom loops. `CoordsSoA::rotate_atoms` rotates four atoms per AVX
  pass with the same fused multiply-adds the compiler emitted for the
  scalar loop. Interleaved A/B against the previous main (n=3, user
  cycles/step) on T4L, CA2, LDH-A, actin and PGK1: default mix -6% to
  -12% (actin 59.7k -> 52.4k), pivot-only -15% to -26%, KIC-only
  unchanged. Final energies, accept sequences and `arch_parity_dump` are
  unchanged.

- **Mu pair walk round 4, bit-identical.** Mu reads its per-pair topology
  flags from a compact table (a band over pairs within four residues plus
  one byte per atom-type pair and a short per-atom exception list, 91-207 KB
  on the test proteins) instead of the N*N byte `topo_flag_` (8-9 MB), whose
  lookups were 28-30% of a Mu move's L2 misses; the compact form is checked
  against every `topo_flag_` entry at setup and is not used if any differs.
  The pair-search drain drops moved partners and `span_mask8` masks short
  spans without branches (together 23-31% of a Mu move's mispredicts), the
  stencil walk adds linear offsets for home cells away from the grid faces
  instead of bounds-testing all 27 cells, and the moved-cell counts are
  cleared with one memset after a large move. Interleaved A/B against the
  previous main (n=3, user cycles/step) on T4L, CA2, LDH-A, actin and PGK1:
  default mix -4.8% to -6.5% (actin 60.1k -> 56.6k), pivot-only -10.1% to
  -11.1%, masked actin (`ignore_all`) -4.5%. Final energies, accept
  sequences and `arch_parity_dump` are unchanged.
- **Per-step fixed costs trimmed, bit-identical.** H-bond pair evaluation
  bins its six orientation angles and makes the CA-CA test on cosines,
  using acos only when a cosine is within 1e-4 of a bin edge, and computes
  each Ramachandran-gate dihedral only when a test reads it (libm acosf and
  atan2f were about 6% of T4L default cycles). The Sturm solver no longer
  zero-fills its 4.6 KB polynomial sequence on every solve.
  `trial_in_bounds` checks a contiguous moved range (every pivot) in one
  branch-free pass. Interleaved A/B against the previous main (n=3, user
  cycles/step) on T4L, CA2, LDH-A, actin and PGK1: default mix -2% to -6%
  (actin 78.3k -> 73.6k), KIC-only -2% to -7%, pivot-only -2% to -5%;
  those numbers also include a cheaper virtual amide H grid reset that the
  H-bond ledger change below has since replaced. Final energies, accept
  sequences and `arch_parity_dump` are unchanged.

- **H-bond moves score each changed pair once.** `HBondPotential` now keeps
  the accepted state's nonzero (donor, acceptor) pair energies on the
  `State` (`HBondStateCache`) and folds each accepted move into it through
  `Potential::commitAcceptedMove`. The old side of the delta is read from
  it, so a move no longer walks the grids at the old H and O positions or
  scores every candidate pair a second time; the affected x affected scan
  now looks only at proposed positions. On an accepted backbone move the
  virtual amide H grid moves only the donors whose N, CA or previous C
  moved, instead of being cleared and refilled for every donor. The set of
  pairs whose energy changes is the same as before on every move checked:
  `Context.set_hbond_ledger_check(True)` re-scores both states the old way
  on every delta and counts disagreements (0 over pivot, KIC, side-chain
  and masked runs on actin, T4L, LDH-A, CA2 and PGK1;
  `tests/physics/test_hbond_ledger_check.py`). The delta is summed in
  double, so it can differ from main in the last bits;
  `scripts/tolerance_check.py` passes, and the interleaved A/B runs below
  ended on the same energies and accept sequences as main. Geometry checks
  per step fall about 3.5x. User-space cycles per step, main -> this
  (n = 3, 40k steps): default mix T4L 68.2k -> 62.1k, CA2 68.5k -> 64.2k,
  LDH-A 84.0k -> 73.9k, actin 76.5k -> 67.9k, PGK1 77.9k -> 68.0k
  (1.07-1.15x); pivot-only 1.05-1.12x; KIC-only 1.09-1.17x; actin with 40
  residues masked 109k -> 101k.

- **KORP moves are about 1.6x faster again.** Two costs dominated a KORP
  step after the pair cache: glibc's correctly rounded `acos` and `atan2`
  (34% of cycles), and table lookups that miss the TLB and the caches (the
  ~330 MB map is read through 4 KB file-backed pages). The pair angles now
  come from dot products with the orthonormal residue frame and a libm-free
  atan2 (the Cephes double atan, error ~1e-16), and each changed residue's
  partners are scored in three passes: a branch-free pass picks the pairs
  that can contribute, a second computes and prefetches their table
  entries, and a third reads them in the original order.

  Interleaved A/B, pivot + KIC, user-space cycles per step (n = 3): actin
  1.19M -> 735k (-38%), PGK1 1.32M -> 818k (-38%), beta-galactosidase
  (AF-P00722) 4.18M -> 2.66M (-36%). Every A/B run ended with the same
  energy and accept count as before; a bin can now differ only for an
  angle within a few ulp of a bin edge. The KORP checks of
  `scripts/tolerance_check.py --mode full` (including `korpe`) pass.

- **The Mu contact walk collects a probe's candidates before scoring
  them.** Each moved atom's stencil walk now reuses the live cell list of
  the previous probe when both sit in the same home cell, and collects
  every slot the distance prefilter keeps, branch-free, before scoring
  them in one loop, in the same order as before. Bit-identical. Branch
  mispredicts per step drop 17-27%; cycles/step: default mix 1.05-1.07x
  (actin, T4L, CA2, LDH-A, PGK1), pivot-only 1.08-1.10x, masked actin
  1.03x.


- **KIC root refinement no longer falls back to Sturm-count bisection.**
  About 40% of the roots of the closure polynomial never meet the
  regula-falsi stop (|f(x)/x| < 1e-15) in 20 iterations, and each then
  bisected its whole isolating interval with ~51 Sturm counts. The
  regula-falsi pass now returns its narrowed bracket and the root is
  finished by bisecting on the sign of the polynomial, to the same 1e-15
  relative width; Sturm counts still isolate the roots. Closure counts are
  unchanged on 10,475 T4L/CA2 solves and coordinates move by at most
  3.7e-9 A (76% bit-identical). KIC-only steps: T4L 139k -> 125k cycles
  (-10%), CA2 115k -> 106k (-8%); actin default mix 91.9k -> 88.8k (-3%).

- **A KIC step no longer allocates.** A closure (`Solution`) held three
  heap vectors of three points, and each step built two closure lists, a
  root list and two more closures, each closure three allocations plus
  copies. `Solution` now holds fixed-size arrays and the root and closure
  lists are reused across steps. Python still sees `Solution.r_n`, `r_a`
  and `r_c` as lists of three points, but setting one now takes exactly
  three; before, a list of another length was accepted and the KIC code
  could read past its end.
  Trajectories are unchanged bit for bit. KIC-only steps: T4L 125k -> 120k
  cycles (-4%), CA2 106k -> 100k (-5%); actin default mix 88.8k -> 87.1k.
- **KORP moves are about twice as fast.** `OrientationalPairPotential` now
  keeps the accepted state's residue frames and pair energies on the `State`
  (`KorpStateCache`) and folds each accepted move into them through a new
  `Potential::commitAcceptedMove` hook. A move rebuilds frames only for the
  residues whose N, CA or C moved, takes the old side of each pair from the
  cache instead of scoring it again, and skips pairs beyond the cutoff with a
  plain distance test before any trigonometry. The CA excluded-volume check
  first asks, per moved residue and without branches, whether any partner is
  near the cutoff, and runs the exact test only then.

  Interleaved A/B, pivot + KIC, user-space cycles per step (n = 3): actin
  2.23M -> 1.13M (1.97x), PGK1 2.53M -> 1.29M (1.97x), beta-galactosidase
  (AF-P00722, 8222 atoms) 8.35M -> 3.73M (2.24x). Trajectories are unchanged
  on every run checked: the same final energies and accept sequences in the
  A/B runs, the same running and recomputed energies after 1M pivot-only steps
  on T4 lysozyme, and `scripts/tolerance_check.py` passes. The cache is
  rebuilt by a full-energy resync and dropped whenever the coordinates are
  replaced, and a copied `State` starts without one.

- **The pair-search layer gains per-replica scratch, a grid registry, a
  pair ledger and a shared move footprint.** Internal refactor,
  bit-identical. The per-cell moved-atom counts and the clash_hot list move
  from Mu's workspace to a `PairScratch` on each Context, with one set of
  counts per registered grid. `NeighborSystem::register_subset_grid` lets a
  term add a grid over its own atoms; the Mu and H-bond grids sit at fixed
  ids in the same registry, and an accepted move now updates each grid from
  the move's moved-atom list instead of scanning every atom; a re-init
  after an atom reorder keeps the registered grids and their ids.
  `OpenCellGrid` is `BasicOpenCellGrid<48>`, with the per-cell capacity a
  template parameter; every grid, subset grids included, still uses 48.
  Mu's contact list is a `PairLedger<float>` with
  `PendingPairs` for the changes a move stages, ready for the H-bond and
  KORP pair energies. `SiteClass` (Fixed, Rigid, Flex) is the one
  definition of how a move affects a site: Mu's rigid moved-moved skip and
  KORP's frame classes both use it.
  The Context now keeps whether its atom permutation is the identity, so
  a pivot no longer walks the permutation to find out. Cycles per step
  against the parent, interleaved, n=3: 0.9-2.6% faster on the default
  move mix and 1.4-3.3% faster pivot-only on T4 lysozyme, CA2, LDH-A,
  actin and PGK1. The commit step's share of an actin default run drops
  from 4.0% to 1.9%.

- **Mu's contact-list delta walks its pairs through a shared pair-search
  layer, and is 5-9% faster.** Internal refactor, bit-identical. The cell
  walks that Mu's contact-list delta wrote out by hand (the clash-first
  pass, the moved-vs-grid contact walk, the moved-moved loop), the 8-slot
  distance prefilter, the per-cell moved-atom counts and the clash_hot
  list now live in header-only templates under
  `include/pymcpu/neighbor/` (`SpanMask.h`, `MovedCells.h`,
  `PairSearch.h`), for the H-bond and KORP terms to use next. Each walk
  is an always-inlined template that takes the term's callback by
  forwarding reference, and the exact distance test stays in Mu's
  callback, so the pairs, their order and every rounding are unchanged.
  The contact walk's per-cell callback, which the compiler had kept out of
  line, is now inlined: cycles per step -4.9% to -5.7% on the default move
  mix and -7.6% to -8.6% pivot-only on T4 lysozyme, CA2, LDH-A, actin and
  PGK1 (interleaved, n=3, against the parent). `scripts/check_inlining.py`
  disassembles the built extension and fails if the hot functions call into
  the layer or a lambda.

- **Ordinary runs no longer print a Mu contact-list NOTE.** The engine
  printed "NOTE: Mu move #1 that cannot use the contact list ..." to stderr
  on the first move that could not use the list, and on every thousandth,
  which happens in normal runs, for example chignolin at T = 0.8. It is a
  developer diagnostic, and it showed up inside notebook cells. It is now
  printed only with `MCPU_VERBOSE=1`, like the engine's other diagnostics.

- **A replica swap no longer pays for O(N^2) Mu passes or a full grid
  clear, and the aromatic energy change computes each ring's geometry once.**
  Trajectories are unchanged bit for bit, including across a two-replica
  swap run compared step by step.

  * The full-energy resync after a swap now fills the Mu contact list even
    when the swap dropped it, and the first move that can use a list adopts
    it instead of rebuilding it. The first 50 steps after a swap take
    48.6 -> 4.5 Mcycles per replica on actin and 55.5 -> 5.4 on PGK1 (417
    residues); `calculate_total_energy(-1)` takes 67.8 -> 62.7 and
    77.2 -> 71.9, partly because the O(N^2) Mu loops now iterate a list of
    the non-amide-H atoms.
  * `set_positions` no longer zeroes the whole neighbour grid when it
    reconfigures it: 33.4 -> 8.1 Mcycles on actin, 29.4 -> 5.0 on PGK1. A
    swap calls it on both replicas.
  * The aromatic term computes every ring's geometry once per state instead
    of inside its pair loop, and returns 0 for a move that touches no ring
    atom. 10,000 steps: -3.2% on actin, -4.7% on PGK1.
  * The Mu full energy and the contact-list rebuild find pairs through a
    cell grid instead of testing every pair of atoms. A full recompute takes
    63.2 -> 20.6 Mcycles on actin and 72.1 -> 24.3 on PGK1. With all of the
    above, an accepted swap plus a full recompute costs one replica 2.8% of
    10,000 MC steps on actin and 3.0% on PGK1, down from 15.2% and 15.4%.
- **The H-bond energy change skips donor-acceptor pairs that are far apart
  in both the old and the new state, which makes 164-417 residue proteins
  2-8% faster.** On pivot moves the H-bond delta was 13-24% of the step and
  the one term whose cost grows with how much of the chain moves. Pairs
  that can score are evaluated at the same point in the same order, so
  trajectories are unchanged bit for bit; only the parity dump's work
  counters drop. Cycles per step, default move mix / pivot-only: T4
  lysozyme -3.3% / -6.2%, CA2 -4.1% / -2.3%, LDH-A -4.4% / -5.6%, actin
  -5.8% / -5.5%, PGK1 -5.2% / -8.2%.
- **`ProposalPatch` checks moved-atom lists set from Python.** Setting
  `moved_indices` to a list with an index outside `[0, num_atoms)` raises
  `IndexError`, and a list that names an atom twice raises `ValueError`;
  `mark_moved` raises `IndexError` for an out-of-range index and ignores an
  atom that is already marked, where it used to append it again. Mu's delta
  compares each grid cell's count of moved atoms with its occupancy, so a
  duplicate made a cell that still held an unmoved atom look fully moved and
  dropped its pairs, and an out-of-range index wrote past `moving_atoms`. The
  engine's own moves never build such a list, so the release engine does not
  check and its results are unchanged; debug builds assert the contract.
- **Release builds pad branches against Intel's JCC erratum.** On Skylake
  through Cascade Lake, a jump that crosses or ends on a 32-byte boundary
  cannot be served from the decoded-uop cache, and about half of the
  engine's uops were coming from the slower legacy decoder. The build now
  passes `-Wa,-mbranches-within-32B-boundaries` to the compile and the LTO
  link when the toolchain honours it. On a Cascade Lake Xeon with GCC 13 it
  cut cycles per step by 4.6% on actin and by 3.3-7.5% on an 8222-atom
  protein (8.5-13.6% compared with a GCC 8 build), and runs stay
  bit-identical. It also removes a code-layout effect that made speed swing
  by about 5% after unrelated edits. The new `MCPU_JCC_PAD` option (`AUTO`,
  the default; `ON`; `OFF`) controls it, and `build_info()["build"]` reports
  `jcc_pad`, `jcc_pad_mode` and `jcc_pad_reason`. `AUTO` turns padding off
  when the assembler is older than binutils 2.34, or when the GCC LTO link
  drops `-Wa` options. GCC 8 drops them without a warning, so CMake now
  tests for this instead of assuming. Wheels are built with
  `MCPU_JCC_PAD=ON`.

- **Runs with a residue energy mask use the Mu contact list.** A mask used
  to switch the Mu term off the list, so every masked move took a slower
  walk of the neighbour grid. Masked pairs score 0, so a list built under a
  mask never holds them; the list is now kept under a mask and rebuilt when
  the mask is set or cleared. Masked runs take 44-64% fewer cycles per step
  on actin and PGK1. The accept sequence is unchanged, but the running
  energy can differ from earlier versions in the last bits (up to 6e-4 in
  50,000 steps) because the move's pairs are summed in a different order.

- **Mu checks a move that falls back to the all-pairs delta for overlaps on
  the grid first, and its clash-first pass stops after the atoms that
  overlapped recently, which makes pivot-only runs 1.8-2.0x faster on actin
  and LDH-A.** Trajectories are unchanged bit for bit.

  * A move that leaves the Mu grid, or carries too far for the contact
    list, takes the all-pairs delta, which tests every moved atom against
    every atom and does not stop at an overlap. Only 0.4-0.6% of pivots on
    actin and LDH-A take it, but they cost over a third of a pivot-only
    step, and nearly all of them end in a steric rejection. Such a move now
    runs the clash-first pass on the grid first; an overlap found there is
    one the delta finds too.
  * 98.6-99.4% of the pivots that overlap are caught on an atom that
    overlapped in a recent rejected move, so the clash-first pass now tests
    only those. The contact walk still rejects any overlap the pass misses,
    and records the atom it stopped on for the next pass.

  Cycles per step before and after, interleaved, n=3, 20,000 steps,
  pivot-only: -45.3% actin, -50.8% LDH-A, -9.3% T4 lysozyme, -5.5% PGK1,
  -5.3% CA2; default move mix: -2.2% LDH-A, -2.2% T4 lysozyme, -0.9% PGK1,
  -0.3% actin, +0.4% CA2.

- **`scripts/job_template.slurm` no longer activates a particular conda
  environment.** It activated `mcpu_dev`, a conda environment from one
  developer's setup, and used that environment's `mpirun`. The job now
  uses the environment it is submitted from, and a marked block shows how
  to load one instead. `scripts/submit.sh` now requires the config
  argument; it defaulted to `inputs/template.yaml`, which does not exist.

- **`scripts/arch_parity_dump.py` no longer fails a comparison on internal
  work counters alone.** Counters such as `hbond_num_candidates_iterated`,
  `neighbor_num_cell_visits` and the Mu candidate counters are printed as a
  note when they differ; energies, accept bits, coordinate hashes, move counts
  and step totals stay strict. A speedup that skips work the result does not
  depend on used to read as a parity failure.

- **Mu's energy change does less work per moved atom, which makes actin
  1.7x faster on the default move mix and 2.3x pivot-only.** Each part
  scores the same pairs in the same order and returns the same answer, so
  trajectories are unchanged bit for bit. Cycles per step on actin's default
  move mix (3000 steps), actin pivot-only (1500) and chignolin (30000), each
  against the step before:

  * Both walks over a moved atom's new neighbours (the clash-first pass and
    the contact walk) skip the cells that hold only atoms the move
    displaced. The grid holds accepted coordinates, so those atoms were
    skipped one by one anyway; on an actin pivot they filled about three
    quarters of the cells visited. -25%, -37% and -7%.
  * The clash-first pass tests first the atoms that overlapped in recent
    rejected moves, then the moved atoms from the end of the move's list
    (for a pivot, side chains before backbone). It used to test about 30%
    of a rejected actin pivot's atoms before it found the overlap. -12% and
    -20% on actin; chignolin's moves are too small for the pass.
  * Both walks test eight slots of a cell at once against the cutoff (AVX2,
    in the default `v3` build) and look at the survivors one by one, as
    before; they used to branch on every slot. They also gather the cells
    to visit before visiting them, without a branch per cell. -11%, -13%
    and -6%. Builds without AVX2 run a scalar loop with the same result.

  All three together: -41% cycles per step on actin's default move mix,
  -57% pivot-only and -12% on chignolin.

- **KIC moves find their closures 2.5x faster, which makes chignolin 11%
  faster on the default move mix.** Most of a closure's time went to the
  Sturm root counts of its degree-16 polynomial, and most of those to the
  bisection that finishes a root. With AVX2 and FMA (the default `v3`
  build) a count evaluates four polynomials of the Sturm sequence per
  vector, with the same fused multiply-adds as before, and the bisection
  takes two steps per round from counts computed together. 37k to 15k
  cycles per solve; every root is unchanged bit for bit, so trajectories
  are too. Cycles per step: -11% on chignolin's default move mix, -4% on
  actin's. Builds without AVX2, and builds with `MCPU_FP_CONTRACT` set to
  anything but `fast`, run the scalar code as before.

  The packed counts hold the cores at a lower AVX clock. With every core of
  a socket running a simulation (22 processes on a Xeon 8268) the clock
  drops from 3.44 to 2.97 GHz, so actin's default move mix makes 3.4% fewer
  steps per second across the socket than with the scalar counts, while
  chignolin, where closures are a larger share of the step, still makes
  11% more. A single process gains on both: 7% on actin, 17% on chignolin.

- **An accepted move copies only the atoms it moved, and the step timers
  read the CPU's time-stamp counter.** Committing a move scanned the whole
  moved-atom mask, one entry per atom of the system; it now walks the list
  of moved atoms (-2% cycles per step on actin's default move mix, -3%
  pivot-only). The per-step timers behind `Integrator.step_stats()` read
  the time-stamp counter on x86, calibrated once against `steady_clock`
  in a 2 ms wait the first time a timer runs, instead of `steady_clock`
  itself (-3% on actin, -5% on chignolin); they report the same times, and
  other platforms keep `steady_clock`.
  Trajectories are unchanged bit for bit.

- **A YAML config with an unknown key is an error.** The flat YAML schema
  ignored any key it did not read, so a misspelled key such as `num_cylces`
  or `checkpoint_intrval` silently left the setting at its default, and
  `checkpointing: false` left checkpointing on. Loading a YAML config now
  raises `ValueError` naming every unknown key, at the top level and in the
  `checkpointing` block, with the closest known key or, for a name from
  the JSON schema such as `report_interval` or `integrator`, what to write
  instead; and `checkpointing` must be a mapping.

  The old `load_yaml` allow-list also let through eight keys that no loader
  read. Two of them, which existing configs carry, still load but warn that
  they have no effect: `output_layout` and `mode` (a YAML config runs
  replica exchange when it lists more than one temperature, and folding
  otherwise). The other six are now errors: `temperature` (write
  `temperatures: [T]`, a list even for one temperature) and the JSON blocks
  `integrator`, `outputs`, `replica_exchange`, `constraints` and
  `checkpoint`.

  `pymcpu.utils.yaml_parser.load_yaml` raises the same error instead of
  warning, and its `KNOWN_FIELDS` list, which had drifted from the keys the
  loader reads, is gone; `pymcpu.config.check_yaml_keys` does the check.
  `replica_grid_dims` no longer counts `n_q_windows`, which the YAML loader
  never read (it is an argument of `ReplicaExchange` and a flag of
  `scripts/run_mcpu_replica_exchange.py`), so `scripts/submit.sh` could
  size a job for more replicas than the run made. `scripts/submit.sh` also
  checks the keys before it submits, so a typo fails at once rather than
  after the job has waited in the queue. JSON configs already rejected
  unknown fields.

- **A structure placed far from the origin runs shifted next to it, and
  coordinates come back as float64.** Coordinates are float32 and every move
  rounds each coordinate it changes at its absolute value, so the rounding
  grows with the distance from the origin (3.8e-6 Å per float step at 50 Å,
  2.4e-4 Å at 4000 Å). Far out that showed: over 200k actin pivot steps,
  backbone bond lengths drifted by up to 6.9e-3 Å at 1000 Å and 2.2e-2 Å at
  4000 Å (2e-4 Å at the origin); chignolin's KIC moves failed their
  reversibility check 193 times in 20k steps at 300 Å and 1967 times at
  4000 Å, where only 10 were accepted (1 failure and about 430 accepted at
  the origin); and from a few thousand Å out a rigidly carried pair could
  slip under its hard-core cutoff and stop the run with `StericClashError`.
  Such inputs are common: cryo-EM models often sit several hundred Å out.
  A `Context` now runs a structure that reaches 64 Å or more from the
  origin in an engine frame shifted next to it: on its first placement it
  shifts each axis whose coordinates all lie on one side of the origin by a
  whole number of Å. That is exact for float32 input, so every distance, and
  every energy term computed from distances, is unchanged bit for bit (the
  virtual amide hydrogens and aromatic ring centres, built from absolute
  positions, now round more finely), and the run behaves as it would at the
  origin. The new read-only `Context.frame_offset` holds the shift, and
  `Context.coords`, XTC files, checkpoints, `get_coords` and
  `EngineSession.coords()` add it back, so callers see their own frame. They
  return float64, because engine + offset is exact only in double: a run
  restored from them, or a replica swap, re-enters bit for bit in a
  `Context` that placed the same start structure first, as every pyMCPU
  driver does (store them as float64 to keep that; an engine coordinate
  within a few 1e-6 Å of zero can come back off by about 2e-13 Å).
  `set_positions` takes float64 too (rounded to float32 once, in the engine
  frame), and a keyword-only `frame_offset` to choose the shift. Structures
  within 64 Å of the origin, or straddling it on every axis, run in the
  engine exactly as before, which covers every input in the test suite
  except the KORP structures, whose energies are unchanged. Python code that
  takes coordinates from `get_coords`, such as REMD's native-contact counts,
  now computes in float64. `get_state().coords` is in the engine frame. If
  coordinates still reach 256 Å in the engine frame, the `Context` prints a
  note (once per process); Mu's far-from-origin note is gone. Integrations
  that store coordinates between runs, such as the pymcpu-westpa add-on,
  should store them as float64.

- **A trial move out of the neighbour grid drops Mu's contact list only if it
  is accepted.** Such a move cannot use the list, and it used to discard it
  at once, so the next move rebuilt it from the coordinates (O(N^2), about
  15 ms on actin), although nearly every such trial is rejected. It is now
  dropped only when such a move is accepted. A 200k-step actin run on the
  default move mix takes 53 µs/step instead of 69 (seed 11). Trajectories
  are unchanged.

- **Rigid pivots no longer re-check the pairs they carry, which makes actin
  1.5x faster on the default move mix and 1.9x pivot-only.** A rigid pivot
  keeps every distance inside the segment it turns, so Mu does not score
  those pairs (`skip_rigid_mm`). It, and the KORP CA-CA guard, still
  re-checked them for a hard-core clash, because the pivot rounds each
  carried coordinate to float and a pair a move had left exactly on its
  cutoff could land just under it; the full energy then found a clash in a
  state the move had accepted. That re-check cost a third of the actin step
  (half of it pivot-only), and it is gone. Moves are still tested against
  the cutoff; a whole state (`calculate_total_energy`, `has_steric_clash`,
  `Simulation`'s clash check, `KORPForceField`'s construction check) is now
  judged against one 0.001 Å looser, `mcpu_core.STATE_CLASH_BUFFER_A`, and a
  pair inside that margin scores as any pair at its distance; the old side of
  a move's energy change takes back the contact energy the running energy
  holds for a pair, with no clash test. With nothing re-checking carried
  pairs, none went more than 1.8e-6 Å under its cutoff in 20M-step chignolin
  and 5M-step actin runs (three seeds each, with the rotation fix above).
  That holds near the origin: a float step grows with the coordinate, and
  with actin moved 4000 Å out a carried pair went through the margin within
  200k pivot-only steps (1000 Å out, none did in 1M). The engine therefore
  keeps its coordinates near the origin (see "A structure placed far from
  the origin..." above). Trajectories are unchanged up to the first move the
  re-check would have rejected; on actin the accepted moves of 2000 steps
  match bit for bit. Two behaviours change:

  * A pivot that carries an overlap the state already holds (coordinates
    from `set_positions`, a `clash_only` run, whose full energy ignores
    clashes, or a pair an `ignore_all` mask allows) is accepted. In 0.1.0 the
    re-check also ignored `ignore_all`, so masked (linker) residues that
    overlapped could get stuck. A move that re-decides an overlapping pair,
    with one atom moved, is still rejected, under `clash_only` too.
  * `PhysicsVerifier` passes a move rejected for a pair that is under the
    move cutoff but not the state cutoff. It asks the potential, through the
    new `Potential::clashesAtMoveCutoff`.

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

- `OrientationalPairPotential.set_rigid_skip_enabled` and
  `rigid_skip_enabled` (KORP). The skip of pairs carried by one rigid pivot
  was off by default because it is not exact for a nearest-bin table in
  float32; KORP now always re-scores them, as it did by default. Delete
  `set_rigid_skip_enabled(False)` calls.

- Python bindings nothing used: `Context.set_coords_from_python` (assign
  `Context.coords`), `has_hard_constraint_violation` (same as
  `has_steric_clash`), `print_neighbor_audit`, `print_neighbor_proxy_stats`
  (deprecated), `mu_cell_size_A` (still in `neighbor_proxy_stats()`),
  `coord_sync_stats`/`reset_coord_sync_stats` (on `Context` and the module;
  the counters behind them go too), `MuPotential.mu_cutoff_sq`,
  `topo_flag_size_mb` and `type_params_size_kb`,
  `NativeContactsBiasPotential.initialize_pair_cache` (it filled a
  temporary copy of its argument), `EnergyWeights.set_legacy_defaults`,
  `set_unweighted` and `outer_weight` (use `set_use_legacy_weights` and
  `weight_for_group`), `BlockIndices.has_sidechain`/`has_hydrogen`/
  `has_oxygen`, six debug fields of `ProposalPatch`
  (`first_affected_residue`, `last_affected_residue` and the
  `*_atom_moved` masks) and `TripeptideSolver.get_xi`/`get_eta`/
  `get_delta`. The `neighbor_proxy_stats()` key `mu_num_pairs_evaluated`,
  an alias of `mu_num_pair_distance_checks`, goes too.

- Developer switches and dumps nobody used: `MCPU_DEBUG_MOVES` (a
  per-step move, energy and RNG trace on stderr), `MCPU_GRID_OCCUPANCY` and
  the HB stencil probe (two one-shot `MCPU_VERBOSE` dumps at grid setup;
  `neighbor_proxy_stats()` reports the same occupancy live), and the CMake
  option `MCPU_EIGEN_HOT_FLAGS` (Release builds already define `NDEBUG`).
  `MCPU_ENERGY_TIMING=0` no longer turns off the per-potential timers in
  `step_stats`: they cost no measurable time (actin, 34.8k vs 34.9k
  cycles/step) and always run.

- The overlap check on the grid before Mu's all-pairs fallback delta
  (`fallback_grid_overlap`), which round 8 also ran under energy masks. It
  served moves that left the grid; with wrapped grids the fallback runs
  only while a grid is off after an overflow (where the check cannot run)
  or for a rigid move whose rounding bound passes the contact list's drift
  budget, which takes coordinates beyond about 2.6e5 A.

- `Context.trial_in_bounds`, `Context.boxBounds`, the `num_trial_fallback`
  and `num_dense_cap_fallback` counters (with `num_trial_fallback` in
  `neighbor_proxy_stats()` and `Context.neighbor_dense_cap_fallbacks()`)
  and `NeighborConfig::max_cells_total`/`max_nx`/`max_ny`/`max_nz`: with
  wrapped grids no move leaves a grid and no grid is refused for its size.

- **The neighbour grid's linked lists and the knobs around them.** Each
  grid kept every cell twice, as a fixed 48-slot block and as a linked list
  that served queries only after a cell overflowed; a full cell now
  switches the grid off instead, so the lists, `MCPU_USE_CONTIGUOUS_CELLS`
  and the per-atom Mu grid walk they fed are gone (a Mu move the contact
  list cannot follow takes the exact all-pairs delta, as under
  `MCPU_CONTACT_LIST=0`). Also gone: the scratch grid of moved atoms that
  walk used; the occupied-stencil modes and `MCPU_OCCUPIED_STENCIL` with
  their timers and the end-of-run `occ_stencil` line under `MCPU_VERBOSE`; and the Mu cell-size knobs `MCPU_MU_CELL_SCALE`,
  `Context.set_mu_cell_size_scale` / `set_mu_cell_size_angstrom` /
  `set_mu_cell_size_min_angstrom`, their getters and
  `effective_mu_cell_size_A`, with the matching `neighbor_proxy_stats()`
  keys. The cell is the Mu cutoff: larger cells were slower in wall time
  and are what the occupancy bound does not cover (10 A cells overflowed).
  `Context.mu_cell_size_A()` still reports it. `neighbor_proxy_stats()`
  drops `mu_grid_contiguous`, and `mu_num_candidates_iterated` and
  `avg_mu_candidates_per_step`, which only the per-atom walk fed.

- **Checkpoint upload: `cloud_sync`, `cloud_bucket` and `cloud_sync_cmd`.**
  After each save it started `<cloud_sync_cmd> last.chk
  <cloud_bucket>/last.chk` in the background and never checked the result.
  A failed upload printed nothing, the run did not wait for the last upload,
  so ending the job could cut it off, and only `last.chk` was copied, never
  the trajectories or logs. Checkpoints are now written only to
  `checkpoint_dir`; copying them elsewhere is left to the user. The
  `--cloud-sync`, `--cloud-bucket` and `--cloud-sync-cmd` flags are gone,
  as are the matching arguments of the runners and `FoldingRunner` /
  `MPIReplicaExchange`, and the `config` argument of `save_checkpoint`. A
  config that still has the keys at their defaults, as copies of the old
  template do, loads with a warning; `cloud_sync: true` is an error.

- `BoxPolicy` with its unreachable `Fixed` and `AutoRecenter` modes,
  `CoordsSoA::recenter` and the always-zero `num_reject_out_of_box` entry of
  `Context.neighbor_proxy_stats()`. Nothing could select those modes;
  `AutoRecenter` would have moved every atom by an inexact float centroid and
  lost the caller's frame.

- **The legacy Mu build (`MCPU_FAST_MU_DELTA=OFF`).** It no longer compiled,
  so its code was dead: the CMake option and its pyproject pin, the
  cached-contact delta (`calculateEnergyChange_legacy`, `ContactData`), the
  per-state N^2 pair bitmap (`State::is_contact_cache`, `StateCacheMode`)
  and the pending flag updates that fed it. `build_info()["features"]` loses
  its `MCPU_FAST_MU_DELTA` key.
  `Integrator.proposal_is_dynamic_only()`, which only repeated
  `use_pooled_proposal()`, and the `proposal_dynamic_only` key of
  `proposal_lifecycle_info()` are gone too. The default build is unchanged.
- The `MCPU_MU_CUTOFF_OVERRIDE` environment variable. It could set the Mu
  neighbour cutoff below the contact list's near-miss band.
- `mcpu_core.build_flags()`, deprecated since `build_info()` replaced it;
  `scripts/install_check.py` now reads `build_info()`.
- `MCPU_MM_GUARD_N2`, with the rigid-move re-check it belonged to (see
  Changed).
- `Context.set_mm_clash_margin` / `mm_clash_margin`,
  `Context.set_mm_double_boundary` / `mm_double_boundary`, the matching
  `MuPotential` properties and the `MCPU_MM_CLASH_MARGIN` and
  `MCPU_MM_DOUBLE_BOUNDARY` environment variables. They were experiments for
  the rigid-move clash problem (see Changed), and only one Mu pair walk,
  since removed, read them. The double-boundary check had become the default
  check without its prefilter, so it changed only speed. The margin rejected
  rigid moves that have no clash, on that one path only, so a masked run's
  trajectory depended on which path a move took. Runs that did not set them
  are bit-identical.
- `MCPUAtom.to_write` and `MCPUAtom.is_sidechain`. They existed to tell
  glycine's second CA slot (see Changed) apart from real atoms. `to_write`
  was then false only for explicit amide hydrogens, exactly when
  `original_index` is -1, and nothing read `is_sidechain`.
  `MCPUForceField.inverse_mapping` is now each atom's `original_index`.
- `mcpu_core.EnergyComponents`. It held the fixed MCPU column set the energy
  reporter used to write; nothing exported or used it.
- **Mu developer diagnostics: `MCPU_DEBUG_MM_DELTA` and
  `MCPU_PIVOT_MU_BREAKDOWN`.** The first printed the moved-moved energy
  change of every rigid move, which is zero by construction. The second ran
  pivots through a third copy of the pair walk that timed its collect,
  distance and pair-energy phases. With it go the keys of
  `Integrator.step_stats()["pivot_mu_breakdown"]` it filled: `cell_walk_ns`,
  `r2_filter_ns`, `eval_pair_ns`, `overhead_ns`, `candidates`, `in_cutoff`,
  `n_pivot_steps`, their `avg_*` forms and `in_cutoff_frac`, plus the
  `avg_walk_*`, `walk_empty_frac` and `avg_atoms_per_nonempty_cell` keys,
  whose counters nothing ever filled.
  Neither ran unless set, so default runs are bit-identical.
- The `MCPU_CLASH_FIRST` and `MCPU_CLASH_FIRST_MIN_MOVED` environment
  variables. The clash-first pass now always runs as it did by default:
  over the atoms that overlapped in recent rejected moves, on the cells its
  radius can reach, for moves of at least
  `Context.clash_first_min_moved()` atoms (set with
  `set_clash_first_min_moved`). The other modes are gone: off (`0`), the
  27-cell stencil with a point-to-box cull (`1`, measured slower) and every
  moved atom (`2`). Mode `1` was the only user of the 27-cell stencil walker,
  which goes too.
- **The Mu Verlet neighbour list.** It was opt-in (skin > 0), ran in no
  default run and was 22-28x slower than the default on actin and an
  8k-atom protein; the live Mu contact list now does the reuse it was meant
  to provide. Gone with it: the `Context` methods `set_mu_skin`, `mu_skin`,
  `set_mu_verlet_enabled`, `mu_verlet_enabled`, `set_verlet_moved_threshold`,
  `verlet_moved_threshold`, `set_verlet_partial_threshold`,
  `verlet_partial_threshold`, `set_invalidate_verlet_on_pivot_accept`,
  `invalidate_verlet_on_pivot_accept`, `maybe_rebuild_verlet` and
  `invalidate_verlet_pivot_accept`; the environment variables
  `MCPU_MU_SKIN` and `MCPU_VERLET_PARTIAL_THRESHOLD`; the
  `neighbor_proxy_stats()` keys `mu_skin`, `mu_verlet_enabled`,
  `invalidate_verlet_on_pivot_accept`, `num_verlet_*`, `num_pivot_accepts*`,
  `verlet_*`, `verlet_use_rate`, `rebuild_rate_per_step`, `low_use_rate` and
  `high_rebuild_rate` (also from its `derived` dict); the
  `Integrator.step_stats()` keys `verlet_used`, `verlet_fallback_cell`,
  `verlet_rebuilds`, `verlet_partial_rebuilds`, `verlet_partial_affected_sum`,
  `verlet_stats`, `verlet_use_rate` and `verlet_rebuild_rate_per_step`; the
  `verlet_mode` and `skin` fields of the `MCPU_DEBUG_MOVES` log; and
  `scripts/parity_verlet_vs_cellonly.py`. Scripts that called
  `set_mu_skin(0.0)` only restated the default and can drop the call. Results
  are bit-identical.
- `Context.set_proxy_print_every()`. The 0.1.0 notes list it as removed, but
  the deprecated no-op binding was still there; it is gone now.
- **The cell-pair Mu walk.** `calculateEnergyChange_fast` carried a second
  pair walk of the dense grid that grouped the moved atoms by cell. The
  contact list replaced it, and it ran 0 times in every audited run, masked
  or not. Gone with it: `Context.set_use_cell_pair` / `use_cell_pair` and
  `Context.set_cell_pair_min_moved` / `cell_pair_min_moved`; the
  `MCPU_USE_CELL_PAIR`, `MCPU_CELL_PAIR_MIN_MOVED`, `MCPU_UNIFORM_SKIPMASK`,
  `MCPU_LIVE_R2`, `MCPU_CLASH_ORDER_BY_DENSITY`, `MCPU_NC_SHARE_DIAG` and
  `MCPU_HOT_COUNTERS` environment variables (the last counted only on that
  walk) and the `MCPU_CP_BREAKDOWN` diagnostic build; the
  `cell_pair_breakdown` and `pivot_mu_breakdown` entries of
  `Integrator.step_stats()` (the latter held only cell-pair counters by
  then: `cell_pairs`, `cell_pairs_empty`, `n_groups`, `group_atoms`,
  `cell_pair_evals` and the averages and fractions built from them); the
  `mu_span_slots_scanned` and `mu_stencil_cells_culled` entries of
  `Context.neighbor_proxy_stats()`; and
  `scripts/parity_cell_pair_vs_per_atom.py`. `MCPU_CONTACT_LIST=0` now sends
  every Mu move to the all-pairs moved-vs-all delta, an exact reference that
  costs O(n_moved x N) per move, for checks only. Each `Context` is about 200
  KB smaller. Default runs are bit-identical.
- **The second, hand-mirrored copy of the Mu pair rules.** `MuPotential`
  could decide which pairs clash or make contacts either from the per-pair
  flag table built in `cache_necessary_data` (the default) or by decoding
  per-atom roles on the fly, the path `MCPU_TOPO_FLAGS=0` selected. The two
  copies had drifted apart before, so the variable changed trajectories, not
  just the code path. The flag table is now the only one, and a Mu rule
  change is a Python change in `MuPotentialBuilder.build_topology_masks`.
  Gone with it: the `MCPU_TOPO_FLAGS` environment variable; the
  `MuPotential` methods and properties `set_topology_atom_meta`,
  `use_topo_flags`, `verify_layered_eval_consistency`,
  `bench_eval_pair_only` and `clash_exception_count`;
  `MuPotentialBuilder.layer1_atom_meta`; and
  `scripts/parity_layered_eval.py`. Results are bit-identical.

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
