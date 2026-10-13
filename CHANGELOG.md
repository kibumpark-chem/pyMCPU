# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Checkpoints record the parameter set and refuse a resume under another
  one.** A run started with one parameter folder used to resume silently with
  any other, for example under another set's tables given through
  `MCPU_PARAMS_DIR`. Checkpoints now record `param_set`, `param_signature`
  and `param_dir`. The signature is a sha256 over the files the force field
  reads: those the set's registry layout names, plus
  `constants/paramset.json` when the folder has one. A resume under another
  set, or with files whose signature differs, stops with an error that says
  which differs, gives both values, and names the `param_set:` or
  `param_dir:` that resumes it. The folder is recorded as its real path, and
  one folder reached through a link counts as itself, so the error does not
  send a resume back to the folder whose files changed. The same files in
  another folder resume. A
  checkpoint written before these fields counts as mcpu08 and resumes under it
  with a one-line warning, as its files cannot be checked. Folding, replica
  exchange and MPI replica exchange all check it. A checkpoint of parameter
  files other than the mcpu08 files 0.1.0 shipped is written as format
  version 3, which pyMCPU 0.1.0 refuses as newer than it understands instead
  of resuming it under its own mcpu08; other checkpoints stay at version 2,
  and this version reads both. The three new
  `CheckpointState` fields are keyword-only, so the other fields keep their
  positional order.
- `MCPUForceField.param_signature` and `pymcpu.params.param_signature`.
  Taking a signature reads every parameter file in full, which a run
  otherwise does not (the tables are mapped, and only the parts a structure
  needs are read): 0.7 GB for mcpu08, about 1.7 s when the files are in the
  page cache, 2.5 to 4 s from network scratch and up to 45 s from a home
  directory when they are not. So it is taken only when needed: by the
  runners when they first save or resume a checkpoint (MPI replica exchange
  on rank 0, which sends it to the other ranks), and by `EngineSession` the
  first time its fingerprint is read. A run without checkpoints, and a force
  field alone, never take it. A file's sha256 and md5 come from one read,
  and are kept under the cache root (`MCPU_CACHE_DIR`, default
  `~/.cache/pymcpu`) in `params/digests.json`, by the file's real path,
  inode, size, and modification and change times: each parameter file is
  read once per cache root while it is unchanged, not once per process. A
  force field refuses to sign parameter files that have changed since it
  read them, as the signature would not describe its tables.
- **A parameter set can carry its own defaults in
  `constants/paramset.json`.** The file (format `pymcpu-paramset/1`) names
  the set and can give its hydrogen-bond sheet factor (`hbond_beta_favor`),
  the factors already multiplied into its tables (`folded_weights`, 0 or
  above, a record only: the engine's group weights do not change) and its
  provenance,
  including the md5 of any of its files, which each file must still have when
  a force field is built. `MCPUForceField(verify_provenance=False)` builds
  without comparing them, and `MCPUForceField.check_provenance` compares
  them later; MPI replica exchange builds so on every rank and compares on
  rank 0 alone, which sends its error to the others. Unknown or repeated keys and values of the wrong
  kind are refused with an error that names the file.
  `pymcpu.params.load_paramset_manifest` reads and checks it, and
  `MCPUForceField.paramset_manifest` holds what the force field read. mcpu08
  has none. The file ships with its set: `scripts/encode_params.py` copies it
  next to the table archive (and `--check` compares it), and a wheel install
  unpacks it with the tables, into a cache folder whose name covers it. So a
  set has the same defaults and signature from a wheel as from a checkout or
  `MCPU_PARAMS_DIR`. `scripts/encode_params.py` also removes a constants file
  from `--out` when `--src` no longer has it, and `--check` reports one that
  only one side has.

### Changed

- `MCPUForceField(hbond_beta_favor=None)` is the default: a force field
  given no factor takes its parameter set's, and 2.0 when the set gives none,
  so mcpu08 runs as before. A factor given as an argument or in
  `forcefield_options` always wins, and checkpoints record the factor the run
  used. `hbond_beta_favor: null` in a config now asks for this default
  instead of being refused. The README's physics defaults and the
  hydrogen-bond physics notes give the default this way.
- `EngineSession.fingerprint` covers the parameter set the force field was
  built from, the signature of its files and the hydrogen-bond sheet factor
  it resolved, so it differs from 0.1.0's for the same engine, and restart
  states from sessions that differ only in the factor no longer match.
  `compute_fingerprint` takes `param_signature` and `hbond_beta_favor`;
  without them the hash is the old one.
- Runs print the parameter set and its folder with the rest of their
  description, and log the start of its signature when they take it.

### Fixed

- **A residue template must name the atoms of every chi angle the engine
  counts.** `MCPUForceField` refuses a parameter folder whose
  `constants/standard_amino_acids.json` gives a residue, for each of its
  `ntorsions` chi angles (a whole number from 0 to 4), anything but one
  `chi_atoms` row of four atom names and one `chi_moved_atoms` row of atom
  names, with an error that names the file and each such residue. Such a
  template used to load: the engine counts chi angles by `ntorsions`, the
  side-chain moves turn only the atoms in the `chi_moved_atoms` rows, and the
  side-chain torsion energy reads each chi from its `chi_atoms` row, so a chi
  angle without its rows never changed and was scored at a placeholder
  angle, with no other sign. Every residue in the template is checked, not
  only those in the structure. mcpu08's template passes, and loads as before.

## [0.1.0] - 2026-10-11

First public release. pyMCPU reimplements MCPU, the Shakhnovich lab's Monte
Carlo protein folding program, as a C++ engine with a Python API. It ships
one force field, `mcpu08`: legacy MCPU's energy function and parameter
tables, with a smaller hydrogen-bond sheet factor by default (see Sheet
factor below). The KORP force field is not included, and the `mcpu26` force
field is planned for 0.2.0. pyMCPU needs Linux on x86-64 and Python 3.12 or
newer.
As this is the first release, this entry lists how pyMCPU differs from
legacy MCPU, the C program on the lab's
[MCPU page](https://shakhnovich.faculty.chemistry.harvard.edu/software/monte-carlo-protein-unfolding-mcpu),
in its version with umbrella sampling, run with its reference configuration
(`cfg_1igd`).

### Same as legacy MCPU

- **Energy.** The same weighted sum of five terms, with the same tables:
  contacts (Mu, `p178_conrange4_potential_0.995054`) × 0.4, backbone torsions
  × 1.35, sidechain torsions × 2.5, backbone hydrogen bonds × 2.7
  (`HBOND_WEIGHT` 1.35 × `RDTHREE_CON` 2.0, sequence-dependent as with
  `SEQ_DEP_HB 1`; only the factor on bonds more than four residues apart
  differs, see Changed), and PHE/TRP ring stacking within 7 Å × 5.0
  (`aromatic_noTYR.energy`).
- **Atoms and contacts.** The same atom types and radii. Two atoms are in
  contact within 1.35 (r_i + r_j) and clash under 0.75 (r_i + r_j)
  (`LAMBDA` 1.8, `ALPHA` 0.75). Residues fewer than five apart make no
  contacts, and neither do two backbone atoms.
- **Temperature and acceptance.** T is the same reduced temperature. Moves
  are accepted by the Metropolis rule, and a move that creates a clash is
  rejected (`NO_NEW_CLASHES 1`).
- **Moves.** The same three kinds:
  - pivot: a Gaussian change of one backbone φ or ψ, including ψ of the
    first residue and φ of the last, never a proline's φ; the shorter end of
    the chain turns;
  - KIC: tripeptide loop closure on a three-residue window; a window whose
    move would change a proline's φ is refused;
  - sidechain: a residue with χ angles, other than proline, gets either a
    rotamer from the `bbind02.May.lib` library (a Gaussian around the
    rotamer's mean χ; see Symmetric χ under Changed) or a Gaussian change of
    each χ.
- **Native contacts.** N counts residue pairs at least `min_seq_sep` apart
  whose CA–CA distance is below the cutoff both in the reference structure
  and in the current one.

### Changed from legacy MCPU

- **Sheet factor.** A hydrogen bond between residues more than four apart is
  multiplied by 2 by default; legacy compiled `beta_favor` 3 into its binary
  (`define.h`). Bonds four apart are not scaled in either. On the native
  structures this raises the hydrogen-bond energy, and so the total, by 0.39
  on CLN025, 28.1 on 1igd and 23.7 on actin (1.3 %, 8.2 % and 4.3 % of the
  total). β-hairpins are less stable: with the 0.1.0 defaults, chignolin
  (1UAO) melts at about 0.6 (midpoints 0.57 to 0.60 from 20,000-cycle
  replica exchange), lower than with legacy's factor.
  The factor is a force-field option: `forcefield_options:
  {hbond_beta_favor: 3.0}` in a config, or
  `MCPUForceField(hbond_beta_favor=3.0)`, restores legacy's. Checkpoints
  record it, and a run resumed with another factor stops with an error.
- **One step is one move.** A legacy MC step was one backbone move plus
  `SIDECHAIN_MOVES` sidechain moves, two moves in the reference configs, so
  double `MC_STEPS`, `MC_REPLICA_STEPS` and print intervals when porting. In
  a replica exchange config, `steps` (or `mc_replica_steps`) is the number of
  steps each replica makes between exchange rounds, like
  `MC_REPLICA_STEPS`; in a folding config, `steps` is the total, like
  `MC_STEPS`.
- **Move mix.** By default 25 % of moves are pivots, 25 % KIC and 50 %
  sidechain moves. Legacy also made half its moves sidechain moves (with
  `SIDECHAIN_MOVES 1`), but split its backbone moves 33 % `LoopBackboneMove`
  (φ and ψ of one residue, 10° Gaussians; `CLUSTER_MOVE`, 0.33 unless set),
  33.5 % KIC and 33.5 % pivot (`YANG_MOVE 0.5`). pyMCPU has no
  `LoopBackboneMove`. A pyMCPU pivot draws its torsion uniformly from the
  chain's 2N − 2 backbone torsions; legacy drew a residue, then φ or ψ, so it
  proposed each end torsion and each proline ψ twice as often.
- **Step sizes.** `step_size_rad` sets the pivot step, default 0.1 rad
  (5.7°). Legacy pivots were always 2°, because legacy overwrites
  `MC_STEP_SIZE` with 2° after reading the config. `kic_step_size_rad` sets
  the width of the KIC driver, default π/6 (30°), and can be tuned; legacy
  drew `YANG_SCALE` × 2°, 60° in the reference configs. In a study of
  driver widths (run at legacy's sheet factor, 3), 20° to 120° sampled the
  same equilibrium about equally fast, while 10° hardly moved the ψ of
  CLN025's proline and 5.7° (0.1 rad) or less never moved it out of its
  basin, so keep the width at 20° or more.
- **Sidechain moves.** The rotamer move, the default, picks a library row by
  its probability, as legacy did with `USE_ROT_PROB 1` (the reference
  configs used 0, a uniform pick), and is corrected for its proposal (see
  Fixed). The width of the continuous move, `sidechain_step_size_rad`
  (Python API only), defaults to `step_size_rad` (legacy: `SIDECHAIN_NOISE`,
  10° in the reference configs).
- **Symmetric χ.** The rotamer move also proposes the other 180° image of the
  two-fold symmetric χ (ASP χ2, GLU χ3, PHE χ2, TYR χ2), and draws each χ from
  a normal truncated to one period (`rotamer_library: dunbrack2002_sym`, the
  default). The library's rows cover one half of those χ. Legacy drew only
  from the rows, so it hardly ever proposed the other half, and with the
  proposal correction (see Fixed) the same draw can leave a residue that
  starts in that half frozen: in 10⁹-step runs with the default moves, at
  legacy's sheet factor 3, it froze 1.5 to 3 residues of CLN025 and 1igd for
  the whole run (TYR38, TYR50 and GLU61 of 1igd, TYR1 and TYR10 of CLN025),
  and the default froze none. With the 0.1.0 defaults chignolin (1UAO) melts
  at 0.57 to 0.60 (midpoints from 20,000-cycle replica exchange), against
  0.58 to 0.62 with legacy's draw at the same sheet factor. The default is
  also the cheaper move: 4 % more steps per second on 1igd and 5 % on actin
  with the default move mix. `rotamer_library: dunbrack2002` gives legacy's
  draw, still corrected, for comparisons with legacy MCPU.
- **Umbrella bias.** pyMCPU adds ½ k_bias (N − N0)² to the energy, so the
  bias is divided by T in the Metropolis and exchange tests. Legacy added
  k_bias (N − s)² to the exponent, with no ½ (its README says ½) and no 1/T.
  The same umbrella therefore needs k_bias = 2T × `K_BIAS`: legacy's 0.02 is
  0.024 at T = 0.6 and 0.04 at T = 1.0. In configs, `k_bias` defaults to 1.0
  when umbrella targets are set, about 40 times the strength of legacy's
  0.02 at T = 0.6, and to 0 (plain temperature replica exchange) when none
  are.
- **Native contacts.** By default N uses CA atoms, a 6.0 Å cutoff and
  `min_seq_sep` 4, the same in every config, runner and `NativeContactsCV`.
  Legacy's reference configs used 6 Å and `MIN_SEQ_SEP` 8.
- **Replica exchange.** After every `mc_replica_steps` steps, pyMCPU tries
  every neighbouring pair along T, then along the umbrella windows, in a
  fixed order. Legacy's rank 0 made `MAX_EXCHANGE` random attempts, each
  pairing a random replica with the next temperature or the next setpoint.
  Temperatures and targets (`n_targets` or `q_targets`) are free lists, and
  replicas can share an MPI rank or run in one process; legacy ran one
  replica per rank on an evenly spaced grid.
- **Precision.** Legacy tested contacts and clashes on coordinates truncated
  to 0.01 Å and summed energies in single precision. pyMCPU compares float32
  squared distances and sums energies in double precision: the running total
  stays within about 1e-10 of a full recompute over 1e7 steps. A move's
  clash cutoff is the hard-core distance rounded to 0.001 Å, less 0.0015 Å,
  so the rounding of PDB coordinates cannot decide a clash.
- **Frame.** Legacy moved the protein's centre to the origin at the start,
  every 10,000 steps and at every exchange round, and wrote its PDBs in that
  frame. pyMCPU writes coordinates in the input frame; it shifts only its
  internal frame, and only once a coordinate reaches 64 Å.
- **Clashes in the start structure.** Legacy cleared the clash flag of pairs
  that overlapped at the start (and again every 10,000 steps and every
  exchange round), but still rejected any move that moved one atom of such a
  pair and left it overlapping. pyMCPU exempts those pairs for the whole
  run. Any other overlap found by a full recompute, a checkpoint restore or
  a replica swap stops the run with `StericClashError`
  (`MCPU_CLASH_FATAL=0` makes it a warning).
- **Random numbers.** pyMCPU uses `std::mt19937`. Configs and runners seed
  it with `seed` (default 42), plus a fixed offset per replica, and the
  exchange decisions use NumPy generators seeded from `seed`, so the same
  seed and build give the same trajectory. SHA-256 is used only by
  `pymcpu.sampling.derive_seed`, which gives independent seeds to drivers
  that clone walkers. An `Integrator` made directly in Python is seeded from
  `std::random_device` unless `set_seed` is called. Legacy seeded Random123
  Threefry from the clock, the process ID and the rank, so a run could not
  be repeated.
- **Secondary structure.** Every residue is coil by default, like a legacy
  `.sec_str` of all `C`; DSSP secondary structure is optional
  (`compute_dssp=True`). `.sec_str` files are not read.

### Fixed relative to legacy MCPU

- **Rotamer moves are corrected for their proposal.** Their acceptance
  includes q(old)/q(new), the ratio of the densities with which the library
  proposes the old and the new χ. Legacy accepted them on exp(−ΔE/T) alone,
  so it sampled the Boltzmann distribution times that density, which is
  close to 0 on the half of each symmetric χ that the library does not cover
  (see Symmetric χ above).
- **KIC moves are reversible and do not depend on orientation.** The
  Jacobian is computed from the six torsion axes of the window, so it does
  not change when the molecule turns. Legacy's Jacobian (`jac_local.h`)
  used lab-frame components of a bond that a φ-driver move turns, so the
  weight of such a move depended on how the molecule was oriented. A move is
  refused when the current window is not one of the closures found before
  the move, so every move has its reverse; legacy counted a window with no
  pre-move closure as having one. The window keeps the start structure's
  peptide ω; legacy set ω to 180° in every window it moved.
- **The last sidechain-torsion window is scored.** Legacy's preparation tool
  (`mcpu_prep/save_triple.c`) wrote each `.sctorsion` table one window
  short, so legacy scored that window at its "never observed" value, +2.5
  after weighting. On chignolin, 1igd, barnase and actin this made legacy's
  sidechain-torsion energy 2.8 to 4.8 higher than pyMCPU's.

### New in pyMCPU

- **Python API and command line.** `System`, `Context`, `Integrator`,
  `Simulation` and reporters; YAML and JSON configs; and an `mcpu` command
  (`run`, `validate`, `materialize-params`). The parameter tables ship in
  the package.
- **No preparation step.** There are no `.triple`, `.sctorsion` or
  `.sec_str` files to make. pyMCPU reads a PDB, removes hydrogens and
  renames protonation variants such as HSD. It stops with an error for a
  residue it has no parameters for, a missing heavy atom, several chains or
  a chain break (a jump in residue numbers, or more than 2 Å from one
  residue's C to the next one's N; `allow_chain_breaks=True` joins the
  pieces). Legacy logged `MISSING RESIDUES!!!` and ran on.
- **Output.** XTC trajectories; an energy CSV with the total (umbrella bias
  included), one column per term, weighted as in the total (legacy's `STEP`
  lines printed the terms unweighted), and acceptance counts per move kind;
  for replica exchange, a data CSV per temperature and window with walker
  IDs, `rex_stats.json`, and optional exchange and state logs and HDF5 or
  NPZ samples for MBAR.
- **Checkpoints**, on by default. A run resumed from a regular checkpoint
  writes the same XTC, CSV and `rex_stats.json` files, byte for byte, as an
  uninterrupted run with the same build and checkpoint interval. Legacy
  could restart only from its PDB snapshots.
- **Native contacts** can use CB atoms, an explicit list of pairs, and a
  separate cutoff for a formed contact (`q_cutoff`).
- **Fixed and linker residues.** Residues can be held fixed by index, and
  linker residues can move with their energies off or with clashes only.
- **Ramachandran pivot.** An optional pivot that draws φ and ψ together from
  a knowledge-based distribution, corrected for its proposal; off by
  default.
- **No compile-time size limits.** Legacy was compiled for at most 8,000
  atoms and 1,000 residues (`define.h`), and 500 residues in `loop.h`.

### Not ported

- `LoopBackboneMove`, including its knowledge-based moves (`USE_CLUSTER`).
  These break detailed balance, as legacy warned, and legacy made them with
  probability 0.5 on every residue marked `H` in `.sec_str`, whatever
  `USE_CLUSTER` said.
- Refitting Mu to the input structure (`READ_POTENTIAL 0`; legacy then
  derived μ = 0.51 for chignolin and 0.997 for actin, where the shipped
  table has 0.995), the Go-model potential (`USE_GO_POTENTIAL`) and soft
  clashes (`NO_NEW_CLASHES 0`).
- Disulfide restraints (`CONSTRAINT_FILE`), the RMSD cap
  (`CONSTRAINT_RMSD`), and template or alignment input (`TEMPLATE_FILE`,
  `ALIGNMENT_FILE`).
- A start structure per replica (`NATIVE_DIRECTORY`); every replica starts
  from one PDB.
- PDB snapshots, `_Emin.pdb`, the lowest-RMSD structure, and the
  simulated-annealing script.

### Known limitations

- **Linux x86-64 only.** The wheels are built for x86-64-v3 CPUs (AVX2, FMA
  and BMI2: Intel Haswell, AMD Excavator or newer). On an older CPU,
  `import pymcpu` raises `ImportError`, naming the missing instructions and
  how to build pyMCPU from source for that CPU (`MCPU_ARCH=v2` or
  `MCPU_ARCH=native`).
- **A trajectory repeats only with the same build**, because compilers fuse
  floating-point operations differently. A resumed run continues the exact
  random-number stream only with the same build or a compatible `libstdc++`.

### Performance and validation

- **Energies.** On the native structures of chignolin, 1igd, barnase and
  actin, pyMCPU's total energies are within 0.24 (under 0.1 %) of legacy's
  (legacy run with `READ_POTENTIAL 1` and corrected `.sctorsion` files;
  pyMCPU with legacy's sheet factor, `hbond_beta_favor` 3).
  Four terms agree within 0.02 (weighted). The contact term differs by up to
  0.23, for reasons not yet pinned down; the contact table itself equals
  legacy's to float32 rounding (largest difference 3e-8). A test checks
  actin's hydrogen-bond energy at sheet factor 3, −172.961, against legacy's
  logged −128.12 × 1.35.
- **KIC** samples its exact distribution in a test that traces every
  closure of a window (ψ-driver windows on chignolin and actin, driver widths
  0.1 and 1.0 rad).
- **Speed.** The table gives the wall time per attempted Monte Carlo move,
  start-up excluded, for legacy MCPU and pyMCPU 0.1.0 on 13 proteins started
  from their native structures, with matched move settings: pivot 2°, KIC
  driver 30° (legacy `YANG_SCALE` 15), continuous 10° sidechain moves, and
  25 % pivot, 25 % KIC and 50 % sidechain moves (legacy: one backbone and
  one sidechain move per step, half the backbone moves KIC, cluster moves
  off), at T = 0.6 with the shipped contact table (`READ_POTENTIAL 1`), all
  residues coil, and legacy's sheet factor, 3, in both codes. One legacy MC
  step counts as two attempts. Both codes
  ran on one core each of the same exclusive node (Intel Xeon Platinum
  8480CL), in 5 paired, interleaved rounds. Costs are medians over the
  rounds; the speedup is the median of the per-round ratios, with its range
  over the rounds. Legacy was built on the node at `-O3 -march=native` with
  GCC 13 and GCC 15, and the faster build (GCC 15) was timed. pyMCPU was
  built as its wheels are (GCC 15, x86-64-v3). Not matched, each in legacy's
  favour: legacy is compiled for the node's own CPU (AVX-512), and on small
  proteins it spends more of its attempts on cheap moves (end torsions, and
  KIC windows that run off the chain). The two KIC closures differ, so the
  table compares cost per attempt, not sampling per attempt. For the 8
  proteins of at least 400 atoms, the cost per attempt grows with the number
  of atoms N as N^(1.04 ± 0.04) for legacy and as N^(0.16 ± 0.04) for pyMCPU
  (least-squares fit of log cost on log N; ± one standard error).

| Protein | Residues | Atoms | Legacy (µs per attempt) | pyMCPU (µs per attempt) | Speedup |
|---|---:|---:|---:|---:|---:|
| chignolin | 10 | 77 | 9.81 | 5.03 | 1.95× (1.84–1.99) |
| trpcage | 20 | 144 | 12.5 | 5.51 | 2.26× (2.25–2.28) |
| villin | 35 | 287 | 22.6 | 7.35 | 3.08× (3.03–3.09) |
| ntl9 | 39 | 300 | 24.9 | 8.71 | 2.87× (2.83–2.88) |
| proteinb | 47 | 362 | 28.1 | 8.15 | 3.44× (3.41–3.54) |
| nug2 | 56 | 435 | 28.4 | 10.4 | 2.74× (2.73–2.86) |
| 1igd | 61 | 468 | 32.0 | 9.06 | 3.52× (3.41–3.58) |
| a3d | 73 | 572 | 33.4 | 8.76 | 3.81× (3.75–3.88) |
| lambda | 80 | 618 | 38.1 | 8.94 | 4.26× (4.12–4.27) |
| barnase | 108 | 864 | 53.4 | 10.2 | 5.20× (5.13–5.25) |
| chey | 128 | 979 | 63.0 | 10.1 | 6.25× (6.12–6.29) |
| sce | 272 | 2,168 | 129.3 | 11.8 | 10.9× (10.8–11.1) |
| actin | 377 | 2,943 | 219.8 | 12.4 | 17.9× (17.5–18.6) |

[0.1.0]: https://github.com/kibumpark-chem/pyMCPU/releases/tag/v0.1.0
