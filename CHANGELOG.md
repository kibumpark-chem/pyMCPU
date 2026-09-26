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

### Fixed

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
  `export MCPU_PARAMS_DIR="$(mcpu materialize-params --set mcpu_v1)"`.
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
