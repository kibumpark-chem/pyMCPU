# Contributing to pyMCPU

Thanks for your interest. pyMCPU is a Monte Carlo protein-folding engine: a
C++20 core behind a Python API. Most contributions touch one of three layers,
and the layer decides what is expected of you.

## Quick start

```bash
git clone https://github.com/kibumpark-chem/pyMCPU && cd pyMCPU
python -m pip install -e ".[dev]"      # builds the C++ extension
python -m pytest -q                    # the default suite
ruff check .
```

The build needs a C++20 compiler (GCC 9+ / Clang 10+) and CMake. Eigen is
fetched automatically if it is not installed. `pip install -e .` rebuilds the
extension when C++ sources change.

## The one rule that matters most

**This engine makes discrete decisions from `float32` arithmetic.** A hard-core
clash test, a contact-membership test and a Metropolis acceptance are all
threshold comparisons on computed floats. A change that looks numerically
harmless can flip one of them and decorrelate a trajectory.

So: **if your change could alter any number, prove that it does not**, or say
plainly that it does and why that is correct.

```bash
# record a fingerprint before your change
python scripts/arch_parity_dump.py --out /tmp/before.json
# ... make your change, rebuild ...
python scripts/arch_parity_dump.py --compare /tmp/before.json   # exit 0 required
```

`arch_parity_dump.py` compares accept-bit streams, per-group energies as hex
floats, coordinate hashes, weights and move counters between two builds, with
no tolerance. Exit 0 = bit-identical, 1 = divergence, 2 = the harness itself
failed.

If your change *should* alter results, say so in the PR, re-derive the affected
goldens as an explicit and dated capture (see the capture history at the top of
`tests/physics/test_coords_soa.py` — it models this well), and note it in
`CHANGELOG.md`.

## Tests

The default suite excludes slow, network, MPI and example tiers. Before
opening a PR, run the slow tier too — it contains the physics goldens:

```bash
python -m pytest -q                                              # fast
python -m pytest -o addopts="" -m "not network and not mpi and not mpi_integration" -q
```

`MCPU_VERIFY_PHYSICS=1` makes the integrator check, on *every* proposal, that
each potential's incremental ΔE equals a from-scratch `E(new) − E(old)`, and
throws on mismatch. Use it when touching any energy term.

## Adding or changing an energy term

Energy terms are C++ (`include/pymcpu/forces/`, `src/pymcpu/forces/`). There is
no Python-authored term: `Potential` has no pybind11 trampoline, deliberately,
because the hot loop cannot afford a Python call per pair.

**Where the file goes is a rule, not a judgement call** — see
[`include/pymcpu/forces/README.md`](include/pymcpu/forces/README.md). In short:
a term shared by every fit of a force-field family goes in
`forces/<family>/common/` (a family is a LINEAGE, e.g. `mcpu/`, not a method),
a term specific to one fit goes in
`forces/<family>/<fit>/` and opens `namespace mcpu::forces::<fit>`, and a
restraint with no fitted tables goes in `forces/bias/`.
`tests/config/test_forces_layout.py` enforces both halves.

A new term must:

1. Subclass `mcpu::Potential` and implement `calculateEnergy` and
   `calculateEnergyChange`. `AromaticPotential` (~165 lines) is the clearest
   template; its delta is simply `E(new) − E(old)`, which is correct by
   construction and fine when an O(N) recompute is affordable.
2. **Agree with itself.** `PhysicsVerifier` asserts the incremental ΔE matches a
   full recompute to 1e-3. This is enforced at runtime, not just in tests.
3. Override `permute_atom_indices` **if it caches atom indices**. The base is a
   silent no-op, so forgetting this produces a term that reads the wrong atoms
   after a locality reorder without any error.
4. Stay below the clash sentinel. `MuPotential` returns `99999.0f` for a
   hard-core overlap and anything `>= 49999.5` is read as a clash, so a large
   but finite energy must stay under that.
5. Claim an energy group in `include/pymcpu/EnergyWeights.h`. Groups 1–5 are
   taken; **group 4 is silently multiplied by 2.0** (the legacy `RDTHREE_CON`),
   so do not reuse it. Groups 8–15 work and are weighted correctly but are
   **silently untimed** — `Context::energy_delta_ns_` is `[8]`.
6. Add a pybind11 class in `src/bindings/bindings.cpp` (unavoidable today) and
   a builder under `pymcpu/forcefields/builders/`.

CMake needs no edit — the source glob uses `CONFIGURE_DEPENDS`.

## Build knobs worth knowing

| variable | meaning |
|---|---|
| `MCPU_ARCH` | CPU baseline: `v2`, `v3` (default), `v4`, `native`, `none`, or a raw `-march` value |
| `MCPU_FP_CONTRACT` | `off` makes results reproducible across compiler versions, at ~3% throughput |
| `MCPU_PARAMS_DIR` | use a parameter directory directly |
| `MCPU_VERIFY_PHYSICS` | per-proposal delta-vs-full checking |

`mcpu_core.build_info()` reports what a build **actually did** — resolved
`-march`, compiler, LTO state, FP flags — read from the branch the build took
rather than from what was requested. If you add a build option, report it the
same way; a report derived from intent will agree with a bug instead of
catching it.

## Style

`ruff check .` must pass. Docs build with `-W` (warnings are errors):

```bash
python -m sphinx -b html -W docs _build/html
```

Documentation is checked for more than formatting: `tests/docs/` asserts that
every API name in a docs code block actually resolves against the package, and
that every autodoc target exists. This exists because the quickstart once
documented an entirely fictional API that passed `sphinx -W` because Sphinx does
not validate code blocks.

## Pull requests

- One logical change per PR.
- Say what you verified, not just what you changed. "Suite green" is less useful
  than "parity oracle exit 0, slow tier green, bench unchanged within noise".
- If you found something surprising, write it down in the PR description —
  measurements and refuted hypotheses are as useful as the change itself.

## Reporting bugs

Please include `mcpu_core.build_info()` output, your platform, and a minimal
reproducer. For anything involving numbers that differ between machines, the
compiler version matters more than the CPU — include both.

Security issues: see [SECURITY.md](SECURITY.md).
