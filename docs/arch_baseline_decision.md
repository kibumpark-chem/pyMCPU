# CPU baseline decision: x86-64-v3 (2026-09-16)

**pyMCPU 0.1.0 ships `-march=x86-64-v3` (AVX2 + FMA + BMI2). The switch from
`skylake-avx512` was verified bit-identical, so no energies, trajectories,
checkpoints or goldens were rebaselined.**

## Why the baseline changed

The pre-release build pinned `-DMCPU_ARCH=skylake-avx512`, and the resulting
extension contained hundreds of `zmm` register operands. That SIGILLs on:

* every AMD Zen 1-3 part,
* Intel consumer CPUs from the 12th generation onward (P-cores ship AVX-512
  fused off),
* every Xeon older than Skylake-SP.

A wheel built that way cannot be published: `pip install pymcpu` would abort
with `Illegal instruction` on a large fraction of machines, with no diagnostic.

## The escape hatch had never worked

`MCPU_ARCH` was documented as an environment variable in four places --
`README.md`, `docs/installation.rst`, `docs/known_issues.md` and
`conda-recipe/meta.yaml` -- but it was declared as a CMake `CACHE STRING`, and
**CMake cache variables are not populated from the environment**. There was no
`$ENV{...}` anywhere in `CMakeLists.txt`.

So every documented `MCPU_ARCH=... pip install` command silently built AVX-512
anyway. The conda recipe was the sharpest case: it set `MCPU_ARCH=x86-64-v3`
specifically to avoid SIGILLs on AMD hardware, and that line did nothing.

Compounding it, the pin lived in `cmake.args`, which is a **list** --
`-Ccmake.args=...` on the command line replaces it wholesale. So the one
override that did reach CMake also silently dropped the baseline.

Both halves are fixed: `MCPU_ARCH` is now an env-backed entry in
`cmake.define` (a *mapping*, which merges), and `CMakeLists.txt` reads
`$ENV{MCPU_ARCH}` as the cache initializer.

## Tiers, not raw `-march` strings

`MCPU_ARCH` takes a tier name and CMake resolves a spelling the compiler
accepts, feature-tested with `check_cxx_compiler_flag`:

| value | ISA | candidate spellings |
|---|---|---|
| `v2` | SSE4.2 + POPCNT | `x86-64-v2`, `nehalem` |
| **`v3`** (default) | AVX2 + FMA + BMI2 | `x86-64-v3`, `haswell` |
| `v4` | adds AVX-512F/BW/DQ/VL/CD | `x86-64-v4`, `skylake-avx512` |
| `native` | this machine | `native` |
| `none` | none | *(no flag; for packagers managing `CXXFLAGS`)* |
| anything else | as given | used verbatim, **no fallback** |

The indirection is not decoration. GCC 8.5 -- the oldest version that builds
this tree -- **rejects `-march=x86-64-v3`** and needs `-march=haswell` for the
same ISA level. Taking the published string literally would have turned a
silently-ignored flag into a hard configure error on that compiler.

Substitution happens only *within* a tier and is logged. Across tiers, and for
a raw string the compiler rejects, the configure fails with `FATAL_ERROR` --
a silent downgrade would reproduce the original bug in different clothing.

Note `haswell` is a strict **superset** of `x86-64-v3`: it adds PCLMUL, RDRND,
FSGSBASE and XSAVEOPT, and tunes differently (`__tune_haswell__` vs `__k8__`).
Equivalent in guaranteed ISA level, not identical in codegen -- which is why
the verification below used the exact `x86-64-v3` spelling rather than the
fallback.

## Verification

`scripts/arch_parity_dump.py` records a bit-exact fingerprint to disk, so two
separately-compiled binaries can be compared. The existing
`scripts/parity_*.py` oracles cannot do this: each imports `pymcpu` once and
runs twice in the same interpreter, toggling a runtime setter.

Both arms GCC 12.2.0, Release with LTO, differing only in `MCPU_ARCH`:

| arm | zmm | kmask | ymm | FMA | `.so` sha256 |
|---|---|---|---|---|---|
| `v4` (`skylake-avx512`) | 2001 | 168 | 1922 | 1262 | `946d1279493da931` |
| `v3` (`x86-64-v3`) | **0** | **0** | 2692 | 1291 | `40a653a67ed1bac9` |

Result: **all four cases bit-identical** -- chignolin (80 atoms) and actin
(2971 atoms; both counts include the second glycine-CA slot of the time, so
77 and 2943 today), seeds 42 and 1337, comparing accept-bit streams, per-group
energies as hex floats, coordinate hashes, move counters and neighbour-list
counters with `==` and no tolerance.

Corroborating, on the `v3` build:

* full suite **577 passed, 0 failed**;
* four of five `parity_*.py` oracles clean, the fifth failing only its
  pre-existing coverage assertion (`elided_rigid_mm == 0`, unchanged) with
  byte-identical output to the AVX-512 run, including `E=-596.214966`;
* per-group energies unchanged, actin
  `mu = -127.932579` matching the earlier GCC 8.5 / AVX-512 run. (That is
  the two-slot glycine layout of the time. With one slot it reads
  `-127.932594`: the same pair terms, summed in a different order.)

That last point is stronger than the A/B alone: those energies are stable
across a **compiler** change *and* a baseline change.

## Why bit-identity was expected, and held

FMA is present in both baselines, GCC does not reassociate floating point
without `-ffast-math` (`build_info()` reports `fast_math: false`), and the hot
loops are element-wise rather than reductions. Narrower vector registers
therefore change how many lanes are processed per iteration, not the arithmetic
performed on each. Removing contraction is not free either: measured with
`-ffp-contract=off`, one run in four decorrelates and the cost is +3.2-3.3%.

The same reasoning explains why the *compiler* dimension does not enjoy this
protection. Changing GCC version changes which expressions get contracted into
an FMA in the first place (895 FMA instructions under GCC 8.5 versus 1281 under
GCC 12.2, at the same `-march`), so the arithmetic itself differs rather than
only its vector width.

## What this does not prove

* **That the shipped wheel is unaffected.** CI and conda-forge build with
  whatever compiler they supply, and the compiler matters **more** than the
  baseline flag -- this has since been measured rather than suspected. At the
  same `-march` and on identical source, GCC 8.5 and GCC 12.2 decorrelate
  `chignolin-s1337` at MC step 1253 (495 vs 512 accepts). The `-march` tier, by
  contrast, does not move the result at all:

  | | `haswell` / `x86-64-v3` | `skylake-avx512` |
  |---|---|---|
  | **GCC 8.5.0** | 495 accepts | 495 accepts |
  | **GCC 12.2.0** | 512 accepts | 512 accepts |

  So the bit-identity established above is real but narrower than it looks: it
  holds across the dimension this project controls, not across the one it does
  not. `-ffp-contract=off` closes the compiler dimension -- under it both
  compilers agree exactly (434 accepts, all four cases bit-identical) -- but
  0.1.0 ships the compiler default, so this limitation is **live** and
  documented in `docs/known_issues.md`. This oracle must also run inside the
  manylinux container.
* **Portability to non-AVX-512 hardware.** That is the point of the change and
  the one thing an AVX-512 host structurally cannot demonstrate. Same ISA,
  different microarchitecture -- it should hold, but that is an argument.
* **That no divergence exists.** Actin performs millions of pair evaluations
  per full Mu recompute, so the case matrix samples a large but finite number of
  threshold comparisons. The honest claim is **no divergence at this sampling
  depth**; the recorded JSON makes the depth auditable.

## A methodological note

The first attempt at this A/B compared a build against **itself**. The
`cmake.args` pin silently overrode `-Ccmake.define.MCPU_ARCH`, both arms built
`skylake-avx512`, and the oracle reported `PASS: every case bit-identical`.

What caught it was the record carrying the `.so` **sha256** and the `objdump`
ISA counts: identical digests and `zmm=10` on both sides. Comparing
`mcpu_core.__file__` would not have caught it, because a rebuild in place keeps
the path and changes the content.

This is why the oracle records build identity, and why a same-digest comparison
prints a warning instead of a clean result.
