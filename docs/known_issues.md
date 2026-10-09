# Known issues and limitations

Current, verified limitations of pyMCPU 0.1.0. Each entry states the symptom,
the cause, and what to do about it.

Issues fixed before the first release are not listed here — see the
[changelog](https://github.com/kibumpark-chem/pyMCPU/blob/main/CHANGELOG.md).

## Linux x86-64 only, with an x86-64-v3 baseline

**Symptom.** No wheel for your platform, or `Illegal instruction` on a CPU
older than roughly 2013.

**Cause.** The published wheel is built for Linux x86-64 with an
`x86-64-v3` baseline (AVX2 + FMA + BMI2 — Haswell, Excavator or newer). macOS
and arm64 are not built for 0.1.0.

**What to do.** Build from source with a baseline your CPU supports:

```bash
MCPU_ARCH=v2     pip install --no-build-isolation -e .   # SSE4.2, pre-2013 CPUs
MCPU_ARCH=native pip install --no-build-isolation -e .   # this machine only
MCPU_ARCH=v4     pip install --no-build-isolation -e .   # AVX-512, if you have it
```

`mcpu_core.build_info()["arch"]` reports which baseline a build actually used.

Changing the baseline tier does **not** change results: `v3` versus `v4` was
verified bit-identical across chignolin and actin at two seeds, comparing
accept-bit streams, per-group energies and coordinate hashes with no tolerance.
See `arch_baseline_decision` for the verification. What *does* change results is
the compiler — see the next entry.

## A trajectory is reproducible per build, not across compilers

**Symptom.** You cannot reproduce a published trajectory, or two colleagues get
different results from the same seed, the same input and the same
`MCPU_ARCH`.

**Cause.** GCC defaults to `-ffp-contract=fast`, which fuses `a*b+c` into a
single FMA with one rounding instead of two. *Which* expressions get fused is a
compiler-version-dependent optimisation decision — 895 FMA instructions under
GCC 8.5 versus 1281 under GCC 12.2 on identical source at the same baseline.

The consequence is **chaotic amplification, not a single fragile comparison.**
Traced on `chignolin-s1337`: the move-generation arithmetic (the rotation and
loop-closure math in `Integrator.cpp`) produces trial coordinates differing by
**one float32 ULP** (4.77e-07 Å) as early as MC step 27. Monte Carlo dynamics
then amplify that difference with an e-folding time of roughly **145 steps**,
reaching 2.2e-03 Å — about 2,300 ULP — by step 1252. Every accept/reject
decision is still identical at that point; the two runs are simply exploring
slightly different conformations. At step 1253 they are far enough apart
(~23,000 ULP in squared distance) that one genuinely has a steric overlap and
the other does not, the move is hard-rejected in one build and accepted in the
other, and the trajectories decorrelate from there.

Measured: 495 accepts under GCC 8.5, 512 under GCC 12.2 (2000 steps, seed
1337), first differing decision at step 1253. GCC 15.2, the default compiler,
parts from GCC 8.5 at step 1188 on the same case (step 939 with seed 42);
its energies of a given structure agree with GCC 8.5's to float rounding.

It is worth being precise about what this is *not*, because the engine does
contain bare threshold comparisons on squared distances and they are the
obvious suspect. A last-bit difference in a squared distance landing exactly
on one of those thresholds is expected roughly once per **10^7** steps at
float32 — four orders of magnitude rarer than the effect above, and not what
happens here. Promoting the distance arithmetic to double was implemented and
measured: it leaves the trajectories bit-identical and does not move the
divergence at all, because the inputs to that arithmetic already differ.

**What to do.** Either compare only runs from the same build, or build with
contraction disabled, which removes the compiler sensitivity:

```bash
MCPU_FP_CONTRACT=off pip install --no-build-isolation -e .
```

Under `off`, GCC 8.5 and GCC 12.2 produce bit-identical trajectories on all
four verification cases (not yet checked for GCC 15). It costs roughly 3% throughput.
`mcpu_core.build_info()["fp"]["fp_contract"]` reports which setting a build
used, so a result can always be attributed.

Not affected: a single build is fully deterministic — the same binary in a
fresh process reproduces its output bit-for-bit.

## Exact RNG restore is tied to the build

**Symptom.** A resumed run diverges from what an uninterrupted run would have
produced, even though coordinates and counters restored correctly.

**Cause.** The Monte Carlo RNG state is serialized through `std::mt19937`'s
stream operators, so the on-disk form is a C++ standard-library text format
rather than one pyMCPU defines. A different standard library can fail to parse
it or parse it differently.

**What to do.** Resume with the same `mcpu_core` build, or one compiled against
a compatible `libstdc++`. Coordinates, cycle counters and exchange state
restore regardless — only continuation of the exact random stream is affected.

## `libstdc++` mismatches fail at import, not at build time

**Symptom.** The package builds successfully, then `import pymcpu` raises
`ImportError: ... version 'CXXABI_x.y.z' not found`.

**Cause.** The extension was compiled against a newer `libstdc++` than the one
present at run time. A conda interpreter carries an RPATH of `$ORIGIN/../lib`,
so the environment's own `libstdc++` takes precedence and `LD_LIBRARY_PATH`
cannot override it.

**What to do.** Let conda supply both the compiler and the runtime
(`gxx_linux-64` plus `libstdcxx-ng`, as `environment.yml` does), so they cannot
drift apart. On RHEL 8 or Rocky 8, build with
`source /opt/rh/gcc-toolset-15/enable`: the toolset compiles the runtime parts
newer than the system's into the extension, which then needs only
`CXXABI_1.3.11` and imports under any Python. A module-loaded GCC does not do
this. A stock Miniforge/Mambaforge base environment ships a `libstdc++`
capped at `CXXABI_1.3.14`, which a module GCC 14 or newer exceeds and GCC 13
does not.

## Memory on large, compact proteins

**Symptom.** The process is killed by the OS or the batch scheduler on a large
globular system.

**Cause.** The neighbour machinery holds a contact shell whose size grows with
how densely connected the structure is. For a compact globular protein of a few
thousand atoms, that shell is close to fully connected, so memory scales worse
than the atom count alone suggests.

**What to do.** Reduce the number of concurrent replicas per node, or raise the
job's memory request. There is no graceful in-process recovery once an
allocation fails.
