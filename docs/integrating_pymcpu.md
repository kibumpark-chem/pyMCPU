# Integrating pyMCPU with your sampling framework

pyMCPU owns one engine: coordinates, energies, Monte Carlo moves, collective
variables. Your framework owns the ensemble: walkers, weights, cloning,
binning, scheduling, storage. This page is the seam between them, and
[`pymcpu-westpa`](https://github.com/kibumpark-chem/pymcpu-westpa)
is a production integration that uses nothing else.

| pyMCPU gives you | you provide |
|---|---|
| Build an engine from a spec | your own configuration format |
| Set/get coordinates, step Monte Carlo | when to step, and for how long |
| Collective variables from a declarative spec | what to do with the values |
| Independent random streams for clones | the cloning policy itself |
| Restart identity and multi-format state loading | where state files live |

There is no plugin API to register with, and that is deliberate. WESTPA
loads its driver classes by dotted path from its own config file, so the
add-on needed **no change to pyMCPU at all** to exist. If your framework can
import a class by name, so do you.

## 1. Describe the engine

```python
from pymcpu.config import EngineSpec

spec = EngineSpec(
    pdb="protein.pdb",
    temperature=0.6,
    step_size_rad=0.1,
    cv=({"type": "native_contacts_q", "reference_pdb": "native.pdb"},),
)
```

Map your framework's config onto this. `EngineSpec` validates eagerly —
missing files, bad move modes, a fixed residue that is also a linker — so
build it in your master process and a bad field is rejected there, not
inside one worker where it looks like a single failed trajectory.

If you already have a pyMCPU config, `EngineSpec.from_simulation_config`
converts it.

**`EngineSpec` has no `seed` field, on purpose.** See step 4.

## 2. Hold one engine per worker

```python
from pymcpu.sampling import EngineSession

session = EngineSession(spec)
session.set_coords(start_coords)      # (3, n_atoms) float32, Angstrom
session.step(1000)
coords = session.coords()
```

Constructing an `EngineSession` is cheap; it builds nothing until first use.
Building the force field reads the parameter tables (~0.5 s, independent of
system size), so create **one session per worker process and reuse it** —
per-trajectory construction turns a large run into a parameter-parsing
benchmark.

pyMCPU does not decide when a new session is needed, because only you know
your process model. If you fork workers, build the session lazily *inside*
the worker and key it on the PID: a session created before the fork holds
C++ state that must not be shared. The add-on's propagator keys on
`(thread, PID)` and defines `__getstate__` so the object cannot be pickled
into a worker by accident.

## 3. Compute a collective variable

```python
from pymcpu.sampling import build_cv, build_forcefield

forcefield, _topology = build_forcefield(spec)
cv = build_cv(spec.cv, forcefield)
values = cv(coords)                    # np.ndarray, shape (cv.ndim,)
```

Built-in `type` values: `native_contacts_q`, `native_contacts_n`,
`ca_rmsd`, `two_state_rmsd`, `two_state_delta`, and `custom`. Several specs
in one list produce a concatenated CV, with `ndim` and `labels` composed
for you.

`custom` takes a dotted path to your own factory, which is the escape hatch
that means a framework-specific CV needs no change here:

```yaml
cv:
  - type: custom
    factory: my_package.cvs.make_end_to_end_distance
    kwargs: {atom_a: 0, atom_b: 41}
```

Anything satisfying `pymcpu.sampling.CollectiveVariable` works — `ndim`,
`labels`, and `__call__(coords_3xn) -> np.ndarray`. It is a
`runtime_checkable` Protocol, so a consumer can accept a CV it has never
heard of.

One rule these follow and yours should too: **raise, do not return a
sentinel.** A Q or an RMSD of exactly `0.0` is physically meaningful, so
substituting it for a failure corrupts a free-energy or flux estimate with
no visible symptom.

## 4. Seed independent streams — the part that is easy to get wrong

```python
from pymcpu.sampling import derive_seed

session.set_seed(derive_seed(base_seed, round_index, stream_index))
```

When your framework clones a walker, every child starts from the **same**
parent state. If you seed a child by restoring the parent's saved RNG
state, every child runs a **bitwise identical** trajectory. The clone
"succeeds": the children are distinct objects, the weights divide
correctly, every log line looks healthy. You have one walker counted N
times, and only the statistics will ever tell you.

So pyMCPU's restart contract is `(coordinates, step)` and deliberately
**not** an RNG state, and `EngineSpec` has no `seed` field for an engine to
pick up at construction time. Derive each stream instead: the result is
unique per `(round_index, stream_index)` and stable across processes and
machines.

`derive_seed` is a frozen wire format — changing it reseeds every run ever
done. Two tests in the pyMCPU repo demonstrate both halves:
`tests/physics/test_stream_independence.py` (clones diverge, *and* the
naive design provably produces identical siblings) and
`tests/unit/test_seed_derivation.py` (the derivation pinned by literal).

## 5. Restart safely

```python
from pymcpu.sampling import compute_fingerprint

coords = session.coords_from_auxref("prior_state.npz")   # or .chk, or .pdb
```

`coords_from_auxref` reads any `.npz` carrying a `coords` array — including
restart files your own code writes — a `.chk` from a completed
`FoldingRunner` or `ReplicaExchange` run, or a plain `.pdb`. Either
coordinate orientation is accepted and normalized, which matters because a
transposed array is not an error, it is a silently wrong structure.

`session.fingerprint` hashes what the engine was built from, so a restart
state loaded against a different system fails loudly instead of producing
nonsense. Record it with your state and check it on load. The hash includes
the engine's atom count, so it changed for every protein with glycine when
glycine's CA went from two engine slots to one; `set_coords` also rejects
coordinates of the wrong size.

## 6. Ship it as your own distribution

You do not need to vendor anything or patch pyMCPU. Depend on it:

```toml
[project]
name = "myframework-pymcpu"
dependencies = ["pymcpu>=0.1.0,<0.2", "myframework"]

[project.scripts]
myframework-pymcpu = "myframework_pymcpu.cli:main"
```

Cap the pyMCPU version while it is pre-1.0. Ship your CLI as its own
console script rather than asking for a subcommand on `mcpu`: argparse
cannot register a subcommand lazily, so anything added there is advertised
to every pyMCPU user whether or not they can run it. A console script
appears exactly when your package is installed.

## The reference implementation, by the numbers

`pymcpu_westpa`, measured:

| module | lines | what is WESTPA-specific |
|---|---|---|
| `propagator.py` | 266 | subclasses `WESTPropagator`; segment/status bookkeeping |
| `tools.py` | 247 | writes and preflights `west.cfg` |
| `state.py` | 187 | **nothing** — the WE restart-file layout, pure numpy |
| `config.py` | 146 | parses the `west.pymcpu` config block |
| `cli.py` | 115 | `mcpu-westpa init` / `check` |
| `system.py` | 68 | subclasses `WESTSystem`; bin mapper |
| `__init__.py` | 73 | exports |
| **total** | **1102** | **46 lines mention a WESTPA symbol at all** |

The two modules that subclass a WESTPA class are 334 lines together. **None
of the 1102 reimplement pyMCPU behaviour** — no energy term, no move, no
integrator loop, no CV maths. That is the claim this page is making, and it
is why the integration is worth reading before writing your own.

## Validating a new integration

In this order, because each step makes the next one debuggable:

1. Build the spec and session; step one short trajectory; check the energy
   is finite and the CV is in range.
2. Run a handful of rounds **serially**. Confirm restart works: tear the
   session down, rebuild it from `(coords, step)`, and continue.
3. Confirm clones diverge. Two children of one parent with different stream
   indices must not produce identical coordinates.
4. Run the same thing in parallel. Confirm the results are unchanged by
   worker count — if they are not, you are sharing engine state across a
   fork.
5. The one that actually matters: compare an equilibrium observable against
   a long unbiased `pymcpu.sampling.FoldingRunner` trajectory. Everything
   above can pass while the reweighting is wrong.
