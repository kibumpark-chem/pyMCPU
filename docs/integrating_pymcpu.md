# Integrating pyMCPU with your sampling framework

pyMCPU owns one engine: coordinates, energies, Monte Carlo moves and
collective variables. Your framework owns the ensemble: walkers, weights,
cloning, binning, scheduling and storage. This page describes the interface
between them.

| pyMCPU gives you | you provide |
|---|---|
| An engine built from a spec | your own configuration format |
| Set and get coordinates, step Monte Carlo | when to step, and for how long |
| Collective variables from a declarative spec | what to do with the values |
| Independent random streams for clones | the cloning policy itself |
| Restart checks and loading from several file formats | where state files live |

There is no plugin API to register with. Your package imports pyMCPU and
calls the classes below, so a new integration needs no change to pyMCPU.

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

Map your framework's configuration onto this. `EngineSpec` checks its fields
when it is created: missing files, an unknown move mode or force field, a
fixed residue that is also a linker. Build it in your main process, so a bad
field fails there rather than inside one worker, where it looks like a
single failed trajectory.

If you already have a pyMCPU config, `EngineSpec.from_simulation_config`
converts it. `EngineSpec` has no `seed` field, on purpose; see step 4.

## 2. Hold one engine per worker

```python
from pymcpu.sampling import EngineSession

session = EngineSession(spec)
session.set_coords(start_coords)      # (3, n_atoms), Å
session.step(1000)
coords = session.coords()
```

Creating an `EngineSession` is cheap: it builds the engine on first use.
Building it takes from under a second for a small protein to several seconds
for a large one, so create **one session per worker process and reuse it**.

pyMCPU does not decide when a new session is needed, because only you know
your process model. If you fork workers, build the session lazily *inside*
each worker, and key it on the process ID (and the thread, if workers are
threads): a session created before the fork holds C++ state that must not be
shared. Make the object that holds the session refuse pickling, for example
with a `__getstate__` that raises, so it cannot be sent to a worker by
accident.

## 3. Compute a collective variable

```python
values = session.compute_cv(session.coords())   # np.ndarray of length cv.ndim
```

The session builds the CV from `spec.cv`. Outside a session,
`pymcpu.sampling.build_cv(spec.cv, forcefield)` builds the same object, with
a force field from `build_forcefield(spec)`.

Built-in `type` values: `native_contacts_q`, `native_contacts_n`, `ca_rmsd`,
`two_state_rmsd`, `two_state_delta` and `custom`. Several specs in one list
give one CV whose values are concatenated, with its `ndim` and `labels`
combined. The two native-contact types take `reference_pdb` and, optionally,
`contact_cutoff` (default 6 Å), `min_seq_sep` (default 4),
`contact_atom_mode` (default `ca`) and `native_contact_pairs`, with the same
defaults as everywhere else in pyMCPU.

`custom` takes a dotted path to your own factory, so a framework-specific CV
needs no change to pyMCPU:

```yaml
cv:
  - type: custom
    factory: my_package.cvs.make_end_to_end_distance
    kwargs: {atom_a: 0, atom_b: 41}
```

The factory can return any object that satisfies
`pymcpu.sampling.CollectiveVariable`: `ndim`, `labels`, and
`__call__(coords_3xn) -> np.ndarray`. It is a `runtime_checkable` Protocol,
so a consumer can accept a CV it has never seen.

The built-in CVs follow one rule, and yours should too: **raise, do not
return a sentinel.** A Q or an RMSD of exactly 0.0 is physically meaningful,
so returning it for a failure corrupts a free-energy or flux estimate with no
visible symptom.
## 4. Seed independent streams

```python
from pymcpu.sampling import derive_seed

session.set_seed(derive_seed(base_seed, round_index, stream_index))
```

When your framework clones a walker, every child starts from the **same**
parent state. If you seed a child by restoring the parent's saved random
state, every child runs a **bitwise identical** trajectory. Nothing looks
wrong: the children are distinct objects, the weights divide correctly, and
every log line looks healthy. You have one walker counted N times, and only
the statistics will ever show it.

So pyMCPU's restart state is the coordinates, their frame offset and the
step count, and deliberately not the random state, and `EngineSpec` has no `seed` field.
Derive a seed for each stream instead: `derive_seed` gives a different seed
for each `(round_index, stream_index)`, the same on every process and
machine. Its output is fixed for good, because changing it would reseed
every existing run.

To replay a single trajectory exactly, `session.get_rng_state()` and
`session.restore_rng_state()` save and restore the random state. Never use
them across a clone.

## 5. Restart safely

```python
session.set_coords(saved_coords, frame_offset=saved_offset)
session.current_step = saved_step
session.set_seed(derive_seed(base_seed, round_index, stream_index))
```

Store `session.coords()` as it comes, as float64, and `session.frame_offset()`
with it. The engine runs a structure far from the origin shifted toward it,
moves the shift with the chain as it drifts, and `coords()` adds it back. A
float32 copy, or coordinates placed without their offset, are rounded once
more, and the restart would no longer continue the run exactly. An exact
replay with `get_rng_state()` also needs the frame to move at the same steps.
A session recomputes the energy every `full_energy_every_steps` steps of its
own and then recentres a chain that has reached 64 Å from the origin, so a
restarted session can recentre at a different step than the original run did;
from there the two trajectories differ, at first only by float32 rounding.

`session.coords_from_auxref(path)` reads starting coordinates from a file:

- `.npz`: any file with a `coords` array, including restart files your own
  code writes. Either orientation, `(3, n_atoms)` or `(n_atoms, 3)`, is
  accepted.
- `.chk`: a pyMCPU checkpoint. It takes the first replica, which in replica
  exchange is the first temperature of the ladder in the first window. Its
  saved frame offset is not used, so the coordinates enter in the session's
  own frame, rounded once to float32, which does not matter for a start
  state.
- `.pdb`: a structure of the same protein as `spec.pdb`.

`session.fingerprint` is a hash of what the engine was built from: the PDB's
absolute path, the parameter set and the atom count. Record it with your
state and compare it on load, so that a state loaded against a different
system fails loudly instead of producing nonsense. Because the path is part
of it, the same files moved to another directory give a different
fingerprint. `set_coords` also rejects coordinates of the wrong size, and
coordinates with overlapping atoms.

## 6. Ship it as your own package

You do not need to vendor or patch pyMCPU. Depend on it:

```toml
[project]
name = "myframework-pymcpu"
dependencies = ["pymcpu>=0.1.0,<0.2", "myframework"]

[project.scripts]
myframework-pymcpu = "myframework_pymcpu.cli:main"
```

Cap the pyMCPU version while it is below 1.0. Ship your command-line tool as
its own console script rather than as an `mcpu` subcommand; it then appears
exactly when your package is installed.

## Validating a new integration

Work through these in order; each one makes the next easier to debug.

1. Build the spec and session, and step one short trajectory. Check that the
   energy is finite and the CV is in range.
2. Run a few rounds **serially**. Check that restart works: tear the session
   down, rebuild it from the coordinates and step count, and continue.
3. Check that clones diverge: two children of one parent with different
   stream indices must not produce identical coordinates.
4. Run the same thing in parallel. The results must not depend on the number
   of workers; if they do, you are sharing engine state across a fork.
5. The one that matters most: compare an equilibrium observable with a long
   unbiased `pymcpu.sampling.FoldingRunner` trajectory. Everything above can
   pass while the reweighting is wrong.
