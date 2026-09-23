# Layout of `forces/`

Three tiers, and the tier a file sits in is the whole of the convention.

```
forces/
  <Role>Potential.h        role interface  — the engine asks for a capability
  bias/                    restraints and biases, not force-field terms
  <family>/                a LINEAGE of force fields
    common/                shared by every fit of that lineage   <- the DEFAULT
    <fit>/                 one fit only
```

Today:

```
forces/
  bias/QBiasPotential.h
  mcpu/
    common/   AromaticPotential  HydrogenBondPotential
              SideChainTripletPotential  TripletPotential
    mcpu08/   MuPotential
```

## A family is a lineage, not a method

`mcpu/`, not `knowledge_based/`. "Knowledge-based" describes *how* a potential
was obtained — by statistics over a structure database — and that is true of
whole families that share no code with this one: DFIRE, GOAP, Rosetta's
statistical terms. A directory named for the method would have to hold all of
them, and they have nothing to share.

So the family tier names the **lineage**: the set of fits that are refits of
each other and therefore *do* share machinery. `forces/mcpu/common/` is code
common to MCPU fits, not to knowledge-based potentials in general. A future
DFIRE implementation is `forces/dfire/`, a peer — not something nested here.

## Where does a new file go?

> A file lives at the **deepest tier that owns it**: `utils/` if anything
> outside `forces/` calls it, `forces/<Role>Potential.h` if the engine must ask
> for it polymorphically, `forces/<family>/<fit>/` if a second fit of that
> lineage would need it byte-different, and `forces/<family>/common/`
> otherwise — which is the default, and where most files belong.

Four questions, first yes wins.

1. **Does anything outside `forces/` call it?** → `include/pymcpu/utils/`.
   The non-obvious case: `utils/virtual_amide_h.h` looks like H-bond code, but
   `neighbor/NeighborSystem.h` calls it to build the H-bond grid, so it stays
   in `utils/`.
2. **Must the engine ask a potential for it polymorphically?** → a role
   interface at `forces/<Role>Potential.h`, namespace `mcpu::forces`, every
   method `virtual` with a no-opinion default. The test for "is this a role":
   the engine owns exactly one of some resource and must configure it from
   whichever term the force field installed.
3. **Would a second fit of the same lineage need this file byte-different?** →
   `forces/<family>/<fit>/`.
4. **Otherwise** → `forces/<family>/common/`.

A term belonging to no force field at all — no fitted tables, parameters
supplied by the caller — goes in `forces/bias/`. `QBiasPotential` qualifies.

## Namespaces

A file's namespace gains a component **if and only if it sits in a fit
directory**. Family and role directories add nothing.

| Path | Namespace |
|---|---|
| `forces/<Role>Potential.h` | `mcpu::forces` |
| `forces/bias/*` | `mcpu::forces` |
| `forces/mcpu/common/*` | `mcpu::forces` |
| `forces/mcpu/mcpu08/*` | `mcpu::forces::mcpu08` |

Not `mcpu::forces::mcpu::mcpu08` — fit tags are globally unique (they are the
Python registry keys and the `"forcefield"` value in `params_registry.json`),
so a family component prevents no collision and only guarantees a `using`
alias at every call site.

The fit directory is named for the **tag** (`mcpu08`), matching the parameter
registry and the Python force-field name, which is worth more than avoiding
the `mcpu/mcpu08` stutter in the path.

The namespace is invisible to Python: every `py::class_` passes an explicit
name string. A forked class is named after its **physics**
(`RadialContactPotential`), never its tag (`Mcpu26MuPotential`) — pybind11's
"already registered" error is then what enforces the rule below.

## Two standing rules

**Do not add a tier speculatively.** Files are born at the deepest tier that
fits and are promoted only when a second consumer exists. Do not add a role
interface until there is a `dynamic_cast` it deletes.

**Never copy a `common/` file into a fit directory for symmetry.** `common/`
holding four files next to a fit directory holding one is the tree telling you
exactly one term forked; that asymmetry is the information. Forking is a
demotion, done deliberately: copy into *both* fit directories, delete the
`common/` copy, add the namespace, rename the diverging class. That is cheap
precisely because sharing here is by directory and not by base class.

## Adding a force field

**Same lineage, new fit** (a refit of the MCPU potentials): add
`mcpu/<tag>/` holding only the terms that genuinely differ, and a parameter
set declaring `"forcefield": "<tag>"`. Everything unchanged is reused from
`mcpu/common/` as-is.

**New lineage** (`forces/dfire/`, `forces/amber/`): a peer of `mcpu/`,
crossing no `#include` into it in either direction. Note such a family should
land with **no** fit subdirectories until one is earned — Amber's `ff14SB`,
`ff19SB` and `charmm36` differ in force constants, torsion barriers and
partial charges, not in one line of C++. They are parameter sets, the same way
GROMACS ships `<name>.ff/` against fixed interaction forms.

`CMakeLists.txt` needs no edit either way: it globs `src/*.cpp` with
`CONFIGURE_DEPENDS`.
