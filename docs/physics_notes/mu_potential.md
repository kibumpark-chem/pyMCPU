# Mu contact potential (KERNEL-1)

**Python API:** :class:`~pymcpu.MuPotential` (see :doc:`/api/forces`).

## What is modeled

The μ-potential is a **knowledge-based pairwise contact energy**.
For each atom-type pair that falls within a contact shell (and
outside the hard-core clash distance), the energy contribution
is a table entry derived from protein structure statistics.
Temperature in the Metropolis criterion that accepts or rejects
moves is a **dimensionless** reduced parameter (typical 0.3–0.6);
μ energies themselves are **unitless** table values.

## Inputs

- Cartesian coordinates (Å, double) for all atoms
- Per-atom MCPU type indices (including the residue-independent
  backbone types)
- Precomputed pair tables: `can_contact`, `can_clash`,
  `contact_r2`, `hardcore_r2`, `mu_energy`
- Default contact parameters: α = 0.75, λ = 1.8
  (see Implementation Notes)

## Energy contribution

When atom pair (i, j) is a live contact:

```
E_pair = mu_energy[i, j]     # unitless table entry
```

The MC delta for a move is the sum of energies of newly formed
contacts minus energies of broken contacts. A hard-core clash returns a
sentinel energy that rejects the move before the Metropolis test is reached;
`Context.has_steric_clash()` reports whether the *current* state contains one.

The default outer weight for this group is **0.4** (energy group 1). Weights
live on `EnergyWeights`, not on the potential object -- read them with
`Context.get_energy_weights()` and set one with
`Context.set_energy_weight(group, w)`.

## Implementation notes

### Coordinates are floating point

Legacy MCPU stored coordinates as integers scaled by a factor of 100 and did
its distance comparisons in integer arithmetic. pyMCPU does not: coordinates
are floating point in Ångströms throughout. The legacy scale factor survives
only where a legacy integer cutoff has to be converted at table-build time.

### Contact and clash distance cutoffs

For each atom-type pair (i, j) with van der Waals
radii $r_i$ and $r_j$:

```
Hard-core clash:  r_hard = α × (r_i + r_j)
Contact upper:    r_contact = λ × α × (r_i + r_j)
```

Default parameter values (`_MU_ALPHA_DEFAULT` / `_MU_LAMBDA_DEFAULT` in
`pymcpu/forcefields/mcpu.py`, applied by
`pymcpu/forcefields/builders/mu_builder.py`):

```
α (MU_ALPHA_DEFAULT)  = 0.75
λ (MU_LAMBDA_DEFAULT) = 1.8
```

These are configurable at System creation time.

### Neighbour enumeration

Candidate pairs come from a cell grid rather than an all-pairs sweep. The Mu
term uses typed grids over backbone/oxygen and sidechain atoms -- the backend
reports itself as `opencell_mu_BBO_SC`, readable at run time via
`Context.mu_backend_name()`.

A Verlet list is available on top of the grid but is **off by default**: the
`mu_skin` parameter is 0, which selects the grid's dense candidate list
directly. Setting a positive skin (`Context.set_mu_skin`) engages the Verlet
path, which trades rebuild work against per-step traversal. Both paths are
required to give identical energies, and `scripts/parity_verlet_vs_cellonly.py`
is the oracle that checks it.

### Backbone atom type encoding

Backbone atoms N, CA, C, O use residue-independent
type indices regardless of which amino acid they belong to:

| Atom | Type | Lookup key |
|------|------|------------|
| N | 79 | `("XXX", "N")` in `atom_types.csv` |
| CA | 80 | `("XXX", "CA")`; GLY CA = type 78 |
| C | 81 | `("XXX", "C")` |
| O | 82 | `("XXX", "O")` |

An earlier revision assigned type −1 (unknown) to backbone atoms of non-GLY
residues, which gave them wrong clash radii; the residue-independent `"XXX"`
lookup above is the fix. GLY CA keeps its own type because its radius and
energy row differ from the generic CA.
