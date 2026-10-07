# KORP 6D orientational potential

**Python API:** {class}`~pymcpu.OrientationalPairPotential` (see {doc}`/api/forces`).
Built by {class}`~pymcpu.KORPForceField`.

Energies are **unitless**, as everywhere else in pyMCPU: sums of knowledge-based
table entries scaled by a dimensionless per-group weight. Temperature is a
reduced parameter, not Kelvin. KORP's tabulated values are already
`-RT ln(P_obs / P_ref)` in the units of its training set, so the group weight
defaults to **1.0** and the local/non-local factor is applied *inside* the term
rather than through the weight.

KORP scores residue **pairs**. Each residue is reduced to a local frame built
from its own N, CA and C, and the pair coordinate is CA-CA. No sidechain atom is
read at any point, which is what makes the potential usable as a backbone-only
force field.

## The residue frame

With `r12 = N - CA` and `r13 = C - CA`, both from the same residue:

| Axis | Definition |
|---|---|
| `vz` | `normalize(r12 + r13)` — the N-CA-C bisector |
| `vy` | `normalize(vz × r13)` |
| `vx` | `vy × vz` |
| origin | **CA** |

The frame is right-handed (`vx × vy = vz`) and `det R = +1`.

:::{warning}
The published Eq. (2) and the code that built the released map **disagree**.
The paper writes `vy ~ vz × r12`; `frameCoord` in the reference source uses
`vz × r13`. Because `r12 = λ·vz − r13`, those two cross products are exact
negatives, so the paper's frame is this one rotated 180° about `vz`. Both are
right-handed, so no invariant catches the difference — but every ψ angle shifts
by π and lands in a different bin. pyMCPU follows the **code**, because the code
is what produced `korp6Dv1.bin`.
:::

## The six coordinates

For an ordered pair (A, B), where **A is the residue with the lower index** —
the map is not symmetric under swapping the partners:

| # | Symbol | Definition | Range |
|---|---|---|---|
| 1 | `d` | `abs(CA_B − CA_A)` | `(min_r, cutoff)` |
| 2 | `θ_A` | angle between `vz_A` and `r_AB` | `[0, π]` |
| 3 | `ψ_A` | `r_AB` projected into A's xy-plane, measured from `vx_A`, signed by `vy_A`, then `+ π` | `[0, 2π]` |
| 4 | `θ_B` | as `θ_A`, in B's frame, using `r_BA = −r_AB` | `[0, π]` |
| 5 | `ψ_B` | as `ψ_A`, in B's frame, using `r_BA` | `[0, 2π]` |
| 6 | `χ` | `π + dihedral(vz_A, normalize(r_AB), vz_B)` | `[0, 2π]` |

The dihedral is the variant whose first cross product is negated; upstream marks
that negation as a fix predating v1 of the map, so the released table was trained
with it.

## Binning

Everything below is **read from the map header**, never assumed — the format
carries no version field, so a differently-binned map that was silently assumed
to match would read neighbouring cells rather than fail. The values shown are
those of the released `korp6Dv1.bin`.

| Quantity | Released map |
|---|---|
| radial shells | 10, boundaries 3.0 → 16.0 Å in 1.3 Å steps |
| angular cells per shell | 36, in 6 polar rings of **1 / 7 / 10 / 10 / 7 / 1** |
| χ bins | 8 |
| residue types | 20 × 20, ordered alphabetically by **one-letter** code |
| sequence-separation slices | 2 |

Lookup is **nearest-bin with no interpolation**, so the energy is a step function
of geometry. Nothing takes a derivative of it, and it makes the incremental ΔE
exact by construction rather than an approximation needing a bound.

## Sequence separation

Separation is computed from **PDB residue numbers**, not array position, and a
pair spanning two chains is always non-bonding. For the released map:

| `abs(i − j)` | Treatment | Weight |
|---|---|---|
| ≤ 1 | excluded entirely | — |
| 2 – 4 | local | 1.8 |
| ≥ 5, or cross-chain | non-local | 1.0 |

## The energy

```
E = sum over pairs a < b with min_r < d < cutoff and slice(sep) >= 0 of
        f[slice] * M[slice][type_a][type_b][shell][cell_a][cell_b][chi]

slice(sep)  from the map's smapping table (-1 means the pair is not scored)
f[slice]    from the map's fmapping table (1.0 non-local, 1.8 local)
```

Accumulated in double, as upstream does.

## Delta-kernel scope

Only pairs with at least one residue whose frame changed can contribute. A
residue is classified by **which of its N, CA and C are in the move's moved
set** — not by whether it was displaced, because a C-terminal φ pivot moves
`C(r)` while leaving `N(r)` and `CA(r)` behind, and an N-terminal pivot carries
frame atoms that sit on the rotation axis and do not move at all.

Pairs with **both** partners carried by the same rigid motion are skipped: all
six coordinates are invariant then, not just the distance, because both frames
transform together and a proper rotation commutes with the cross products the
frame is built from. Cost is `O(n_changed × N)`; a sidechain-only move costs
nothing, since no frame atom moves.

### Speed

A step costs time roughly in proportion to chain length: about 480 µs at 200
residues, 1.1 ms at 400 and 2.2 ms at 686, with pivot moves and KORP alone.

:::{note}
The elision is exact in real arithmetic. In float32 a pair within a rounding
error of a bin boundary could in principle fall in a neighbouring bin on a full
recompute while the delta assumed no change. Measured on actin over 1e7 steps,
the running total and a full recompute agreed exactly (a difference of 0.0), and
the periodic recompute (`Simulation.full_energy_every_steps`) would report such
a flip as drift.
:::

## Excluded volume

KORP carries none: nothing in a potential fitted to real structures says a 1 Å
CA-CA contact is impossible, because no such contact appears in the PDB. Driving
MC with KORP alone lets a chain collapse through itself.
{class}`~pymcpu.CalphaExcludedVolumePotential` (energy group 8) supplies the
floor as a pure filter — exactly zero in every accepted state, the clash
sentinel otherwise — so it deletes configurations without shifting the ensemble.
`KORPForceField` installs it by default (`steric_guard=True`). It checks CA
pairs at least `min_separation` residues apart (default 3) against
`min_distance` (default 3.2 Å). As with Mu, moves are tested against the floor
and a whole state against a floor 0.001 Å lower, for the rounding of pairs a
rigid pivot carries without re-checking them (see the hard-core section of
{doc}`mc_acceptance`).

Its 3.2 Å default is measured, not assumed: across 1CEO, 1DOS, T0860D1, actin
and chignolin the closest CA-CA contact at three or more apart in sequence is
**3.53 Å**, and those are genuine packing contacts (actin has no numbering gaps
at all and still reaches 3.88 Å at separation 9). A 4.0 Å floor rejects native
structures outright.

One sphere per residue at CA prevents collapse, but it does not rigorously stop
one strand threading through another, as an all-backbone-atom guard would.

## Differences from the reference implementation

There are none in the energy: pyMCPU reproduces the reference `korpe` scorer to
5e-9 relative on four structures, and the compiled term to 4e-8 (the residual is
the term returning a float). What differs is everything around it:

- **The map is not distributed** with pyMCPU; {ref}`korp-map` says where to get
  it. Different copies of a map score differently, so record which one you
  used.
- **Residues with an incomplete backbone raise** rather than being skipped. KORP
  has no redundancy — without all of N, CA and C there is no frame.
- **Non-increasing residue numbering raises** by default
  (`strict_residue_numbering=True`), since it would silently change the
  local/non-local split.
- Geometry is computed in double where upstream uses float; the energy
  accumulation matches.

## Source

J. R. López-Blanco and P. Chacón, "KORP: knowledge-based 6D potential for fast
protein and loop modeling", *Bioinformatics* 35, 3013–3019 (2019),
https://doi.org/10.1093/bioinformatics/btz026. Reference implementation:
`github.com/chaconlab/Korp`, `sbg/src/libenergy/korpe.cpp`.
