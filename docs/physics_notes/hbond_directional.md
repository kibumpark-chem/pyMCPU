# Hydrogen-bond potential (mcpu08)

**Python API:** {class}`~pymcpu.HBondPotential`; see {doc}`/api/forces`.

The hydrogen-bond term scores backbone hydrogen bonds: the N–H of one residue,
the donor i, to the C=O of another, the acceptor j. Each bond's energy is a
table value looked up from seven quantities describing its geometry, scaled by
a factor for the two residue types. The weight is 2.7 (energy group 4): 1.35
times a fixed factor of 2.0 carried over from legacy MCPU.

## Which pairs count

A pair forms a hydrogen bond when all of these hold:

- Neither residue is the first or last of the chain, and the donor is not a
  proline, which has no amide hydrogen.
- They are at least four residues apart in sequence.
- The amide H of i is within 2.5 Å of the O of j.
- CA(i−1) and CA(i) each have CA(j) or CA(j+1) within 5.8 Å, and the closest
  of these four CA–CA pairs is within 5.5 Å. For pairs more than four apart,
  the limits are 6.0 Å and 5.4 Å.
- For pairs more than four apart, the backbone at both ends is β-like: φ of i
  and of j+1 at most −30°, and ψ of i−1 and of j outside −150° to 30°.

With secondary structure from DSSP (`MCPUForceField(compute_dssp=True)`),
helix residues also cannot form bonds more than four apart, and strand
residues cannot form i, i+4 bonds with helix-like φ and ψ. By default every
residue counts as coil, and these two checks never apply.

## The hydrogen

By default the amide H is not stored. It is placed 1.0 Å from N, opposite the
sum of the N→CA and N→C(i−1) bond vectors, whenever it is needed. With
`MCPUForceField(virtual_amide_h=False)` the hydrogens are explicit atoms that
move with the chain.

## The seven quantities

| # | Quantity |
|---|----------|
| 1 | Class: 0 for a pair four apart (helix-like); otherwise 1 (parallel) if the chain directions at the two ends, CA(i−1)→CA(i+1) and CA(j−1)→CA(j+1), are less than 90° apart, and 2 (antiparallel) if not. |
| 2, 3 | The angle between the planes of residues i and j, each through N, CA and C, and the angle between their N–CA–C bisectors. |
| 4, 5 | The same two angles for residues i−1 and j+1. |
| 6, 7 | The same two angles for the planes around the N–H and C=O groups: C(i−1), N(i), CA(i), and CA(j), C(j), N(j+1). |

Each angle lies between 0° and 180° and is binned in 20° bins, 9 per angle, so
the table has 3 × 9⁶ = 1,594,323 entries.

## Energy

```
E_pair = table[class][six bins] × s[class][donor type][acceptor type] × f
```

s is a factor for the two residue types, and f is 3.0 for a pair more than
four apart (parallel or antiparallel) and 1.0 for a helix-like pair. The term
is the sum over pairs divided by 1000; the weight of 2.7 then applies.

## Energy changes

A move re-evaluates only the pairs near residues it moved, finding partners
within 2.5 Å on a grid. A sidechain move cannot change this term, which reads
only backbone atoms.
