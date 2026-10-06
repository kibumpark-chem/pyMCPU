# Backbone and sidechain torsion potentials (mcpu08)

**Python API:** {class}`~pymcpu.TripletPotential` and
{class}`~pymcpu.SidechainTripletPotential`; see {doc}`/api/forces`.

Both terms score the conformation of one residue with a table chosen by the
types of three consecutive residues: the residue and its two neighbours. That
is why they are called triplet potentials. The first and last residues of the
chain have no triplet, so they get no torsion energy. Both energies are table
values divided by 1000.

## Backbone torsion

For each residue r with a neighbour on each side, four angles index the table
of the triplet (r−1, r, r+1):

| Angle | Definition | Bins |
|-------|------------|------|
| φ | dihedral C(r−1)–N(r)–CA(r)–C(r) | 6 of 60°, from −180° |
| ψ | dihedral N(r)–CA(r)–C(r)–N(r+1) | 6 of 60°, from −180° |
| pCA | angle between the planes through N, CA and O of residues r−1 and r+1 | 6 of 30°, 0° to 180° |
| bCA | angle between the bisectors of CA→N and CA→O at residues r−1 and r+1 | 6 of 30°, 0° to 180° |

The table has 6⁴ = 1296 cells per triplet. The weight is 1.35 (energy
group 2).

## Sidechain torsion

For each residue r with a neighbour on each side and at least one χ angle, its
χ angles index the table of the triplet (r−1, r, r+1). Each χ is binned in 12
bins of 30°, centred on −180°, −150°, …, 150°; a residue with fewer than four
χ angles uses bin 0 for the missing ones. Glycine and alanine have no χ angles
and get no sidechain torsion energy. The table has 12⁴ = 20,736 cells per
triplet. The weight is 2.5 (energy group 3).

## The tables

Values run from −1000 to +1000. Combinations that the structure statistics
never saw hold +1000, the largest penalty; most cells are of this kind (96% of
the backbone table and over 99% of the sidechain table).

## Energy changes

A move re-evaluates only the residues whose angles it changes.
