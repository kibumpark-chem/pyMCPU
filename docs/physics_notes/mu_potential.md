# Mu contact potential (mcpu08)

**Python API:** {class}`~pymcpu.MuPotential`; see {doc}`/api/forces`.

The Mu potential scores contacts between pairs of atoms. Each heavy atom has
one of 84 atom types, and each pair of types has a contact energy, a table
value derived from the statistics of known protein structures. A pair of atoms
in contact adds its types' energy; a pair closer than its hard-core distance
is a clash, and a move that makes one is rejected. The term's weight is 0.4
(energy group 1).

## Contact and clash distances

Each atom type has a radius. For atoms with radii r_i and r_j:

| Cutoff | Distance |
|--------|----------|
| Hard core | r_hard = α (r_i + r_j) |
| Contact | r_contact = λ α (r_i + r_j) |

with α = 0.75 and λ = 1.8, which `MCPUForceField` fixes. A pair is in contact
when it is closer than r_contact and does not clash.

A move is rejected if it puts a pair under r_hard, rounded to 0.001 Å, less
0.0015 Å, so that noise at the precision of a PDB file cannot decide a clash.
A whole state (`calculate_total_energy`, `Context.has_steric_clash()`) is
judged against a cutoff 0.001 Å looser, `mcpu_core.STATE_CLASH_BUFFER_A`; see
the hard-core section of {doc}`mc_acceptance`. Pairs that already overlap in
the starting structure are exempt from the clash test for the whole run.

## Which pairs count

- Contacts count only between atoms whose residues are more than four apart
  in sequence, and never between two backbone atoms (N, CA, C, O, OXT).
- Clashes are tested between almost all pairs. The exceptions are pairs that
  covalent geometry holds close, mostly within a residue and across the
  peptide bond.
- Two cysteine SG atoms more than four residues apart count as a contact and
  never as a clash, which allows a disulfide.
- Explicit hydrogens (`virtual_amide_h=False`) take no part.

## Atom types

Backbone atoms have the same type whatever their residue: N 79, CA 80, C 81,
O 82, and the terminal OXT 83. Glycine's CA has a type of its own, 78.
Sidechain atoms have a type for each residue and atom name. The types and
radii are listed in `atom_types.csv` in the mcpu08 parameter set.

## Energy changes

A move changes the Mu energy by the energy of the contacts it makes minus that
of the contacts it breaks. Pairs in which neither atom moved keep their
energy, so the cost of a move grows with the atoms it moves; candidate
partners come from a cell grid, not a search over every atom.
