# Aromatic orientation potential (mcpu08)

**Python API:** {class}`~pymcpu.AromaticPotential`; see {doc}`/api/forces`.

The aromatic term scores how pairs of aromatic rings are oriented. Only PHE
and TRP take part; TYR and HIS do not. The weight is 5.0 (energy group 5).

## Rings

Each ring is represented by three atoms:

| Residue | Ring atoms |
|---------|------------|
| PHE | CG, CE1, CE2 |
| TRP | CG, CZ2, CZ3 |

The ring centre is the mean of the three atoms, and the normal is
(second atom − CG) × (third atom − CG). A residue missing one of these atoms
is left out, with a warning.

## Energy

Each pair of rings whose centres are closer than 7.0 Å adds one table value,
chosen by the angle between their normals. The angle is folded into 0°–90°,
capped at 89.9° and binned in 10° bins. The mcpu08 values, divided by 1000 and
before the weight:

| Angle between normals | Energy |
|-----------------------|--------|
| 0–10° | 1.000 |
| 10–20° | 0.921 |
| 20–30° | 0.659 |
| 30–40° | 0.528 |
| 40–50° | 0.481 |
| 50–60° | 0.032 |
| 60–70° | 0.032 |
| 70–80° | 0 |
| 80–90° | 0.044 |

So the term penalises nearby rings whose planes are close to parallel, by up
to 5.0 after weighting, and barely affects rings tilted by 50° or more.

## Energy changes

Every move recomputes all ring pairs, which is cheap: proteins have few PHE
and TRP residues.
