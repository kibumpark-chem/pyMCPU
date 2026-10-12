# Rotamer proposal

**Python API:** {func}`~pymcpu.forcefields.builders.rotamer_builder.load_rotamer_library`,
`System.set_rotamer_library`; config key `rotamer_library`.

The default sidechain move (`sidechain_move_mode: rotamer_library`) picks a
residue with χ angles (not proline) and draws all its χ at once from
Dunbrack's 2002 backbone-independent rotamer library (`bbind02.May.lib`): a
mixture over the residue type's rotamers, each a normal distribution per χ
around the rotamer's mean angle. The library only proposes; the energy is
always the force field's. The move is accepted with
min(1, exp(−ΔE / T) q(old) / q(new)) (see {doc}`mc_acceptance`), so the
sampled ensemble does not depend on which proposal is used, as long as it can
reach every state. How fast it samples does.

`rotamer_library` chooses between two proposals made from the same rows:

| `rotamer_library` | Draw | Symmetric χ |
|---|---|---|
| `dunbrack2002_sym` (default) | each χ from a normal truncated to one period | period 180°, random 180° image |
| `dunbrack2002` | each χ from an untruncated normal, as legacy MCPU draws it | only the half the rows cover |

## The symmetric χ

ASP χ2, GLU χ3, PHE χ2 and TYR χ2 are two-fold symmetric: turning the
carboxylate or the ring by 180° gives the same chemical group. The 2002 rows
therefore cover only half of the circle (TYR χ2 means lie between −17° and
98°, for example). A proposal that draws only from the rows hardly ever
proposes the other half, and that is what legacy MCPU did and what
`dunbrack2002` still does. The other half is not equivalent in the mcpu08 energy, which does not
treat the two images alike: in the validation runs described below, ASP27 of
1igd spent 12 % of a full-move run at T = 0.8 in the half the rows miss.

With the Hastings correction, a residue whose symmetric χ is in that half is
trapped. Its q(old) is close to 0, so a move out of the state is accepted
with probability about exp(−ΔE / T) q(old) / q(new), also close to 0: the
expected stay is about q(new) / q(old) attempts. All χ of the residue stay
where they are, because the move proposes them together, and backbone moves
do not help, because q does not depend on the backbone. Without the
correction (legacy MCPU) the residue leaves at its first accepted move and is
hardly ever proposed back, so legacy barely samples that half.

`dunbrack2002_sym` gives each symmetric χ period 180°: its normal is drawn
within ±90° of the row mean, and with probability 1/2 the draw is turned by
180°. Both halves are then proposed with the density of the rows, and the
residue is never trapped by its starting half.

## Exactly the density that is drawn

The density in the Hastings ratio must be that of the draw. For
`dunbrack2002_sym` the two agree by construction:

```
q(χ) = Σ_k w_k Π_i g_ki(χ_i)
```

over the residue type's rows k with a positive probability (the 13 rows of
LYS and ARG with probability 0 are left out: they are never drawn and add
nothing to q).

* **Truncated normals.** χ_i is drawn from a normal with the row's mean and
  width, redrawn when it falls more than half a period from the mean, and
  g_ki carries the normal's mass inside that window as its normaliser.
  Because the window is exactly one period, g_ki is that normal at the image
  of χ_i nearest the mean, with no mass wrapped past the antipode. For a
  symmetric χ the period is 180°, and g_ki has a factor 1/2 for the random
  image.
* **The row.** The row is chosen by comparing one 32-bit random integer with
  integer thresholds, and w_k in q is the probability of exactly that draw.

`dunbrack2002` keeps legacy's draw: an untruncated normal, wrapped onto the
circle, whose density is evaluated at the nearest image only. That misses the
mass a row wraps past its antipode, at most 2e-5 (the widest 2002 row, TRP χ2
at 42°). For a residue type without a symmetric χ the two proposals therefore
have the same density to within 2e-5, and only the random numbers they use
differ.

## Why it is the default

A validation study at legacy's sheet factor 3 ran CLN025 and 1igd with each
proposal and with the continuous move, from native, unfolded and
randomised-sidechain starts, with the backbone fixed and with the default
move mix (10⁹ steps per run, 5 × 10⁸ for CLN025 with the backbone fixed),
and timed them, and actin, on one exclusive node.

* **Frozen residues.** With `dunbrack2002`, 1.5 to 3 residues per full-move
  run never moved for all 10⁹ steps: TYR38 and TYR50 of 1igd (4 of 4 runs at
  T = 0.8 and at 1.0) and TYR1 and TYR10 of CLN025 (3 or 4 of 4 runs at
  T = 0.45, 0.59 and 0.73) and the C-terminal GLU61 of 1igd (4 of 4); each
  had its symmetric χ in the half the rows miss. With `dunbrack2002_sym` no
  residue froze in any full-move run.
* **Unsampled regions.** At T = 1.0 the two 30° bins of TYR8's χ2 from −180°
  to −120° held 1 % and 0 % of the samples with `dunbrack2002`, against 6 %
  and 11 % with `dunbrack2002_sym` and 3 % and 11 % with the continuous move,
  which needs no correction.
* **Sampling per CPU second.** In default full-move runs, `dunbrack2002_sym`
  gave 1.9 (1igd, T = 0.8) and 6.5 (1igd, T = 1.0) times the effective χ
  samples per CPU second (median over χ), 3 to 80 times on CLN025, and 10
  to 105 % more transitions between χ1 wells per CPU second.
* **Cost.** It is the cheapest sidechain move: `dunbrack2002` takes 21 %
  longer per sidechain move on 1igd and actin, and 3 % longer on CLN025.
  With the default move mix this gave 6 % more steps per second on 1igd
  and 8 % on actin, and the same on CLN025 (study runs at sheet factor 3).
  At the default sheet factor 2, a build with the new default runs 4 % more
  steps per second on 1igd than the build before it, which had only
  `dunbrack2002` (139,000 against 132,000), and 5 % more on actin (116,000
  against 110,000).

In the study's full-move runs, energies, Q and Rg, which the backbone moves
limit, showed no difference between the proposals that the runs could
resolve: over whole runs, 1igd at T = 1.0 had a 0.4 Å smaller mean Rg with
`dunbrack2002`, but the second halves of the runs agree. Replica exchange of
chignolin (1UAO) at sheet factor 2 does resolve one. The melting midpoint is
0.57 to 0.60 with `dunbrack2002_sym` and 0.58 to 0.62 with `dunbrack2002`
(twelve 20,000-cycle runs each), on average 0.016 higher with `dunbrack2002`
where Q crosses 0.5 and 0.019 higher at the heat-capacity peak (Mann-Whitney
p = 0.001). Both proposals target the same ensemble, so this is a difference
in sampling: `dunbrack2002` samples only part of the ensemble. A better
sidechain proposal buys sidechain equilibration and the absence of frozen
residues, not faster folding.

**Remaining weaknesses.** mcpu08 has no sidechain-torsion energy for the first
and last residue, so their χ have an almost flat target, and the library's
narrow rotamers explore it slowly: in the full-move runs MET1 of 1igd had not
converged (the χ histograms of different seeds differed by a total variation
of 0.20 to 0.37, against 0.01 to 0.10 with the continuous move), and with the
backbone fixed it stayed in its starting state in 3 of 9 runs. Both
proposals also hardly ever propose χ far from the mean of every row, even
where the energy allows them: in 1igd held fixed except for THR16's
sidechain, at T = 1.5, about 9 % of THR16's Boltzmann weight lies at
χ1 −130° to −90°, where each proposal put less than 3 % of its samples.
`sidechain_move_mode: continuous` moves the chain-end residues, and a rule
that makes some sidechain moves continuous, a planned follow-up, would cover
both weaknesses.

## Choosing the legacy draw

`rotamer_library: dunbrack2002` (YAML or JSON config, `IntegratorConfig`,
`EngineSpec`, and the runner and sampler keyword) gives legacy MCPU's draw,
still with the Hastings correction, for comparisons with legacy MCPU. In
Python:

```python
from pymcpu.forcefields.builders.rotamer_builder import load_rotamer_library

system.set_rotamer_library(load_rotamer_library("dunbrack2002"))
```

Checkpoints record the library, and a run resumed with the other one stops
with an error that names the setting to use; a checkpoint written before the
library was recorded counts as `dunbrack2002`.

## How it is tested

`tests/physics/moves/test_rotamer_detailed_balance.py` puts actin in pairs of
χ states of residues with one to four χ, with each symmetric χ and at both
chain ends, forces the move between them in both directions, and checks
π(a) q(b) α(a, b) = π(b) q(a) α(b, a) with q computed independently from
`bbind02.May.lib`; a wrong Hastings term (none, the wrong sign, period 360°
for a symmetric χ, or the `dunbrack2002` density) fails the same check. The
table, the density and the sampler are checked against the file and against
each other in `tests/physics/forcefield/test_rotamer_library_tables.py`.
