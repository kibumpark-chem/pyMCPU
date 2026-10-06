# Monte Carlo acceptance

**Python API:** {class}`~pymcpu.Integrator`; see {doc}`/api/integrator`.

Each step tries one move (see the moves in {doc}`background`). A move that
would put two atoms closer than their hard-core distance is rejected outright.
Any other move, with energy change ΔE, is accepted with probability

```
P = min(1, exp(−ΔE / T) × w)
```

where T is the reduced temperature and w corrects for a move that is not
proposed symmetrically:

| Move | w |
|------|---|
| Pivot | 1 |
| Sidechain, `'continuous'` mode | 1 |
| Sidechain, rotamer library (the default) | q(old) / q(new) |
| Pivot, Ramachandran library | q(old) / q(new) |
| KIC | (J_new / J_old) × (n_new / n_old) |

The continuous moves draw a symmetric Gaussian step, so w = 1. The library
moves draw new angles from a distribution q that does not depend on the
current ones, and q(old) / q(new) makes up for q favouring some angles over
others. For KIC, J is the Jacobian of the loop closure and n the number of
closures, before and after the move; see {doc}`kic_jacobian`.

A move that cannot be made, such as a KIC window with no closure, counts as an
attempt that was not accepted, so it lowers that move's acceptance rate.

## Hard-core clashes

A move that puts a pair of atoms under its hard-core cutoff is rejected before
the acceptance test, so no accepted move creates an overlap. Pairs that move
together in a rigid pivot are not re-checked: the rotation keeps their
distance, except for float32 rounding of a few 1e-6 Å. A whole state is
therefore judged against cutoffs 0.001 Å looser
(`mcpu_core.STATE_CLASH_BUFFER_A`). {doc}`/api/simulation` describes the
running energy, its full recompute, and what happens if a recompute finds a
clash anyway.

Rounding grows with distance from the origin, so a `Context` runs a structure
that reaches 64 Å or more from the origin in an engine frame shifted next to
it; see `Context.frame_offset` in {doc}`/api/context`.
