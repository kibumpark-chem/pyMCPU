# MC acceptance criterion

**Python API:** :class:`~pymcpu.Integrator` (see :doc:`/api/integrator`).

pyMCPU uses the Metropolis–Hastings criterion implemented in
`metropolis_accept()` (`mc_integrator.hpp`). For a proposed move
with energy change $\Delta E$, dimensionless temperature $T$, and
Jacobian ratio $J$ (`proposal_meta.jacobian_ratio`):

```
if delta_E >= clash_sentinel:  reject immediately
if delta_E <= 0:               accept always
else:  accept if rng() < exp(-delta_E / T) * jacobian_ratio
```

The Jacobian enters as a **multiplicative** factor on the
acceptance probability, **not** as a log term in the exponent.

For pivot and sidechain moves, `jacobian_ratio = 1.0f`. For KIC
moves the ratio is set in `kic_move.hpp` as:

```
jacobian_ratio = (jacobi_after / jacobi_before)
                 × (n_soln / soln_no_before)
```

See [kic_jacobian](kic_jacobian.md) for how `jacobi_*` are computed.

## Temperature

$T$ is a **dimensionless** reduced parameter (typical range
0.3–0.6). Energy $\Delta E$ is unitless (knowledge-based table
sum). Neither quantity is expressed in physical units.

## Implementation Details

### Clash sentinel

Moves that produce a hard-core clash are rejected before
the Metropolis test. The sentinel value is:

```
clash sentinel = 99999.0f  (PhysicsVerifier::kHardCorePenalty)
```

Any delta-energy return at or above half the sentinel
causes immediate rejection without evaluating `exp(-dE/T)`.
This matches legacy behavior where clashed moves bypass
the acceptance criterion entirely.

### KIC jacobian_ratio field

The `MoveProposal` struct carries:

```
float jacobian_ratio = 1.0f  (default: neutral)
```

For KIC moves, `apply_kic_move()` sets:

```
jacobian_ratio = (jacobi_after / jacobi_before)
                 × (n_soln / soln_no_before)
```

For pivot and sidechain moves: `jacobian_ratio` stays `1.0f`.
The Metropolis criterion is:

```
accept = exp(-delta_E / T) * jacobian_ratio >= uniform_rng()
```

(equivalently: `rng() < exp(-delta_E / T) * jacobian_ratio`).

### soln_no_before approximation

When the pre-rotation KIC closure finds zero solutions
(`soln_no_before == 0`), it is forced to 1 to avoid division
by zero in the detailed balance ratio. This is a known
approximation (from legacy `loop.h:582–583`). The effect is
that the `n_soln/soln_no_before` factor becomes `n_soln` rather
than the theoretically correct value.

### Random solution selection

When KIC finds multiple valid closure solutions (`n_soln > 1`),
one is selected uniformly at random:

```
index = (int)(rng() * n_soln)
```

This is required for detailed balance. RMSD-based selection
exists in the legacy code but is overridden by this random
selection (legacy `loop.h:685`).

### Periodic contact cache rebuild

To prevent floating-point drift in the contact CSR over
long simulations, the full contact list is recomputed
from geometry every N steps:

```
contact_rebuild_interval = 1,000,000  (default)
```

This matches the legacy `fold.h` hygiene step and is
set through the `Integrator` constructor.

### Hard-core clashes

A proposal that puts a pair under its hard-core cutoff is rejected before the
Metropolis test runs, so no accepted move introduces an overlap.

A rigid pivot does not re-check the pairs it carries (both atoms moved): the
rotation keeps their distances, which is what makes the move cheap. It rounds
each carried coordinate to float, though, so a pair a move left exactly on its
cutoff can drift a few 1e-6 Å under it. A whole state is therefore judged --
by the full recompute, `Context.has_steric_clash()` and the check below --
against cutoffs 0.001 Å looser (`mcpu_core.STATE_CLASH_BUFFER_A`), and a pair
inside that margin scores as any pair at its distance. With nothing
re-checking carried pairs, none went more than 1.8e-6 Å under its cutoff in
5M-step actin and 20M-step chignolin runs. The KORP CA-CA guard works the
same way. (The legacy `MCPU_FAST_MU_DELTA=OFF` build of Mu has one exact
cutoff for both.)

A clash in an accepted state therefore means coordinates that did not come
from a move, or a pair a delta path missed. `Simulation` checks
`Context.has_steric_clash()` on its periodic full-energy recompute and raises
`StericClashError` by default. Setting `MCPU_CLASH_FATAL=0` downgrades this to
a counted warning; continuing costs one exchange attempt made with a slightly
stale energy.

Overlaps in the structure a force field is built from are handled up front:
Mu exempts those pairs for the whole run (native-structure exceptions), and
`KORPForceField` refuses such a structure. Coordinates set later with an
overlap (`set_positions`, a restore) fail the check above; with
`MCPU_CLASH_FATAL=0` the run goes on, but every move that moves one atom of
an overlapping pair and leaves it overlapping is rejected (moves that leave the
pair alone, or carry both atoms rigidly, are not), so relax or repair the
structure before sampling.

### Q-bias native contacts

`NativeContactsBiasPotential` builds the native CA–CA contact list at
`Context` construction. The minimum sequence separation
`min_seq_sep` is configurable (default **5**, matching the
historical pyMCPU convention). Legacy MCPU `template.cfg`
uses **8** — set `NativeContactsBiasPotential(min_seq_sep=8)` when matching
a legacy run.
