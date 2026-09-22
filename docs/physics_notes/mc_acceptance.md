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

A proposal whose delta-energy path detects a hard-core overlap is rejected
before the Metropolis test runs, so a clashing structure can never be
*introduced* by an accepted move.

That makes a clash in an already-accepted state an invariant violation rather
than a condition to tolerate: it means the incremental path failed to detect
something the full recompute does see. `Simulation` therefore checks
`Context.has_steric_clash()` on its periodic full-energy recompute and raises
`StericClashError` by default. Setting `MCPU_CLASH_FATAL=0` downgrades this to
a counted warning, which is what long production runs use -- the event is rare
(order one per 10^7-10^8 steps) and halting a multi-day run costs more than
one exchange attempt made with a slightly stale energy.

There is no separate "freeze pre-existing clashes" step. A starting structure
with overlapping atoms will simply reject most proposals until the overlap
relaxes; relax or repair the structure before sampling if that happens.

### Q-bias native contacts

`NativeContactsBiasPotential` builds the native CA–CA contact list at
`Context` construction. The minimum sequence separation
`min_seq_sep` is configurable (default **5**, matching the
historical pyMCPU convention). Legacy MCPU `template.cfg`
uses **8** — set `NativeContactsBiasPotential(min_seq_sep=8)` when matching
a legacy run.
