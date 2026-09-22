# KIC loop-closure Jacobian

**Python API:** KIC moves are selected via
:class:`~pymcpu.Integrator` move probabilities
(see :doc:`/api/integrator` and :doc:`mc_acceptance`).

Temperature $T$ in the Metropolis factor below is a
**dimensionless** reduced parameter (typical 0.3–0.6).

## Overview

Kinematic closure (KIC) in pyMCPU closes a three-residue backbone fragment by solving
for internal half-angle tangents that satisfy distance constraints, then accepting or
rejecting the move with a Jacobian correction for detailed balance.

The implementation lives in header-only `mcpu::kic` kernels (LOCK-6):

| File | Role |
|------|------|
| `physics/integrators/kic/tripeptide_closure.hpp` | Polynomial assembly, Sturm root finding, coordinate reconstruction |
| `physics/integrators/kic/sturm_solver.hpp` | Graphics Gems Sturm-sequence root bracketing (float) |
| `physics/integrators/kic/jacobian.hpp` | Loop-closure acceptance Jacobian |

## Connectivity

Atoms along the tripeptide path:

```
N1 — A1 — C1 — N2 — A2 — C2 — N3 — A3 — C3
```

Closure inputs are the fixed anchors `N1, A1, A3, C3`. The solver reconstructs
intermediate backbone atoms for each real root of a degree-16 polynomial in the
half-angle tangent `t0`.

## Algorithm pipeline

1. **`initialize_loop_closure(b_len, b_ang, t_ang)`** — store ideal bond lengths,
   angles, and peptide torsions; precompute virtual-bond limits `aa13_min_sqr` /
   `aa13_max_sqr`.
2. **`get_input_angles`** — from anchor Cartesian coordinates, compute internal
   angles `xi`, `eta`, `delta`, `alpha`, `theta` and verify cone intersection
   feasibility.
3. **`get_poly_coeff`** — build the 16th-degree closure polynomial via determinant
   expansion (Seok 2003).
4. **`solve_sturm`** — isolate real roots using a Sturm sequence (Hook & McAree,
   Graphics Gems 1990). Float tolerances: `SMALL=1e-10f`, `RELERROR=1e-6f`.
5. **`coord_from_poly_roots`** — for each root, recover `(t0,t2,t1)`, build residue
   coordinates, and rotate into the global frame.
6. **`loop_jacobian`** — compute `1/|det J|` for the concerted six-torsion move;
   singular determinants trigger one deterministic LCG rotation retry.

## Jacobian structure

`loop_jacobian` follows Dinner 2000 / Dobbs Appendix B.6–B.9:

- Six rotation axes: φ and ψ for each of three residues (N→CA and CA→C directions).
- Rows 1–3: `axis × (r_CA3 − pivot)` for the first four axes.
- Rows 4–5: selected components of `axis × (r_CA3→C3)`; if the CA3→C3 vector is
  parallel to the z-axis (`|z| < 1e-10`), use x and z components instead of x and y
  to avoid linear dependence.
- Row 6 is implicitly `[0,0,0,0,0,-1]`; the determinant is evaluated via a 4×4
  cofactor expansion on the upper-left block (`det3` helper, no Eigen).

The Jacobian ratio enters the Metropolis criterion
multiplicatively:

```
acceptance_crit = exp(-delta_E / T) * jacobian_ratio
where:
  jacobian_ratio = (jacobi_after / jacobi_before)
                   × (n_soln / soln_no_before)
```

`jacobi_after` and `jacobi_before` are each computed as
`1/|det(J)|` by `loop_jacobian()`. Their ratio therefore
represents `|det(J_before)| / |det(J_after)|`, which
is the standard KIC detailed balance correction.
Non-positive Jacobians reject the proposal.

Note: `log_jacobian` was an earlier field name in this
codebase. It was renamed to `jacobian_ratio` to avoid
the implication that a log was taken.

## Numerical policy

| Quantity | Tolerance vs legacy double dump |
|----------|----------------------------------|
| Polynomial coefficients | rel_error < 1e-4 |
| Sturm roots | rel_error < 1e-4 |
| Jacobian values / ratio | rel_error < 1e-3 |

Float Sturm settings intentionally relax secant/bisection tolerances relative to legacy
`1e-15` double settings to match production `float` KIC kernels.

### Float precision: measured errors

Float precision was validated on a reference α-helix geometry
(condition number ~8.15×10²). Measured errors vs double:

| Quantity | Error |
|----------|-------|
| Polynomial coefficients | max \|float − double\| = 2.93×10⁻⁶ |
| Root values | max \|float − double\| = 1.69×10⁻⁶ |
| Jacobian determinant | relative error = 1.77×10⁻⁶ |

All errors are well below the regression tolerance of 1×10⁻⁴.
Float is demonstrably safe on well-conditioned geometries.
For unusual backbone geometries (tight turns, near-degenerate
closures), double precision may be needed — see `known_issues.md`.

### Eigen PolynomialSolver evaluation

`Eigen::PolynomialSolver<float, 16>` was evaluated as a
replacement for the Sturm solver. On the reference polynomial:

| Solver | Timing |
|--------|--------|
| Eigen | 56,237 ns/call |
| Sturm | 19,004 ns/call |

Eigen is ~3× slower with identical root count and was rejected
on performance grounds. The Sturm solver (Graphics Gems 1990,
modified by Seok 2003) remains the implementation.

### no-fast-math pragma rationale

All three KIC files (`sturm_solver.hpp`, `jacobian.hpp`,
`tripeptide_closure.hpp`) are wrapped in:

```
#pragma GCC push_options
#pragma GCC optimize("O2")
#pragma GCC optimize("no-fast-math")
...
#pragma GCC pop_options
```

The Sturm sequence sign-count algorithm and the Jacobian
cofactor expansion require strict IEEE 754 arithmetic.
`-ffast-math` can reorder floating-point operations in ways
that produce incorrect sign changes in the Sturm sequence.
The `_pymcpu_core` extension intentionally omits `-ffast-math`
for the same reason; the per-file pragma additionally protects
these headers if they are included from a TU compiled with
aggressive FP flags (e.g. MSVC `/fp:fast` on legacy targets).

## Random selection

When multiple roots exist, `select_solution_index(n_soln, seed)` picks a branch
deterministically from a fractional seed (unit tests / reproducible benchmarks).
Production MC uses `std::uniform_int_distribution` over valid solutions.

## References

- Chaok Seok, Evangelos Coutsias, Matthew Jacobson, Ken Dill (2003) tripeptide closure
- D.G. Hook, P.R. McAree, Graphics Gems (1990) Sturm solver
- A.R. Dinner et al., J. Comput. Chem. (2000) — Jacobian for loop moves
