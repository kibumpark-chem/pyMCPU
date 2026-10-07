# KIC loop closure

**Python API:** the share of KIC moves is set with
`Integrator.set_move_weights`; see {doc}`/api/integrator`.

A KIC (kinematic closure) move changes the backbone of a window of three
residues, r, r+1 and r+2, while the chain on both sides stays put. The
window's bond lengths, bond angles and peptide ω angles stay at the starting
structure's values; only its six φ and ψ torsions change.

## The move

1. Pick a residue r and a driver: the φ of residue r+3 or the ψ of residue
   r−1, with equal probability. The window must fit in the chain:
   1 ≤ r ≤ n−4 for a φ driver, 2 ≤ r ≤ n−3 for a ψ driver.
2. Find every closure of the current window: the positions of C(r), N(r+1),
   CA(r+1), C(r+1) and N(r+2) that join the fixed N(r), CA(r) to CA(r+2),
   C(r+2) with the window's bond lengths and angles. Closures are the real
   roots of a degree-16 polynomial (Coutsias et al. 2004), so there are at
   most 16. Call their number n_old.
3. Rotate the driver by a Gaussian angle of width `step_size_rad`. This moves
   one end of the window: CA(r+2) and C(r+2) for a φ driver, N(r) and CA(r)
   for a ψ driver.
4. Find every closure for the new end, n_new of them, and pick one uniformly
   at random.
5. Accept or reject with w = (J_new / J_old) × (n_new / n_old); see
   {doc}`mc_acceptance`.

The sidechains of the three residues move rigidly with their backbone.

## Why the weight

The closure maps the six window torsions to the position and orientation of
the window's far end. For detailed balance, a move between closures is
weighted by the inverse of that map's Jacobian determinant, J = 1/|det M|. M is
the 6 × 6 matrix whose columns are the twists (u_i, (p_i − o) × u_i) of the six
torsion axes: u_i along the axis, p_i a point on it, o = CA(r). J does not
depend on how the molecule sits in space. The factor n_new / n_old corrects
for picking one of n_new closures, where the reverse move would pick one of
n_old.

## When a move is refused

A KIC attempt is refused, and counts as an attempt that was not accepted,
when:

- the window does not fit, or would change the φ of a proline (residues r to
  r+3 for a φ driver, r to r+2 for a ψ driver);
- the current window has no closure, or is not one of its own closures to
  within 0.001 Å, so the move could not be reversed (float32 rounding causes
  this in a small fraction of attempts);
- the new end has no closure;
- a Jacobian is not finite and positive (|det M| < 1e-10).

A window that would move a fixed residue is not refused but never drawn: the
move draws its window and driver again (see `Integrator.set_fixed_residues`).

A closure whose three N–CA–C angles miss their targets by more than 1e-6 rad
is discarded, in both solves. {doc}`/api/integrator` lists counters for most
of these cases.

## Reference

E. A. Coutsias, C. Seok, M. P. Jacobson and K. A. Dill, "A kinematic view of
loop closure", *J. Comput. Chem.* 25, 510–528 (2004),
https://doi.org/10.1002/jcc.10416.
