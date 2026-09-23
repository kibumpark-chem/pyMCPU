"""Shared numeric constants for the ``tests/physics`` suite.

Centralized here so sibling test files that both reason about "is this
energy a steric-clash sentinel" don't hardcode two different, silently
drifting literals for the same underlying concept (this happened before the
rewrite: ``test_force_energy.py`` used ``99999.0`` while
``test_energy_mask.py`` used ``90000.0`` for what was conceptually the same
check).
"""

from __future__ import annotations

#: MuPotential's hard-core steric-clash sentinel energy
#: (``kHardCorePenalty`` in ``src/pymcpu/forces/mcpu/mcpu08/MuPotential.cpp``).
#: A raw Mu energy at or above this value means a clash was detected.
MU_CLASH_SENTINEL = 99999.0

#: Safety margin used by "this should NOT be a clash" assertions. Kept
#: strictly below ``MU_CLASH_SENTINEL`` (rather than reusing it directly)
#: so those assertions still pass if the sentinel's exact value ever shifts
#: by a small amount -- what they actually care about is "far below clash
#: territory", not bit-equality with the sentinel.
MU_CLASH_SAFETY_MARGIN = 90000.0
