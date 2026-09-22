"""Basic sanity checks on a freshly-built ``Context``, independent of any
one potential's formula. Kept separate from the delta/full energy-consistency
tests (``test_energy_conservation.py``) since this is a much cheaper,
narrower "did construction even produce something physically sane" check.
"""

from __future__ import annotations

import pytest

# Building ``chignolin_context`` constructs a real forcefield/neighbor list
# from a full PDB (actin by default, see tests/fixtures/context_builders.py)
# -- slow for the same reason every other file that uses this fixture is.
pytestmark = pytest.mark.slow


def test_folded_structure_total_energy_is_negative(chignolin_context) -> None:
    energy = chignolin_context.calculate_total_energy(-1)
    # Not a pinned numeric target: a correctly-folded, low-energy reference
    # structure should have net-favorable (negative) total energy under the
    # forcefield. Any specific number would be incidental to the test PDB.
    assert energy < 0.0
