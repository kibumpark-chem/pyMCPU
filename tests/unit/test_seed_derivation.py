"""``pymcpu.sampling.derive_seed`` -- a frozen wire format.

The values below are **literals on purpose**. ``derive_seed`` determines the
Monte Carlo stream of every cloned trajectory any external sampler runs, so
changing the packing or the truncation silently reseeds every run ever done
with pyMCPU: the new results stay perfectly plausible and stop being
comparable to anything published from an earlier version.

That makes this the opposite of a characterization test. If it fails, the
answer is almost never "re-record the expected values" -- it is that a
change to the derivation needs to be a deliberate, documented, versioned
decision.
"""

from __future__ import annotations

import pytest

from pymcpu.sampling import derive_seed

# Captured from the implementation before it moved out of the WESTPA
# integration into core, and verified identical across 378 input triples at
# the time of the move. Pairs that differ in one argument only (stream 10/11,
# base 12345/0, round 0/1) also pin that each argument changes the seed.
_FROZEN = {
    (12345, 0, 0): 3457032058,
    (12345, 3, 5): 4294913211,
    (99, 5, 10): 4178159205,
    (99, 5, 11): 763092638,
    (0, 0, 0): 3482226845,
    (2026, 0, 0): 1198683059,
    (2026, 1, 0): 1034913220,
    # Negative stream indices are allowed: a sampler can number generated
    # initial states -1, -2, ...
    (7, 0, -1): 1331302366,
    (7, 0, -2): 1737723508,
}


@pytest.mark.parametrize(("args", "expected"), sorted(_FROZEN.items()))
def test_derive_seed_matches_its_frozen_values(args: tuple, expected: int) -> None:
    assert derive_seed(*args) == expected


def test_derive_seed_fits_uint32() -> None:
    """``Integrator.set_seed`` takes an unsigned 32-bit value."""
    for base in (0, 1, 42, 2**31, 2**62):
        for round_index in range(4):
            for stream_index in range(-2, 4):
                seed = derive_seed(base, round_index, stream_index)
                assert isinstance(seed, int)
                assert 0 <= seed < 2**32
