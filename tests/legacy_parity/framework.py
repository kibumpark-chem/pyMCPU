"""The one, official way this suite compares pyMCPU's numbers against
legacy MCPU (the C/MPI reference implementation at
``dbfold_actin/MCPU/src_mpi_umbrella/``).

A legacy comparison can fail for two very different reasons: the physics is
actually wrong, or the two implementations legitimately compute the same
physics slightly differently (float32 vs. legacy's accumulation order, a
discretized lookup table's boundary rounding, a small already-investigated
residual). Collapsing both into one unexplained ``abs=1e-2`` invites exactly
the failure mode this project has hit before: a real regression absorbed by
too loose a tolerance, or a benign rounding difference chased as a bug.

Every comparison here must state its :class:`ParityTolerance` tier and, in
``reason``, *why* that tier applies -- so a reviewer (or a future engineer
staring at a failure) can tell at a glance which kind of mismatch they're
looking at.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import pytest


class ParityTolerance(Enum):
    """Named tolerance tiers. Pick the tightest one that's actually true --
    do not reach for a looser tier just to make a test pass."""

    #: Bitwise identical as float32. The ONLY tier with no slack at all:
    #: ``struct.pack('f', actual) == struct.pack('f', ref)``. Use it for
    #: constants that are transcribed rather than computed -- weights,
    #: cutoffs, table dimensions -- where any difference is a transcription
    #: error, not float noise.
    BITWISE = auto()

    #: Same computation, different implementation -- agrees to within 1e-4
    #: absolute.
    #:
    #: NOTE the name is historical and overstates what this tier does. It is
    #: an ABSOLUTE 1e-4, which at float32 magnitudes is very loose: 1e-4
    #: around 0.4f spans 3,355 ULP, and around 5.0f spans 210 ULP. A
    #: transcription error like 0.4 -> 0.40001 passes it. If you are
    #: comparing a transcribed constant rather than a computed quantity, use
    #: ``BITWISE``.
    EXACT = auto()

    #: The comparison sums many small floating-point terms in a different
    #: order than legacy (e.g. neighbor-iteration order, SIMD lane grouping).
    #: float32 addition isn't associative, so a small residual is expected
    #: and does not indicate a physics difference.
    FLOAT32_ACCUMULATION = auto()

    #: The energy term is looked up from a discretized table (angle bins,
    #: distance bins). A value sitting near a bin edge can land in an
    #: adjacent bin under a tiny coordinate/rounding difference between the
    #: two implementations, without either being "wrong".
    BINNING_EDGE = auto()

    #: A specific, already-investigated discrepancy with a known, bounded
    #: magnitude that isn't worth chasing further (documented in ``reason``
    #: with a pointer to the investigation). Not a catch-all -- if you can't
    #: point to the investigation, the mismatch isn't understood yet and
    #: this is the wrong tier to reach for.
    KNOWN_RESIDUAL = auto()


_ATOL = {
    # BITWISE is deliberately absent: it is not an atol comparison at all.
    # Putting a number here would silently turn it back into one.
    ParityTolerance.EXACT: 1e-4,
    ParityTolerance.FLOAT32_ACCUMULATION: 1e-2,
    ParityTolerance.BINNING_EDGE: 5e-2,
    ParityTolerance.KNOWN_RESIDUAL: 1e-1,
}


@dataclass(frozen=True)
class LegacyReference:
    """One pinned legacy MCPU reference value, with mandatory provenance.

    ``source`` must be traceable: a file:line in ``dbfold_actin``, or a
    literal quoted line from a legacy log/output file. "I computed this once
    and it looked right" is not a source.
    """

    value: float
    source: str
    tolerance: ParityTolerance
    reason: str


def assert_legacy_parity(actual: float, ref: LegacyReference) -> None:
    """Assert ``actual`` (computed by pyMCPU) matches ``ref`` within its
    declared tolerance tier. Fails with the full provenance and reason
    inline, so a failure is diagnosable from the pytest output alone."""
    if ref.tolerance is ParityTolerance.BITWISE:
        import struct

        a = struct.pack("f", actual)
        b = struct.pack("f", ref.value)
        assert a == b, (
            f"{actual!r} is not BITWISE equal to legacy reference "
            f"{ref.value!r}\n"
            f"  actual float32 bits: {a.hex()}\n"
            f"  legacy float32 bits: {b.hex()}\n"
            f"  source: {ref.source}\n"
            f"  reason: {ref.reason}"
        )
        return

    atol = _ATOL[ref.tolerance]
    assert actual == pytest.approx(ref.value, abs=atol), (
        f"{actual!r} != legacy reference {ref.value!r}\n"
        f"  source: {ref.source}\n"
        f"  tolerance: {ref.tolerance.name} (atol={atol})\n"
        f"  reason: {ref.reason}"
    )
