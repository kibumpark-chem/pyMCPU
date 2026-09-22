"""The tolerance tiers must actually differ. A tier that cannot fail where a
looser one passes is decorative.

Background: ``ParityTolerance.EXACT`` is an ABSOLUTE 1e-4, but its name and
docstring both read as "bitwise". At float32 magnitudes 1e-4 is very loose --
3,355 ULP around 0.4f -- so the six published energy-weight constants were
being "pinned" by a check that a 0.4 -> 0.40001 transcription error passes.
``BITWISE`` was added for transcribed constants; these tests prove the two
tiers are not the same check.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.legacy_parity.framework import (
    LegacyReference,
    ParityTolerance,
    assert_legacy_parity,
)


def _ref(value: float, tol: ParityTolerance) -> LegacyReference:
    return LegacyReference(
        value=value, source="test", tolerance=tol, reason="tier self-test"
    )


def test_bitwise_rejects_what_exact_accepts() -> None:
    """The discriminating case: a plausible transcription error."""
    typo = 0.40001  # 0.4 mistyped
    assert_legacy_parity(typo, _ref(0.4, ParityTolerance.EXACT))  # passes: 1e-4
    with pytest.raises(AssertionError, match="not BITWISE equal"):
        assert_legacy_parity(typo, _ref(0.4, ParityTolerance.BITWISE))


def test_bitwise_rejects_a_one_ulp_difference() -> None:
    """One float32 ULP of 0.4 is 2.98e-08 -- four orders inside EXACT's 1e-4."""
    mu = np.float32(0.4)
    one_ulp_up = float(np.nextafter(mu, np.float32(1.0)))
    assert one_ulp_up != float(mu)
    assert_legacy_parity(one_ulp_up, _ref(float(mu), ParityTolerance.EXACT))
    with pytest.raises(AssertionError, match="not BITWISE equal"):
        assert_legacy_parity(one_ulp_up, _ref(float(mu), ParityTolerance.BITWISE))


def test_bitwise_accepts_a_genuine_match() -> None:
    for v in (0.4, 1.35, 2.50, 5.0, 2.0, 2.7):
        assert_legacy_parity(v, _ref(v, ParityTolerance.BITWISE))


def test_bitwise_error_names_both_bit_patterns() -> None:
    """A failure must be diagnosable without a debugger."""
    with pytest.raises(AssertionError) as exc:
        assert_legacy_parity(0.40001, _ref(0.4, ParityTolerance.BITWISE))
    msg = str(exc.value)
    assert "actual float32 bits" in msg and "legacy float32 bits" in msg


def test_bitwise_has_no_atol_entry() -> None:
    """Giving BITWISE a number in _ATOL would silently make it approximate."""
    from tests.legacy_parity.framework import _ATOL

    assert ParityTolerance.BITWISE not in _ATOL
