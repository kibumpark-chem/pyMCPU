"""Legacy MCPU's per-energy-group "outer" energy weights, and pyMCPU's
"legacy weights" feature that reproduces them.

Legacy MCPU (``dbfold_actin/MCPU/src_mpi_umbrella/define.h``) hardcodes a
fixed outer weight per energy term at compile time:

    POTNTL_WEIGHT 0.4    -> Mu / contact              -> energy group 1
    TOR_WEIGHT    1.35   -> backbone torsion           -> energy group 2
    SCT_WEIGHT    2.50   -> sidechain torsion          -> energy group 3
    HBOND_WEIGHT  1.35   -> HBond outer weight   \\
    RDTHREE_CON   2.0    -> applied inside HydrogenBonds() -> together, group 4
    ARO_WEIGHT    5.0    -> aromatic                   -> energy group 5

(``hbonds.h``'s ``HydrogenBonds``/``FoldHydrogenBonds``: ``E_hbond =
(table_sum / 1000) * RDTHREE_CON``, then the outer ``HBOND_WEIGHT`` is
applied on top in ``backbone.c``/``energy.h`` -- so group 4's effective
weight is ``HBOND_WEIGHT * RDTHREE_CON = 1.35 * 2.0 = 2.7``.)

pyMCPU ports these as compile-time constants in
``include/pymcpu/EnergyWeights.h`` (``kLegacyMu``/``kLegacyBbTor``/etc.),
enabled by default (``use_legacy_weights() == True``). This file asserts
those two independently-maintained constant tables still agree.
"""

from __future__ import annotations

import pytest

# NOT slow-marked. These are six transcribed constants and the whole file
# runs in milliseconds -- it has no business being deselected by the default
# addopts. It was in the slow tier while being the ONLY direct test of the
# published energy weights, so `pytest -q` never checked them.
# (no pytestmark: nothing here needs deselecting)

from tests.fixtures.context_builders import ATOL
from tests.legacy_parity.framework import LegacyReference, ParityTolerance, assert_legacy_parity

_DEFINE_H = "dbfold_actin/MCPU/src_mpi_umbrella/define.h"

# Both compile-time constants, ported by literal value (not by any float
# computation on either side), so they must agree exactly -- this is exactly
# what ParityTolerance.BITWISE is for.
_REASON = (
    "compile-time weight constant copied by literal value from legacy's "
    "define.h into pyMCPU's EnergyWeights.h; no computation on either side "
    "that could introduce float32 accumulation or binning-edge drift"
)

LEGACY_GROUP_WEIGHTS = {
    1: LegacyReference(
        value=0.4, source=f"{_DEFINE_H}:24 POTNTL_WEIGHT",
        tolerance=ParityTolerance.BITWISE, reason=_REASON,
    ),
    2: LegacyReference(
        value=1.35, source=f"{_DEFINE_H}:26 TOR_WEIGHT",
        tolerance=ParityTolerance.BITWISE, reason=_REASON,
    ),
    3: LegacyReference(
        value=2.50, source=f"{_DEFINE_H}:27 SCT_WEIGHT",
        tolerance=ParityTolerance.BITWISE, reason=_REASON,
    ),
    4: LegacyReference(
        value=2.7,
        source=f"{_DEFINE_H}:25,40 HBOND_WEIGHT (1.35) * RDTHREE_CON (2.0)",
        tolerance=ParityTolerance.BITWISE,
        reason=(
            "HBond's effective outer weight is the product of two legacy "
            "constants applied at two different points (backbone.c's "
            "outer weighting and hbonds.h's inline RDTHREE_CON scaling); "
            + _REASON
        ),
    ),
    5: LegacyReference(
        value=5.0, source=f"{_DEFINE_H}:28 ARO_WEIGHT",
        tolerance=ParityTolerance.BITWISE, reason=_REASON,
    ),
}

LEGACY_RDTHREE_CON = LegacyReference(
    value=2.0, source=f"{_DEFINE_H}:40 RDTHREE_CON",
    tolerance=ParityTolerance.BITWISE, reason=_REASON,
)


def test_legacy_weights_enabled_by_default(chignolin_context) -> None:
    assert chignolin_context.use_legacy_weights() is True


def test_get_energy_weights_matches_legacy_constants(chignolin_context) -> None:
    w = chignolin_context.get_energy_weights()
    for group, ref in LEGACY_GROUP_WEIGHTS.items():
        assert_legacy_parity(w[group], ref)
    assert_legacy_parity(w["hbond_rdthree"], LEGACY_RDTHREE_CON)


def test_weighted_energy_scales_raw_energy_by_legacy_factors(chignolin_context) -> None:
    """calculate_total_energy_raw stays unweighted regardless of the
    use_legacy_weights toggle; calculate_total_energy(group) scales it by
    exactly the legacy factor above when legacy weighting is on."""
    ctx = chignolin_context
    ctx.set_use_legacy_weights(False)
    ctx.calculate_total_energy(-1)

    raw_by_group = {g: float(ctx.calculate_total_energy_raw(g)) for g in range(1, 6)}
    raw_total = float(ctx.calculate_total_energy_raw(-1))
    assert sum(raw_by_group.values()) == pytest.approx(raw_total, abs=ATOL)

    # Unweighted: total == raw.
    assert float(ctx.calculate_total_energy(-1)) == pytest.approx(raw_total, abs=ATOL)

    ctx.set_use_legacy_weights(True)
    for group, ref in LEGACY_GROUP_WEIGHTS.items():
        weighted = float(ctx.calculate_total_energy(group))
        assert weighted == pytest.approx(raw_by_group[group] * ref.value, abs=ATOL)
        # Raw API is unaffected by the weighting toggle.
        assert float(ctx.calculate_total_energy_raw(group)) == pytest.approx(
            raw_by_group[group], abs=ATOL
        )

    expected_weighted = sum(raw_by_group[g] * ref.value for g, ref in LEGACY_GROUP_WEIGHTS.items())
    assert float(ctx.calculate_total_energy(-1)) == pytest.approx(expected_weighted, abs=ATOL)

    bd_raw = ctx.energy_breakdown(weighted=False)
    bd_w = ctx.energy_breakdown(weighted=True)
    assert bd_raw["raw_total"] == pytest.approx(raw_total, abs=ATOL)
    assert bd_w["weighted_total"] == pytest.approx(expected_weighted, abs=ATOL)
