"""Numeric parity ladder between pyMCPU's H-bond potential and the
legacy MCPU reference (``dbfold_actin/MCPU/src_mpi_umbrella/hbonds.h``).

This file is oracle-only: every assertion runs
``tests/legacy_parity/helpers/legacy_hbond_oracle.py`` (an independent NumPy
transliteration of the legacy C++ formula that reads the legacy *text*
parameter files directly, not pyMCPU's compiled ``.bin`` tables) and never
touches the compiled ``pymcpu.mcpu_core`` extension -- that comparison lives
in ``test_hbond_engine_parity.py`` so a stale/mismatched compiled build can't
make these fast, pure-Python checks fail too.

Two different kinds of numbers appear below and must not be confused:

* Stages A/1/2 are *not* legacy values. They pin pyMCPU's own historical
  (buggy) behavior at each step of the fix -- see ``docs/hbond_legacy_parity.md``
  -- as fixed regression markers of "what the code used to do". They are
  plain ``pytest.approx`` pins, deliberately not run through
  ``assert_legacy_parity`` since there is nothing legacy to cite for them.
* ``test_oracle_matches_legacy_log`` and Stage 3 (full parity) *are* genuine
  legacy comparisons and go through ``assert_legacy_parity`` with an explicit
  ``LegacyReference``.

A note on precision: the only independently-citable legacy source for the
hbond energy is a completed legacy run's log (``acta_T_0.600.log``,
not distributed -- see docs/hbond_legacy_parity.md), which
prints the STEP-0 "hbond" column to 2 decimals only (``-128.12``) -- the
legacy C binary itself is not rebuildable in this environment (see
``scripts/validate_energy.py``'s "legacy binary may not be runnable" note).
A prior version of this file carried a 6-decimal constant
(``-128.119058``); that extra precision was not independently confirmed
against any legacy source -- it was the oracle's own output, copy-pasted
back in as if it were a tighter legacy ground truth. This rewrite cites only
the 2-decimal log value as the legacy reference and is explicit, via
``ParityTolerance.KNOWN_RESIDUAL``, about the resulting slack.
"""

from __future__ import annotations

from pathlib import Path

import mdtraj as md
import pytest

from tests.legacy_parity.framework import LegacyReference, ParityTolerance, assert_legacy_parity
from tests.legacy_parity.helpers.legacy_hbond_oracle import (
    HBOND_WEIGHT,
    legacy_hbond_energy,
    load_jpl3h_table,
    load_seq_dep_table,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
LEGACY_CFG = REPO_ROOT / "examples" / "actin" / "legacy" / "config_files"
ACTA_PDB = REPO_ROOT / "examples" / "actin" / "input_pdb" / "acta.pdb"
LEGACY_LOG = REPO_ROOT / "examples" / "actin" / "legacy" / "outputs" / "acta_T_0.600.log"

# Legacy run acta_T_0.600.log (not distributed), line 502, STEP 0's "hbond"
# column -- confirmed genuine (not oracle-derived) by reconciling the same
# line's printed weighted total: 0.40*(-320.34) + 5.00*(1.58) + 1.35*(-128.12)
# + 1.35*(-29.20) + 2.50*(-85.56) = -546.518, matching the printed
# ENERGY = -546.51 to the log's own 2-decimal precision.
LEGACY_HBOND_LOG_RAW = -128.12
LEGACY_HBOND_LOG_WEIGHTED = LEGACY_HBOND_LOG_RAW * HBOND_WEIGHT

# The log's 2-decimal print rounds the true raw value to within +/-0.005;
# 1.35x that is +/-0.00675 on the weighted value. KNOWN_RESIDUAL's atol=1e-1
# comfortably covers this bounded, understood slack without pretending to a
# precision the source doesn't have.
_LOG_PRECISION_REASON = (
    "the only independently-citable legacy source is the printed log line "
    f"({LEGACY_LOG.relative_to(REPO_ROOT)}, STEP 0), which prints the raw "
    "hbond energy to 2 decimals only; the legacy C binary is not rebuildable "
    "in this environment to get a higher-precision figure (see "
    "scripts/validate_energy.py). This bounds the achievable agreement to "
    "~0.01, not a physics or implementation difference."
)


# The legacy MCPU input tables are NOT distributed with pyMCPU. They are
# inputs to a C binary that is not in this repository and is not rebuildable
# here, so shipping them would look like reproducibility evidence while
# being unusable -- and three of the four `jPL3h*` variants differ only in
# parameterization, so a reader could not tell which produced the cited
# number. They are retained in the maintainer's archive.
#
# Everything these five tests assert about pyMCPU's OWN H-bond energy is
# pinned without them by test_hbond_engine_parity.py and
# test_legacy_weight_constants.py, which need no fixtures. What is lost here
# is only the re-derivation of the legacy side.
_LEGACY_INPUTS = [
    LEGACY_CFG / "jPL3h.energy",
    LEGACY_CFG / "seq_dep_hb_mu_low.energy",
]
_missing = [p for p in _LEGACY_INPUTS if not p.is_file()]
pytestmark = pytest.mark.skipif(
    bool(_missing),
    reason=(
        "legacy MCPU input tables are not distributed: "
        + ", ".join(str(p.relative_to(REPO_ROOT)) for p in _missing)
        + ". See docs/hbond_legacy_parity.md for the recorded values and how "
        "they were reconciled."
    ),
)


@pytest.fixture(scope="module")
def acta_traj() -> md.Trajectory:
    return md.load(str(ACTA_PDB))


@pytest.fixture(scope="module")
def jpl3h_table():
    return load_jpl3h_table(LEGACY_CFG / "jPL3h.energy")


@pytest.fixture(scope="module")
def seq_dep_table():
    return load_seq_dep_table(LEGACY_CFG / "seq_dep_hb_mu_low.energy")


def test_oracle_matches_legacy_log(acta_traj, jpl3h_table, seq_dep_table) -> None:
    """The oracle, with every correction enabled and secstr=all-'C' (acta's
    actual .sec_str), must reproduce the legacy log's printed value."""
    e, n_bonds = legacy_hbond_energy(acta_traj, jpl3h_table, seq_dep_table)
    assert_legacy_parity(
        e,
        LegacyReference(
            value=LEGACY_HBOND_LOG_RAW,
            source=f"{LEGACY_LOG.relative_to(REPO_ROOT)}, STEP 0, 'hbond' column = -128.12",
            tolerance=ParityTolerance.KNOWN_RESIDUAL,
            reason=_LOG_PRECISION_REASON,
        ),
    )
    # Regression baseline: count of geometry-valid H-bonds the oracle finds
    # for acta.pdb (not printed directly in the legacy log); refresh only if
    # the H-bond geometry/gate criteria or the input structure change.
    assert n_bonds == 113


def test_oracle_ablation_stage_a_matches_current_pymcpu(acta_traj, jpl3h_table) -> None:
    """Stage A: no seq_dep, no rama gates, no beta_favor, buggy ang_CACA --
    i.e. an emulation of pyMCPU's behavior *before* any fix in this
    workstream. Fixed historical reference point (not a legacy value --
    do not re-derive it from dbfold_actin), also recorded in CHANGELOG.md
    and docs/hbond_legacy_parity.md's ablation table."""
    e, _ = legacy_hbond_energy(
        acta_traj, jpl3h_table, seq_dep_table=None,
        caca_fixed=False, rama_gates=False, seq_dep=False, beta_favor=False,
    )
    # abs=1e-2: this is a pin of the oracle's own current output, not a
    # legacy comparison -- tight tolerance is appropriate and achievable.
    assert e * HBOND_WEIGHT == pytest.approx(-216.4995, abs=1e-2)


def test_oracle_ablation_stage_1_ang_caca_fix(acta_traj, jpl3h_table) -> None:
    """Stage 1: + fix ang_CACA (units + transposed operands). Intermediate
    diagnostic point on the fix ladder, matching docs/hbond_legacy_parity.md
    row 1 -- not itself a legacy value."""
    e, _ = legacy_hbond_energy(
        acta_traj, jpl3h_table, seq_dep_table=None,
        caca_fixed=True, rama_gates=False, seq_dep=False, beta_favor=False,
    )
    assert e * HBOND_WEIGHT == pytest.approx(-224.0406, abs=1e-2)


def test_oracle_ablation_stage_2_rama_gates(acta_traj, jpl3h_table) -> None:
    """Stage 2: + Ramachandran gates + correct 'H'-gate placement, matching
    docs/hbond_legacy_parity.md row 2 -- still not a legacy value (that's
    Stage 3, below)."""
    e, n_bonds = legacy_hbond_energy(
        acta_traj, jpl3h_table, seq_dep_table=None,
        caca_fixed=True, rama_gates=True, seq_dep=False, beta_favor=False,
    )
    assert e * HBOND_WEIGHT == pytest.approx(-217.1853, abs=1e-2)
    # Same oracle H-bond-count baseline as test_oracle_matches_legacy_log
    # (rama_gates doesn't change which pairs are geometry-valid here).
    assert n_bonds == 113


def test_oracle_ablation_stage_3_full_parity(acta_traj, jpl3h_table, seq_dep_table) -> None:
    """Stage 3: + sequence-dependent scaling + beta_favor = full legacy
    parity. This is the genuine legacy comparison the ablation ladder builds
    up to."""
    e, _ = legacy_hbond_energy(
        acta_traj, jpl3h_table, seq_dep_table,
        caca_fixed=True, rama_gates=True, seq_dep=True, beta_favor=True,
    )
    assert_legacy_parity(
        e * HBOND_WEIGHT,
        LegacyReference(
            value=LEGACY_HBOND_LOG_WEIGHTED,
            source=(
                f"{LEGACY_LOG.relative_to(REPO_ROOT)}, STEP 0: "
                "-128.12(hbond) * 1.35(HBOND_WEIGHT) = -172.962"
            ),
            tolerance=ParityTolerance.KNOWN_RESIDUAL,
            reason=_LOG_PRECISION_REASON,
        ),
    )
