"""End-to-end H-bond legacy parity through the *compiled* engine
(``pymcpu.mcpu_core``), as opposed to ``test_hbond_ablation_ladder.py``'s
pure-Python oracle checks.

Kept in its own file so that a stale or ABI-mismatched compiled extension
(e.g. missing a method added later in the same source tree) produces a
clean, explicit skip here rather than an opaque ``AttributeError`` that
would otherwise also take down the fast oracle-only tests in the same
module.

This is a plain numeric assertion, not an ``xfail``: it fails outright if
the compiled engine's H-bond energy drifts from the legacy target, so
update the target here once a verified fix lands rather than letting this
go stale.
"""

from __future__ import annotations

from pathlib import Path

import mdtraj as md
import numpy as np
import pytest

from tests.legacy_parity.framework import LegacyReference, ParityTolerance, assert_legacy_parity
from tests.legacy_parity.helpers.legacy_hbond_oracle import HBOND_WEIGHT

REPO_ROOT = Path(__file__).resolve().parents[2]
ACTA_PDB = REPO_ROOT / "examples" / "actin" / "input_pdb" / "acta.pdb"
LEGACY_LOG = REPO_ROOT / "examples" / "actin" / "legacy" / "outputs" / "acta_T_0.600.log"

# Same legacy citation as test_hbond_ablation_ladder.py's Stage 3 -- see that
# file's module docstring for why this is 2-decimal-precision-limited rather
# than a higher-precision figure.
LEGACY_HBOND_LOG_WEIGHTED = -128.12 * HBOND_WEIGHT


def _measure_engine_hbond_energy(pdb_path: Path) -> float:
    """Build a heavy-atom-only ``Context`` for ``pdb_path`` and return the
    engine's weighted group-4 (hydrogen_bond) energy at frame 0."""
    from pymcpu import mcpu_core
    from pymcpu.forcefields.mcpu import MCPUForceField

    traj = md.load(str(pdb_path))
    heavy = traj.atom_slice(traj.topology.select("not element H"))
    forcefield = MCPUForceField(heavy, param_set="mcpu08")
    system = forcefield.create_system(heavy.topology)
    context = mcpu_core.Context(system)
    context.set_positions((forcefield.coords[0] * 10.0).T.astype(np.float32))
    context.calculate_total_energy(-1)
    return context.calculate_total_energy(4)  # group 4 = hydrogen_bond


def test_engine_matches_legacy_log() -> None:
    """The compiled engine's group-4 energy on acta.pdb must match the same
    legacy-log-derived reference as the oracle's full-parity stage."""
    try:
        e_hbond = _measure_engine_hbond_energy(ACTA_PDB)
    except AttributeError as exc:
        pytest.skip(
            "compiled pymcpu.mcpu_core extension is missing an API this "
            f"test needs (stale build vs. current source tree?): {exc}"
        )
    assert_legacy_parity(
        e_hbond,
        LegacyReference(
            value=LEGACY_HBOND_LOG_WEIGHTED,
            source=(
                f"{LEGACY_LOG.relative_to(REPO_ROOT)}, STEP 0: "
                "-128.12(hbond) * 1.35(HBOND_WEIGHT) = -172.962"
            ),
            tolerance=ParityTolerance.KNOWN_RESIDUAL,
            reason=(
                "same 2-decimal log-precision bound as the oracle's Stage 3 "
                "(test_hbond_ablation_ladder.py), plus this path adds a "
                "float32 compiled engine vs. float64 NumPy oracle -- both "
                "comfortably inside KNOWN_RESIDUAL's atol=1e-1."
            ),
        ),
    )
