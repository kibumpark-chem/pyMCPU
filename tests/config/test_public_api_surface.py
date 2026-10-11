"""Public API surface tests for the top-level ``pymcpu`` package.

These are pure import/``hasattr``/``callable`` checks -- no simulation is run
and no physics is exercised. They exist to catch software-engineering
regressions (a class silently un-exported, a method renamed/removed) that
would break every downstream user of the OpenMM-style API, independent of
whether the underlying physics is correct. Methods that other tests call
are not listed here; ``Simulation.describe`` is, because it is documented
and nothing else calls it.
"""

from __future__ import annotations

import pymcpu
import pytest

EXPECTED_TOP_LEVEL_EXPORTS = (
    "System",
    "Context",
    "Integrator",
    "Simulation",
    "MCPUForceField",
    "MuPotential",
    "HBondPotential",
    "TripletPotential",
    "SidechainTripletPotential",
    "AromaticPotential",
    "NativeContactsBiasPotential",
    "XtcReporter",
    "EnergyReporter",
    "SimulationReporter",
    "Reporter",
)


@pytest.mark.parametrize("name", EXPECTED_TOP_LEVEL_EXPORTS)
def test_top_level_name_is_exported(name: str) -> None:
    assert hasattr(pymcpu, name), f"pymcpu.{name} missing from public API"
    # pymcpu/__init__.py's __all__ is the authoritative API surface.
    assert name in pymcpu.__all__


def test_simulation_has_describe_method() -> None:
    assert callable(getattr(pymcpu.Simulation, "describe", None))
