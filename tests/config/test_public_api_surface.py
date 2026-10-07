"""Public API surface tests for the top-level ``pymcpu`` package.

These are pure import/``hasattr``/``callable`` checks -- no simulation is run
and no physics is exercised. They exist to catch software-engineering
regressions (a class silently un-exported, a method renamed/removed) that
would break every downstream user of the OpenMM-style API, independent of
whether the underlying physics is correct.

This file merges what used to be two separate files: the old
``tests/test_openmm_style.py`` (this file) and the trailing
``test_public_api_exports`` test from the old ``tests/test_simulation_reporters.py``
(whose actual subject -- reporter list/step-offset mechanics -- now lives in
``tests/physics/test_reporters_and_stepping.py``). Both files asserted an
overlapping-but-not-identical set of exported class names; here they're
combined into one export list (the union of both) checked by a single
parametrized test, instead of two near-duplicate whole-list assertions.
"""

from __future__ import annotations

import pymcpu
import pytest

# Union of the class names checked by the old test_openmm_style.py
# (Simulation, System, Context, Integrator, MCPUForceField, EnergyReporter,
# XtcReporter, SimulationReporter) and test_simulation_reporters.py's
# test_public_api_exports (which additionally covered the force/potential
# classes: MuPotential, HBondPotential, TripletPotential, SidechainTripletPotential,
# AromaticPotential). Keeping the union means dropping either half of the
# merge would be caught here.
EXPECTED_TOP_LEVEL_EXPORTS = (
    "Simulation",
    "System",
    "Context",
    "Integrator",
    "MCPUForceField",
    "EnergyReporter",
    "XtcReporter",
    "SimulationReporter",
    "MuPotential",
    "HBondPotential",
    "TripletPotential",
    "SidechainTripletPotential",
    "AromaticPotential",
    "KORPForceField",
    "OrientationalPairPotential",
    "CalphaExcludedVolumePotential",
)

# Simulation methods grouped by the sub-API they belong to, mirroring the
# old file's per-topic tests (reporter management vs. fixed-residue control).
_REPORTER_METHODS = (
    "add_reporter",
    "remove_reporter",
    "clear_reporters",
    "add_xtc_reporter",
    "add_energy_reporter",
    "add_simulation_reporter",
)
_FIXED_RESIDUE_METHODS = ("set_fixed_residues", "clear_fixed_residues", "get_fixed_residues")


def test_simulation_importable() -> None:
    assert pymcpu.Simulation is not None


@pytest.mark.parametrize("name", EXPECTED_TOP_LEVEL_EXPORTS)
def test_top_level_class_exported(name: str) -> None:
    assert hasattr(pymcpu, name), f"pymcpu.{name} missing from public API"


def test_simulation_has_step_method() -> None:
    assert callable(getattr(pymcpu.Simulation, "step", None))


def test_simulation_has_describe_method() -> None:
    assert callable(getattr(pymcpu.Simulation, "describe", None))


def test_simulation_has_reporter_methods() -> None:
    for method in _REPORTER_METHODS:
        assert callable(getattr(pymcpu.Simulation, method, None)), f"Missing: {method}"


def test_simulation_has_fixed_residue_methods() -> None:
    for method in _FIXED_RESIDUE_METHODS:
        assert callable(getattr(pymcpu.Simulation, method, None)), f"Missing: {method}"


def test_integrator_has_set_seed() -> None:
    assert callable(getattr(pymcpu.Integrator, "set_seed", None))

