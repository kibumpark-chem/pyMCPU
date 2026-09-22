"""Fast import smoke tests: confirms ``pymcpu`` and its key submodules import
cleanly and expose the expected public API surface. No forcefield, PDB, or
network access -- if this file is slow or touches disk beyond ``import``,
something regressed.
"""

from __future__ import annotations


def test_package_version() -> None:
    import pymcpu

    assert isinstance(pymcpu.__version__, str)
    assert pymcpu.__version__
    assert pymcpu.PACKAGE_ROOT
    assert pymcpu.mcpu_core is not None
    assert hasattr(pymcpu, "System")


def test_public_toplevel_exports() -> None:
    import pymcpu as mc

    for name in (
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
    ):
        assert hasattr(mc, name), name
        assert name in mc.__all__  # pymcpu/__init__.py's __all__ is the authoritative API surface


def test_core_symbols() -> None:
    from pymcpu import mcpu_core

    for name in ("System", "Context", "Integrator", "HBondPotential", "MuPotential"):
        assert hasattr(mcpu_core, name), name


def test_forcefield_class_importable() -> None:
    from pymcpu.forcefields.mcpu import MCPUForceField

    assert MCPUForceField is not None


def test_simulation_importable() -> None:
    from pymcpu.simulation import Simulation

    assert Simulation is not None


def test_params_module_importable() -> None:
    from pymcpu import params

    assert callable(params.ensure_params)
    assert callable(params.get_cache_dir)
