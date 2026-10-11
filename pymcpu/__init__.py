"""pyMCPU: Monte Carlo protein folding engine."""

from __future__ import annotations

import glob
import importlib.util
import os
import sys

__version__ = "0.1.0"

PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))


def _mcpu_core_abi_tag() -> str:
    return f"cpython-{sys.version_info.major}{sys.version_info.minor}"


def _find_mcpu_core_so() -> str | None:
    """Locate an ABI-matching mcpu_core extension on disk."""
    tag = _mcpu_core_abi_tag()
    candidates: list[str] = []
    search_roots = list(globals().get("__path__", [PACKAGE_ROOT]))
    for entry in list(sys.path):
        if not entry:
            continue
        search_roots.append(os.path.join(entry, "pymcpu"))
    seen: set[str] = set()
    for root in search_roots:
        root = os.path.abspath(root)
        if root in seen or not os.path.isdir(root):
            continue
        seen.add(root)
        for path in glob.glob(os.path.join(root, "mcpu_core*.so")):
            if tag in os.path.basename(path):
                candidates.append(path)
    return candidates[0] if candidates else None


def _load_mcpu_core():
    """Import the C++ extension, with a clear error if it is missing for this Python."""
    # Editable installs put the compiled .so under site-packages/pymcpu while
    # Python sources stay in the repo; ``__path__`` must include that directory.
    # A bare PYTHONPATH=repo import (without a matching build) fails with a
    # misleading "circular import" ImportError from CPython.
    #
    # First make sure this CPU can run it: an extension built for x86-64-v3 would
    # otherwise stop Python with an illegal instruction on an older CPU.
    from ._cpu_check import check_cpu_baseline

    check_cpu_baseline()
    try:
        from . import mcpu_core as _core  # noqa: F401

        return _core
    except ImportError as first_err:
        so_path = _find_mcpu_core_so()
        if so_path is None:
            raise ImportError(
                "Failed to import pymcpu.mcpu_core (C++ extension).\n"
                f"  Python: {sys.executable} ({sys.version.split()[0]})\n"
                f"  Package paths: {list(globals().get('__path__', [PACKAGE_ROOT]))}\n"
                f"  Expected an extension matching '*{_mcpu_core_abi_tag()}*.so'.\n"
                "  This is NOT a circular import — the compiled module is missing "
                "for this interpreter.\n"
                "  Fix: install a wheel built for this interpreter, or rebuild\n"
                "  from a source checkout against THIS python:\n"
                f"    {sys.executable} -m pip install --no-build-isolation -e '.[dev]'\n"
                "  Building needs a C++20 compiler and a matching libstdc++ on\n"
                "  the loader path:\n"
                "    conda/mamba: conda install -c conda-forge "
                "gxx_linux-64 libstdcxx-ng\n"
                "    HPC modules: load the same compiler you built with, and\n"
                "                 make sure it is no NEWER than the libstdc++\n"
                "                 this interpreter loads -- a newer one gives\n"
                "                 \"CXXABI_x.y.z not found\" at import time\n"
                "  Or activate the env where mcpu_core was already built for "
                "this ABI.\n"
                "  Avoid: pointing PYTHONPATH at the repo from a different "
                "Python than the build."
            ) from first_err

        so_dir = os.path.dirname(so_path)
        pkg_path = list(globals().get("__path__", [PACKAGE_ROOT]))
        if so_dir not in pkg_path:
            pkg_path.append(so_dir)
            __path__[:] = pkg_path  # type: ignore[name-defined]

        spec = importlib.util.spec_from_file_location(
            "pymcpu.mcpu_core",
            so_path,
            submodule_search_locations=[],
        )
        if spec is None or spec.loader is None:
            raise ImportError(
                f"Could not create import spec for {so_path}"
            ) from first_err
        module = importlib.util.module_from_spec(spec)
        sys.modules["pymcpu.mcpu_core"] = module
        spec.loader.exec_module(module)
        return module


mcpu_core = _load_mcpu_core()

from .forcefields.mcpu import MCPUForceField  # noqa: E402
from .mcpu_core import (  # noqa: E402
    AromaticPotential,
    Context,
    EnergyReporter,
    HBondPotential,
    Integrator,
    MuPotential,
    NativeContactsBiasPotential,
    Reporter,
    SidechainTripletPotential,
    SimulationReporter,
    System,
    TripletPotential,
    XtcReporter,
)
from .simulation import Simulation  # noqa: E402

__all__ = [
    "PACKAGE_ROOT",
    "AromaticPotential",
    "Context",
    "EnergyReporter",
    "HBondPotential",
    "Integrator",
    "MCPUForceField",
    "MuPotential",
    "NativeContactsBiasPotential",
    "Reporter",
    "SidechainTripletPotential",
    "Simulation",
    "SimulationReporter",
    "System",
    "TripletPotential",
    "XtcReporter",
    "__version__",
    "mcpu_core",
]
