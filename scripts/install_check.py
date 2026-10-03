#!/usr/bin/env python3
"""Quick installation verification script."""
import sys


def check_import():
    try:
        import pymcpu
        print(f"[OK] pymcpu imported successfully (version: {pymcpu.__version__})")
        return True
    except ImportError as e:
        print(f"[FAIL] Cannot import pymcpu: {e}")
        return False


def check_cpp_core():
    try:
        import pymcpu
        core = pymcpu.mcpu_core
        print(f"[OK] C++ core loaded: {core.__file__}")
        return True
    except (AttributeError, ImportError) as e:
        print(f"[FAIL] C++ core not accessible: {e}")
        return False


def check_energy_reporter():
    try:
        import pymcpu
        import tempfile
        import os
        fd, path = tempfile.mkstemp(suffix=".csv")
        os.close(fd)
        try:
            pymcpu.EnergyReporter(path, 1)
            print("[OK] EnergyReporter instantiated")
            return True
        finally:
            os.unlink(path)
    except Exception as e:
        print(f"[FAIL] EnergyReporter: {e}")
        return False


def check_simulation_reporter():
    try:
        import pymcpu
        pymcpu.SimulationReporter(1)
        print("[OK] SimulationReporter instantiated")
        return True
    except Exception as e:
        print(f"[FAIL] SimulationReporter: {e}")
        return False


def check_xtc_reporter():
    try:
        import pymcpu
        import tempfile
        import os
        fd, path = tempfile.mkstemp(suffix=".xtc")
        os.close(fd)
        try:
            pymcpu.XtcReporter(path, 1)
            print("[OK] XtcReporter instantiated")
            return True
        finally:
            os.unlink(path)
    except Exception as e:
        print(f"[FAIL] XtcReporter: {e}")
        return False


def check_build_info():
    try:
        import pymcpu
        features = pymcpu.mcpu_core.build_info()["features"]
        print(f"[OK] Build flags: POOLED_PROPOSAL={features['MCPU_USE_POOLED_PROPOSAL']}")
        return True
    except Exception as e:
        print(f"[FAIL] build_info: {e}")
        return False


if __name__ == "__main__":
    print("MCPU Installation Check")
    print("=" * 50)
    checks = [
        check_import,
        check_cpp_core,
        check_energy_reporter,
        check_simulation_reporter,
        check_xtc_reporter,
        check_build_info,
    ]
    results = [c() for c in checks]
    print("=" * 50)
    if all(results):
        print("All checks passed. pyMCPU is correctly installed.")
        sys.exit(0)
    else:
        print(f"{results.count(False)} check(s) failed.")
        sys.exit(1)
