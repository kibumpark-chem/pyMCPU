#!/usr/bin/env python3
"""Acceptance test for an installed pyMCPU wheel.

Run against the INSTALLED package from a directory that is not the source
tree, with no network. cibuildwheel invokes it as CIBW_TEST_COMMAND; it is
also runnable by hand:

    cd /tmp && MCPU_NO_DOWNLOAD=1 python /path/to/scripts/ci_check_wheel.py

Why a dedicated script rather than `pytest` against the wheel: the things
that distinguish a *shipped* artifact from a working developer checkout are
not covered by the suite, because the suite runs in the source tree with a
populated parameter directory and a locally-built extension. Every check
below is one that has actually failed, or could only fail, in the packaged
configuration:

  * LTO was silently off in every containerised build for as long as the
    project has had wheels, because `check_ipo_supported()` tripped over
    Fortran that FetchContent'd Eigen enables. Local `cmake` builds kept LTO,
    so no benchmark and no test ever saw the shipped configuration.
  * The CPU baseline decides whether the wheel runs at all on a given
    machine. A pin to `skylake-avx512` would SIGILL on every AMD Zen 1-3 and
    every 12th-gen-or-later Intel consumer part, with no diagnostic.
    (docs/arch_baseline_decision.md)
  * Parameter resolution step 5 -- the in-wheel compact archive -- is the
    only step a `pip install` user exercises, and it is below the dev-tree
    and MCPU_PARAMS_DIR steps that every developer hits instead.

Exit codes: 0 all checks passed, 1 a check failed.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

FAILURES: list[str] = []


def check(name: str):
    def deco(fn):
        print(f"\n--- {name}")
        try:
            fn()
        except Exception:  # noqa: BLE001
            FAILURES.append(name)
            print(traceback.format_exc().rstrip())
            print(f"    FAIL: {name}")
        else:
            print(f"    ok: {name}")
        return fn

    return deco


print("pyMCPU wheel acceptance check")
print(f"  cwd:        {os.getcwd()}")
print(f"  python:     {sys.version.split()[0]}")
print(f"  MCPU_* env: {sorted(k for k in os.environ if k.startswith('MCPU_'))}")


@check("imports from outside the source tree")
def _import() -> None:
    import pymcpu

    pkg = Path(pymcpu.__file__).resolve().parent
    print(f"    pymcpu {pymcpu.__version__} from {pkg}")

    # Being outside the checkout is NOT sufficient: an editable install
    # resolves to the source tree from any cwd, so a naive cwd check passes
    # while testing exactly the thing this script exists not to test.
    installed = any(
        part in ("site-packages", "dist-packages") for part in pkg.parts
    )
    if not installed:
        msg = (
            f"{pkg} is not under site-packages -- this is a source/editable "
            f"import, so none of the checks below say anything about a built "
            f"wheel."
        )
        if os.environ.get("MCPU_WHEEL_CHECK_ALLOW_SOURCE") == "1":
            print(f"    WARNING (allowed by override): {msg}")
        else:
            raise AssertionError(msg)


@check("build_info reports a Release build with LTO")
def _build_info() -> None:
    from pymcpu import mcpu_core

    info = mcpu_core.build_info()
    print("    " + json.dumps({k: info[k] for k in ("arch", "build", "fp")}))
    assert info["build"]["type"] == "Release", info["build"]["type"]
    assert info["build"]["lto"] is True, (
        "Release wheel reports lto=False. Check the configure log for "
        "'LTO/IPO not supported'; if it names a language this project does "
        "not use, a dependency enabled it and check_ipo_supported() needs "
        "LANGUAGES CXX."
    )
    assert info["fp"]["fast_math"] is False, "wheel built with -ffast-math"
    assert info["fp"]["associative_math"] is False
    assert info["fp"]["reciprocal_math"] is False


@check("the CPU baseline is portable (no AVX-512 in a published wheel)")
def _baseline() -> None:
    from pymcpu import mcpu_core

    info = mcpu_core.build_info()
    arch, isa = info["arch"], info["isa"]
    print(f"    tier={arch.get('tier')} march={arch.get('march')}")
    assert arch.get("tier"), "no CPU baseline recorded"
    avx512 = [k for k, v in isa.items() if "AVX512" in k.upper() and v]
    assert not avx512, (
        f"wheel advertises AVX-512 features {avx512}. It will SIGILL with no "
        f"diagnostic on AMD Zen 1-3 and on 12th-gen-or-later Intel consumer "
        f"parts. See docs/arch_baseline_decision.md."
    )


@check("parameters resolve with no network (in-wheel archive, step 5)")
def _params() -> None:
    from pymcpu.params import ensure_params

    root = Path(ensure_params("mcpu08"))
    print(f"    resolved to {root}")
    assert root.is_dir(), root
    from pymcpu.params import required_files

    for role, rel in required_files("mcpu08").items():
        assert (root / rel).is_file(), f"missing {role} -> {rel}"
    print(f"    all {len(required_files('mcpu08'))} required files present")


@check("a short simulation runs and conserves its running energy")
def _simulate() -> None:
    import numpy as np

    import pymcpu as mc
    from pymcpu.forcefields.mcpu import MCPUForceField
    from pymcpu.runners import default_example_pdb

    import mdtraj as md

    traj = md.load(str(default_example_pdb()))
    heavy = traj.atom_slice(traj.topology.select("not element H"))
    ff = MCPUForceField(heavy, param_set="mcpu08")
    system = ff.create_system(heavy.topology)
    integ = mc.mcpu_core.Integrator(temperature=0.6, step_size_rad=0.1)
    integ.set_seed(42)
    sim = mc.Simulation(heavy.topology, system, integ)
    sim.context.set_positions((ff.coords[0] * 10.0).T.astype(np.float32))

    sim.step(200)
    running = sim.context.get_state().current_energy
    exact = sim.context.calculate_total_energy(-1)
    print(f"    running={running!r} exact={exact!r}")
    assert np.isfinite(running) and np.isfinite(exact)
    assert abs(running - exact) < 1e-2, (
        f"running energy drifted from an exact recompute by "
        f"{abs(running - exact)}"
    )


@check("the CLI entry point is installed and runs")
def _cli() -> None:
    import subprocess

    out = subprocess.run(
        [sys.executable, "-m", "pymcpu.cli", "--help"],
        capture_output=True, text=True, check=False,
    )
    assert out.returncode == 0, out.stderr[-500:]


print()
if FAILURES:
    print(f"FAILED {len(FAILURES)} check(s): {', '.join(FAILURES)}")
    sys.exit(1)
print("all wheel acceptance checks passed")
