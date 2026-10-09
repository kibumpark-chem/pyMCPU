"""pymcpu refuses to load an extension that this CPU cannot run.

The published wheels are built for the x86-64-v3 baseline (AVX2, FMA, BMI1/2,
...). On an older CPU the extension used to stop Python with an illegal
instruction and no hint why. CMake now records each build's baseline in
pymcpu/_build_arch.py, and ``import pymcpu`` compares it with the CPU's flags
(/proc/cpuinfo) before it loads the extension, so the import fails with a
message that names what is missing and how to build for an older tier.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import pymcpu
from pymcpu import _cpu_check

V3_CPU = frozenset(_cpu_check.TIER_FLAGS["v3"])
PACKAGE_PARENT = str(Path(pymcpu.__file__).resolve().parent.parent)


def test_each_tier_includes_the_one_below() -> None:
    tiers = _cpu_check.TIER_FLAGS
    assert set(tiers["v2"]) < set(tiers["v3"]) < set(tiers["v4"])
    assert {"avx2", "fma", "bmi1", "bmi2", "movbe", "f16c"} <= set(tiers["v3"])


def test_a_cpu_with_the_baseline_passes() -> None:
    _cpu_check.check_cpu_baseline("v3", V3_CPU)
    _cpu_check.check_cpu_baseline("v2", V3_CPU)
    _cpu_check.check_cpu_baseline("v3", V3_CPU | {"avx512f"})


def test_a_cpu_without_avx2_is_refused_with_a_clear_message() -> None:
    with pytest.raises(ImportError) as info:
        _cpu_check.check_cpu_baseline("v3", V3_CPU - {"avx2", "fma"})
    message = str(info.value)
    assert "x86-64-v3" in message
    assert "AVX2" in message and "FMA" in message
    assert "MCPU_ARCH=native" in message and "MCPU_ARCH=v2" in message
    assert _cpu_check.SKIP_ENV in message


def test_flags_spelled_differently_in_cpuinfo_get_their_usual_names() -> None:
    with pytest.raises(ImportError, match=r"LZCNT \(abm\)"):
        _cpu_check.check_cpu_baseline("v3", V3_CPU - {"abm"})


def test_a_v4_build_needs_avx512() -> None:
    with pytest.raises(ImportError, match="AVX512F"):
        _cpu_check.check_cpu_baseline("v4", V3_CPU)


@pytest.mark.parametrize("tier", ["none", "native", "custom", ""])
def test_a_build_without_a_fixed_baseline_is_not_checked(tier: str) -> None:
    _cpu_check.check_cpu_baseline(tier, frozenset())


@pytest.mark.parametrize("value", ["1", "yes", "TRUE"])
def test_the_check_can_be_skipped(monkeypatch, value: str) -> None:
    monkeypatch.setenv(_cpu_check.SKIP_ENV, value)
    _cpu_check.check_cpu_baseline("v3", frozenset())


@pytest.mark.parametrize("value", ["", "0", "false", "No", " off "])
def test_a_false_value_does_not_skip_the_check(monkeypatch, value: str) -> None:
    monkeypatch.setenv(_cpu_check.SKIP_ENV, value)
    with pytest.raises(ImportError, match="AVX2"):
        _cpu_check.check_cpu_baseline("v3", frozenset(_cpu_check.TIER_FLAGS["v3"]) - {"avx2"})


def test_the_flags_come_from_cpuinfo(tmp_path) -> None:
    cpuinfo = tmp_path / "cpuinfo"
    cpuinfo.write_text(
        "processor\t: 0\nflags\t\t: fpu avx avx2\n\nprocessor\t: 1\nflags\t\t: fpu\n"
    )
    assert _cpu_check.cpu_flags(str(cpuinfo)) == frozenset({"fpu", "avx", "avx2"})
    assert _cpu_check.cpu_flags(str(tmp_path / "missing")) is None


def _import_pymcpu(tier: str, flags: str, env: dict[str, str] | None = None) -> str:
    """Import pymcpu in a fresh interpreter with the build's baseline set to
    ``tier`` and the CPU's flags to the expression ``flags``; report whether
    the import got as far as the extension."""
    code = textwrap.dedent(
        f"""
        import importlib.util, sys, types
        spec = importlib.util.spec_from_file_location(
            "pymcpu._cpu_check", {_cpu_check.__file__!r})
        check = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(check)
        assert "pymcpu.mcpu_core" not in sys.modules  # the check alone loads nothing
        check.cpu_flags = lambda path="/proc/cpuinfo": frozenset({flags})
        sys.modules["pymcpu._cpu_check"] = check
        arch = types.ModuleType("pymcpu._build_arch")
        arch.TIER = {tier!r}
        sys.modules["pymcpu._build_arch"] = arch
        print("ORIGIN", importlib.util.find_spec("pymcpu").origin)
        try:
            import pymcpu
        except ImportError as error:
            loaded = "pymcpu.mcpu_core" in sys.modules
            print("REFUSED", "LOADED-EXTENSION" if loaded else "", str(error).splitlines()[0])
        else:
            print("IMPORTED", pymcpu.mcpu_core.__name__)
        """
    )
    run_env = {**os.environ, "PYTHONPATH": PACKAGE_PARENT, **(env or {})}
    run_env.pop(_cpu_check.SKIP_ENV, None)
    run_env.update(env or {})
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=run_env,
        cwd=PACKAGE_PARENT, timeout=300,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    origin, _, outcome = result.stdout.strip().partition("\n")
    origin = origin.removeprefix("ORIGIN ")
    # Another install of pymcpu (an editable one, say) can win over PYTHONPATH;
    # the test would then check that install instead of this tree.
    assert Path(origin).resolve().is_relative_to(PACKAGE_PARENT), (
        f"the child process imported pymcpu from {origin}, not from {PACKAGE_PARENT}"
    )
    return outcome.strip()


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc/cpuinfo")
def test_the_import_stops_before_the_extension_loads() -> None:
    out = _import_pymcpu("v3", f"{sorted(V3_CPU - {'avx2'})!r}")
    assert out.startswith("REFUSED"), out
    assert "LOADED-EXTENSION" not in out
    assert "AVX2" in out


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc/cpuinfo")
def test_the_import_goes_on_when_the_cpu_has_the_baseline_or_the_check_is_skipped() -> None:
    assert _import_pymcpu("v3", f"{sorted(V3_CPU)!r}").startswith("IMPORTED")
    skipped = _import_pymcpu("v3", "[]", env={_cpu_check.SKIP_ENV: "1"})
    assert skipped.startswith("IMPORTED")
