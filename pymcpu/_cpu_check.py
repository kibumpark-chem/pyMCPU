"""Refuse to load the extension on a CPU without the instructions it was built for.

The published wheels are built for the x86-64-v3 baseline (AVX2, FMA, BMI1/2,
...). On an older CPU the extension would stop Python with an illegal
instruction, with no hint why. CMake records the baseline of each build in
``pymcpu/_build_arch.py`` (from ``cmake/_build_arch.py.in``), and
``pymcpu/__init__.py`` calls :func:`check_cpu_baseline` before it imports the
extension, so the import fails instead with a message that names what the CPU
lacks.

Nothing is checked when the build names no fixed baseline (``MCPU_ARCH``
``native``, ``none`` or a raw ``-march`` value, and Debug builds), when there
is no ``_build_arch.py`` (an extension copied into a source tree by hand),
when the CPU's flags cannot be read (anything but Linux), or when
``MCPU_SKIP_CPU_CHECK`` is set (to anything but empty, ``0``, ``false``, ``no``
or ``off``).

This module must not import the extension, or anything that does.
"""

from __future__ import annotations

import os
import sys
from typing import Iterable

# The x86-64 psABI micro-architecture levels, as Linux /proc/cpuinfo flags.
# Each level includes the one below it. OSXSAVE (v3) has no flag of its own in
# /proc/cpuinfo; the kernel clears "xsave" and "avx" when it does not enable it.
_V2 = ("cx16", "lahf_lm", "popcnt", "pni", "sse4_1", "sse4_2", "ssse3")
_V3 = _V2 + ("abm", "avx", "avx2", "bmi1", "bmi2", "f16c", "fma", "movbe", "xsave")
_V4 = _V3 + ("avx512bw", "avx512cd", "avx512dq", "avx512f", "avx512vl")
TIER_FLAGS: dict[str, tuple[str, ...]] = {"v2": _V2, "v3": _V3, "v4": _V4}

# The usual names of the flags whose /proc/cpuinfo spelling is not the name.
_FLAG_NAMES = {
    "abm": "LZCNT",
    "cx16": "CMPXCHG16B",
    "lahf_lm": "LAHF/SAHF",
    "pni": "SSE3",
}

SKIP_ENV = "MCPU_SKIP_CPU_CHECK"
# Values of SKIP_ENV that leave the check on (compared in lower case).
_SKIP_OFF = ("", "0", "false", "no", "off")


def cpu_flags(cpuinfo_path: str = "/proc/cpuinfo") -> frozenset[str] | None:
    """The first ``flags`` line of ``cpuinfo_path``, or None if there is none."""
    try:
        with open(cpuinfo_path, encoding="utf-8", errors="replace") as handle:
            for line in handle:
                key, sep, value = line.partition(":")
                if sep and key.strip() == "flags":
                    return frozenset(value.split())
    except OSError:
        return None
    return None


def missing_flags(tier: str, flags: Iterable[str]) -> list[str]:
    """The flags of baseline ``tier`` ("v2", "v3", "v4") that ``flags`` lacks.

    Empty for any other tier, which names no fixed instruction set.
    """
    have = set(flags)
    return [flag for flag in TIER_FLAGS.get(tier, ()) if flag not in have]


def _describe(flags: Iterable[str]) -> str:
    return ", ".join(
        f"{_FLAG_NAMES[f]} ({f})" if f in _FLAG_NAMES else f.upper() for f in flags
    )


def baseline_error(tier: str, missing: Iterable[str]) -> str:
    """The message for a CPU that lacks ``missing`` from baseline ``tier``."""
    lower = "v2" if tier in ("v3", "v4") else None
    alternative = (
        f"  MCPU_ARCH={lower} also gives a build that runs on any x86-64-{lower} CPU.\n"
        if lower
        else ""
    )
    return (
        f"pymcpu: this CPU cannot run the installed extension (pymcpu.mcpu_core). "
        f"It was built for the x86-64-{tier} baseline, and this CPU lacks "
        f"{_describe(missing)}. Loading it would stop Python with an illegal "
        f"instruction.\n"
        f"Build pyMCPU for this CPU from a source checkout instead:\n"
        f"  MCPU_ARCH=native pip install --no-build-isolation -e .\n"
        f"{alternative}"
        f"See 'Linux x86-64 only, with an x86-64-v3 baseline' in the known issues "
        f"of the documentation. {SKIP_ENV}=1 skips this check."
    )


def _build_tier() -> str | None:
    try:
        from pymcpu import _build_arch  # written by CMake at build time
    except ImportError:
        return None
    return str(getattr(_build_arch, "TIER", "none"))


def check_cpu_baseline(
    tier: str | None = None, flags: Iterable[str] | None = None
) -> None:
    """Raise ImportError if this CPU lacks the extension's baseline.

    ``tier`` defaults to the installed build's (``pymcpu._build_arch.TIER``)
    and ``flags`` to this CPU's (``/proc/cpuinfo``); either one unknown means
    there is nothing to check.
    """
    if os.environ.get(SKIP_ENV, "").strip().lower() not in _SKIP_OFF:
        return
    if tier is None:
        tier = _build_tier()
    if tier not in TIER_FLAGS:
        return
    if flags is None:
        if not sys.platform.startswith("linux"):
            return
        flags = cpu_flags()
        if flags is None:
            return
    missing = missing_flags(tier, flags)
    if missing:
        raise ImportError(baseline_error(tier, missing))
