"""Regenerate ``abi_surface.txt`` from the currently-built ``mcpu_core``.

Run this after adding or renaming a pybind11 binding::

    python tests/extension_abi/_regen_abi_surface.py

``test_compiled_surface_matches_snapshot`` fails when the built extension is
missing an entry the snapshot lists, which is what turns a stale build into
one clear error instead of dozens of confusing ``AttributeError``s elsewhere.
Adding a binding therefore requires regenerating this file -- deliberately,
so the snapshot cannot silently drift away from the bindings.
"""

from __future__ import annotations

from pathlib import Path

from pymcpu import mcpu_core

# The bound classes the Python layer actually builds on. Deliberately a fixed
# list rather than ``dir(mcpu_core)``: a new class appearing should not silently
# widen the canary's surface without someone looking at it.
CLASSES = [
    "System", "Context", "Integrator", "State", "Potential",
    "MuPotential", "HBondPotential",
    "TripletPotential", "SidechainTripletPotential", "AromaticPotential",
    "NativeContactsBiasPotential", "EnergyReporter", "SimulationReporter",
    "XtcReporter", "Reporter", "RamaMixtureLibrary", "RotamerLibrary",
    "EnergyWeights", "BlockIndices",
    "OrientationalPairMap", "OrientationalPairPotential",
    "CalphaExcludedVolumePotential",
]

HEADER = [
    "# Snapshot of the compiled mcpu_core surface the Python layer depends on.",
    "# Regenerate with: python tests/extension_abi/_regen_abi_surface.py",
    "# A missing entry means the built extension is STALE relative to the",
    "# Python source -- rebuild before investigating any other failure.",
]


def render() -> str:
    lines = list(HEADER)
    for class_name in CLASSES:
        cls = getattr(mcpu_core, class_name, None)
        if cls is None:
            continue
        lines += [
            f"{class_name}.{attr}"
            for attr in sorted(a for a in dir(cls) if not a.startswith("_"))
        ]
    return "\n".join(lines) + "\n"


def main() -> int:
    out = Path(__file__).with_name("abi_surface.txt")
    out.write_text(render(), encoding="utf-8")
    print(f"wrote {out} ({len(render().splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
