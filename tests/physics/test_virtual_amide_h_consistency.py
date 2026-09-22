"""Virtual vs. explicit amide-H: two internal HBond encodings, one physics.

pyMCPU supports two internal encodings of the backbone amide hydrogen for
HBond-energy purposes:

* the default "virtual" amide-H -- no explicit H atom is added; its position
  is inferred geometrically inside the HBond term.
* an "explicit escape hatch" -- a real H atom is added to the system and
  used directly.

These are two code paths of the *same* engine, expected to agree exactly (up
to floating point). This file is a self-consistency check between them --
NOT a legacy-comparison test. (An earlier version of this file's docstring
described it as parity against legacy CheckHBond geometry; that motivated the
virtual-H design but no assertion here reads a legacy MCPU reference value or
runs dbfold_actin, so it belongs in physics_internal, not legacy_parity.)
"""

from __future__ import annotations

import mdtraj as md
import numpy as np
import pytest

from pymcpu import mcpu_core
from pymcpu.forcefields.mcpu import MCPUForceField
from tests.fixtures.context_builders import ATOL, resolve_test_pdb

pytestmark = pytest.mark.slow

# Floating-point equality tolerance for comparing the same energy computed by
# two independent code paths (virtual vs. explicit H) that should agree
# exactly in exact arithmetic -- looser than machine epsilon to absorb
# differing operation order, still far tighter than ATOL (which is meant for
# incremental-vs-full-recompute consistency, a different kind of comparison).
_HBOND_PATH_ATOL = 1e-5


def _build(virtual: bool) -> tuple[mcpu_core.Context, MCPUForceField]:
    pdb = resolve_test_pdb()
    traj = md.load(str(pdb))
    indices = traj.topology.select("not element H")
    filtered = traj.atom_slice(indices)
    ff = MCPUForceField(filtered, virtual_amide_h=virtual)
    system = ff.create_system(filtered.topology)
    ctx = mcpu_core.Context(system)
    coords = ff.coords[0] * 10.0
    ctx.set_positions(coords.T.astype(np.float32))
    ctx.set_use_legacy_weights(False)  # compare raw HBond components
    ctx.calculate_total_energy(-1)
    return ctx, ff


def test_default_has_no_explicit_amide_h() -> None:
    ctx, ff = _build(virtual=True)
    assert ff.virtual_amide_h is True
    assert ff.total_h_atoms == 0
    assert all(a.name != "H" for a in ff.ordered_atom_list)
    # Raw group-4 (HBond) energy must equal its own energy_breakdown entry --
    # a self-consistency check that the two internal energy accessors agree.
    assert ctx.calculate_total_energy_raw(4) == pytest.approx(
        float(ctx.energy_breakdown(weighted=False)["by_group"].get(4, 0.0)),
        abs=ATOL,
    )


def test_explicit_escape_hatch_still_adds_h() -> None:
    ctx, ff = _build(virtual=False)
    assert ff.virtual_amide_h is False
    assert ff.total_h_atoms > 0
    assert any(a.name == "H" for a in ff.ordered_atom_list)


def test_virtual_vs_explicit_hbond_energy_parity() -> None:
    ctx_v, _ = _build(virtual=True)
    ctx_e, _ = _build(virtual=False)
    e_v = float(ctx_v.calculate_total_energy_raw(4))
    e_e = float(ctx_e.calculate_total_energy_raw(4))
    assert e_v == pytest.approx(e_e, abs=_HBOND_PATH_ATOL)


def test_virtual_vs_explicit_delta_parity() -> None:
    ctx_v, _ = _build(virtual=True)
    ctx_e, _ = _build(virtual=False)
    for ctx in (ctx_v, ctx_e):
        integ = mcpu_core.Integrator(temperature=300.0, step_size_rad=0.05)
        integ.set_seed(7)
        integ.verify_physics_consistency(ctx, num_steps=20, atol=ATOL)


@pytest.mark.parametrize("skin", [0.0, 1.0])
def test_actin_verify_virtual_default_at_mu_skin(skin: float) -> None:
    ctx, _ = _build(virtual=True)
    ctx.set_use_legacy_weights(True)
    ctx.calculate_total_energy(-1)
    ctx.set_mu_skin(skin)
    integ = mcpu_core.Integrator(temperature=300.0, step_size_rad=0.05)
    integ.verify_physics_consistency(ctx, num_steps=25, atol=ATOL)
