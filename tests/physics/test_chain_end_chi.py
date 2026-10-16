"""A Context caches the chi angles of its coordinates at both chain ends too.

``State.sidechain_torsions`` holds each residue's chi angles. Placing
coordinates (``Context.set_positions``, the ``Context.coords`` setter, and
through them a run's start, a checkpoint resume, a replica-exchange swap and
``recenter``) measures them from the coordinates, and a sidechain move
measures those of the residue it turns. A placement used to measure chi only
for the residues with a neighbour on each side, which phi, psi, pCA and bCA
need, so the first and the last residue kept the chi of the conformation
placed before (-pi on a new Context) until a sidechain move at that end was
accepted.

The sidechain torsion energy scores residues 1 to N-2 only, as legacy
``sctenergy`` does (the ends have no triplet), and the rotamer move measures
the chi it starts from on the coordinates, so no energy and no move changed;
what read the cache did, such as the chi a run records with each frame.

Crambin (PDB 3NIR) starts with THR (one chi) and ends with ASN (two). The
checkpoint resumes are in
``tests/integration/checkpointing/test_chain_end_chi_resume.py``.
"""

from __future__ import annotations

from pathlib import Path

import mdtraj as md
import numpy as np
import pytest

from pymcpu import mcpu_core
from pymcpu.forcefields.mcpu import MCPUForceField
from pymcpu.sampling.replica_exchange_core import get_coords, swap_context_coordinates
from tests.helpers.chi_cache import (
    angle_gap,
    cached_chi,
    chi_from_coordinates,
    stale_chi,
)

CRAMBIN_PDB = Path(__file__).parent / "forcefield" / "data" / "3nir_heavy.pdb"
LAYOUTS = ["off", "init_only"]


def _build(reorder: str = "off"):
    structure = md.load(str(CRAMBIN_PDB))
    ff = MCPUForceField(structure)
    ctx = mcpu_core.Context(ff.create_system(structure.topology))
    ctx.set_atom_reorder_mode(reorder)
    ctx.set_positions((ff.coords[0] * 10.0).T.astype(np.float32))
    ctx.calculate_total_energy(-1)
    return ff, ctx


def _turn_sidechains(ctx, *, mode: str = "rotamer_library", seed: int = 11, steps: int = 3000):
    """Sidechain moves only, at T 1: both end sidechains turn in 3000 steps."""
    integ = mcpu_core.Integrator(temperature=1.0, step_size_rad=0.1)
    integ.set_seed(seed)
    integ.set_sidechain_move_mode(mode)
    integ.set_move_weights(0.0, 0.0, 1.0)
    integ.run(ctx, steps)
    return integ


def _ends(ctx) -> list[tuple[int, int]]:
    """(residue, chi count) of the first and the last residue."""
    n_chi = list(ctx.get_system().get_torsions_per_residue())
    return [(0, n_chi[0]), (len(n_chi) - 1, n_chi[-1])]


def _end_turn(ff, a, b) -> float:
    """Smallest change of any end chi between coordinates a and b."""
    ca, cb = chi_from_coordinates(ff, a), chi_from_coordinates(ff, b)
    return min(float(angle_gap(ca[r, :k], cb[r, :k]).max()) for r, k in ((0, 1), (len(ca) - 1, 2)))


@pytest.mark.parametrize("reorder", LAYOUTS)
def test_a_new_context_caches_the_chi_of_every_residue(reorder: str) -> None:
    ff, ctx = _build(reorder)
    assert [k for _, k in _ends(ctx)] == [1, 2]  # THR 1, ASN 46
    assert stale_chi(ctx, ff) == []


@pytest.mark.parametrize("reorder", LAYOUTS)
@pytest.mark.parametrize("placement", ["set_positions", "coords"])
def test_placing_another_conformation_caches_its_chi(placement: str, reorder: str) -> None:
    """A Context whose ends turned takes another conformation: every residue's
    cached chi is the new coordinates', the two ends included."""
    ff, here = _build(reorder)
    _, there = _build(reorder)
    _turn_sidechains(there)
    start, turned = np.asarray(here.coords), np.asarray(there.coords)
    assert _end_turn(ff, start, turned) > 0.1
    assert stale_chi(there, ff) == []  # accepted moves at both ends measured them

    if placement == "set_positions":
        there.set_positions(start, frame_offset=np.asarray(here.frame_offset))
    else:
        there.coords = start
    assert stale_chi(there, ff) == []
    np.testing.assert_array_equal(cached_chi(there), cached_chi(here))


@pytest.mark.parametrize("reorder", LAYOUTS)
def test_a_replica_swap_caches_the_chi_each_context_receives(reorder: str) -> None:
    ff, here = _build(reorder)
    _, there = _build(reorder)
    _turn_sidechains(there)
    a, b = get_coords(here), get_coords(there)
    assert _end_turn(ff, a, b) > 0.1
    chi_a, chi_b = cached_chi(here), cached_chi(there)

    swap_context_coordinates(here, there, a, b)
    assert stale_chi(here, ff) == []
    assert stale_chi(there, ff) == []
    np.testing.assert_array_equal(cached_chi(here), chi_b)
    np.testing.assert_array_equal(cached_chi(there), chi_a)


@pytest.mark.parametrize("reorder", LAYOUTS)
def test_a_new_context_reads_what_a_no_op_placement_reads(reorder: str) -> None:
    """After a run, placing its coordinates again (a no-op) and placing them in
    a new Context give the same cached torsions and the same energy, term by
    term."""
    ff, ctx = _build(reorder)
    integ = mcpu_core.Integrator(temperature=1.0, step_size_rad=0.1)
    integ.set_seed(31)
    integ.set_move_weights(0.25, 0.25, 0.5)
    start = np.asarray(ctx.coords)
    integ.run(ctx, 4000)
    coords, offset = np.asarray(ctx.coords), np.asarray(ctx.frame_offset)
    assert _end_turn(ff, start, coords) > 0.01

    ctx.set_positions(coords, frame_offset=offset)
    e_again = ctx.calculate_total_energy(-1)
    _, fresh = _build(reorder)
    fresh.set_positions(coords, frame_offset=offset)
    e_fresh = fresh.calculate_total_energy(-1)

    np.testing.assert_array_equal(cached_chi(fresh), cached_chi(ctx))
    assert stale_chi(fresh, ff) == []
    assert e_fresh == e_again
    assert dict(fresh.energy_breakdown(weighted=True)["by_name"]) == dict(
        ctx.energy_breakdown(weighted=True)["by_name"])


@pytest.mark.parametrize("mode", ["rotamer_library", "continuous"])
def test_a_short_run_turns_both_end_sidechains(mode: str) -> None:
    """Both ends are sidechain move sites: a short run of sidechain moves
    turns every chi of THR 1 and ASN 46, and leaves the cache on the
    coordinates."""
    ff, ctx = _build()
    start = chi_from_coordinates(ff, np.asarray(ctx.coords))
    integ = _turn_sidechains(ctx, mode=mode)
    key = "rotamer" if mode == "rotamer_library" else "sidechain"
    assert integ.move_counts()[key][0] > 0  # accepted
    end = chi_from_coordinates(ff, np.asarray(ctx.coords))
    for r, k in _ends(ctx):
        assert angle_gap(end[r, :k], start[r, :k]).min() > 0.01, (r, end[r], start[r])
    assert stale_chi(ctx, ff) == []
