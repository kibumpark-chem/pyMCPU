"""Exact-formula unit tests for the two knowledge-based triplet potentials
(backbone ``TripletPotential`` and sidechain ``SidechainTripletPotential``).

Unlike the finite/sanity checks in ``test_energy_consistency.py``, these
pin an exact expected energy computed by hand from known input parameters
on a hand-built minimal system with all torsions forced into bin 0 -- so
they're precise numeric tests, not smoke tests (hence the separate file;
the old ``test_force_energy.py`` mixed both kinds under a "smoke tests"
docstring, which was misleading).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

# No ``pytestmark = pytest.mark.slow`` here: unlike the chignolin-fixture
# tests in this cluster, these build a tiny 16-atom synthetic system directly
# (no PDB parsing / real forcefield construction), so they're cheap.
from pymcpu import mcpu_core
from tests.fixtures.context_builders import ATOL
from tests.physics.helpers.minimal_system_builders import (
    setup_minimal_bb_system,
    setup_minimal_sc_system,
)

_PI = math.pi


def _set_zero_bb_torsions(context: mcpu_core.Context, n_res: int) -> None:
    """Force phi=psi=-pi, pCA=bCA=0 (bin 0) on every interior residue."""
    state = context.get_state()
    for r in range(1, n_res):
        t = state.backbone_torsions[r]
        t.phi = -_PI
        t.psi = -_PI
        t.p_ca = 0.0
        t.b_ca = 0.0
        state.backbone_torsions[r] = t


def _set_zero_sc_torsions(context: mcpu_core.Context, n_pos: int) -> None:
    """Force all four chi angles to -pi (bin 0) on every sidechain position."""
    state = context.get_state()
    for r in range(1, n_pos + 1):
        sc = state.sidechain_torsions[r]
        sc.chi_angles = [-_PI, -_PI, -_PI, -_PI]
        state.sidechain_torsions[r] = sc


def test_bb_triplet_known_bins_sum_to_expected_energy() -> None:
    """Three nonzero bin-0 params must combine to sum(params)/1000.

    ``TripletPotential.cpp`` divides its raw table lookup by 1000 before
    returning (comment: "Must scale down by 1000 to match Total E") -- so
    -2500 + -3500 + -4000 = -10000 at 3 residue positions (all landing in
    bin 0) becomes -10.0.
    """
    n_res = 4
    n_atoms = 16
    system, context = setup_minimal_bb_system(n_res, n_atoms)
    context.set_use_legacy_weights(False)

    n_pos = 3
    flat_size = n_pos * 1296
    params = [0.0] * flat_size
    params[0] = -2500.0
    params[1296] = -3500.0
    params[2592] = -4000.0

    potential = mcpu_core.TripletPotential(params)
    potential.set_energy_group(2)
    system.add_potential(potential)

    coords = np.zeros((3, n_atoms), dtype=np.float32)
    context.set_positions(coords)
    _set_zero_bb_torsions(context, n_res)

    energy = context.calculate_total_energy(2)
    # Freshly re-derived by running this exact recipe against the current
    # engine: matches the pre-rewrite pinned value (-10.0) exactly.
    assert energy == pytest.approx(-10.0, abs=ATOL)


def test_sc_triplet_known_bins_sum_to_expected_energy() -> None:
    """Same /1000 scaling as the BB triplet potential, confirmed directly
    in ``SideChainTripletPotential.cpp`` (same "Must scale down by 1000.0"
    comment): -1200 + -2400 + -3600 = -7200 at 3 sidechain positions (bin 0)
    becomes -7.2.
    """
    n_res = 4
    n_atoms = 16
    system, context = setup_minimal_sc_system(n_res, n_atoms)
    context.set_use_legacy_weights(False)

    n_pos = 3
    stride_res = 20736
    flat_size = n_pos * stride_res
    params = [0.0] * flat_size
    params[0 * stride_res] = -1200.0
    params[1 * stride_res] = -2400.0
    params[2 * stride_res] = -3600.0

    potential = mcpu_core.SidechainTripletPotential(params)
    potential.set_energy_group(3)
    system.add_potential(potential)

    coords = np.zeros((3, n_atoms), dtype=np.float32)
    context.set_positions(coords)
    _set_zero_sc_torsions(context, n_pos)

    energy = context.calculate_total_energy(3)
    # Freshly re-derived: matches the pre-rewrite pinned value (-7.2) exactly
    # (up to float32 precision -- the raw output is -7.199999809265137).
    assert energy == pytest.approx(-7.2, abs=ATOL)
