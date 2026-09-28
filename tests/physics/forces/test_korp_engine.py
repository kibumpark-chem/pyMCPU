"""The compiled KORP terms: energy, incremental delta, and the steric guard.

`test_korp_reference_parity.py` pins the *Python* reference implementation
against the upstream `korpe` binary. This file pins the **C++** one against the
same numbers, and then goes after the part that has no upstream counterpart at
all: the incremental energy the MC integrator actually uses.

The delta is where the risk is. `calculateEnergy` is a straightforward double
loop and either matches upstream or does not. `calculateEnergyChange` skips
every residue pair that moved rigidly *together*, on the grounds that all six
KORP coordinates are then invariant -- and if that classification is ever wrong
the error is small, silent, and accumulates over a run rather than showing up
as an obvious failure. So it is checked two independent ways: against a full
recompute over real moves, and against the same delta with the skip disabled.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

HELPERS = Path(__file__).resolve().parents[1] / "helpers"
if str(HELPERS) not in sys.path:
    sys.path.insert(0, str(HELPERS))

from korp_pdb import parse_backbone  # noqa: E402
from korp_system import build_backbone_system, residue_atom_indices  # noqa: E402

from pymcpu import mcpu_core  # noqa: E402
from pymcpu.forcefields.builders.korp_builder import KorpPotentialBuilder  # noqa: E402
from pymcpu.forcefields.korp_map import load_korp_map, score_structure  # noqa: E402

KORP_GROUP = 7
GUARD_GROUP = 8

REFERENCE_ENERGIES = {
    "CASP12DCsel20/T0860D1.pdb": -3693.586739,
    "CASP12DCsel20/T0860D1_s026m1.pdb": -1004.099531,
    "CASP12DCsel20/T0860D1_s119m1.pdb": -2444.948077,
    "rcd6/1CEO.pdb": -11463.957486,
}

#: Looser than the Python reference's 1e-7 for one reason only: the engine's
#: `Potential::calculateEnergy` returns float, so a total of order 1e4 is
#: quantised at ~1e-3 absolute however carefully it was accumulated. 1e-6
#: relative is still far tighter than any convention error could hide in.
ENGINE_TOLERANCE = 1e-6


def _map_path():
    path = os.environ.get("KORP_MAP_PATH")
    if not path:
        pytest.skip("set KORP_MAP_PATH to the korp6Dv1.bin energy map")
    return Path(path)


def _structure(rel):
    pdb = _map_path().parent / rel
    if not pdb.is_file():
        pytest.skip(f"{rel} not found next to the map (needs the KORP bundle)")
    return pdb


@pytest.fixture(scope="module")
def korp_map():
    return load_korp_map(_map_path())


@pytest.fixture(scope="module")
def engine_map(korp_map):
    return KorpPotentialBuilder.build_map(korp_map)


def _assemble(engine_map, coords, names, res_seq, chain_ids, *, with_guard=False):
    n_res = len(names)
    system, context = build_backbone_system(coords)
    n_atom, ca_atom, c_atom = residue_atom_indices(n_res)

    if with_guard:
        # Inserted BEFORE the KORP term on purpose: System::evaluateDeltaEnergy
        # walks the potentials in insertion order and stops at the first hard
        # rejection, so a clashing proposal never pays for the 16 A pair loop.
        guard = KorpPotentialBuilder.build_steric_guard(
            ca_atom=ca_atom, res_seq=res_seq, chain_ids=chain_ids)
        guard.set_energy_group(GUARD_GROUP)
        system.add_potential(guard)
    else:
        guard = None

    potential = KorpPotentialBuilder.build_pair_potential(
        engine_map, n_atom=n_atom, ca_atom=ca_atom, c_atom=c_atom,
        res_names=names, res_seq=res_seq, chain_ids=chain_ids)
    potential.set_energy_group(KORP_GROUP)
    system.add_potential(potential)
    return system, context, potential, guard


def _korp_energy(context):
    return context.energy_breakdown(weighted=False)["by_group"][KORP_GROUP]


@pytest.mark.parametrize("rel,expected", sorted(REFERENCE_ENERGIES.items()))
def test_engine_matches_reference_korpe(engine_map, rel, expected):
    coords, names, res_seq, chain_ids = parse_backbone(_structure(rel))
    _, context, _, _ = _assemble(engine_map, coords, names, res_seq, chain_ids)
    assert _korp_energy(context) == pytest.approx(expected, rel=ENGINE_TOLERANCE)


def test_engine_agrees_with_the_python_reference(engine_map, korp_map):
    """Same inputs through two independent implementations."""
    coords, names, res_seq, chain_ids = parse_backbone(
        _structure("CASP12DCsel20/T0860D1.pdb"))
    _, context, _, _ = _assemble(engine_map, coords, names, res_seq, chain_ids)
    reference = score_structure(korp_map, coords, names,
                                res_seq=res_seq, chain_ids=chain_ids)
    assert _korp_energy(context) == pytest.approx(reference, rel=ENGINE_TOLERANCE)


def test_energy_is_invariant_under_rigid_motion(engine_map):
    """The property the whole delta design rests on, measured end to end."""
    coords, names, res_seq, chain_ids = parse_backbone(
        _structure("CASP12DCsel20/T0860D1.pdb"))
    _, context, _, _ = _assemble(engine_map, coords, names, res_seq, chain_ids)
    before = _korp_energy(context)

    rng = np.random.default_rng(3)
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q @ np.diag(np.sign(np.diag(r)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1.0
    moved = coords @ q.T + rng.normal(scale=25.0, size=3)

    _, moved_context, _, _ = _assemble(engine_map, moved, names, res_seq, chain_ids)
    assert _korp_energy(moved_context) == pytest.approx(before, rel=1e-5)


def _run_verifier(engine_map, n_res=45, steps=200, seed=11, rigid_skip=True):
    coords, names, res_seq, chain_ids = parse_backbone(
        _structure("CASP12DCsel20/T0860D1.pdb"))
    coords = coords[:n_res]
    names, res_seq = names[:n_res], res_seq[:n_res]
    chain_ids = chain_ids[:n_res]

    _, context, potential, _ = _assemble(
        engine_map, coords, names, res_seq, chain_ids, with_guard=True)
    potential.set_rigid_skip_enabled(rigid_skip)

    integrator = mcpu_core.Integrator(temperature=0.6, step_size_rad=0.05)
    integrator.set_seed(seed)
    integrator.set_move_weights(0.5, 0.5, 0.0)   # backbone-only: no chi angles
    integrator.verify_physics_consistency(context, steps, 1e-3)


def test_incremental_delta_matches_a_full_recompute(engine_map):
    """Over real pivot and KIC moves produced by the integrator itself."""
    _run_verifier(engine_map, rigid_skip=True)


def test_delta_is_consistent_with_the_rigid_skip_disabled(engine_map):
    """Same check with the elision off, so the skip is not grading its own work."""
    _run_verifier(engine_map, rigid_skip=False)


def test_rigid_skip_does_not_change_the_trajectory(engine_map):
    """The elision must not alter sampling -- checked over several seeds.

    The moved-moved skip is exact in real arithmetic. In float32 it is exact
    *almost* always: the table lookup is a step function, so a pair sitting
    within a rounding error of a bin boundary can land in a neighbouring bin
    on a full recompute while the delta assumed no change. That is rare -- it
    showed up on one seed in five over 500 steps -- and when it happens it
    moves the bookkeeping total by ~0.3 out of ~3700.

    Over THIS short, cold run it does not change which moves are accepted, and
    that is all this test establishes. It does not hold in general: at T = 8
    over 1e5 steps a hidden bin flip mis-scores accepted moves by several
    units and the trajectories diverge, which is why the elision is now off
    by default -- see test_korp_exact_delta.py for the long, hot guard.
    """
    coords, names, res_seq, chain_ids = parse_backbone(
        _structure("CASP12DCsel20/T0860D1.pdb"))
    coords, names = coords[:60], names[:60]
    res_seq, chain_ids = res_seq[:60], chain_ids[:60]

    def run(rigid_skip, seed):
        _, context, potential, _ = _assemble(
            engine_map, coords, names, res_seq, chain_ids, with_guard=True)
        potential.set_rigid_skip_enabled(rigid_skip)
        integrator = mcpu_core.Integrator(temperature=0.6, step_size_rad=0.05)
        integrator.set_seed(seed)
        integrator.set_move_weights(0.5, 0.5, 0.0)
        integrator.run(context, 400)
        # The RECOMPUTED energy, not the incrementally maintained one: the
        # question is whether the two runs ended in the same place.
        return list(integrator.last_accept_bits()), context.calculate_total_energy(-1)

    for seed in (3, 17, 42, 101, 2024):
        bits_on, energy_on = run(True, seed)
        bits_off, energy_off = run(False, seed)
        assert bits_on == bits_off, f"seed {seed}: the skip changed the trajectory"
        assert energy_on == pytest.approx(energy_off, rel=1e-6), (
            f"seed {seed}: the two runs ended at different structures"
        )
