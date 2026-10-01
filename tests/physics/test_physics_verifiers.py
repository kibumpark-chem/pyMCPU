"""Coverage for the two ``PhysicsVerifier`` / Mu self-check entry points that
were bound but never exercised.

Both verify an invariant the incremental path can genuinely violate, which is
why they are worth pinning rather than deleting:

* ``PhysicsVerifier.verify_all_potential_deltas`` -- the vector form of the
  per-group delta check. ``verify_potential_delta`` (the scalar form) is already
  covered by ``test_potential_delta_consistency.py``; this pins that the "all"
  form agrees with it for every added energy group, so the two cannot drift.
* ``MuPotential.verify_layered_eval_consistency`` -- compares
  ``eval_pair_layered_v1`` (``use_topo_flags_=false``) against
  ``eval_pair_layered_v2`` (``use_topo_flags_=true``) over all pairs x six
  test r^2 values. A disagreement would mean the topo-flag fast path and the
  reference path score contacts differently, which is exactly the class of bug
  that produced the accepted-state clash documented in
  ``pymcpu/simulation.py``'s steric-clash handler.
"""

from __future__ import annotations

import numpy as np
import pytest

from pymcpu import mcpu_core
from tests.fixtures.context_builders import ATOL
from tests.physics.helpers.minimal_system_builders import setup_minimal_bb_system


def _bb_proposal(context, n_res: int, n_atoms: int):
    """A single-residue backbone torsion perturbation, as used by
    ``test_potential_delta_consistency.py``."""
    coords = np.zeros((3, n_atoms), dtype=np.float32)
    context.set_positions(coords)

    old_state = mcpu_core.State(context.get_state())
    proposed_state = mcpu_core.State(context.get_state())
    proposed_state.backbone_torsions[1].phi = 0.5
    proposed_state.backbone_torsions[1].psi = -0.3

    patch = mcpu_core.ProposalPatch(n_atoms)
    patch.is_valid = True
    patch.add_distorted_bb_residue(1)
    return old_state, proposed_state, patch


def test_verify_all_potential_deltas_covers_every_added_group() -> None:
    """The vector form must return one check per added energy group, all
    passing, and each must agree with the scalar ``verify_potential_delta``."""
    n_res, n_atoms = 4, 16
    system, context = setup_minimal_bb_system(n_res, n_atoms)

    # Two groups so the "all" form has something to iterate over.
    bb_triplet = mcpu_core.TripletPotential([0.0] * (n_res * 1296))
    bb_triplet.set_energy_group(2)
    system.add_potential(bb_triplet)

    old_state, proposed_state, patch = _bb_proposal(context, n_res, n_atoms)

    checks = mcpu_core.PhysicsVerifier.verify_all_potential_deltas(
        context, old_state, proposed_state, patch, ATOL
    )

    # Non-vacuous: the verifier must actually have looked at something.
    assert len(checks) >= 1, "verify_all_potential_deltas returned no checks"

    groups = {int(c.energy_group) for c in checks}
    assert 2 in groups, f"energy group 2 not checked; got {sorted(groups)}"

    for check in checks:
        assert check.passed, (
            f"energy_group={check.energy_group}: "
            f"delta_incremental={check.delta_incremental} "
            f"delta_direct={check.delta_direct} message={check.message}"
        )
        # The two deltas are what the check compares; pin the tolerance too so
        # a future change to `passed` cannot mask a real divergence.
        assert abs(check.delta_incremental - check.delta_direct) <= ATOL

    # Agreement with the already-trusted scalar form, group by group.
    for group in sorted(groups):
        scalar = mcpu_core.PhysicsVerifier.verify_potential_delta(
            context, old_state, proposed_state, patch, group, ATOL
        )
        vector = next(c for c in checks if int(c.energy_group) == group)
        assert scalar.passed == vector.passed
        assert scalar.delta_incremental == pytest.approx(
            vector.delta_incremental, abs=1e-6
        )
        assert scalar.delta_direct == pytest.approx(vector.delta_direct, abs=1e-6)


def test_mu_layered_eval_v1_matches_v2(chignolin_pdb_path, capfd) -> None:
    """The topo-flag fast path (v2) must score every pair identically to the
    reference path (v1).

    Uses chignolin explicitly rather than the shared ``chignolin_context``
    fixture, which defaults to actin: this verifier is O(N^2 x 6), which is
    ~18k pair evaluations at N=77 but ~26M at N=2943.
    """
    mdtraj = pytest.importorskip("mdtraj")
    from pymcpu.forcefields.mcpu import MCPUForceField

    traj = mdtraj.load(chignolin_pdb_path)
    heavy = traj.atom_slice(traj.topology.select("not element H"))
    forcefield = MCPUForceField(heavy, param_set="mcpu08")
    system = forcefield.create_system(heavy.topology)
    context = mcpu_core.Context(system)
    context.set_positions((forcefield.coords[0] * 10.0).T.astype(np.float32))

    mu = next(
        (f for f in system.get_potentials() if hasattr(f, "verify_layered_eval_consistency")),
        None,
    )
    assert mu is not None, "no MuPotential found on the system"

    # Guard against a VACUOUS pass: the verifier early-returns (printing an
    # ERROR) when Layer-1 meta or topo_flag_ is missing, in which case it
    # compares nothing at all. Assert both preconditions via the bound
    # size properties before trusting a silent run.
    assert float(mu.type_params_size_kb) > 0.0, "Layer-1 type_params_ not populated"
    assert float(mu.topo_flag_size_mb) > 0.0, "topo_flag_ not populated"

    # A clean run is SILENT unless MCPU_VERBOSE is set (and that env var is
    # cached in a function-local static on first use, so it cannot be flipped
    # from here). Verified manually with MCPU_VERBOSE=1 on this system that the
    # verifier does report "eval_pair_layered consistency (v1 vs v2): PASS",
    # i.e. it really does walk the pairs -- the precondition assertions above
    # are what keep this test from passing vacuously.
    capfd.readouterr()  # drop setup chatter
    mu.verify_layered_eval_consistency()
    err = capfd.readouterr().err

    assert "LAYERED_MISMATCH" not in err, f"v1/v2 disagree:\n{err}"
    assert "ERROR" not in err, f"verifier reported an error:\n{err}"
