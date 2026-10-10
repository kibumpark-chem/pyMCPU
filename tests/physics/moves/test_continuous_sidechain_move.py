"""Continuous sidechain move: now perturbs all chi1-4, not only chi1.

The continuous move (``MCIntegrator::apply_sidechain_at``,
``sidechain_move_mode="continuous"``) used to rigidly rotate the whole
sidechain block about the CA-CB axis -- mathematically equivalent to
perturbing chi1 alone, leaving chi2/chi3/chi4 (for multi-chi residues like
ARG, LYS, GLN, GLU, MET) untouched. It has been rewritten to draw an
independent, zero-mean Gaussian delta for each of the residue's chi angles
and apply the same per-chi cascading rotation the rotamer-library move uses
(``apply_chi_cascade``), so a single move now perturbs every chi
simultaneously.

Purely an internal check of pyMCPU's own move-proposal machinery --
mirrors ``test_rotamer_move.py``'s structure for the sibling move.
"""

from __future__ import annotations

import pytest

from pymcpu import mcpu_core
from tests.fixtures.context_builders import build_raw_context

pytestmark = pytest.mark.slow


def _gly_ala_residues(ctx) -> list[int]:
    system = ctx.get_system()
    return [
        r
        for r in range(system.get_num_residues())
        if system.get_torsions_per_residue()[r] == 0
    ]


def test_debug_force_sc_at_gly_ala_rejects() -> None:
    """The fast-reject criterion moved from BlockIndices (num_sc_atoms<2) to
    residue topology (ntorsions<=0) -- confirm it's still equivalent for
    Gly/Ala."""
    ctx, _ = build_raw_context(virtual_amide_h=True)
    no_chi = _gly_ala_residues(ctx)
    assert no_chi, "test PDB should contain a Gly or Ala residue"
    integ = mcpu_core.Integrator(temperature=300.0, step_size_rad=0.1)
    integ.set_seed(7)
    assert integ.debug_force_sc(ctx, no_chi[0]) is False


def test_continuous_move_perturbs_multiple_chi_angles_in_one_step() -> None:
    """Steps one MC move at a time (continuous mode forced); whenever an
    accepted step is a Sidechain-slot move on a residue with >= 2 chi
    angles, at least 2 of its chi values must have changed simultaneously --
    this is impossible under the old chi1-only rigid rotation, where at most
    one chi index could ever change in a single move."""
    ctx, _ = build_raw_context(virtual_amide_h=True)
    system = ctx.get_system()
    ntorsions = system.get_torsions_per_residue()

    integ = mcpu_core.Integrator(temperature=0.6, step_size_rad=0.3)
    integ.set_sidechain_move_mode("continuous")
    integ.set_seed(99)

    n_res = system.get_num_residues()
    found_multi_chi_change = False
    for _ in range(1500):
        before = [tuple(t.chi_angles) for t in ctx.get_state().sidechain_torsions]
        integ.run(ctx, 1)
        if integ.last_move_kind() != "Sidechain":
            continue
        after = [tuple(t.chi_angles) for t in ctx.get_state().sidechain_torsions]
        for r in range(n_res):
            if ntorsions[r] < 2:
                continue
            n_changed = sum(
                1 for k in range(ntorsions[r]) if before[r][k] != after[r][k]
            )
            if n_changed >= 2:
                found_multi_chi_change = True
                break
        if found_multi_chi_change:
            break

    assert found_multi_chi_change, (
        "no accepted Sidechain-slot step changed 2+ chi angles for any "
        "multi-chi residue over 1500 steps"
    )
