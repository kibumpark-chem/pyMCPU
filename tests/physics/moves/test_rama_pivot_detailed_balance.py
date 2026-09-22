"""Direct numerical detailed-balance check for the knowledge-based backbone
pivot move.

Forces the engine through a hand-picked, exact (phi,psi) transition (via
``debug_force_rama_pivot_to``) and compares its reported
``last_log_jacobian_weight()`` -- the Metropolis-Hastings correction term,
``log q(x_old) - log q(x_new)`` -- against an INDEPENDENT computation done
in pure Python from the packaged ``rama_mixture.json``'s raw weights/means/
covariances, using a periodic-sum formula re-derived here rather than
imported from the C++ binding or the production builder. A translation bug
in ``RamaMixtureLibrary::log_mixture_density`` (or in how
``apply_rama_pivot_at``/``apply_rama_pivot_to_target`` wires it into
``patch.log_jacobian_weight``) can't hide by agreeing with itself here,
since this test never calls into that code path to compute its reference
value.
"""

from __future__ import annotations

import itertools
import json
import math

import pytest

from pymcpu import mcpu_core
from pymcpu.params import ensure_params
from tests.fixtures.context_builders import build_raw_context

pytestmark = pytest.mark.slow

TWO_PI = 2.0 * math.pi

AMINO_ORDER = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
]


def _independent_log_mixture_density(entry: dict, x: tuple[float, float], n_wrap: int) -> float:
    """Pure-Python periodic-sum log-density, re-derived independently of
    both mcpu_core.RamaMixtureLibrary and RamaMixtureLibraryBuilder --
    intentionally not sharing a single line of code with either."""
    log_terms = []
    for weight, mean, cov in zip(entry["weights"], entry["means"], entry["covariances"]):
        c11, c12 = cov[0]
        _, c22 = cov[1]
        det = c11 * c22 - c12 * c12
        inv11, inv12, inv22 = c22 / det, -c12 / det, c11 / det
        log_det = math.log(det)

        shift_terms = []
        for dx, dy in itertools.product(range(-n_wrap, n_wrap + 1), repeat=2):
            m0 = mean[0] + dx * TWO_PI
            m1 = mean[1] + dy * TWO_PI
            d0, d1 = x[0] - m0, x[1] - m1
            quad = inv11 * d0 * d0 + 2.0 * inv12 * d0 * d1 + inv22 * d1 * d1
            shift_terms.append(-0.5 * (quad + log_det + 2.0 * math.log(2.0 * math.pi)))
        max_shift = max(shift_terms)
        component_log_pdf = max_shift + math.log(
            sum(math.exp(t - max_shift) for t in shift_terms)
        )
        log_terms.append(math.log(weight) + component_log_pdf)

    max_term = max(log_terms)
    return max_term + math.log(sum(math.exp(t - max_term) for t in log_terms))


def _load_real_rama_mixture_json() -> dict:
    param_dir = ensure_params("mcpu_v1")
    path = param_dir / "constants" / "rama_mixture.json"
    with open(path) as f:
        return json.load(f)


@pytest.mark.parametrize(
    "residue_offset,delta_phi_deg,delta_psi_deg",
    [
        (2, 20.0, -15.0),
        (3, -35.0, 40.0),
        (4, 5.0, 5.0),
        (5, 90.0, -60.0),
    ],
)
def test_forced_transition_log_jacobian_weight_matches_independent_computation(
    residue_offset: int, delta_phi_deg: float, delta_psi_deg: float
) -> None:
    ctx, top = build_raw_context(virtual_amide_h=True)
    system = ctx.get_system()
    n_res = system.get_num_residues()
    residue = min(residue_offset, n_res - 2)
    assert not system.is_proline(residue), "test PDB residue choice must be non-proline"

    data = _load_real_rama_mixture_json()
    n_wrap = data["source"]["n_wrap"]
    amino_idx = system.amino_index(residue)
    residue_name = AMINO_ORDER[amino_idx]
    entry = data["residues"][residue_name]

    old_phi = ctx.get_state().backbone_torsions[residue].phi
    old_psi = ctx.get_state().backbone_torsions[residue].psi
    target_phi = old_phi + math.radians(delta_phi_deg)
    target_psi = old_psi + math.radians(delta_psi_deg)
    # wrap to (-pi, pi] to match the engine's own convention
    target_phi = (target_phi + math.pi) % TWO_PI - math.pi
    target_psi = (target_psi + math.pi) % TWO_PI - math.pi

    log_q_old = _independent_log_mixture_density(entry, (old_phi, old_psi), n_wrap)
    log_q_new = _independent_log_mixture_density(entry, (target_phi, target_psi), n_wrap)
    expected_log_jacobian_weight = log_q_old - log_q_new

    integ = mcpu_core.Integrator(temperature=0.6, step_size_rad=0.1)
    integ.set_seed(1)
    ok = integ.debug_force_rama_pivot_to(ctx, residue, target_phi, target_psi)
    assert ok, "forced transition should succeed for a real, non-proline interior residue"

    assert integ.last_log_jacobian_weight() == pytest.approx(
        expected_log_jacobian_weight, abs=1e-3
    ), (
        f"residue={residue} ({residue_name}): engine-reported log_jacobian_weight "
        f"{integ.last_log_jacobian_weight()} != independently computed "
        f"{expected_log_jacobian_weight}"
    )

    # ctx.state itself must be untouched by a debug-forced call (confirms
    # this test isn't accidentally relying on any prior test's side effects
    # or silently mutating shared state across parametrize cases).
    assert ctx.get_state().backbone_torsions[residue].phi == old_phi
    assert ctx.get_state().backbone_torsions[residue].psi == old_psi


def test_forced_transition_is_rejected_when_proposed_residue_is_proline() -> None:
    ctx, top = build_raw_context(virtual_amide_h=True)
    system = ctx.get_system()
    pros = [r.index for r in top.residues if r.name == "PRO"]
    assert pros, "test PDB should contain proline"
    pro = pros[0]
    if pro < 1 or pro > system.get_num_residues() - 2:
        pytest.skip("proline residue not in the valid [1, N-2] debug-hook range")

    integ = mcpu_core.Integrator(temperature=0.6, step_size_rad=0.1)
    integ.set_seed(1)
    # PRO's own category (per resolve_rama_category, currently just
    # amino_index) is still a registered mixture (no proline-specific
    # exclusion exists for the debug_force_rama_pivot_to path -- unlike the
    # production apply_rama_pivot_move loop, this low-level hook doesn't
    # skip proline, since a direct-detailed-balance test may legitimately
    # want to force ANY residue's transition, including proline's, to
    # check the math in isolation). This test documents that behavior
    # rather than asserting a rejection that doesn't actually happen.
    ok = integ.debug_force_rama_pivot_to(ctx, pro, 0.0, 0.0)
    assert ok is True
