"""SoA (structure-of-arrays) coordinate storage regression: determinism and
physics unchanged.

All three tests here are self-vs-self: same-seed determinism, the internal
``verify_physics_consistency`` checker, and a frozen regression baseline of
pyMCPU's *own* output (not a legacy MCPU comparison, despite the baseline
history below referencing legacy hbonds.h parity work as the reason the
physics changed). The Python-exposed ``state.coords`` API roundtrip is a
pure software-contract check with no physics content, so it lives separately
in ``tests/config/test_coords_api.py``.
"""

from __future__ import annotations

import pytest

from pymcpu import mcpu_core
from tests.fixtures.context_builders import (
    ATOL,
    build_test_context,
    require_safe_math_for_accept_determinism,
)
from tests.physics.helpers.hotpath_runs import run_hotpath

pytestmark = pytest.mark.slow

# Frozen regression baseline of the CURRENT engine's own output under a fixed
# recipe (seed=42, 100-step warmup then reseed to 42+1_000_003, 1000 measured
# steps, temperature=0.6, step_size_rad=0.1, pooled proposal
# enabled) -- not independently derivable from physics first principles or
# from legacy MCPU. This is the fourth capture of this baseline; each prior
# refresh followed a genuine physics-changing engine fix (stale proposal
# torsions on distorted residues under pooled reuse; ang_CACA cross-chain
# operands vs. wrong radian/degree threshold; the four Ramachandran
# hard-reject gates plus relocating the secondary-structure 'H'-gate; and
# Integrator::set_seed not resetting angle_dist's cached spare Gaussian on
# mid-run reseed) -- see docs/hbond_legacy_parity.md for the hbond-angle fix
# specifically. Full history retired from this comment into that doc; refresh
# by literally re-running the recipe above via `run_hotpath`, not by
# hand-deriving a number. Values are host/-march=native sensitive.
#
# Refreshed 2026-08-18 after a genuine Mu-potential fix: mu_builder.py's
# skip_local_contact_range boundary used a strict "<" where legacy's
# CheckCorrelation (init.h) uses "<=", so i,i+4 residue pairs -- the
# canonical alpha-helix spacing -- wrongly fell through into the long-range
# contact-eligible branch instead of being excluded. Confirmed against the
# legacy dbfold_actin binary directly (fold_potential_mpi_actin, with
# READ_POTENTIAL=1 to load the real pretrained potential table rather than
# deriving one from the input structure): on a helical decoy-benchmark
# protein (1a32), raw Mu went from 36% off legacy (-76.85 vs -56.53) to ~1%
# off (-55.976 vs -56.53) after this fix; a beta-hairpin control (chignolin)
# was unaffected (-8.645 vs -8.65 legacy, matching before and after), as
# expected since it has no i,i+4 pairs within the contact cutoff.
#
# Refreshed again 2026-08-19 after a genuine sidechain-torsion (chi angle)
# fix: computeTorsions()/recompute_sidechain_torsion() computed chi
# POSITIONALLY (Nth stored sidechain atom = Nth atom in the dihedral chain),
# which is wrong for any branched sidechain -- confirmed concretely for ILE
# (real chi2 is CA-CB-CG1-CD1, positional formula read CA-CB-CG1-CG2, the
# 3rd stored sidechain atom, an improper angle) and for atom_reorder_mode
# consistency (TYR chi flipped between "off" and "init_only"). Fixed by
# resolving chi atoms by NAME (System::getChiAtomIndices(), sourced from
# legacy's amino_torsion.data via standard_amino_acids.json's "chi_atoms")
# instead of positional sc_start+k offsets. Verified: engine's ILE chi2 now
# matches an independent mdtraj CA-CB-CG1-CD1 dihedral computation to 1e-6
# rad across all 30 ILEs in acta.pdb, and chi angles are now bit-identical
# across atom_reorder_mode settings for every residue (see
# tests/physics/forcefield/test_chi_topology.py). This was the sixth capture
# of this baseline.
#
# Refreshed again 2026-08-19 after two compounding, genuine sidechain-move
# changes: (1) Integrator's default sidechain_move_mode_ flipped from
# Continuous to RotamerLibrary (a discrete rotamer-library proposal now
# runs by default in the "Sidechain" move slot instead of the old
# CA-CB-axis rotation -- see MCIntegrator::apply_rotamer_at), and (2) the
# Continuous move itself was rewritten to perturb every chi angle
# (chi1..chi4) simultaneously via the same per-chi cascading rotation,
# instead of only chi1 -- so even a hypothetical run pinned to
# sidechain_move_mode="continuous" would need a fresh baseline too, since
# the RNG-draw count per sidechain-move attempt changed from a fixed 1 to
# 1-4 depending on the residue's chi count. run_hotpath (this file's harness)
# never calls set_sidechain_move_mode, so it exercises whatever the engine's
# current default is. This was the seventh capture of this baseline.
#
# EIGHTH capture, 2026-09-16. E_HBOND only; ACCEPT was already 258 and did not
# move. The previous value, -175.09228515625, is NOT REPRODUCIBLE from any
# commit, which is why this refresh is a re-capture rather than a bisected fix:
#
#   * It was recorded by e57ac48 ("Checkpoint working tree as the known-good
#     pre-optimization baseline"), a 126-file / 63-new-file hand-staged commit.
#     That commit added the import at pymcpu/forcefields/mcpu.py:34
#     (`from .builders.rotamer_builder import RotamerLibraryBuilder`) but did
#     not add rotamer_builder.py, which stayed untracked until 5fd9433 --
#     NINE commits later.
#   * So `import pymcpu` fails at e57ac48 and at the eight commits after it.
#     The value came from the author's working tree, which had the untracked
#     file; no commit reproduces it, and the range is not bisectable.
#   * Nothing could have noticed: the file was present in every developer's
#     working tree, and the only CI workflow at the time was lint.yml, whose
#     two steps are `pip install ruff` and `ruff check .` -- it never installed
#     the package, built the extension, imported pymcpu, or ran a test.
#
# The value below is deterministic and build-independent: -176.66957092285156
# was obtained identically from GCC 8.5 and GCC 12.2, at -march=haswell and
# -march=skylake-avx512, with geometry arithmetic in float and in double, and
# from both the pre- and post-Phase-6 extensions. It is a stable property of
# the current engine, not an artifact of one build.
#
# NINTH capture, 2026-09-28, after the KIC loop-closure fix (CHANGELOG
# [Unreleased] -> Fixed). KIC now takes its closure targets from the start
# structure, back-substitutes without poles, rounds its anchors to float,
# skips prolines and irreversible windows and uses an orientation-free
# Jacobian, and N-terminal psi pivots no longer swing O(r), so every
# trajectory that runs these moves changes. The previous values, ACCEPT 258
# and E_HBOND -176.66957092285156, still come out of the unfixed engine (GCC
# 14.2, same recipe). Replaying both engines one step at a time, the
# coordinates are bit-identical through warmup step 3; step 4 is an accepted
# KIC move that moves the same 16 atoms in both, and 11 of them land up to
# 7.6e-6 A apart (float32 rounding of the new closure arithmetic), after which
# the runs separate chaotically. Captured by re-running the recipe above via
# `run_hotpath` on the fixed engine.
#
# A GCC 8.5 build gave ACCEPT 271 and E_HBOND -180.06430053710938 here until
# the rotation fix of 2026-10-02 (rigid rotations done in double and rounded
# once), and gives exactly these values since.
BASELINE_ACCEPT = 264
BASELINE_E_HBOND = -177.14263916015625

# See tests/physics/test_hbond_delta_hotpath.py's DETERMINISM_ATOL for why
# this is an empirical repeatability allowance, not a physics constant.
DETERMINISM_ATOL = 1e-5

# Looser than DETERMINISM_ATOL: this compares against a baseline frozen at an
# earlier point in time, so it must tolerate cross-BUILD float drift, not just
# same-run-to-run FP noise.
#
# Corrected 2026-09-16: this used to say "-march=native float drift". That is
# measurably wrong -- the ISA baseline does NOT change results. v3 versus v4
# was verified bit-identical across chignolin and actin at two seeds
# (docs/arch_baseline_decision.md), and 8.5-at-haswell versus
# 8.5-at-skylake-avx512 gives the same accept count. What DOES drift is the
# COMPILER: GCC's default -ffp-contract=fast fuses multiply-adds differently
# between versions (895 FMA instructions under GCC 8.5 versus 1281 under
# GCC 12.2 on identical source), which perturbs move-generation arithmetic and
# amplifies chaotically: measured at ~145 steps per e-folding.
BASELINE_ATOL = 1e-4


def test_soa_coords_deterministic_repeat() -> None:
    require_safe_math_for_accept_determinism()
    a = run_hotpath(seed=42, steps=150, warmup=0, step_size_rad=0.1)
    b = run_hotpath(seed=42, steps=150, warmup=0, step_size_rad=0.1)
    assert a.bits == b.bits
    assert a.accept == b.accept
    assert a.energy == pytest.approx(b.energy, abs=DETERMINISM_ATOL)
    assert a.e_hbond == pytest.approx(b.e_hbond, abs=DETERMINISM_ATOL)
    assert a.proxy_stat("mu_num_pair_distance_checks") == b.proxy_stat(
        "mu_num_pair_distance_checks"
    )
    assert a.proxy_stat("mu_num_pairs_within_rcut") == b.proxy_stat(
        "mu_num_pairs_within_rcut"
    )


def test_soa_actin_verify() -> None:
    ctx, _ = build_test_context(with_qbias=False)
    integ = mcpu_core.Integrator(temperature=300.0, step_size_rad=0.05)
    integ.verify_physics_consistency(ctx, num_steps=30, atol=ATOL)


def test_soa_baseline_accept_and_hbond_energy() -> None:
    require_safe_math_for_accept_determinism()
    r = run_hotpath(seed=42, steps=1000, warmup=100, step_size_rad=0.1)
    assert r.accept == BASELINE_ACCEPT
    assert r.e_hbond == pytest.approx(BASELINE_E_HBOND, abs=BASELINE_ATOL)
