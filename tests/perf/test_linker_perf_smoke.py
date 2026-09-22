"""Wall-clock smoke test: residue energy-masking must not pathologically
slow down MC.

This is a non-functional (performance) guardrail, not a physics-correctness
check -- split out from ``tests/physics/test_linker_energy_mask.py`` (which
covers the masking feature's actual physics/engine consistency) because
timing assertions are a different concern and belong in the perf-smoke
tier of the taxonomy. Marked ``slow`` since it builds the full actin
context and runs several hundred MC steps three times over.
"""

from __future__ import annotations

import time

import pytest

from pymcpu import mcpu_core
from tests.fixtures.context_builders import build_test_context

pytestmark = pytest.mark.slow

#: Deliberately loose bound: masked MC must not be catastrophically slower
#: than unmasked, not "exactly as fast". Wall-clock timing on shared/NFS
#: cluster hosts is inherently noisy, so this is a guardrail against a
#: pathological regression (e.g. an O(n^2) mask-lookup bug), not a tight
#: performance target -- tighten only after re-timing on dedicated hardware.
_PERF_MULTIPLIER = 2.0
_PERF_SLOP_SECONDS = 0.05


def _time_mc_steps(ctx, mask_res: list[int], mode: str | None, steps: int = 200) -> float:
    system = ctx.get_system()
    system.clear_energy_ignored_residues()
    if mode is not None:
        system.set_energy_ignored_residues(mask_res, mode)
    integrator = mcpu_core.Integrator(0.6, 0.1)
    integrator.set_seed(123)  # arbitrary: only wall-clock cost is measured, not the trajectory
    integrator.run(ctx, 20)  # warm up (JIT-like caches, first-touch page faults)
    t0 = time.perf_counter()
    integrator.run(ctx, steps)
    return time.perf_counter() - t0


def test_energy_mask_not_pathologically_slower_than_unmasked() -> None:
    ctx, _ = build_test_context(with_qbias=False)
    n_res = ctx.get_system().get_num_residues()
    lo = n_res // 3
    hi = min(n_res, lo + max(2, n_res // 5))
    mask_res = list(range(lo, hi))

    t_none = _time_mc_steps(ctx, mask_res, None)
    t_ignore = _time_mc_steps(ctx, mask_res, "ignore_all")
    t_clash = _time_mc_steps(ctx, mask_res, "clash_only")

    bound = t_none * _PERF_MULTIPLIER + _PERF_SLOP_SECONDS
    assert t_ignore < bound, f"ignore_all too slow: {t_ignore:.3f}s vs unmasked {t_none:.3f}s"
    assert t_clash < bound, f"clash_only too slow: {t_clash:.3f}s vs unmasked {t_none:.3f}s"
