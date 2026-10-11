"""Fixed-seed ``Integrator`` run harness for ``test_coords_soa.py``.

``run_hotpath`` builds a context, runs the ``Integrator`` for N steps under a
fixed seed, and reads back the accept-bit stream, the energies and the
``neighbor_proxy_stats()`` counters, for same-seed determinism checks and the
frozen baseline.
"""

from __future__ import annotations

from dataclasses import dataclass

from pymcpu import mcpu_core
from tests.fixtures.context_builders import build_test_context


@dataclass(frozen=True)
class HotpathRun:
    """Result of one fixed-seed ``Integrator`` run, for self-vs-self
    determinism comparisons."""

    bits: list[int]
    accept: int
    energy: float
    e_hbond: float
    proxy: dict[str, object]


def run_hotpath(
    *,
    seed: int,
    steps: int,
    warmup: int = 0,
    reseed_offset: int = 1_000_003,
    temperature: float = 0.6,
    step_size_rad: float = 0.05,
) -> HotpathRun:
    """Build a fresh test context, run the Integrator, and collect stats.

    With ``warmup`` > 0: runs a warmup phase under ``seed`` first, then
    reseeds to ``seed + reseed_offset`` and resets neighbor proxy stats
    before the measured phase. This gives a clean, reproducible cut between
    warmup noise and the phase actually being measured -- see
    ``set_seed``'s Gaussian-cache-reset fix in the coords_soa baseline
    history for why a mid-run reseed needs to be genuinely clean.
    """
    ctx, _ = build_test_context(with_qbias=False)
    integ = mcpu_core.Integrator(temperature=temperature, step_size_rad=step_size_rad)
    integ.set_seed(seed)

    if warmup:
        integ.run(ctx, warmup)
        integ.set_seed(seed + reseed_offset)
    ctx.reset_neighbor_proxy_stats()

    integ.run(ctx, steps)
    bits = list(integ.last_accept_bits())
    proxy = dict(ctx.neighbor_proxy_stats())
    return HotpathRun(
        bits=bits,
        accept=int(sum(bits)),
        energy=float(ctx.get_state().current_energy),
        e_hbond=float(ctx.calculate_total_energy(4)),  # group 4 == hbond, see EnergyGroup
        proxy=proxy,
    )
