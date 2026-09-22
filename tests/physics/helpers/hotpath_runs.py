"""Shared fixed-seed Integrator run harness for the delta-energy hotpath
regression tests (mu, hbond, proposal-pooling, SoA-coords).

Each of those test files exercises the same underlying operation: build a
context, run ``Integrator`` for N steps under a fixed seed, and read back the
accept-bit stream plus whichever ``neighbor_proxy_stats()`` counters that
hotpath cares about. Before this module existed, each file reimplemented this
loop by hand with slightly different local variable names and dict-key
strings -- a real source of copy-paste drift flagged by the test survey.
``run_hotpath`` factors the common part; callers still choose their own
seed/steps/skin and read whichever proxy-stat keys are relevant to them via
``HotpathRun.proxy_stat`` (a strict lookup -- see its docstring for why that
matters over a bare ``dict.get``).
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

    def proxy_stat(self, key: str) -> int:
        """Strict lookup into ``neighbor_proxy_stats()``.

        Deliberately raises ``KeyError`` on a typo'd key instead of the
        ``proxy.get(key, 0)`` pattern the old hotpath files used -- with a
        silent-zero default, a typo in one of the two runs being compared
        would degrade the assertion to "0 == 0", which always passes and
        checks nothing.
        """
        return int(self.proxy[key])


def run_hotpath(
    *,
    seed: int,
    steps: int,
    skin: float = 0.0,
    warmup: int = 0,
    reseed_offset: int = 1_000_003,
    temperature: float = 0.6,
    step_size_rad: float = 0.05,
    pooled_proposal: bool = True,
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
    ctx.set_mu_skin(skin)
    integ = mcpu_core.Integrator(temperature=temperature, step_size_rad=step_size_rad)
    integ.set_seed(seed)
    if pooled_proposal and hasattr(integ, "set_use_pooled_proposal"):
        integ.set_use_pooled_proposal(True)

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
