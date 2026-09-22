"""Deterministic, independent random streams for externally-driven sampling.

Any framework that clones a trajectory -- weighted ensemble, forward flux,
adaptive seeding, a branching MSM sampler -- faces the same hazard, and it
is not obvious until it has already cost a study.

When such a framework splits a walker, every child starts from the *same*
parent state. If the child's Monte Carlo stream is seeded by restoring the
parent's saved RNG state verbatim, every child then executes a **bitwise
identical** trajectory: the ensemble looks like N independent walkers and
is really one walker counted N times. Nothing errors, no log line looks
wrong, and the weights still sum to one. Only the statistics are wrong.

So pyMCPU's restart contract is ``(coordinates, step)`` and deliberately
NOT an RNG state, with each clone's stream derived here instead --
independent by construction because the derivation is unique per
``(round, stream)``.

``tests/physics/test_stream_independence.py`` demonstrates both halves: that
two clones with different stream indices diverge, and that restoring a
parent RNG state instead would have produced identical siblings.
"""

from __future__ import annotations

import hashlib
import struct

__all__ = ["derive_seed"]


def derive_seed(base_seed: int, round_index: int, stream_index: int) -> int:
    """Derive an independent 32-bit seed for one stream of one round.

    ``round_index`` is the outer iteration (a WE iteration, an FFS stage, a
    sampling round); ``stream_index`` identifies the walker within it. The
    result is unique per pair, stable across processes, machines and runs,
    and fits the unsigned 32-bit range ``Integrator.set_seed`` expects.

    In WESTPA terms this is ``derive_seed(base_seed, n_iter, seg_id)``.

    **This is a frozen wire format.** Changing the packing or the truncation
    below silently reseeds every existing run -- results would stay
    plausible and stop being comparable to anything published from an
    earlier version. ``tests/unit/test_seed_derivation.py`` pins specific
    outputs as literals for exactly this reason; it is not a
    characterization test to be re-baselined when it fails.
    """
    payload = struct.pack("<qqq", int(base_seed), int(round_index), int(stream_index))
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:4], "little")
