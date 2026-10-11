"""Clones of one state must sample independently.

Any framework that splits a walker -- weighted ensemble, forward flux,
adaptive seeding -- hands every child the *same* parent state. If a child's
Monte Carlo stream comes from restoring the parent's saved RNG state rather
than from :func:`pymcpu.sampling.derive_seed`, every sibling executes a
**bitwise identical** trajectory. The split still "succeeds": the children
are distinct objects, the weights divide correctly, every log line looks
healthy. The ensemble is simply one walker counted N times, and only the
statistics show it.

So this lives in ``tests/physics/`` and not with any one framework's tests:
it guards a core promise that core's public API now makes.

See ``tests/physics/test_engine_determinism.py`` for the bare-engine
determinism guarantees this relies on.
"""

from __future__ import annotations

import numpy as np
import pytest

from pymcpu.config import EngineSpec
from pymcpu.sampling import EngineSession, derive_seed

# An arbitrary run-level seed. In WESTPA terms this is `base_seed`; the core
# API calls the three arguments (base_seed, round_index, stream_index).
_BASE_SEED = 99


@pytest.fixture
def spec(engine_spec_factory) -> EngineSpec:
    return engine_spec_factory()


def test_many_siblings_are_pairwise_distinct(spec: EngineSpec) -> None:
    """A split into N > 2 children must not just avoid *one* collision --
    check several siblings pairwise."""
    parent_coords = EngineSession(spec).coords_from_auxref(spec.pdb)
    round_index = 7
    finals = []
    for stream_index in range(6):
        eng = EngineSession(spec)
        eng.set_coords(parent_coords)
        eng.set_seed(derive_seed(_BASE_SEED, round_index, stream_index))
        eng.step(200)
        finals.append(eng.coords())

    for i in range(len(finals)):
        for j in range(i + 1, len(finals)):
            assert not np.array_equal(finals[i], finals[j]), (
                f"siblings {i} and {j} produced identical trajectories "
                "-- this is exactly the silent-clone failure mode this test guards against"
            )
